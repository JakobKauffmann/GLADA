# hierda.py
# -*- coding: utf-8 -*-
"""
Hierarchical Disentangled Domain Adaptation (HierDA) for Audio Deepfake Detection
with Layer-Wise Domain Adversarial Training (LWDA)

Architecture:
  Raw Audio -> RawBoost (train) -> SSL Backbone (frozen or unfrozen with LLRD)
  -> Per-Layer LayerNorm -> Hierarchical Attention Pooling (grouped layers + ASP)
  -> [LWDA: Per-group GRL + Domain Discriminators] (NOVEL)
  -> VIB Bottleneck -> Task Classifier + Speaker Adversary (GRL)

Loss: L_spoof + alpha*L_LWDA + beta*L_KL + gamma*L_speaker + delta*L_CKA

Primary novel contribution: Layer-Wise Domain Adversarial Training (LWDA) — applying
separate domain discriminators with GRL at each SSL layer group to enforce domain
invariance at every level of the representation hierarchy.

All components are individually toggleable for ablation studies.

Author: Jakob Kauffmann
Thesis: Cross-Domain Generalization in Audio Deepfake Detection (SJSU)
"""

import os, random, json, time, warnings, logging, math, copy
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

import librosa
from transformers import AutoModel, AutoConfig
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm.auto import tqdm

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True


# ============================================================================
# BACKBONE REGISTRY
# ============================================================================

BACKBONE_REGISTRY = {
    "wavlm-large": {
        "hf_id": "microsoft/wavlm-large",
        "dim": 1024, "num_layers": 24, "num_groups": 4,
    },
    "xlsr-300m": {
        "hf_id": "facebook/wav2vec2-xls-r-300m",
        "dim": 1024, "num_layers": 24, "num_groups": 4,
    },
    "data2vec-large": {
        "hf_id": "facebook/data2vec-audio-large-960h",
        "dim": 1024, "num_layers": 24, "num_groups": 4,
    },
    "wav2vec2-robust": {
        "hf_id": "facebook/wav2vec2-large-robust",
        "dim": 1024, "num_layers": 24, "num_groups": 4,
    },
    "wavlm-base-plus": {
        "hf_id": "microsoft/wavlm-base-plus",
        "dim": 768, "num_layers": 12, "num_groups": 3,
    },
    "xlsr-1b": {
        "hf_id": "facebook/wav2vec2-xls-r-1b",
        "dim": 1280, "num_layers": 48, "num_groups": 6,
    },
}


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class HierDAConfig:
    # --- Experiment ---
    experiment_name: str = "hierda"
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    seed: int = 42

    # --- Backbone ---
    backbone: str = "wavlm-large"  # key into BACKBONE_REGISTRY

    # --- Architecture ---
    bottleneck_dim: int = 256
    vib_beta: float = 1e-4          # KL weight
    num_speakers: int = 107         # ASVspoof 2019 LA train speakers

    # --- Loss weights ---
    domain_loss_weight: float = 0.1  # alpha
    speaker_loss_weight: float = 0.1 # gamma (max, scheduled)
    cka_loss_weight: float = 0.01    # delta

    # --- GRL scheduling ---
    grl_warmup_epochs: int = 5       # domain GRL warmup
    speaker_grl_warmup_epochs: int = 10  # speaker GRL warmup (slower)

    # --- Ablation switches ---
    use_vib: bool = True              # Toggle VIB (deterministic projection if False)
    use_speaker_adversary: bool = True  # Toggle speaker GRL
    use_global_domain_disc: bool = False  # Single domain disc on z (baseline comparison)
    use_cka_loss: bool = True         # Toggle CKA diversity between groups
    use_hierarchical: bool = True     # Grouped attention (flat weighted sum if False)
    use_rawboost: bool = True         # Toggle RawBoost augmentation
    use_lwda: bool = True             # Per-group domain discriminators (NOVEL)

    # --- LWDA architecture ---
    lwda_disc_hidden: int = 128       # Hidden dim for per-group domain discriminators

    # --- Training ---
    phase1_epochs: int = 5
    phase2_epochs: int = 45
    batch_size: int = 24             # per domain
    effective_batch_size: int = 384
    grad_accum_steps: int = 1
    head_lr: float = 1e-3
    backbone_lr: float = 1e-5       # phase 2 peak
    llrd_decay: float = 0.95
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    patience: int = 15
    use_amp: bool = True
    amp_dtype: str = "bfloat16"     # bfloat16 for A100, float16 for T4

    # --- Data ---
    max_audio_len: int = 64000      # 4s at 16kHz
    rawboost_prob: float = 0.5
    num_workers: int = 0

    # --- Paths (set by notebook after extraction, no defaults) ---
    drive_log_dir: str = "/content/drive/MyDrive/Jakob/Experiments"

    def __post_init__(self):
        bb = BACKBONE_REGISTRY[self.backbone]
        self.hf_model_id = bb["hf_id"]
        self.hidden_dim = bb["dim"]
        self.num_layers = bb["num_layers"]
        self.num_groups = bb["num_groups"]
        # +1 for CNN embedding layer
        self.total_hidden_states = self.num_layers + 1
        # grad accumulation
        actual_batch = self.batch_size * 2  # source + target
        self.grad_accum_steps = max(1, self.effective_batch_size // actual_batch)
        self.total_epochs = self.phase1_epochs + self.phase2_epochs

    def get_run_dir(self):
        return os.path.join(self.drive_log_dir, self.experiment_name, self.run_id)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# RAWBOOST AUGMENTATION (Tak et al. 2022)
# ============================================================================

class RawBoost:
    """
    RawBoost: 3 types of raw waveform augmentation.
    Type 1: Linear/non-linear convolutive noise
    Type 2: Impulsive signal-dependent additive noise
    Type 3: Stationary signal-independent additive noise
    """

    @staticmethod
    def _lnl_convolutive(x, N_f=5, nBands=5, minF=20, maxF=8000, minBW=100,
                         maxBW=1000, minG=-12, maxG=12, sr=16000):
        """Type 1: Linear and non-linear convolutive noise."""
        length = len(x)
        # Generate random notch filter
        freqs = np.random.uniform(minF, maxF, nBands)
        bws = np.random.uniform(minBW, maxBW, nBands)
        gains = np.random.uniform(minG, maxG, nBands)

        # Build frequency response
        f_axis = np.fft.rfftfreq(length, d=1.0/sr)
        H = np.ones(len(f_axis))
        for f0, bw, g in zip(freqs, bws, gains):
            notch = 1.0 - np.exp(-0.5 * ((f_axis - f0) / (bw / 2.0 + 1e-8))**2)
            H *= (1.0 + (10**(g/20.0) - 1.0) * (1.0 - notch))

        # Apply in frequency domain
        X = np.fft.rfft(x)
        X_filtered = X * H
        y = np.fft.irfft(X_filtered, n=length)

        # Non-linear component
        for _ in range(N_f):
            y = y + 0.01 * y * np.random.randn(length)

        return y.astype(np.float32)

    @staticmethod
    def _impulsive_noise(x, prob=0.05, g_sd=0.1):
        """Type 2: Impulsive signal-dependent additive noise."""
        length = len(x)
        mask = np.random.binomial(1, prob, length)
        noise = mask * x * np.random.normal(0, g_sd, length)
        return (x + noise).astype(np.float32)

    @staticmethod
    def _stationary_noise(x, snr_min=15, snr_max=30):
        """Type 3: Stationary signal-independent additive noise."""
        sig_power = np.mean(x**2) + 1e-10
        snr = np.random.uniform(snr_min, snr_max)
        noise_power = sig_power / (10**(snr / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_power), len(x))
        return (x + noise).astype(np.float32)

    @staticmethod
    def augment(x):
        """Apply a random RawBoost augmentation type."""
        aug_type = np.random.randint(1, 4)
        if aug_type == 1:
            return RawBoost._lnl_convolutive(x)
        elif aug_type == 2:
            return RawBoost._impulsive_noise(x)
        else:
            return RawBoost._stationary_noise(x)


# ============================================================================
# GRADIENT REVERSAL LAYER
# ============================================================================

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class PerLayerNorm(nn.Module):
    """Apply LayerNorm to each hidden state before aggregation.
    Normalizes magnitude across layers to prevent gradient dominance."""

    def __init__(self, num_layers, hidden_dim):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(self, hidden_states):
        """hidden_states: list of (B, T, D) tensors"""
        return [norm(h) for norm, h in zip(self.norms, hidden_states)]


class AttentiveStatisticsPooling(nn.Module):
    """Attentive Statistics Pooling: weighted mean + weighted std."""

    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        """x: (B, T, D) -> (B, 2*D)"""
        # Attention weights: (B, T, 1)
        w = F.softmax(self.attention(x), dim=1)
        # Weighted mean: (B, D)
        mu = (x * w).sum(dim=1)
        # Weighted std: (B, D)
        sigma = torch.sqrt(((x - mu.unsqueeze(1))**2 * w).sum(dim=1) + 1e-6)
        return torch.cat([mu, sigma], dim=1)


def compute_cka(X, Y):
    """Linear CKA between two representation matrices.
    X, Y: (B, D) feature matrices. Returns scalar similarity in [0, 1]."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic_xy = (X @ X.T * (Y @ Y.T)).sum()
    hsic_xx = (X @ X.T * (X @ X.T)).sum()
    hsic_yy = (Y @ Y.T * (Y @ Y.T)).sum()
    return hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-8)


class HierarchicalAttentionPooling(nn.Module):
    """
    Group SSL hidden states into K groups.
    Per-group: learnable softmax-weighted layer sum -> ASP.
    Concatenate group outputs -> linear projection.
    CKA diversity loss between group representations.
    """

    def __init__(self, num_hidden_states, hidden_dim, num_groups, output_dim):
        super().__init__()
        self.num_groups = num_groups
        self.hidden_dim = hidden_dim

        # Assign layers to groups (evenly, last group gets remainder)
        layers_per_group = num_hidden_states // num_groups
        self.group_assignments = []
        for g in range(num_groups):
            start = g * layers_per_group
            end = start + layers_per_group if g < num_groups - 1 else num_hidden_states
            self.group_assignments.append(list(range(start, end)))

        # Per-group learnable layer weights
        self.group_weights = nn.ParameterList([
            nn.Parameter(torch.ones(len(layers)) / len(layers))
            for layers in self.group_assignments
        ])

        # Per-group ASP
        self.group_asp = nn.ModuleList([
            AttentiveStatisticsPooling(hidden_dim)
            for _ in range(num_groups)
        ])

        # Projection: concat of K groups * 2*D -> output_dim
        self.projection = nn.Linear(num_groups * 2 * hidden_dim, output_dim)

    def forward(self, hidden_states):
        """
        hidden_states: list of (B, T, D) tensors (already layer-normed)
        Returns: pooled (B, output_dim), cka_loss scalar, group_outputs list of (B, 2*D)
        """
        group_outputs = []

        for g, (layer_ids, weights) in enumerate(
                zip(self.group_assignments, self.group_weights)):
            # Stack group layers: (num_layers_in_group, B, T, D)
            layers = torch.stack([hidden_states[i] for i in layer_ids])
            # Softmax weights: (num_layers, 1, 1, 1)
            w = F.softmax(weights, dim=0).view(-1, 1, 1, 1)
            # Weighted sum: (B, T, D)
            group_feat = (layers * w).sum(dim=0)
            # ASP: (B, 2*D)
            group_pooled = self.group_asp[g](group_feat)
            group_outputs.append(group_pooled)

        # Concatenate: (B, K * 2*D)
        concat = torch.cat(group_outputs, dim=1)
        # Project: (B, output_dim)
        pooled = self.projection(concat)

        # CKA diversity loss: penalize high similarity between groups
        cka_loss = torch.tensor(0.0, device=pooled.device)
        n_pairs = 0
        for i in range(len(group_outputs)):
            for j in range(i + 1, len(group_outputs)):
                cka_loss = cka_loss + compute_cka(group_outputs[i], group_outputs[j])
                n_pairs += 1
        if n_pairs > 0:
            cka_loss = cka_loss / n_pairs

        return pooled, cka_loss, group_outputs


class LayerWiseDomainDiscriminators(nn.Module):
    """Per-group domain discriminators for Layer-Wise Domain Adversarial (LWDA) training.
    Each layer group's ASP output gets its own GRL + lightweight domain discriminator.
    This is the primary novel contribution: enforcing domain invariance at each
    abstraction level of the SSL representation hierarchy."""

    def __init__(self, num_groups, input_dim, hidden_dim=128):
        super().__init__()
        self.discriminators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 2),
            )
            for _ in range(num_groups)
        ])

    def forward(self, group_outputs, grl_alpha):
        """
        Args:
            group_outputs: list of (B, 2*D) tensors from per-group ASP
            grl_alpha: float, GRL reversal strength
        Returns:
            list of (B, 2) domain logits per group
        """
        logits = []
        for disc, feat in zip(self.discriminators, group_outputs):
            feat_rev = GradientReversal.apply(feat, grl_alpha)
            logits.append(disc(feat_rev))
        return logits


class VIBBottleneck(nn.Module):
    """Variational Information Bottleneck.
    Maps pooled features to (mu, logvar), samples z, computes KL loss."""

    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        self.mu_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, bottleneck_dim),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, bottleneck_dim),
        )

    def forward(self, x, training=True):
        """x: (B, D) -> z: (B, bottleneck_dim), kl_loss: scalar"""
        mu = self.mu_head(x)
        logvar = self.logvar_head(x)

        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu

        # KL divergence: KL(q(z|x) || N(0,I))
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return z, kl_loss


# ============================================================================
# FULL MODEL
# ============================================================================

class HierDAModel(nn.Module):
    def __init__(self, config: HierDAConfig):
        super().__init__()
        self.config = config

        # --- Backbone ---
        self.backbone = AutoModel.from_pretrained(config.hf_model_id)
        # Freeze CNN encoder permanently
        if hasattr(self.backbone, 'feature_extractor'):
            self.backbone.feature_extractor._freeze_parameters()
        elif hasattr(self.backbone, 'feature_projection'):
            for p in self.backbone.feature_projection.parameters():
                p.requires_grad = False

        # --- Per-layer normalization ---
        self.layer_norms = PerLayerNorm(config.total_hidden_states, config.hidden_dim)

        # --- Hierarchical Attention Pooling ---
        self.pooling = HierarchicalAttentionPooling(
            num_hidden_states=config.total_hidden_states,
            hidden_dim=config.hidden_dim,
            num_groups=config.num_groups,
            output_dim=config.hidden_dim,
        )

        # --- VIB Bottleneck (or deterministic projection) ---
        if config.use_vib:
            self.vib = VIBBottleneck(config.hidden_dim, config.bottleneck_dim)
        else:
            self.vib = None
            self.deterministic_proj = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.bottleneck_dim),
            )

        # --- Task Classifier (spoof detection) ---
        self.task_classifier = nn.Sequential(
            nn.Linear(config.bottleneck_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )

        # --- Speaker Adversary (deeper, 3-layer) ---
        if config.use_speaker_adversary:
            self.speaker_adversary = nn.Sequential(
                nn.Linear(config.bottleneck_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, config.num_speakers),
            )
        else:
            self.speaker_adversary = None

        # --- Global Domain Discriminator (baseline, off by default) ---
        if config.use_global_domain_disc:
            self.domain_disc = nn.Sequential(
                nn.Linear(config.bottleneck_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 2),
            )
        else:
            self.domain_disc = None

        # --- LWDA: Per-group Domain Discriminators (NOVEL) ---
        if config.use_lwda:
            self.lwda_disc = LayerWiseDomainDiscriminators(
                num_groups=config.num_groups,
                input_dim=2 * config.hidden_dim,  # ASP output: mean + std
                hidden_dim=config.lwda_disc_hidden,
            )
        else:
            self.lwda_disc = None

        # GRL alphas (set by trainer)
        self.domain_grl_alpha = 0.0
        self.speaker_grl_alpha = 0.0

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        # Re-freeze CNN (it stays frozen always)
        if hasattr(self.backbone, 'feature_extractor'):
            self.backbone.feature_extractor._freeze_parameters()

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        # Keep CNN frozen
        if hasattr(self.backbone, 'feature_extractor'):
            self.backbone.feature_extractor._freeze_parameters()
        # Enable gradient checkpointing
        self.backbone.gradient_checkpointing_enable()

    def forward(self, input_values, attention_mask=None):
        # 1. Backbone with all hidden states
        outputs = self.backbone(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = list(outputs.hidden_states)  # [h0, h1, ..., hL]

        # 2. Per-layer normalization
        hidden_states = self.layer_norms(hidden_states)

        # 3. Hierarchical pooling + CKA loss + per-group outputs
        pooled, cka_loss, group_outputs = self.pooling(hidden_states)
        if not self.config.use_cka_loss:
            cka_loss = torch.tensor(0.0, device=pooled.device)

        # 4. VIB or deterministic projection
        if self.vib is not None:
            z, kl_loss = self.vib(pooled, training=self.training)
        else:
            z = self.deterministic_proj(pooled)
            kl_loss = torch.tensor(0.0, device=pooled.device)

        # 5. Task classifier
        spoof_logits = self.task_classifier(z)

        result = {
            "spoof_logits": spoof_logits,
            "kl_loss": kl_loss,
            "cka_loss": cka_loss,
            "features": z,
        }

        # 6. Global domain discriminator (baseline, off by default)
        if self.domain_disc is not None:
            z_domain = GradientReversal.apply(z, self.domain_grl_alpha)
            result["domain_logits"] = self.domain_disc(z_domain)

        # 7. LWDA: per-group domain discriminators (NOVEL)
        if self.lwda_disc is not None:
            result["lwda_domain_logits"] = self.lwda_disc(group_outputs, self.domain_grl_alpha)
            result["group_outputs"] = group_outputs

        # 8. Speaker adversary
        if self.speaker_adversary is not None:
            z_speaker = GradientReversal.apply(z, self.speaker_grl_alpha)
            result["speaker_logits"] = self.speaker_adversary(z_speaker)

        return result


# ============================================================================
# DATASETS
# ============================================================================

def load_audio(path, max_len=64000):
    """Load audio with soundfile, pad/truncate to max_len."""
    try:
        if HAS_SOUNDFILE:
            wav, sr = sf.read(path, dtype='float32')
            if len(wav.shape) > 1:
                wav = wav.mean(axis=1)
            if sr != 16000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            wav = torch.from_numpy(wav)
        else:
            import torchaudio
            wav, sr = torchaudio.load(path)
            if sr != 16000:
                wav = torchaudio.transforms.Resample(sr, 16000)(wav)
            wav = wav.mean(0) if wav.shape[0] > 1 else wav.squeeze(0)
    except Exception:
        wav = torch.zeros(max_len)

    # Pad or random crop
    if wav.shape[0] > max_len:
        start = random.randint(0, wav.shape[0] - max_len)
        wav = wav[start:start + max_len]
    else:
        wav = F.pad(wav, (0, max(0, max_len - wav.shape[0])))
    return wav


class ASVspoofDatasetWithSpeaker(Dataset):
    """ASVspoof 2019 LA with speaker ID extraction + RawBoost augmentation.
    Protocol format: speaker_id filename _ system label"""

    def __init__(self, protocol_path, audio_dir, max_len=64000,
                 rawboost_prob=0.0, max_samples=None):
        self.audio_dir = Path(audio_dir)
        self.max_len = max_len
        self.rawboost_prob = rawboost_prob
        self.samples = []
        self.speaker_to_id = {}

        with open(protocol_path) as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 5:
                    continue
                speaker = p[0]
                if speaker not in self.speaker_to_id:
                    self.speaker_to_id[speaker] = len(self.speaker_to_id)
                self.samples.append({
                    'filename': p[1],
                    'speaker': speaker,
                    'speaker_id': self.speaker_to_id[speaker],
                    'label': 1 if p[4] == 'bonafide' else 0,
                })

        if max_samples:
            self.samples = self.samples[:max_samples]
        self.num_speakers = len(self.speaker_to_id)
        print(f"  ASVspoof: {len(self.samples)} samples, "
              f"{self.num_speakers} speakers, rawboost={rawboost_prob}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        wav = load_audio(self.audio_dir / f"{s['filename']}.flac", self.max_len)

        # RawBoost augmentation (training only)
        if self.rawboost_prob > 0 and random.random() < self.rawboost_prob:
            wav_np = wav.numpy()
            wav_np = RawBoost.augment(wav_np)
            wav = torch.from_numpy(wav_np)

        return {
            'input_values': wav,
            'attention_mask': torch.ones(self.max_len, dtype=torch.long),
            'task_label': torch.tensor(s['label'], dtype=torch.long),
            'speaker_id': torch.tensor(s['speaker_id'], dtype=torch.long),
        }


class InTheWildDataset(Dataset):
    """In-the-Wild dataset for domain adaptation / evaluation."""
    BONAFIDE = {'bona-fide', 'bonafide', 'real', '1', 'true'}

    def __init__(self, metadata_path, audio_dir, max_len=64000,
                 use_labels=False, max_samples=None):
        self.audio_dir = Path(audio_dir)
        self.max_len = max_len
        self.use_labels = use_labels
        self.samples = []

        with open(metadata_path) as f:
            header = next(f).strip().lower().split(',')
            label_idx = next((i for i, c in enumerate(header) if c in ['label', 'class']), -1)
            for line in f:
                p = line.strip().split(',')
                raw = p[label_idx].strip().lower() if label_idx >= 0 else 'spoof'
                label = 1 if raw in self.BONAFIDE else 0
                self.samples.append({'filename': p[0], 'label': label})

        if max_samples:
            self.samples = self.samples[:max_samples]
        print(f"  ITW: {len(self.samples)} samples ({'labeled' if use_labels else 'unlabeled'})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        wav = load_audio(self.audio_dir / s['filename'], self.max_len)
        return {
            'input_values': wav,
            'attention_mask': torch.ones(self.max_len, dtype=torch.long),
            'task_label': torch.tensor(s['label'] if self.use_labels else 0, dtype=torch.long),
        }


class ASVspoof2021Dataset(Dataset):
    """ASVspoof 2021 LA / DF eval dataset. Eval-only (no speaker labels)."""

    def __init__(self, keys_path, audio_dir, max_len=64000, max_samples=None):
        self.audio_dir = Path(audio_dir)
        self.max_len = max_len
        self.samples = []

        with open(keys_path) as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 2:
                    continue
                filename = p[1]
                label = 1 if 'bonafide' in line.lower() else 0
                self.samples.append({'filename': filename, 'label': label})

        if max_samples:
            self.samples = self.samples[:max_samples]
        print(f"  2021 Eval: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # Try .flac first, then .wav
        path = self.audio_dir / f"{s['filename']}.flac"
        if not path.exists():
            path = self.audio_dir / f"{s['filename']}.wav"
        wav = load_audio(path, self.max_len)
        return {
            'input_values': wav,
            'attention_mask': torch.ones(self.max_len, dtype=torch.long),
            'task_label': torch.tensor(s['label'], dtype=torch.long),
        }


class ASVspoofEvalDataset(Dataset):
    """ASVspoof 2019 LA eval (same protocol format, no speaker needed)."""

    def __init__(self, protocol_path, audio_dir, max_len=64000, max_samples=None):
        self.audio_dir = Path(audio_dir)
        self.max_len = max_len
        self.samples = []

        with open(protocol_path) as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    self.samples.append({
                        'filename': p[1],
                        'label': 1 if p[4] == 'bonafide' else 0,
                    })

        if max_samples:
            self.samples = self.samples[:max_samples]
        print(f"  ASVspoof Eval: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        wav = load_audio(self.audio_dir / f"{s['filename']}.flac", self.max_len)
        return {
            'input_values': wav,
            'attention_mask': torch.ones(self.max_len, dtype=torch.long),
            'task_label': torch.tensor(s['label'], dtype=torch.long),
        }


# ============================================================================
# LLRD OPTIMIZER BUILDER
# ============================================================================

def build_llrd_optimizer(model, config: HierDAConfig):
    """Build AdamW optimizer with layer-wise learning rate decay for backbone."""
    param_groups = []

    # --- Head parameters (highest LR) ---
    head_params = []
    for module_name in ['layer_norms', 'pooling', 'vib', 'deterministic_proj',
                        'task_classifier', 'speaker_adversary', 'domain_disc',
                        'lwda_disc']:
        module = getattr(model, module_name, None)
        if module:
            head_params.extend(list(module.parameters()))

    param_groups.append({
        "params": head_params,
        "lr": config.head_lr,
        "weight_decay": config.weight_decay,
        "name": "heads",
    })

    # --- Backbone parameters with LLRD ---
    layer_groups = {}
    for name, param in model.backbone.named_parameters():
        if not param.requires_grad:
            continue
        if "feature_extractor" in name or "feature_projection" in name:
            layer_id = 0
        elif "encoder.layers" in name:
            try:
                layer_id = int(name.split("encoder.layers.")[1].split(".")[0]) + 1
            except (IndexError, ValueError):
                layer_id = 0
        else:
            layer_id = 0

        if layer_id not in layer_groups:
            layer_groups[layer_id] = []
        layer_groups[layer_id].append(param)

    if layer_groups:
        max_layer = max(layer_groups.keys())
        for layer_id, params in layer_groups.items():
            scale = config.llrd_decay ** (max_layer - layer_id)
            param_groups.append({
                "params": params,
                "lr": config.backbone_lr * scale,
                "weight_decay": config.weight_decay,
                "name": f"backbone_layer_{layer_id}",
            })

    return torch.optim.AdamW(param_groups)


# ============================================================================
# TRAINER
# ============================================================================

class HierDATrainer:
    def __init__(self, config: HierDAConfig, source_loader, target_loader,
                 eval_loader, extra_eval_loaders=None, num_speakers=None,
                 resume_from: Optional[str] = None):
        """
        Args:
            config: HierDAConfig
            source_loader: DataLoader for ASVspoof source domain (with speaker_id)
            target_loader: DataLoader for target domain (ITW, unlabeled)
            eval_loader: DataLoader for primary evaluation (ITW, labeled)
            extra_eval_loaders: dict of {name: DataLoader} for additional eval sets
            num_speakers: actual number of speakers (auto-detected from dataset)
            resume_from: path to checkpoint for resuming
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_seed(config.seed)

        # Update num_speakers if provided (from actual dataset)
        if num_speakers is not None and num_speakers != config.num_speakers:
            print(f"  Updating num_speakers: {config.num_speakers} -> {num_speakers}")
            config.num_speakers = num_speakers

        # --- Build model ---
        print(f"\nBuilding HierDA model: {config.backbone}")
        self.model = HierDAModel(config)
        self.model.to(self.device)

        # --- Store dataloaders (created externally by notebook) ---
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.eval_loader = eval_loader
        self.extra_eval_loaders = extra_eval_loaders or {}

        # --- Training state ---
        self.run_dir = config.get_run_dir()
        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, "config.json"), 'w') as f:
            json.dump(asdict(config), f, indent=2, default=str)

        self.ce_loss = nn.CrossEntropyLoss()
        self.speaker_ce_loss = nn.CrossEntropyLoss()
        amp_dtype = torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
        self.amp_dtype = amp_dtype
        self.scaler = GradScaler() if (config.use_amp and amp_dtype == torch.float16) else None

        self.best_eer = float('inf')
        self.best_epoch = -1
        self.patience_counter = 0
        self.history = []
        self.start_epoch = 0
        self.current_phase = 1

        # --- Initialize Phase 1 ---
        self._setup_phase1()

        # --- Resume ---
        if resume_from and os.path.exists(resume_from):
            self._resume(resume_from)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable: {trainable:,} / {total:,}")
        print(f"  Run dir: {self.run_dir}")

    def _setup_phase1(self):
        """Phase 1: Freeze backbone, train heads only."""
        self.current_phase = 1
        self.model.freeze_backbone()
        # Simple optimizer for heads only
        head_params = []
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                head_params.append(p)
        self.optimizer = torch.optim.AdamW(
            head_params, lr=self.config.head_lr, weight_decay=self.config.weight_decay)
        steps_per_epoch = len(self.source_loader)
        total_steps = steps_per_epoch * self.config.phase1_epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.config.head_lr,
            total_steps=max(total_steps, 1),
            pct_start=0.1,
        )

    def _setup_phase2(self):
        """Phase 2: Unfreeze backbone, rebuild optimizer with LLRD."""
        self.current_phase = 2
        self.model.unfreeze_backbone()
        self.optimizer = build_llrd_optimizer(self.model, self.config)
        steps_per_epoch = len(self.source_loader)
        total_steps = steps_per_epoch * self.config.phase2_epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.config.head_lr,
            total_steps=max(total_steps, 1),
            pct_start=0.05,
        )
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n  Phase 2: Backbone unfrozen. Trainable: {trainable:,}")

    def _resume(self, path):
        print(f"  Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.start_epoch = ckpt.get('epoch', 0)
        self.best_eer = ckpt.get('best_eer', float('inf'))
        self.best_epoch = ckpt.get('best_epoch', -1)
        # Determine phase
        if self.start_epoch >= self.config.phase1_epochs:
            self._setup_phase2()
        hist_path = os.path.join(os.path.dirname(path), "history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                self.history = json.load(f)
        print(f"  Resumed at epoch {self.start_epoch}, best EER: {self.best_eer:.2f}%")

    def _get_grl_alphas(self, epoch):
        """Compute GRL alpha values with warmup scheduling."""
        total_epoch = epoch  # absolute epoch across both phases
        # Domain GRL: warmup over grl_warmup_epochs
        domain_alpha = min(1.0, total_epoch / max(self.config.grl_warmup_epochs, 1))
        # Speaker GRL: warmup over speaker_grl_warmup_epochs, starts at 0
        speaker_alpha = min(
            self.config.speaker_loss_weight,
            self.config.speaker_loss_weight * total_epoch / max(self.config.speaker_grl_warmup_epochs, 1)
        )
        return domain_alpha, speaker_alpha

    def train_epoch(self, epoch):
        self.model.train()
        domain_alpha, speaker_alpha = self._get_grl_alphas(epoch)
        self.model.domain_grl_alpha = domain_alpha
        self.model.speaker_grl_alpha = speaker_alpha

        metrics_acc = {
            'spoof_loss': 0, 'domain_loss': 0, 'lwda_loss': 0,
            'speaker_loss': 0, 'kl_loss': 0, 'cka_loss': 0, 'total_loss': 0,
            'spoof_acc': 0, 'speaker_acc': 0,
        }
        steps = 0
        target_iter = iter(self.target_loader)
        self.optimizer.zero_grad(set_to_none=True)

        for step, s_batch in enumerate(
                pbar := tqdm(self.source_loader, desc=f"Ep {epoch+1}")):
            # Source batch
            s_inp = s_batch['input_values'].to(self.device, non_blocking=True)
            s_mask = s_batch['attention_mask'].to(self.device, non_blocking=True)
            s_label = s_batch['task_label'].to(self.device, non_blocking=True)
            s_spk = s_batch['speaker_id'].to(self.device, non_blocking=True)

            # Target batch
            try:
                t_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(self.target_loader)
                t_batch = next(target_iter)
            t_inp = t_batch['input_values'].to(self.device, non_blocking=True)
            t_mask = t_batch['attention_mask'].to(self.device, non_blocking=True)

            # Combined forward
            c_inp = torch.cat([s_inp, t_inp])
            c_mask = torch.cat([s_mask, t_mask])
            n_src = s_inp.size(0)

            with autocast('cuda', enabled=self.config.use_amp, dtype=self.amp_dtype):
                out = self.model(c_inp, attention_mask=c_mask)

                # Spoof loss (source only)
                spoof_loss = self.ce_loss(out['spoof_logits'][:n_src], s_label)

                # Domain labels (shared by global disc and LWDA)
                domain_labels = torch.cat([
                    torch.zeros(n_src, dtype=torch.long, device=self.device),
                    torch.ones(t_inp.size(0), dtype=torch.long, device=self.device),
                ])

                # Global domain loss (baseline, off by default)
                domain_loss = torch.tensor(0.0, device=self.device)
                if 'domain_logits' in out:
                    domain_loss = self.ce_loss(out['domain_logits'], domain_labels)

                # LWDA: per-group domain losses (NOVEL)
                lwda_loss = torch.tensor(0.0, device=self.device)
                if 'lwda_domain_logits' in out:
                    group_losses = [self.ce_loss(lg, domain_labels)
                                    for lg in out['lwda_domain_logits']]
                    lwda_loss = torch.stack(group_losses).mean()

                # Speaker loss (source only)
                speaker_loss = torch.tensor(0.0, device=self.device)
                if 'speaker_logits' in out:
                    speaker_loss = self.speaker_ce_loss(
                        out['speaker_logits'][:n_src], s_spk)

                # VIB KL loss
                kl_loss = out['kl_loss']

                # CKA loss
                cka_loss = out['cka_loss']

                # Total loss
                total_loss = spoof_loss + self.config.vib_beta * kl_loss
                if self.config.use_lwda:
                    total_loss = total_loss + self.config.domain_loss_weight * lwda_loss
                if self.config.use_global_domain_disc:
                    total_loss = total_loss + self.config.domain_loss_weight * domain_loss
                if self.config.use_speaker_adversary:
                    total_loss = total_loss + speaker_alpha * speaker_loss
                if self.config.use_cka_loss:
                    total_loss = total_loss + self.config.cka_loss_weight * cka_loss
                total_loss = total_loss / self.config.grad_accum_steps

            # Backward
            if self.scaler:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            if (step + 1) % self.config.grad_accum_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip)
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            # Track metrics
            with torch.no_grad():
                spoof_pred = out['spoof_logits'][:n_src].argmax(1)
                spoof_acc = (spoof_pred == s_label).float().mean().item()
                spk_acc = 0.0
                if 'speaker_logits' in out:
                    spk_pred = out['speaker_logits'][:n_src].argmax(1)
                    spk_acc = (spk_pred == s_spk).float().mean().item()

            metrics_acc['spoof_loss'] += spoof_loss.item()
            metrics_acc['domain_loss'] += domain_loss.item()
            metrics_acc['lwda_loss'] += lwda_loss.item()
            metrics_acc['speaker_loss'] += speaker_loss.item()
            metrics_acc['kl_loss'] += kl_loss.item()
            metrics_acc['cka_loss'] += cka_loss.item()
            metrics_acc['total_loss'] += total_loss.item() * self.config.grad_accum_steps
            metrics_acc['spoof_acc'] += spoof_acc
            metrics_acc['speaker_acc'] += spk_acc
            steps += 1

            pbar.set_postfix({
                'L': f"{metrics_acc['total_loss']/steps:.3f}",
                'spf': f"{spoof_acc:.2f}",
                'spk': f"{spk_acc:.2f}",
            })

        return {k: v / max(steps, 1) for k, v in metrics_acc.items()}

    @torch.no_grad()
    def evaluate(self, loader=None, name="ITW"):
        """Evaluate on a dataset. Returns EER, AUC, accuracy."""
        self.model.eval()
        if loader is None:
            loader = self.eval_loader
        scores, labels = [], []

        for batch in loader:
            inp = batch['input_values'].to(self.device, non_blocking=True)
            mask = batch['attention_mask'].to(self.device, non_blocking=True)
            with autocast('cuda', enabled=self.config.use_amp, dtype=self.amp_dtype):
                out = self.model(inp, attention_mask=mask)
            probs = F.softmax(out['spoof_logits'], dim=1)[:, 1]
            scores.extend(probs.cpu().numpy())
            labels.extend(batch['task_label'].numpy())

        labels, scores = np.array(labels), np.array(scores)
        if len(np.unique(labels)) < 2:
            return {"eer": 50.0, "auc": 0.5, "acc": 0.0, "name": name}

        fpr, tpr, thresholds = roc_curve(labels, scores)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2 * 100
        auc = roc_auc_score(labels, scores)

        # Accuracy at EER threshold
        threshold = thresholds[eer_idx]
        preds = (scores >= threshold).astype(int)
        acc = (preds == labels).mean() * 100

        return {"eer": eer, "auc": auc, "acc": acc, "name": name}

    def evaluate_all(self):
        """Evaluate on all configured datasets."""
        results = {}
        # Primary: ITW
        results["ITW"] = self.evaluate(self.eval_loader, "ITW")
        # Extra eval sets
        for name, loader in self.extra_eval_loaders.items():
            results[name] = self.evaluate(loader, name)
        return results

    def _save_checkpoint(self, epoch, is_best=False):
        ckpt = {
            'epoch': epoch + 1,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_eer': self.best_eer,
            'best_epoch': self.best_epoch,
            'phase': self.current_phase,
            'config': asdict(self.config),
        }
        if is_best:
            path = os.path.join(self.run_dir, "best_model.pt")
        else:
            path = os.path.join(self.run_dir, f"ckpt_ep{epoch+1:02d}.pt")
        torch.save(ckpt, path)
        return path

    def train(self):
        """Run full two-phase training."""
        cfg = self.config
        print(f"\n{'='*60}")
        print(f"HierDA Training | {cfg.backbone} | seed={cfg.seed}")
        print(f"  Phase 1: {cfg.phase1_epochs} ep (frozen) | Phase 2: {cfg.phase2_epochs} ep (unfrozen)")
        components = ["spoof"]
        if cfg.use_lwda: components.append(f"{cfg.domain_loss_weight}*LWDA({cfg.num_groups}grp)")
        if cfg.use_global_domain_disc: components.append(f"{cfg.domain_loss_weight}*domain")
        if cfg.use_vib: components.append(f"{cfg.vib_beta}*KL")
        if cfg.use_speaker_adversary: components.append(f"{cfg.speaker_loss_weight}*speaker")
        if cfg.use_cka_loss: components.append(f"{cfg.cka_loss_weight}*CKA")
        print(f"  Loss: {' + '.join(components)}")
        print(f"{'='*60}")

        start_time = time.time()

        for epoch in range(self.start_epoch, cfg.total_epochs):
            # Phase transition
            if epoch == cfg.phase1_epochs and self.current_phase == 1:
                print(f"\n{'='*40} PHASE 2: Unfreezing backbone {'='*40}")
                self._setup_phase2()

            # Train
            train_m = self.train_epoch(epoch)

            # Evaluate all datasets
            eval_results = self.evaluate_all()
            primary_eer = eval_results["ITW"]["eer"]

            # Log
            epoch_log = {
                "epoch": epoch + 1,
                "phase": self.current_phase,
                "train": train_m,
                "eval": {k: v for k, v in eval_results.items()},
                "domain_alpha": self.model.domain_grl_alpha,
                "speaker_alpha": self.model.speaker_grl_alpha,
                "lr": self.optimizer.param_groups[0]['lr'],
            }
            self.history.append(epoch_log)
            with open(os.path.join(self.run_dir, "history.json"), 'w') as f:
                json.dump(self.history, f, indent=2)

            # Best tracking
            is_best = primary_eer < self.best_eer
            if is_best:
                self.best_eer = primary_eer
                self.best_epoch = epoch + 1
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Checkpoint
            self._save_checkpoint(epoch, is_best=is_best)

            # Print summary
            eval_str = " | ".join(
                f"{r['name']}={r['eer']:.2f}%" for r in eval_results.values())
            best_str = " BEST" if is_best else ""
            spk_warning = ""
            if train_m['speaker_acc'] > 0.80 and epoch > cfg.speaker_grl_warmup_epochs:
                spk_warning = " [WARN: speaker_acc high, consider increasing gamma]"
            print(f"  Ep {epoch+1} P{self.current_phase}: "
                  f"L={train_m['total_loss']:.3f} spf_acc={train_m['spoof_acc']:.2f} "
                  f"spk_acc={train_m['speaker_acc']:.2f} | {eval_str}{best_str}{spk_warning}")

            # Early stopping (only in phase 2)
            if self.current_phase == 2 and self.patience_counter >= cfg.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

            torch.cuda.empty_cache()

        elapsed = (time.time() - start_time) / 60
        print(f"\n{'='*60}")
        print(f"Done in {elapsed:.1f} min | Best ITW EER: {self.best_eer:.2f}% (Ep {self.best_epoch})")
        print(f"{'='*60}")

        return {
            "best_eer": self.best_eer,
            "best_epoch": self.best_epoch,
            "run_dir": self.run_dir,
            "history": self.history,
        }


# ============================================================================
# MULTI-DATASET EVALUATOR
# ============================================================================

class MultiDatasetEvaluator:
    """Standalone evaluator that loads a checkpoint and runs eval on multiple datasets."""

    def __init__(self, checkpoint_path, config: HierDAConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model
        self.model = HierDAModel(config)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

        self.amp_dtype = torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16

    @torch.no_grad()
    def evaluate_dataset(self, loader, name="dataset"):
        scores, labels = [], []
        for batch in tqdm(loader, desc=f"Eval {name}"):
            inp = batch['input_values'].to(self.device, non_blocking=True)
            mask = batch['attention_mask'].to(self.device, non_blocking=True)
            with autocast('cuda', enabled=self.config.use_amp, dtype=self.amp_dtype):
                out = self.model(inp, attention_mask=mask)
            probs = F.softmax(out['spoof_logits'], dim=1)[:, 1]
            scores.extend(probs.cpu().numpy())
            labels.extend(batch['task_label'].numpy())

        labels, scores = np.array(labels), np.array(scores)
        if len(np.unique(labels)) < 2:
            return {"name": name, "eer": 50.0, "auc": 0.5, "acc": 0.0, "n_samples": len(labels)}

        fpr, tpr, thresholds = roc_curve(labels, scores)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2 * 100
        auc = roc_auc_score(labels, scores)
        threshold = thresholds[eer_idx]
        preds = (scores >= threshold).astype(int)
        acc = (preds == labels).mean() * 100

        # Bootstrap 95% CI on EER
        n_bootstrap = 1000
        eer_samples = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(labels), len(labels), replace=True)
            b_labels, b_scores = labels[idx], scores[idx]
            if len(np.unique(b_labels)) < 2:
                continue
            b_fpr, b_tpr, _ = roc_curve(b_labels, b_scores)
            b_fnr = 1 - b_tpr
            b_eer_idx = np.nanargmin(np.abs(b_fpr - b_fnr))
            eer_samples.append((b_fpr[b_eer_idx] + b_fnr[b_eer_idx]) / 2 * 100)

        ci_low = np.percentile(eer_samples, 2.5) if eer_samples else eer
        ci_high = np.percentile(eer_samples, 97.5) if eer_samples else eer

        return {
            "name": name, "eer": eer, "auc": auc, "acc": acc,
            "n_samples": len(labels),
            "ci_low": ci_low, "ci_high": ci_high,
        }

    def evaluate_all(self, datasets: Dict[str, DataLoader]):
        """Run eval on all datasets and return results table."""
        results = {}
        for name, loader in datasets.items():
            results[name] = self.evaluate_dataset(loader, name)
            r = results[name]
            print(f"  {name}: EER={r['eer']:.2f}% [{r['ci_low']:.2f}, {r['ci_high']:.2f}] "
                  f"AUC={r['auc']:.4f} Acc={r['acc']:.1f}% (n={r['n_samples']})")
        return results

    @staticmethod
    def results_to_latex(results):
        """Generate LaTeX table from results dict."""
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Cross-domain evaluation results}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Dataset & EER (\%) & 95\% CI & AUC & Samples \\",
            r"\midrule",
        ]
        for name, r in results.items():
            lines.append(
                f"{name} & {r['eer']:.2f} & [{r.get('ci_low', r['eer']):.2f}, "
                f"{r.get('ci_high', r['eer']):.2f}] & {r['auc']:.4f} & {r['n_samples']:,} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines)

    @staticmethod
    def results_to_markdown(results):
        """Generate Markdown table from results dict."""
        lines = [
            "| Dataset | EER (%) | 95% CI | AUC | Samples |",
            "|---|---|---|---|---|",
        ]
        for name, r in results.items():
            lines.append(
                f"| {name} | {r['eer']:.2f} | [{r.get('ci_low', r['eer']):.2f}, "
                f"{r.get('ci_high', r['eer']):.2f}] | {r['auc']:.4f} | {r['n_samples']:,} |"
            )
        return "\n".join(lines)


# ============================================================================
# UTILITIES
# ============================================================================

def create_balanced_sampler(ds):
    labels = [s['label'] for s in ds.samples]
    counts = [labels.count(0), labels.count(1)]
    weights = [1.0 / (counts[l] + 1e-5) for l in labels]
    return WeightedRandomSampler(weights, len(labels))


def smoke_test(config=None):
    """Quick smoke test: 2 epochs with tiny batch on CPU/GPU."""
    if config is None:
        config = HierDAConfig(
            backbone="wavlm-base-plus",
            phase1_epochs=1,
            phase2_epochs=1,
            batch_size=4,
            effective_batch_size=8,
        )

    print("\n=== SMOKE TEST ===")
    model = HierDAModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Fake input
    B = 4
    x = torch.randn(B, config.max_audio_len, device=device)
    mask = torch.ones(B, config.max_audio_len, dtype=torch.long, device=device)

    # Phase 1: frozen
    model.freeze_backbone()
    model.train()
    model.domain_grl_alpha = 0.5  # set for testing
    model.speaker_grl_alpha = 0.1
    out = model(x, mask)
    print(f"  spoof_logits: {out['spoof_logits'].shape}")
    print(f"  kl_loss: {out['kl_loss'].item():.4f}")
    print(f"  cka_loss: {out['cka_loss'].item():.4f}")

    # LWDA outputs
    if 'lwda_domain_logits' in out:
        print(f"  LWDA: {len(out['lwda_domain_logits'])} groups, "
              f"each {out['lwda_domain_logits'][0].shape}")
    if 'domain_logits' in out:
        print(f"  Global domain_logits: {out['domain_logits'].shape}")
    if 'speaker_logits' in out:
        print(f"  speaker_logits: {out['speaker_logits'].shape}")

    # Check backward with all active losses
    loss = out['spoof_logits'].sum() + out['kl_loss'] + out['cka_loss']
    if 'lwda_domain_logits' in out:
        for lg in out['lwda_domain_logits']:
            loss = loss + lg.sum()
    loss.backward()
    print(f"  Backward OK")

    # Verify LWDA discriminator gradients
    if hasattr(model, 'lwda_disc') and model.lwda_disc is not None:
        lwda_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.lwda_disc.parameters())
        print(f"  LWDA disc gradients flowing: {lwda_has_grad}")

    # Phase 2: unfreeze
    model.unfreeze_backbone()
    model.zero_grad()
    out2 = model(x, mask)
    loss2 = out2['spoof_logits'].sum() + out2['kl_loss']
    loss2.backward()

    # Check backbone gradients
    has_grad = False
    for name, p in model.backbone.named_parameters():
        if 'encoder.layers.0' in name and p.grad is not None:
            if p.grad.abs().sum() > 0:
                has_grad = True
                break
    print(f"  Backbone gradients flowing: {has_grad}")

    # Memory
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak GPU memory: {mem:.2f} GB")

    # LayerNorm sanity
    model.eval()
    with torch.no_grad():
        out3 = model.backbone(x, attention_mask=mask, output_hidden_states=True)
        raw_norms = [h.norm(dim=-1).mean().item() for h in out3.hidden_states]
        normed = model.layer_norms(list(out3.hidden_states))
        normed_norms = [h.norm(dim=-1).mean().item() for h in normed]
    print(f"  Raw layer norms range: [{min(raw_norms):.1f}, {max(raw_norms):.1f}]")
    print(f"  Normed layer norms range: [{min(normed_norms):.1f}, {max(normed_norms):.1f}]")

    # Component summary
    total_params = sum(p.numel() for p in model.parameters())
    lwda_params = sum(p.numel() for p in model.lwda_disc.parameters()) if model.lwda_disc else 0
    print(f"  Total params: {total_params:,}")
    print(f"  LWDA disc params: {lwda_params:,} ({100*lwda_params/total_params:.2f}%)")

    print("=== SMOKE TEST PASSED ===\n")
    return True


if __name__ == "__main__":
    smoke_test()
