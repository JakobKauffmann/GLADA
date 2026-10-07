# domain_adaptation_ablation.py
# -*- coding: utf-8 -*-
"""
Stable Domain Adaptation for Audio Deepfake Detection - Ablation Study
=======================================================================

Built on domain_adaptation_e2e.py with targeted improvements for discriminator stability.

CRITICAL DESIGN DECISIONS (from 15.34% EER breakthrough):
1. END-TO-END TRAINING with RAW AUDIO - NOT pre-extracted embeddings!
   - Pre-extracted embeddings broke domain adaptation (~30.8% EER)
   - Raw audio through frozen WavLM is essential (15.34% EER)
2. Frozen WavLM backbone (NOT LoRA fine-tuning)
3. domain_loss_weight = 0.1 (NOT 0.5)

STABILITY ENHANCEMENTS:
1. Stronger discriminator (3 layers, spectral norm option)
2. Multiple GRL schedules (linear, exponential, delayed, cosine)
3. Separate discriminator learning rate (higher LR maintains tension)
4. Gradient penalty option for stability
5. Label smoothing for domain labels
6. Comprehensive per-epoch logging (all metrics, all seeds)

A100 OPTIMIZATIONS:
- TF32 precision for faster matmuls
- cuDNN benchmark mode for optimized convolutions
- torch.compile() support for graph optimization
- Optimized batch sizes for 40GB VRAM

GOAL: Achieve ~15% EER with stable training (domain loss stays in 0.3-1.0 range)

Author: Jakob Kauffmann
"""

import os
import random
import json
import csv
import time
import warnings
import logging
import math
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict, Any, Literal

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Audio loading - soundfile is more reliable in Colab
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    import torchaudio

import librosa
from torch.amp import autocast, GradScaler
from transformers import AutoModel
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm.auto import tqdm

# ============================================================================
# A100/H100 OPTIMIZATIONS
# ============================================================================
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability()[0]
    if capability >= 8:  # Ampere (A100) or newer
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # Optimize convolution algorithms
        print(f"A100/H100 detected: TF32 + cuDNN benchmark enabled")
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AblationConfig:
    """Configuration for domain adaptation ablation study."""

    # Experiment identification
    experiment_name: str = "dann_ablation"
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    seed: int = 42

    # Adaptation method
    adaptation_method: str = "dann"  # none, dann, mmd, coral

    # Model
    model_name: str = "microsoft/wavlm-base-plus"
    hidden_size: int = 768

    # Domain adaptation weights
    domain_loss_weight: float = 0.1  # CRITICAL: NOT 0.5!

    # === NEW: GRL Schedule Options ===
    grl_schedule: str = "linear"  # linear, exponential, delayed, cosine
    grl_warmup_epochs: int = 5
    grl_max_alpha: float = 1.0
    grl_delay_epochs: int = 0  # For delayed schedule: epochs before GRL starts

    # === NEW: Discriminator Configuration ===
    disc_type: str = "standard"  # standard, strong
    disc_hidden_dims: Tuple[int, ...] = (256,)  # standard: (256,), strong: (512, 256, 128)
    disc_dropout: float = 0.1
    disc_use_spectral_norm: bool = False
    disc_lr_multiplier: float = 1.0  # NEW: separate discriminator LR

    # === NEW: Stability Options ===
    use_gradient_penalty: bool = False
    gradient_penalty_weight: float = 10.0
    domain_label_smoothing: float = 0.0  # 0.0 = no smoothing, 0.1 = recommended

    # MMD options
    mmd_kernel: str = "multiscale"

    # Training
    batch_size: int = 48
    effective_batch_size: int = 384
    grad_accum_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 50
    warmup_epochs: int = 1

    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_epochs: int = 5  # Don't stop before this

    # System / A100 Optimization
    num_workers: int = 0  # 0 avoids multiprocessing issues in Colab
    use_amp: bool = True  # Automatic mixed precision
    compile_model: bool = False  # torch.compile() - can cause issues, disable if errors
    max_audio_len: int = 64000  # 4 seconds at 16kHz

    # Logging
    drive_log_dir: str = "/content/drive/MyDrive/Jakob/Experiments"
    save_every_epoch: bool = True
    log_gradient_norms: bool = True  # NEW: track gradient norms

    def __post_init__(self):
        uses_da = self.adaptation_method in ["dann", "mmd", "coral"]
        actual = self.batch_size * (2 if uses_da else 1)
        self.grad_accum_steps = max(1, self.effective_batch_size // actual)

        # Set discriminator architecture based on type
        if self.disc_type == "strong" and self.disc_hidden_dims == (256,):
            self.disc_hidden_dims = (512, 256, 128)

        print(f"Batch: {self.batch_size} x {2 if uses_da else 1} x {self.grad_accum_steps} = {actual * self.grad_accum_steps}")
        print(f"GRL Schedule: {self.grl_schedule} (warmup={self.grl_warmup_epochs}, delay={self.grl_delay_epochs})")
        if self.adaptation_method == 'dann':
            print(f"Discriminator: {self.disc_type} {self.disc_hidden_dims}, SN={self.disc_use_spectral_norm}, LR_mult={self.disc_lr_multiplier}")
            if self.use_gradient_penalty:
                print(f"Gradient Penalty: weight={self.gradient_penalty_weight}")
            if self.domain_label_smoothing > 0:
                print(f"Label Smoothing: {self.domain_label_smoothing}")

    def get_run_dir(self) -> str:
        return os.path.join(self.drive_log_dir, self.experiment_name, self.run_id)

    def get_config_hash(self) -> str:
        """Generate a short hash for this configuration (for deduplication)."""
        import hashlib
        key_params = f"{self.adaptation_method}_{self.grl_schedule}_{self.disc_type}_{self.disc_lr_multiplier}_{self.use_gradient_penalty}_{self.domain_label_smoothing}_{self.seed}"
        return hashlib.md5(key_params.encode()).hexdigest()[:8]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# DATASETS (unchanged from domain_adaptation_e2e.py)
# ============================================================================

class ASVspoofDataset(Dataset):
    """
    ASVspoof 2019 LA dataset - returns RAW AUDIO for end-to-end training.

    Returns:
        dict with:
        - 'input_values': Raw audio waveform tensor (max_len,) at 16kHz
        - 'attention_mask': Tensor of ones (max_len,)
        - 'task_label': 1 for bonafide, 0 for spoof

    NOTE: This returns raw audio, NOT pre-extracted embeddings!
    """
    def __init__(self, protocol_path, audio_dir, max_len=64000, max_samples=None):
        self.audio_dir, self.max_len = Path(audio_dir), max_len
        self.samples = []
        with open(protocol_path) as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    self.samples.append({'filename': p[1], 'label': 1 if p[4] == 'bonafide' else 0})
        if max_samples:
            self.samples = self.samples[:max_samples]
        print(f"  ASVspoof: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def _load_audio(self, path):
        try:
            if HAS_SOUNDFILE:
                wav, sr = sf.read(path, dtype='float32')
                if len(wav.shape) > 1:
                    wav = wav.mean(axis=1)
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                return torch.from_numpy(wav)
            else:
                wav, sr = torchaudio.load(path)
                if sr != 16000:
                    wav = torchaudio.transforms.Resample(sr, 16000)(wav)
                return wav.mean(0) if wav.shape[0] > 1 else wav.squeeze(0)
        except Exception:
            return None

    def __getitem__(self, idx):
        s = self.samples[idx]
        wav = self._load_audio(self.audio_dir / f"{s['filename']}.flac")
        if wav is None:
            wav = torch.zeros(self.max_len)
        wav = F.pad(wav, (0, max(0, self.max_len - wav.shape[0])))[:self.max_len]
        return {
            'input_values': wav,
            'attention_mask': torch.ones(self.max_len, dtype=torch.long),
            'task_label': torch.tensor(s['label'], dtype=torch.long)
        }


class InTheWildDataset(Dataset):
    """
    In-the-Wild dataset (target domain) - returns RAW AUDIO for end-to-end training.

    Returns:
        dict with:
        - 'input_values': Raw audio waveform tensor (max_len,) at 16kHz
        - 'attention_mask': Tensor of ones (max_len,)
        - 'task_label': 1 for bonafide, 0 for spoof (only if use_labels=True)

    NOTE: This returns raw audio, NOT pre-extracted embeddings!
    """
    BONAFIDE = {'bona-fide', 'bonafide', 'real', '1', 'true'}
    SPOOF = {'fake', 'spoof', 'deepfake', '0', 'false'}

    def __init__(self, metadata_path, audio_dir, max_len=64000, use_labels=False, max_samples=None):
        self.audio_dir, self.max_len, self.use_labels = Path(audio_dir), max_len, use_labels
        self.samples = []
        counts = {'bonafide': 0, 'spoof': 0}
        with open(metadata_path) as f:
            header = next(f).strip().lower().split(',')
            label_idx = next((i for i, c in enumerate(header) if c in ['label', 'class']), -1)
            for line in f:
                p = line.strip().split(',')
                raw = p[label_idx].strip().lower() if label_idx >= 0 else 'spoof'
                label = 1 if raw in self.BONAFIDE else 0
                counts['bonafide' if label else 'spoof'] += 1
                self.samples.append({'filename': p[0], 'label': label})
        if max_samples:
            self.samples = self.samples[:max_samples]
        print(f"  ITW: {len(self.samples)} ({'labeled' if use_labels else 'unlabeled'}, {counts})")

    def __len__(self):
        return len(self.samples)

    def _load_audio(self, path):
        try:
            if HAS_SOUNDFILE:
                wav, sr = sf.read(path, dtype='float32')
                if len(wav.shape) > 1:
                    wav = wav.mean(axis=1)
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                return torch.from_numpy(wav)
            else:
                wav, sr = torchaudio.load(path)
                if sr != 16000:
                    wav = torchaudio.transforms.Resample(sr, 16000)(wav)
                return wav.mean(0) if wav.shape[0] > 1 else wav.squeeze(0)
        except Exception:
            return None

    def __getitem__(self, idx):
        s = self.samples[idx]
        wav = self._load_audio(self.audio_dir / s['filename'])
        if wav is None:
            wav = torch.zeros(self.max_len)
        wav = F.pad(wav, (0, max(0, self.max_len - wav.shape[0])))[:self.max_len]
        return {
            'input_values': wav,
            'attention_mask': torch.ones(self.max_len, dtype=torch.long),
            'task_label': torch.tensor(s['label'] if self.use_labels else 0, dtype=torch.long)
        }


# ============================================================================
# GRL SCHEDULING
# ============================================================================

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.alpha, None


def compute_grl_alpha(epoch: int, config: AblationConfig) -> float:
    """Compute GRL alpha based on schedule type."""

    # Handle delay
    if epoch < config.grl_delay_epochs:
        return 0.0

    effective_epoch = epoch - config.grl_delay_epochs
    warmup = config.grl_warmup_epochs
    max_alpha = config.grl_max_alpha

    if config.grl_schedule == "linear":
        # Linear warmup
        return min(max_alpha, max_alpha * effective_epoch / warmup)

    elif config.grl_schedule == "exponential":
        # DANN paper schedule: 2/(1 + exp(-gamma*p)) - 1
        # where p = epoch / total_epochs (we use warmup as total for this)
        if warmup == 0:
            return max_alpha
        p = min(1.0, effective_epoch / warmup)
        gamma = 10.0  # Standard DANN gamma
        return max_alpha * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)

    elif config.grl_schedule == "delayed":
        # Step function: 0 until delay, then jump to max
        return max_alpha if effective_epoch >= 0 else 0.0

    elif config.grl_schedule == "cosine":
        # Cosine annealing warmup
        if warmup == 0:
            return max_alpha
        p = min(1.0, effective_epoch / warmup)
        return max_alpha * 0.5 * (1.0 - math.cos(math.pi * p))

    else:
        raise ValueError(f"Unknown GRL schedule: {config.grl_schedule}")


# ============================================================================
# DOMAIN ADAPTATION LOSSES
# ============================================================================

def compute_mmd_loss(src, tgt, kernel='multiscale'):
    """MMD with multiple kernel options."""
    def gauss(x, y, sigma):
        return torch.exp(-torch.cdist(x, y, p=2)**2 / (2 * sigma**2 + 1e-8))

    n_s, n_t = src.size(0), tgt.size(0)

    if kernel == 'multiscale':
        bws = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        K_ss = sum(gauss(src, src, b) for b in bws) / len(bws)
        K_tt = sum(gauss(tgt, tgt, b) for b in bws) / len(bws)
        K_st = sum(gauss(src, tgt, b) for b in bws) / len(bws)
    elif kernel == 'rbf':
        with torch.no_grad():
            sigma = torch.median(torch.cdist(torch.cat([src, tgt]), torch.cat([src, tgt]))).clamp(min=0.1)
        K_ss, K_tt, K_st = gauss(src, src, sigma), gauss(tgt, tgt, sigma), gauss(src, tgt, sigma)
    elif kernel == 'linear':
        K_ss, K_tt, K_st = src @ src.T, tgt @ tgt.T, src @ tgt.T
    elif kernel == 'poly':
        K_ss, K_tt, K_st = (src @ src.T + 1)**2, (tgt @ tgt.T + 1)**2, (src @ tgt.T + 1)**2
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    mmd = (K_ss.sum() - K_ss.trace()) / (n_s * (n_s - 1) + 1e-8)
    mmd += (K_tt.sum() - K_tt.trace()) / (n_t * (n_t - 1) + 1e-8)
    mmd -= 2 * K_st.sum() / (n_s * n_t + 1e-8)
    return mmd.clamp(min=0)


def compute_coral_loss(src, tgt):
    """CORAL: align covariance matrices between domains."""
    d, n_s, n_t = src.size(1), src.size(0), tgt.size(0)
    src_c, tgt_c = src - src.mean(0, keepdim=True), tgt - tgt.mean(0, keepdim=True)
    cov_s, cov_t = src_c.T @ src_c / (n_s - 1 + 1e-8), tgt_c.T @ tgt_c / (n_t - 1 + 1e-8)
    return ((cov_s - cov_t)**2).sum() / (4 * d * d)


def compute_gradient_penalty(discriminator: nn.Module, src_feat: torch.Tensor,
                             tgt_feat: torch.Tensor, device: torch.device) -> torch.Tensor:
    """WGAN-GP style gradient penalty for discriminator stability."""
    batch_size = min(src_feat.size(0), tgt_feat.size(0))
    src_feat = src_feat[:batch_size].detach()
    tgt_feat = tgt_feat[:batch_size].detach()

    # Random interpolation
    alpha = torch.rand(batch_size, 1, device=device)
    interpolated = alpha * src_feat + (1 - alpha) * tgt_feat
    interpolated.requires_grad_(True)

    # Discriminator output
    d_interpolated = discriminator(interpolated)

    # Gradient w.r.t. interpolated
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]

    # Gradient penalty: (||grad|| - 1)^2
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()

    return penalty


# ============================================================================
# DISCRIMINATOR ARCHITECTURES
# ============================================================================

def apply_spectral_norm(module: nn.Module) -> nn.Module:
    """Apply spectral normalization to all Linear layers."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, nn.utils.spectral_norm(child))
        else:
            apply_spectral_norm(child)
    return module


class StandardDiscriminator(nn.Module):
    """Standard 1-hidden-layer discriminator (from domain_adaptation_e2e.py)."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.net(x)


class StrongDiscriminator(nn.Module):
    """
    Deeper discriminator with optional spectral normalization.
    Harder to fool = maintains adversarial tension longer.
    """

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = (512, 256, 128),
                 dropout: float = 0.3, use_spectral_norm: bool = True):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),  # LeakyReLU more stable for discriminators
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 2))
        self.net = nn.Sequential(*layers)

        if use_spectral_norm:
            apply_spectral_norm(self.net)

    def forward(self, x):
        return self.net(x)


# ============================================================================
# MODEL
# ============================================================================

class DomainAdaptationModel(nn.Module):
    """
    End-to-end domain adaptation model for audio deepfake detection.

    CRITICAL: This model takes RAW AUDIO WAVEFORMS as input, NOT pre-extracted embeddings!
    - input_values: Raw audio tensor of shape (batch, num_samples) at 16kHz
    - The WavLM backbone processes raw audio end-to-end
    - Pre-extracted embeddings broke domain adaptation (~30.8% EER vs 15.34% with raw audio)

    Architecture:
    - WavLM (frozen): Raw audio -> 768-dim frame embeddings -> mean pooling -> utterance embedding
    - Classifier: 768 -> 256 -> 2 (bonafide/spoof)
    - Domain Discriminator: 768 -> [configurable] -> 2 (source/target)
    """

    def __init__(self, config: AblationConfig):
        super().__init__()
        self.config = config

        # WavLM backbone (FROZEN - this is critical for good performance)
        # Processes raw audio waveforms, NOT pre-extracted embeddings
        self.wavlm = AutoModel.from_pretrained(config.model_name)
        self.wavlm.feature_extractor._freeze_parameters()
        for p in self.wavlm.parameters():
            p.requires_grad = False

        # Task classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

        # Domain discriminator
        self.domain_disc = None
        if config.adaptation_method == 'dann':
            if config.disc_type == 'strong':
                self.domain_disc = StrongDiscriminator(
                    input_dim=config.hidden_size,
                    hidden_dims=config.disc_hidden_dims,
                    dropout=config.disc_dropout,
                    use_spectral_norm=config.disc_use_spectral_norm
                )
            else:
                self.domain_disc = StandardDiscriminator(
                    input_dim=config.hidden_size,
                    hidden_dim=config.disc_hidden_dims[0] if config.disc_hidden_dims else 256,
                    dropout=config.disc_dropout
                )

        self.grl_alpha = 0.0

    def forward(self, input_values, attention_mask=None):
        """
        Forward pass with RAW AUDIO input (end-to-end training).

        Args:
            input_values: Raw audio waveform tensor (batch, num_samples) at 16kHz
                         NOT pre-extracted embeddings!
            attention_mask: Optional mask for variable-length audio

        Returns:
            dict with 'task_logits', 'features', and optionally 'domain_logits'
        """
        # WavLM processes raw audio -> frame embeddings -> mean pooling
        features = self.wavlm(input_values, attention_mask=attention_mask).last_hidden_state.mean(1)

        result = {
            "task_logits": self.classifier(features),
            "features": features
        }

        if self.domain_disc is not None:
            reversed_features = GradientReversal.apply(features, self.grl_alpha)
            result["domain_logits"] = self.domain_disc(reversed_features)

        return result


# ============================================================================
# LOSS
# ============================================================================

class AblationLoss(nn.Module):
    """Loss function with label smoothing support."""

    def __init__(self, config: AblationConfig):
        super().__init__()
        self.config = config
        self.ce = nn.CrossEntropyLoss()

        # Domain loss with label smoothing
        if config.domain_label_smoothing > 0:
            self.domain_ce = nn.CrossEntropyLoss(label_smoothing=config.domain_label_smoothing)
        else:
            self.domain_ce = nn.CrossEntropyLoss()

    def forward(self, out, task_labels, domain_labels=None, src_feat=None, tgt_feat=None):
        task_loss = self.ce(out["task_logits"][:task_labels.size(0)], task_labels)
        adapt_loss = torch.tensor(0.0, device=task_loss.device)
        gp_loss = torch.tensor(0.0, device=task_loss.device)

        if self.config.adaptation_method == 'dann' and domain_labels is not None:
            adapt_loss = self.domain_ce(out["domain_logits"], domain_labels)

            # Gradient penalty
            if self.config.use_gradient_penalty and src_feat is not None and tgt_feat is not None:
                # Need to access discriminator without GRL for GP
                # We'll compute this in the trainer instead
                pass

        elif self.config.adaptation_method == 'mmd' and src_feat is not None:
            adapt_loss = compute_mmd_loss(src_feat, tgt_feat, self.config.mmd_kernel)

        elif self.config.adaptation_method == 'coral' and src_feat is not None:
            adapt_loss = compute_coral_loss(src_feat, tgt_feat)

        total = task_loss + self.config.domain_loss_weight * adapt_loss

        return total, {
            "task_loss": task_loss.item(),
            "adapt_loss": adapt_loss.item(),
            "total": total.item()
        }


# ============================================================================
# COMPREHENSIVE LOGGER
# ============================================================================

class AblationLogger:
    """Comprehensive logging for ablation study."""

    def __init__(self, config: AblationConfig):
        self.config = config
        self.run_dir = config.get_run_dir()
        os.makedirs(self.run_dir, exist_ok=True)

        # Save config
        with open(os.path.join(self.run_dir, "config.json"), 'w') as f:
            json.dump(asdict(config), f, indent=2, default=str)

        # CSV files for detailed logging
        self.epoch_csv = os.path.join(self.run_dir, "epoch_metrics.csv")
        self.step_csv = os.path.join(self.run_dir, "step_metrics.csv")
        self._epoch_header_written = False
        self._step_header_written = False

        # In-memory history
        self.history = []

        print(f"Logging to: {self.run_dir}")

    def log_epoch(self, epoch: int, train_metrics: Dict, eval_metrics: Dict,
                  grl_alpha: float, learning_rate: float, disc_lr: float,
                  epoch_time: float, extra: Optional[Dict] = None):
        """Log comprehensive epoch metrics."""

        record = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "grl_alpha": grl_alpha,
            "learning_rate": learning_rate,
            "disc_learning_rate": disc_lr,
            "epoch_time_sec": round(epoch_time, 2),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"eval_{k}": v for k, v in eval_metrics.items()},
        }

        if extra:
            record.update(extra)

        # Write to CSV
        write_header = not self._epoch_header_written
        with open(self.epoch_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if write_header:
                writer.writeheader()
                self._epoch_header_written = True
            writer.writerow(record)

        # Update history
        self.history.append(record)
        self._save_history()

        return record

    def log_step(self, epoch: int, step: int, metrics: Dict):
        """Log step-level metrics (optional, for debugging)."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "step": step,
            **metrics
        }

        write_header = not self._step_header_written
        with open(self.step_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if write_header:
                writer.writeheader()
                self._step_header_written = True
            writer.writerow(record)

    def _save_history(self):
        with open(os.path.join(self.run_dir, "history.json"), 'w') as f:
            json.dump(self.history, f, indent=2)

    def save_final_summary(self, best_eer: float, best_epoch: int, total_time: float):
        """Save final experiment summary."""
        summary = {
            "config": asdict(self.config),
            "best_eer": best_eer,
            "best_epoch": best_epoch,
            "total_time_sec": total_time,
            "completed_at": datetime.now().isoformat()
        }
        with open(os.path.join(self.run_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2, default=str)


# ============================================================================
# TRAINER
# ============================================================================

class AblationTrainer:
    """Trainer with comprehensive logging and stability features."""

    def __init__(self, model: DomainAdaptationModel, config: AblationConfig,
                 source_loader: DataLoader, target_loader: DataLoader,
                 eval_loader: DataLoader, resume_from: Optional[str] = None):

        self.model = model
        self.config = config
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.eval_loader = eval_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Logger
        self.logger = AblationLogger(config)

        # Move model to device
        self.model.to(self.device)

        # Optional: torch.compile() for A100 optimization
        if config.compile_model and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model, mode="reduce-overhead")

        self.criterion = AblationLoss(config)

        # Separate optimizers for classifier and discriminator
        classifier_params = list(self.model.classifier.parameters())

        if config.adaptation_method == 'dann' and self.model.domain_disc is not None:
            disc_params = list(self.model.domain_disc.parameters())

            # Main optimizer (classifier)
            self.optimizer = torch.optim.AdamW(
                classifier_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )

            # Discriminator optimizer (potentially higher LR)
            self.disc_optimizer = torch.optim.AdamW(
                disc_params,
                lr=config.learning_rate * config.disc_lr_multiplier,
                weight_decay=config.weight_decay
            )
        else:
            # Single optimizer for non-DANN methods
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
            self.disc_optimizer = None

        # Scheduler
        total_steps = len(source_loader) * config.num_epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=config.warmup_epochs / config.num_epochs
        )

        if self.disc_optimizer:
            self.disc_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.disc_optimizer,
                max_lr=config.learning_rate * config.disc_lr_multiplier,
                total_steps=total_steps,
                pct_start=config.warmup_epochs / config.num_epochs
            )
        else:
            self.disc_scheduler = None

        # AMP scaler
        self.scaler = GradScaler() if config.use_amp else None

        # State
        self.best_eer = float('inf')
        self.best_epoch = -1
        self.patience_counter = 0
        self.start_epoch = 0

        # Resume if provided
        if resume_from and os.path.exists(resume_from):
            self._load_checkpoint(resume_from)

        # Print info
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable: {trainable:,} / {total:,}")

    def _load_checkpoint(self, path: str):
        print(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if 'disc_optimizer' in ckpt and self.disc_optimizer:
            self.disc_optimizer.load_state_dict(ckpt['disc_optimizer'])
        self.start_epoch = ckpt.get('epoch', 0)
        self.best_eer = ckpt.get('best_eer', float('inf'))
        self.best_epoch = ckpt.get('best_epoch', -1)
        print(f"  Resumed at epoch {self.start_epoch}, best EER: {self.best_eer:.2f}%")

    def _save_checkpoint(self, epoch: int, eer: float, is_best: bool):
        ckpt = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'eer': eer,
            'best_eer': self.best_eer,
            'best_epoch': self.best_epoch,
            'config': asdict(self.config)
        }
        if self.disc_optimizer:
            ckpt['disc_optimizer'] = self.disc_optimizer.state_dict()

        # Save epoch checkpoint
        if self.config.save_every_epoch:
            torch.save(ckpt, os.path.join(self.logger.run_dir, f"ckpt_ep{epoch:02d}.pt"))

        # Save best
        if is_best:
            torch.save(ckpt, os.path.join(self.logger.run_dir, "best_model.pt"))

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()

        # Set GRL alpha
        grl_alpha = compute_grl_alpha(epoch, self.config)
        self.model.grl_alpha = grl_alpha

        # Accumulators
        metrics_acc = {"task_loss": 0, "adapt_loss": 0, "gp_loss": 0, "domain_acc": 0}
        grad_norms = {"classifier": 0, "discriminator": 0}
        steps = 0

        target_iter = iter(self.target_loader) if self.target_loader else None

        self.optimizer.zero_grad(set_to_none=True)
        if self.disc_optimizer:
            self.disc_optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(self.source_loader, desc=f"Ep {epoch+1}")

        for step, batch in enumerate(pbar):
            # Source batch
            s_inp = batch['input_values'].to(self.device, non_blocking=True)
            s_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            s_label = batch['task_label'].to(self.device, non_blocking=True)
            s_dom = torch.zeros(s_inp.size(0), dtype=torch.long, device=self.device)

            # Target batch (for domain adaptation)
            if target_iter and self.config.adaptation_method != 'none':
                try:
                    t_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(self.target_loader)
                    t_batch = next(target_iter)

                t_inp = t_batch['input_values'].to(self.device, non_blocking=True)
                t_mask = t_batch['attention_mask'].to(self.device, non_blocking=True)
                c_inp = torch.cat([s_inp, t_inp])
                c_mask = torch.cat([s_mask, t_mask])
                c_dom = torch.cat([s_dom, torch.ones(t_inp.size(0), dtype=torch.long, device=self.device)])
            else:
                c_inp, c_mask, c_dom = s_inp, s_mask, None

            # Forward pass
            with autocast('cuda', enabled=self.config.use_amp):
                out = self.model(c_inp, attention_mask=c_mask)

                src_feat = out['features'][:s_inp.size(0)] if self.config.adaptation_method in ['mmd', 'coral'] else None
                tgt_feat = out['features'][s_inp.size(0):] if src_feat is not None else None

                loss, m = self.criterion(out, s_label, c_dom, src_feat, tgt_feat)

                # Gradient penalty (computed separately to avoid GRL)
                gp_loss = torch.tensor(0.0, device=self.device)
                if (self.config.use_gradient_penalty and
                    self.config.adaptation_method == 'dann' and
                    self.model.domain_disc is not None and
                    c_dom is not None):

                    src_f = out['features'][:s_inp.size(0)]
                    tgt_f = out['features'][s_inp.size(0):]
                    gp_loss = compute_gradient_penalty(
                        self.model.domain_disc, src_f, tgt_f, self.device
                    )
                    loss = loss + self.config.gradient_penalty_weight * gp_loss * 0.01

                loss = loss / self.config.grad_accum_steps

            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation step
            if (step + 1) % self.config.grad_accum_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    if self.disc_optimizer:
                        self.scaler.unscale_(self.disc_optimizer)

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.classifier.parameters(), 1.0)
                if self.model.domain_disc:
                    torch.nn.utils.clip_grad_norm_(self.model.domain_disc.parameters(), 1.0)

                # Log gradient norms
                if self.config.log_gradient_norms:
                    grad_norms["classifier"] += self._compute_grad_norm(self.model.classifier)
                    if self.model.domain_disc:
                        grad_norms["discriminator"] += self._compute_grad_norm(self.model.domain_disc)

                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    if self.disc_optimizer:
                        self.scaler.step(self.disc_optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                    if self.disc_optimizer:
                        self.disc_optimizer.step()

                self.scheduler.step()
                if self.disc_scheduler:
                    self.disc_scheduler.step()

                self.optimizer.zero_grad(set_to_none=True)
                if self.disc_optimizer:
                    self.disc_optimizer.zero_grad(set_to_none=True)

            # Accumulate metrics
            metrics_acc["task_loss"] += m["task_loss"]
            metrics_acc["adapt_loss"] += m["adapt_loss"]
            metrics_acc["gp_loss"] += gp_loss.item() if isinstance(gp_loss, torch.Tensor) else gp_loss

            # Domain accuracy
            if "domain_logits" in out and c_dom is not None:
                with torch.no_grad():
                    dom_pred = out["domain_logits"].argmax(dim=1)
                    metrics_acc["domain_acc"] += (dom_pred == c_dom).float().mean().item()

            steps += 1

            pbar.set_postfix({
                'task': f"{metrics_acc['task_loss']/steps:.3f}",
                'adapt': f"{metrics_acc['adapt_loss']/steps:.4f}",
                'alpha': f"{grl_alpha:.2f}"
            })

        # Average metrics
        result = {k: v / steps for k, v in metrics_acc.items()}
        if self.config.log_gradient_norms:
            result["grad_norm_classifier"] = grad_norms["classifier"] / max(1, steps // self.config.grad_accum_steps)
            result["grad_norm_discriminator"] = grad_norms["discriminator"] / max(1, steps // self.config.grad_accum_steps)

        return result

    def _compute_grad_norm(self, module: nn.Module) -> float:
        total_norm = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        scores, labels = [], []

        for batch in self.eval_loader:
            with autocast('cuda', enabled=self.config.use_amp):
                out = self.model(
                    batch['input_values'].to(self.device),
                    batch['attention_mask'].to(self.device)
                )
            scores.extend(F.softmax(out['task_logits'], dim=1)[:, 1].cpu().numpy())
            labels.extend(batch['task_label'].numpy())

        labels, scores = np.array(labels), np.array(scores)

        if len(np.unique(labels)) < 2:
            return {"eer": 50.0, "auc": 0.5, "threshold": 0.5}

        fpr, tpr, thresholds = roc_curve(labels, scores)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2 * 100

        return {
            "eer": eer,
            "auc": roc_auc_score(labels, scores),
            "threshold": float(thresholds[eer_idx])
        }

    def train(self) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"{self.config.experiment_name} | {self.config.adaptation_method.upper()}")
        print(f"GRL: {self.config.grl_schedule} | Disc: {self.config.disc_type}")
        print(f"{'='*60}")

        start_time = time.time()

        for epoch in range(self.start_epoch, self.config.num_epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Evaluate
            eval_metrics = self.evaluate()
            eer = eval_metrics["eer"]

            epoch_time = time.time() - epoch_start

            # GRL alpha
            grl_alpha = compute_grl_alpha(epoch, self.config)

            # Learning rates
            lr = self.scheduler.get_last_lr()[0]
            disc_lr = self.disc_scheduler.get_last_lr()[0] if self.disc_scheduler else lr

            # Log
            extra = {}
            if self.config.log_gradient_norms:
                extra["grad_norm_classifier"] = train_metrics.get("grad_norm_classifier", 0)
                extra["grad_norm_discriminator"] = train_metrics.get("grad_norm_discriminator", 0)

            self.logger.log_epoch(
                epoch=epoch + 1,
                train_metrics=train_metrics,
                eval_metrics=eval_metrics,
                grl_alpha=grl_alpha,
                learning_rate=lr,
                disc_lr=disc_lr,
                epoch_time=epoch_time,
                extra=extra
            )

            # Best model
            is_best = eer < self.best_eer
            if is_best:
                self.best_eer = eer
                self.best_epoch = epoch + 1
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            self._save_checkpoint(epoch + 1, eer, is_best)

            # Print summary
            dom_status = ""
            if self.config.adaptation_method == 'dann':
                dom_loss = train_metrics['adapt_loss']
                if dom_loss < 0.3:
                    dom_status = " [disc weak]"
                elif dom_loss > 1.0:
                    dom_status = " [unstable]"
                else:
                    dom_status = " [balanced]"

            marker = " BEST" if is_best else ""
            print(f"  Ep {epoch+1}: task={train_metrics['task_loss']:.3f} "
                  f"adapt={train_metrics['adapt_loss']:.4f}{dom_status} "
                  f"alpha={grl_alpha:.2f} | EER={eer:.2f}%{marker}")

            # Early stopping
            if (self.config.early_stopping and
                epoch + 1 >= self.config.min_epochs and
                self.patience_counter >= self.config.patience):
                print(f"Early stop at epoch {epoch + 1}")
                break

            torch.cuda.empty_cache()

        total_time = time.time() - start_time

        # Save final summary
        self.logger.save_final_summary(self.best_eer, self.best_epoch, total_time)

        print(f"\n{'='*60}")
        print(f"Done {total_time/60:.1f}min | Best: {self.best_eer:.2f}% (Ep {self.best_epoch})")
        print(f"{'='*60}")

        return {
            "best_eer": self.best_eer,
            "best_epoch": self.best_epoch,
            "run_dir": self.logger.run_dir,
            "history": self.logger.history,
            "total_time": total_time
        }


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def create_balanced_sampler(ds):
    labels = [s['label'] for s in ds.samples]
    counts = [labels.count(0), labels.count(1)]
    return WeightedRandomSampler([1.0 / (counts[l] + 1e-5) for l in labels], len(labels))


def run_single_experiment(
    method: str,
    seed: int,
    src_loader: DataLoader,
    tgt_loader: DataLoader,
    eval_loader: DataLoader,
    grl_schedule: str = "linear",
    disc_type: str = "standard",
    disc_lr_multiplier: float = 1.0,
    use_gradient_penalty: bool = False,
    domain_label_smoothing: float = 0.0,
    disc_use_spectral_norm: bool = False,
    num_epochs: int = 50,
    resume_from: Optional[str] = None,
    run_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Run a single experiment with specified configuration."""

    set_seed(seed)

    config = AblationConfig(
        experiment_name=f"{method}_{grl_schedule}_{disc_type}_dlr{disc_lr_multiplier}_gp{int(use_gradient_penalty)}_ls{domain_label_smoothing}_seed{seed}",
        adaptation_method=method,
        seed=seed,
        grl_schedule=grl_schedule,
        disc_type=disc_type,
        disc_lr_multiplier=disc_lr_multiplier,
        use_gradient_penalty=use_gradient_penalty,
        domain_label_smoothing=domain_label_smoothing,
        disc_use_spectral_norm=disc_use_spectral_norm,
        num_epochs=num_epochs,
        **kwargs
    )

    if run_id:
        config.run_id = run_id

    model = DomainAdaptationModel(config)
    trainer = AblationTrainer(model, config, src_loader, tgt_loader, eval_loader, resume_from=resume_from)

    return trainer.train()


def run_ablation_study(
    src_loader: DataLoader,
    tgt_loader: DataLoader,
    eval_loader: DataLoader,
    phase: str = "screening",  # "screening" or "validation"
    methods: List[str] = None,
    grl_schedules: List[str] = None,
    disc_configs: List[Dict] = None,
    seeds: List[int] = None,
    num_epochs: int = 50,
    skip_existing: bool = True,
    base_dir: str = "/content/drive/MyDrive/Jakob/Experiments"
) -> Dict[str, Any]:
    """
    Run ablation study with screening/validation phases.

    Phase 1 (screening): Run each config with 1 seed
    Phase 2 (validation): Run promising configs with 3 seeds
    """

    # Defaults
    if methods is None:
        methods = ['none', 'dann', 'mmd', 'coral']

    if grl_schedules is None:
        grl_schedules = ['linear', 'exponential', 'delayed']

    if disc_configs is None:
        disc_configs = [
            {"disc_type": "standard", "disc_lr_multiplier": 1.0, "use_gradient_penalty": False, "domain_label_smoothing": 0.0, "disc_use_spectral_norm": False},
            {"disc_type": "strong", "disc_lr_multiplier": 3.0, "use_gradient_penalty": False, "domain_label_smoothing": 0.0, "disc_use_spectral_norm": True},
            {"disc_type": "strong", "disc_lr_multiplier": 3.0, "use_gradient_penalty": True, "domain_label_smoothing": 0.1, "disc_use_spectral_norm": True},
        ]

    if seeds is None:
        seeds = [42] if phase == "screening" else [42, 123, 456]

    results = []

    print(f"\n{'='*70}")
    print(f"ABLATION STUDY | Phase: {phase.upper()} | Seeds: {seeds}")
    print(f"Methods: {methods}")
    print(f"GRL Schedules: {grl_schedules}")
    print(f"Disc Configs: {len(disc_configs)}")
    print(f"{'='*70}\n")

    # Generate all configurations
    configs = []
    for method in methods:
        if method == 'none':
            # Baseline doesn't need GRL/disc configs
            for seed in seeds:
                configs.append({
                    "method": method,
                    "seed": seed,
                    "grl_schedule": "linear",
                    **disc_configs[0]  # Use default disc config
                })
        elif method == 'dann':
            # DANN: ablate GRL schedule and discriminator
            for grl_schedule in grl_schedules:
                for disc_config in disc_configs:
                    for seed in seeds:
                        configs.append({
                            "method": method,
                            "seed": seed,
                            "grl_schedule": grl_schedule,
                            **disc_config
                        })
        else:
            # MMD/CORAL: no discriminator, just method
            for seed in seeds:
                configs.append({
                    "method": method,
                    "seed": seed,
                    "grl_schedule": "linear",
                    **disc_configs[0]
                })

    print(f"Total configurations: {len(configs)}")

    # Run experiments
    for i, cfg in enumerate(configs):
        method = cfg["method"]
        seed = cfg["seed"]
        grl_schedule = cfg["grl_schedule"]
        disc_type = cfg.get("disc_type", "standard")

        exp_name = f"{method}_{grl_schedule}_{disc_type}_dlr{cfg['disc_lr_multiplier']}_gp{int(cfg['use_gradient_penalty'])}_ls{cfg['domain_label_smoothing']}_seed{seed}"
        exp_dir = os.path.join(base_dir, exp_name)

        print(f"\n[{i+1}/{len(configs)}] {exp_name}")

        # Check for existing completed run
        if skip_existing and os.path.exists(exp_dir):
            runs = sorted([d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))], reverse=True)

            for run in runs:
                summary_path = os.path.join(exp_dir, run, "summary.json")
                if os.path.exists(summary_path):
                    with open(summary_path) as f:
                        summary = json.load(f)
                    print(f"  SKIP - completed: EER={summary['best_eer']:.2f}%")
                    results.append({
                        **cfg,
                        "best_eer": summary["best_eer"],
                        "best_epoch": summary["best_epoch"],
                        "run_dir": os.path.join(exp_dir, run),
                        "skipped": True
                    })
                    break
            else:
                # No completed run found, check for partial
                # ... (resume logic similar to domain_adaptation_e2e.py)
                r = run_single_experiment(
                    method=method,
                    seed=seed,
                    src_loader=src_loader,
                    tgt_loader=tgt_loader,
                    eval_loader=eval_loader,
                    grl_schedule=grl_schedule,
                    disc_type=disc_type,
                    disc_lr_multiplier=cfg["disc_lr_multiplier"],
                    use_gradient_penalty=cfg["use_gradient_penalty"],
                    domain_label_smoothing=cfg["domain_label_smoothing"],
                    disc_use_spectral_norm=cfg.get("disc_use_spectral_norm", False),
                    num_epochs=num_epochs
                )
                r.update(cfg)
                r["skipped"] = False
                results.append(r)
        else:
            r = run_single_experiment(
                method=method,
                seed=seed,
                src_loader=src_loader,
                tgt_loader=tgt_loader,
                eval_loader=eval_loader,
                grl_schedule=grl_schedule,
                disc_type=disc_type,
                disc_lr_multiplier=cfg["disc_lr_multiplier"],
                use_gradient_penalty=cfg["use_gradient_penalty"],
                domain_label_smoothing=cfg["domain_label_smoothing"],
                disc_use_spectral_norm=cfg.get("disc_use_spectral_norm", False),
                num_epochs=num_epochs
            )
            r.update(cfg)
            r["skipped"] = False
            results.append(r)

    # Summary
    print(f"\n{'='*70}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*70}")

    # Group by method
    for method in methods:
        method_results = [r for r in results if r['method'] == method]
        if method_results:
            eers = [r['best_eer'] for r in method_results]
            print(f"\n{method.upper()}:")
            print(f"  Count: {len(eers)}")
            print(f"  Mean EER: {np.mean(eers):.2f}% +/- {np.std(eers):.2f}%")
            print(f"  Best EER: {np.min(eers):.2f}%")

            # Best config for DANN
            if method == 'dann':
                best_idx = np.argmin(eers)
                best_cfg = method_results[best_idx]
                print(f"  Best config: grl={best_cfg['grl_schedule']}, disc={best_cfg['disc_type']}, "
                      f"dlr={best_cfg['disc_lr_multiplier']}, gp={best_cfg['use_gradient_penalty']}")

    return {
        "results": results,
        "phase": phase,
        "total_configs": len(configs)
    }


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    print("Import this module and use run_single_experiment() or run_ablation_study()")
    print("\nExample:")
    print("  from domain_adaptation_ablation import run_single_experiment, AblationConfig")
    print("  result = run_single_experiment('dann', seed=42, ...)")
