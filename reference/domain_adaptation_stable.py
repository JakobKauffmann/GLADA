"""
Domain Adaptation with Stable Adversarial Training
===================================================
Goal: Achieve 15% EER with stable training (no collapse)

Key improvements over basic DANN:
1. Stronger discriminator (3 layers + spectral norm)
2. Discriminator warmup (pretrain before GRL)
3. Gradient penalty for stability
4. Separate discriminator optimizer with higher LR
5. Domain loss target range monitoring
"""

import os
import json
import math
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import roc_curve

# Audio loading
try:
    import soundfile as sf
    import librosa
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    import torchaudio

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class StableDANNConfig:
    """Configuration for stable adversarial domain adaptation."""
    
    # Model
    model_name: str = "microsoft/wavlm-base-plus"
    hidden_size: int = 768
    
    # Training
    num_epochs: int = 50
    batch_size: int = 48
    effective_batch_size: int = 384
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_audio_length: int = 64000  # 4 seconds
    
    # Domain adaptation - STABLE settings
    domain_loss_weight: float = 0.1
    grl_warmup_epochs: int = 10          # Slower warmup (was 5)
    grl_max_alpha: float = 1.0
    disc_pretrain_epochs: int = 2        # NEW: Pretrain discriminator
    gradient_penalty_weight: float = 10.0  # NEW: GP for stability
    
    # Discriminator - STRONGER
    disc_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    disc_dropout: float = 0.3
    disc_lr_multiplier: float = 3.0      # NEW: Higher LR for discriminator
    use_spectral_norm: bool = True       # NEW: Spectral normalization
    
    # Target domain loss range (for monitoring)
    target_domain_loss_min: float = 0.3  # Below this = discriminator too weak
    target_domain_loss_max: float = 1.0  # Above this = might collapse
    
    # Early stopping
    patience: int = 10
    min_epochs: int = 15                 # NEW: Don't stop too early
    
    # System
    num_workers: int = 0
    output_dir: str = "./runs_stable"
    
    def __post_init__(self):
        self.grad_accum_steps = max(1, self.effective_batch_size // (self.batch_size * 2))
        print(f"📊 Batch: {self.batch_size} × 2 × {self.grad_accum_steps} = {self.batch_size * 2 * self.grad_accum_steps}")


# ============================================================================
# Spectral Normalization Helper
# ============================================================================

def apply_spectral_norm(module: nn.Module) -> nn.Module:
    """Apply spectral normalization to all linear layers."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, nn.utils.spectral_norm(child))
        else:
            apply_spectral_norm(child)
    return module


# ============================================================================
# Gradient Reversal Layer
# ============================================================================

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversal(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.0
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)
    
    def set_alpha(self, alpha):
        self.alpha = alpha


# ============================================================================
# Strong Domain Discriminator
# ============================================================================

class StrongDiscriminator(nn.Module):
    """
    Deeper discriminator with spectral normalization.
    Harder to fool = better gradient signal for domain adaptation.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.3,
                 use_spectral_norm: bool = True):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),  # LeakyReLU more stable than GELU for discriminators
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
# Model with Stable DANN
# ============================================================================

class StableDANNModel(nn.Module):
    """WavLM + Classifier + Strong Discriminator with GRL."""
    
    def __init__(self, config: StableDANNConfig):
        super().__init__()
        self.config = config
        
        # Load WavLM (frozen)
        from transformers import WavLMModel
        self.wavlm = WavLMModel.from_pretrained(config.model_name)
        for param in self.wavlm.parameters():
            param.requires_grad = False
        
        # Task classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )
        
        # Gradient reversal
        self.grl = GradientReversal()
        
        # Strong discriminator
        self.discriminator = StrongDiscriminator(
            input_dim=config.hidden_size,
            hidden_dims=config.disc_hidden_dims,
            dropout=config.disc_dropout,
            use_spectral_norm=config.use_spectral_norm
        )
    
    def forward(self, audio: torch.Tensor, return_features: bool = False):
        """Forward pass through WavLM and heads."""
        # WavLM features (frozen)
        with torch.no_grad():
            outputs = self.wavlm(audio)
            features = outputs.last_hidden_state.mean(dim=1)
        
        # Task prediction
        task_logits = self.classifier(features)
        
        # Domain prediction (through GRL)
        domain_features = self.grl(features)
        domain_logits = self.discriminator(domain_features)
        
        if return_features:
            return task_logits, domain_logits, features
        return task_logits, domain_logits
    
    def get_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract features without task/domain heads."""
        with torch.no_grad():
            outputs = self.wavlm(audio)
            return outputs.last_hidden_state.mean(dim=1)


# ============================================================================
# Gradient Penalty
# ============================================================================

def compute_gradient_penalty(discriminator: nn.Module, real_features: torch.Tensor,
                             fake_features: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    WGAN-GP style gradient penalty for discriminator stability.
    Interpolate between real (source) and fake (target) features.
    """
    batch_size = min(real_features.size(0), fake_features.size(0))
    real_features = real_features[:batch_size]
    fake_features = fake_features[:batch_size]
    
    # Random interpolation
    alpha = torch.rand(batch_size, 1, device=device)
    interpolated = alpha * real_features + (1 - alpha) * fake_features
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
# Dataset Classes (same as before)
# ============================================================================

class ASVspoofDataset(Dataset):
    """ASVspoof 2019 LA dataset."""
    
    def __init__(self, root: str, split: str = "train", max_samples: Optional[int] = None,
                 max_length: int = 64000, balanced: bool = True):
        self.max_length = max_length
        self.samples = []
        
        protocol_map = {"train": "train", "dev": "dev", "eval": "eval"}
        protocol_file = Path(root) / f"ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{protocol_map[split]}.trn.txt"
        if not protocol_file.exists():
            protocol_file = Path(root) / f"ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{protocol_map[split]}.trl.txt"
        
        flac_dir = Path(root) / f"ASVspoof2019_LA_{split}/flac"
        
        bonafide, spoof = [], []
        with open(protocol_file) as f:
            for line in f:
                parts = line.strip().split()
                utt_id, label = parts[1], parts[-1]
                path = flac_dir / f"{utt_id}.flac"
                if path.exists():
                    if label == "bonafide":
                        bonafide.append((str(path), 0))
                    else:
                        spoof.append((str(path), 1))
        
        if balanced:
            n = min(len(bonafide), len(spoof))
            if max_samples:
                n = min(n, max_samples // 2)
            random.shuffle(bonafide)
            random.shuffle(spoof)
            self.samples = bonafide[:n] + spoof[:n]
        else:
            self.samples = bonafide + spoof
            if max_samples:
                random.shuffle(self.samples)
                self.samples = self.samples[:max_samples]
        
        random.shuffle(self.samples)
        print(f"  ASVspoof {split}: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def _load_audio(self, path: str) -> torch.Tensor:
        if HAS_SOUNDFILE:
            wav, sr = sf.read(path, dtype='float32')
            if len(wav.shape) > 1:
                wav = wav.mean(axis=1)
            if sr != 16000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            wav = torch.from_numpy(wav)
        else:
            wav, sr = torchaudio.load(path)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            wav = wav.squeeze(0)
        
        if wav.shape[0] > self.max_length:
            start = random.randint(0, wav.shape[0] - self.max_length)
            wav = wav[start:start + self.max_length]
        elif wav.shape[0] < self.max_length:
            wav = F.pad(wav, (0, self.max_length - wav.shape[0]))
        
        return wav
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        wav = self._load_audio(path)
        return wav, label


class InTheWildDataset(Dataset):
    """In-the-Wild dataset for target domain / evaluation."""
    
    def __init__(self, root: str, split: str = "train", max_samples: Optional[int] = None,
                 max_length: int = 64000, balanced: bool = True):
        self.max_length = max_length
        self.samples = []
        
        meta_file = Path(root) / "meta.csv"
        audio_dir = Path(root) / "release_in_the_wild"
        
        bonafide, spoof = [], []
        with open(meta_file) as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    filename, label = parts[0], parts[1]
                    path = audio_dir / filename
                    if path.exists():
                        if label == "bona-fide":
                            bonafide.append((str(path), 0))
                        else:
                            spoof.append((str(path), 1))
        
        if balanced:
            n = min(len(bonafide), len(spoof))
            if max_samples:
                n = min(n, max_samples // 2)
            random.shuffle(bonafide)
            random.shuffle(spoof)
            self.samples = bonafide[:n] + spoof[:n]
        else:
            self.samples = bonafide + spoof
            if max_samples:
                random.shuffle(self.samples)
                self.samples = self.samples[:max_samples]
        
        random.shuffle(self.samples)
        print(f"  In-the-Wild {split}: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def _load_audio(self, path: str) -> torch.Tensor:
        if HAS_SOUNDFILE:
            wav, sr = sf.read(path, dtype='float32')
            if len(wav.shape) > 1:
                wav = wav.mean(axis=1)
            if sr != 16000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            wav = torch.from_numpy(wav)
        else:
            wav, sr = torchaudio.load(path)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            wav = wav.squeeze(0)
        
        if wav.shape[0] > self.max_length:
            start = random.randint(0, wav.shape[0] - self.max_length)
            wav = wav[start:start + self.max_length]
        elif wav.shape[0] < self.max_length:
            wav = F.pad(wav, (0, self.max_length - wav.shape[0]))
        
        return wav
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        wav = self._load_audio(path)
        return wav, label


# ============================================================================
# Evaluation
# ============================================================================

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Equal Error Rate."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Evaluate model and return EER."""
    model.eval()
    all_labels, all_scores = [], []
    
    for audio, labels in loader:
        audio = audio.to(device)
        task_logits, _ = model(audio)
        scores = F.softmax(task_logits, dim=1)[:, 1].cpu().numpy()
        all_scores.extend(scores)
        all_labels.extend(labels.numpy())
    
    return compute_eer(np.array(all_labels), np.array(all_scores)) * 100


# ============================================================================
# Stable DANN Trainer
# ============================================================================

class StableDANNTrainer:
    """
    Trainer with stable adversarial domain adaptation.
    
    Key features:
    1. Discriminator pretraining (before GRL)
    2. Gradient penalty for stability
    3. Separate optimizer for discriminator
    4. Domain loss monitoring with target range
    """
    
    def __init__(self, config: StableDANNConfig, model: StableDANNModel,
                 source_loader: DataLoader, target_loader: DataLoader,
                 eval_loader: DataLoader, device: torch.device,
                 run_id: Optional[str] = None):
        
        self.config = config
        self.model = model.to(device)
        self.device = device
        
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.eval_loader = eval_loader
        
        # Separate optimizers
        classifier_params = list(model.classifier.parameters())
        discriminator_params = list(model.discriminator.parameters())
        
        self.optimizer_task = torch.optim.AdamW(
            classifier_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.optimizer_disc = torch.optim.AdamW(
            discriminator_params,
            lr=config.learning_rate * config.disc_lr_multiplier,
            weight_decay=config.weight_decay
        )
        
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Setup output directory
        run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(config.output_dir) / f"stable_dann_{run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)
        
        self.history = []
        self.best_eer = float('inf')
        self.best_epoch = 0
    
    def _get_grl_alpha(self, epoch: int) -> float:
        """Compute GRL alpha with warmup."""
        if epoch < self.config.disc_pretrain_epochs:
            return 0.0  # No reversal during discriminator pretraining
        
        effective_epoch = epoch - self.config.disc_pretrain_epochs
        warmup_epochs = self.config.grl_warmup_epochs
        
        if effective_epoch >= warmup_epochs:
            return self.config.grl_max_alpha
        
        # Linear warmup
        return self.config.grl_max_alpha * (effective_epoch / warmup_epochs)
    
    def pretrain_discriminator(self):
        """Pretrain discriminator to recognize domains before adversarial training."""
        print(f"\n🎯 Pretraining discriminator for {self.config.disc_pretrain_epochs} epochs...")
        
        self.model.grl.set_alpha(0.0)  # No gradient reversal
        
        for epoch in range(self.config.disc_pretrain_epochs):
            self.model.train()
            total_loss = 0
            total_acc = 0
            steps = 0
            
            target_iter = iter(self.target_loader)
            
            pbar = tqdm(self.source_loader, desc=f"Disc Pretrain {epoch+1}")
            for src_audio, _ in pbar:
                # Get target batch
                try:
                    tgt_audio, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(self.target_loader)
                    tgt_audio, _ = next(target_iter)
                
                src_audio = src_audio.to(self.device)
                tgt_audio = tgt_audio.to(self.device)
                
                # Get features
                src_features = self.model.get_features(src_audio)
                tgt_features = self.model.get_features(tgt_audio)
                
                # Domain labels: 0=source, 1=target
                src_labels = torch.zeros(src_features.size(0), dtype=torch.long, device=self.device)
                tgt_labels = torch.ones(tgt_features.size(0), dtype=torch.long, device=self.device)
                
                features = torch.cat([src_features, tgt_features], dim=0)
                labels = torch.cat([src_labels, tgt_labels], dim=0)
                
                # Forward through discriminator (no GRL)
                domain_logits = self.model.discriminator(features)
                loss = self.ce_loss(domain_logits, labels)
                
                # Gradient penalty
                gp = compute_gradient_penalty(
                    self.model.discriminator, src_features, tgt_features, self.device
                )
                loss = loss + self.config.gradient_penalty_weight * gp
                
                # Update discriminator only
                self.optimizer_disc.zero_grad()
                loss.backward()
                self.optimizer_disc.step()
                
                # Accuracy
                preds = domain_logits.argmax(dim=1)
                acc = (preds == labels).float().mean().item()
                
                total_loss += loss.item()
                total_acc += acc
                steps += 1
                
                pbar.set_postfix(loss=f"{total_loss/steps:.3f}", acc=f"{total_acc/steps:.1%}")
            
            print(f"  Pretrain Ep {epoch+1}: Loss={total_loss/steps:.3f}, Acc={total_acc/steps:.1%}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch with stable DANN."""
        self.model.train()
        
        # Set GRL alpha
        alpha = self._get_grl_alpha(epoch)
        self.model.grl.set_alpha(alpha)
        
        metrics = {"task_loss": 0, "domain_loss": 0, "gp_loss": 0, "domain_acc": 0}
        steps = 0
        
        target_iter = iter(self.target_loader)
        
        self.optimizer_task.zero_grad()
        self.optimizer_disc.zero_grad()
        
        pbar = tqdm(self.source_loader, desc=f"Ep {epoch+1}")
        
        for batch_idx, (src_audio, src_labels) in enumerate(pbar):
            # Get target batch
            try:
                tgt_audio, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(self.target_loader)
                tgt_audio, _ = next(target_iter)
            
            src_audio = src_audio.to(self.device)
            src_labels = src_labels.to(self.device)
            tgt_audio = tgt_audio.to(self.device)
            
            # Forward pass
            src_task_logits, src_domain_logits, src_features = self.model(src_audio, return_features=True)
            _, tgt_domain_logits, tgt_features = self.model(tgt_audio, return_features=True)
            
            # Task loss (source only)
            task_loss = self.ce_loss(src_task_logits, src_labels)
            
            # Domain loss
            domain_labels = torch.cat([
                torch.zeros(src_domain_logits.size(0), dtype=torch.long, device=self.device),
                torch.ones(tgt_domain_logits.size(0), dtype=torch.long, device=self.device)
            ])
            domain_logits = torch.cat([src_domain_logits, tgt_domain_logits], dim=0)
            domain_loss = self.ce_loss(domain_logits, domain_labels)
            
            # Gradient penalty (for discriminator stability)
            gp = compute_gradient_penalty(
                self.model.discriminator,
                src_features.detach(),
                tgt_features.detach(),
                self.device
            )
            
            # Total loss
            total_loss = task_loss + self.config.domain_loss_weight * domain_loss
            total_loss = total_loss + self.config.gradient_penalty_weight * gp * 0.01  # Small GP weight
            total_loss = total_loss / self.config.grad_accum_steps
            
            total_loss.backward()
            
            # Domain accuracy (for monitoring)
            with torch.no_grad():
                domain_preds = domain_logits.argmax(dim=1)
                domain_acc = (domain_preds == domain_labels).float().mean().item()
            
            metrics["task_loss"] += task_loss.item()
            metrics["domain_loss"] += domain_loss.item()
            metrics["gp_loss"] += gp.item()
            metrics["domain_acc"] += domain_acc
            steps += 1
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer_task.step()
                self.optimizer_disc.step()
                self.optimizer_task.zero_grad()
                self.optimizer_disc.zero_grad()
            
            pbar.set_postfix(
                task=f"{metrics['task_loss']/steps:.3f}",
                dom=f"{metrics['domain_loss']/steps:.3f}",
                α=f"{alpha:.2f}"
            )
        
        # Average metrics
        for k in metrics:
            metrics[k] /= steps
        metrics["grl_alpha"] = alpha
        
        return metrics
    
    def train(self):
        """Full training loop with stable DANN."""
        print(f"\n{'='*60}")
        print(f"🚀 Stable DANN Training")
        print(f"   Disc pretrain: {self.config.disc_pretrain_epochs} epochs")
        print(f"   GRL warmup: {self.config.grl_warmup_epochs} epochs")
        print(f"   Gradient penalty: {self.config.gradient_penalty_weight}")
        print(f"{'='*60}\n")
        
        # Step 1: Pretrain discriminator
        if self.config.disc_pretrain_epochs > 0:
            self.pretrain_discriminator()
        
        # Step 2: Main training with DANN
        print(f"\n🎯 Starting adversarial training...")
        
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Evaluate
            eer = evaluate(self.model, self.eval_loader, self.device)
            
            # Domain loss status
            dom_loss = train_metrics["domain_loss"]
            if dom_loss < self.config.target_domain_loss_min:
                status = "⚠️ disc weak"
            elif dom_loss > self.config.target_domain_loss_max:
                status = "⚠️ unstable"
            else:
                status = "✅ balanced"
            
            # Log
            is_best = eer < self.best_eer
            if is_best:
                self.best_eer = eer
                self.best_epoch = epoch + 1
                patience_counter = 0
                self._save_checkpoint(epoch + 1, eer, is_best=True)
                print(f"  Ep {epoch+1}: task={train_metrics['task_loss']:.3f} "
                      f"dom={dom_loss:.3f} ({status}) α={train_metrics['grl_alpha']:.2f} "
                      f"| EER={eer:.2f}% 🏆")
            else:
                patience_counter += 1
                print(f"  Ep {epoch+1}: task={train_metrics['task_loss']:.3f} "
                      f"dom={dom_loss:.3f} ({status}) α={train_metrics['grl_alpha']:.2f} "
                      f"| EER={eer:.2f}%")
            
            # History
            self.history.append({
                "epoch": epoch + 1,
                "train_task_loss": train_metrics["task_loss"],
                "train_domain_loss": train_metrics["domain_loss"],
                "train_domain_acc": train_metrics["domain_acc"],
                "grl_alpha": train_metrics["grl_alpha"],
                "eval_eer": eer,
                "is_best": is_best
            })
            self._save_history()
            
            # Early stopping (but not before min_epochs)
            if epoch + 1 >= self.config.min_epochs and patience_counter >= self.config.patience:
                print(f"⏹️ Early stop at epoch {epoch+1}")
                break
        
        print(f"\n{'='*60}")
        print(f"✅ Done | Best: {self.best_eer:.2f}% (Epoch {self.best_epoch})")
        print(f"{'='*60}")
        
        return {
            "best_eer": self.best_eer,
            "best_epoch": self.best_epoch,
            "history": self.history
        }
    
    def _save_checkpoint(self, epoch: int, eer: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_task_state_dict": self.optimizer_task.state_dict(),
            "optimizer_disc_state_dict": self.optimizer_disc.state_dict(),
            "eer": eer,
            "best_eer": self.best_eer,
            "best_epoch": self.best_epoch
        }
        
        if is_best:
            torch.save(checkpoint, self.run_dir / "best_model.pt")
        torch.save(checkpoint, self.run_dir / f"ckpt_ep{epoch:02d}.pt")
    
    def _save_history(self):
        """Save training history."""
        with open(self.run_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)


# ============================================================================
# Main Entry Point
# ============================================================================

def run_stable_dann(
    source_loader: DataLoader,
    target_loader: DataLoader,
    eval_loader: DataLoader,
    config: Optional[StableDANNConfig] = None,
    seed: int = 42
) -> Dict:
    """Run stable DANN training."""
    
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Config
    config = config or StableDANNConfig()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")
    
    # Model
    model = StableDANNModel(config)
    
    # Train
    trainer = StableDANNTrainer(
        config=config,
        model=model,
        source_loader=source_loader,
        target_loader=target_loader,
        eval_loader=eval_loader,
        device=device
    )
    
    results = trainer.train()
    results["run_dir"] = str(trainer.run_dir)
    
    return results


if __name__ == "__main__":
    print("Import this module and call run_stable_dann()")
