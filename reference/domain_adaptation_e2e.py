# domain_adaptation_e2e.py
# -*- coding: utf-8 -*-
"""
End-to-End Domain Adaptation for Audio Deepfake Detection

KEY INSIGHTS FROM 15.34% EER BREAKTHROUGH:
1. End-to-end training (NOT pre-extracted embeddings)
2. Frozen WavLM backbone (NOT LoRA fine-tuning)
3. domain_loss_weight = 0.1 (NOT 0.5)
4. Linear GRL warmup over 5 epochs

Author: Jakob Kauffmann
Thesis: Cross-Domain Generalization in Audio Deepfake Detection
"""

import os, random, json, time, warnings, logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

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

import librosa  # For resampling if needed
from torch.amp import autocast, GradScaler
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm.auto import tqdm

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print("🚀 Ampere+ GPU: TF32 enabled")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DAConfig:
    experiment_name: str = "domain_adaptation"
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    adaptation_method: str = "dann"  # none, dann, mmd, coral
    model_name: str = "microsoft/wavlm-base-plus"
    hidden_size: int = 768
    use_lora: bool = False  # CRITICAL: frozen worked best!
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Tuple[str, ...] = ("q_proj", "v_proj", "k_proj", "out_proj")
    domain_loss_weight: float = 0.1  # CRITICAL: NOT 0.5!
    grl_warmup_epochs: int = 5
    mmd_kernel: str = "multiscale"  # rbf, linear, poly, multiscale
    batch_size: int = 48
    effective_batch_size: int = 384
    grad_accum_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 50  # Early stopping will halt if converged
    warmup_epochs: int = 1
    early_stopping: bool = True
    patience: int = 10  # Stop after 10 epochs without improvement
    num_workers: int = 0  # 0 avoids multiprocessing issues in Colab
    use_amp: bool = True
    max_audio_len: int = 64000
    drive_log_dir: str = "/content/drive/MyDrive/Jakob/Experiments"
    save_every_epoch: bool = True
    seed: int = 42
    
    def __post_init__(self):
        uses_da = self.adaptation_method in ["dann", "mmd", "coral"]
        actual = self.batch_size * (2 if uses_da else 1)
        self.grad_accum_steps = max(1, self.effective_batch_size // actual)
        print(f"📊 Batch: {self.batch_size} x {2 if uses_da else 1} x {self.grad_accum_steps} = {actual * self.grad_accum_steps}")
    
    def get_run_dir(self): return os.path.join(self.drive_log_dir, self.experiment_name, self.run_id)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ============================================================================
# DATASETS
# ============================================================================

class ASVspoofDataset(Dataset):
    def __init__(self, protocol_path, audio_dir, max_len=64000, max_samples=None):
        self.audio_dir, self.max_len = Path(audio_dir), max_len
        self.samples = []
        with open(protocol_path) as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5: self.samples.append({'filename': p[1], 'label': 1 if p[4] == 'bonafide' else 0})
        if max_samples: self.samples = self.samples[:max_samples]
        print(f"  📂 ASVspoof: {len(self.samples)} samples")
    
    def __len__(self): return len(self.samples)
    
    def _load_audio(self, path):
        """Load audio using soundfile (primary) or torchaudio (fallback)."""
        try:
            if HAS_SOUNDFILE:
                wav, sr = sf.read(path, dtype='float32')
                if len(wav.shape) > 1:  # Stereo to mono
                    wav = wav.mean(axis=1)
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                return torch.from_numpy(wav)
            else:
                wav, sr = torchaudio.load(path)
                if sr != 16000:
                    wav = torchaudio.transforms.Resample(sr, 16000)(wav)
                return wav.mean(0) if wav.shape[0] > 1 else wav.squeeze(0)
        except Exception as e:
            return None
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        wav = self._load_audio(self.audio_dir / f"{s['filename']}.flac")
        if wav is None:
            wav = torch.zeros(self.max_len)
        wav = F.pad(wav, (0, max(0, self.max_len - wav.shape[0])))[:self.max_len]
        return {'input_values': wav, 'attention_mask': torch.ones(self.max_len, dtype=torch.long), 'task_label': torch.tensor(s['label'], dtype=torch.long)}

class InTheWildDataset(Dataset):
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
        if max_samples: self.samples = self.samples[:max_samples]
        print(f"  📂 ITW: {len(self.samples)} ({'labeled' if use_labels else 'unlabeled'}, {counts})")
    
    def __len__(self): return len(self.samples)
    
    def _load_audio(self, path):
        """Load audio using soundfile (primary) or torchaudio (fallback)."""
        try:
            if HAS_SOUNDFILE:
                wav, sr = sf.read(path, dtype='float32')
                if len(wav.shape) > 1:  # Stereo to mono
                    wav = wav.mean(axis=1)
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                return torch.from_numpy(wav)
            else:
                wav, sr = torchaudio.load(path)
                if sr != 16000:
                    wav = torchaudio.transforms.Resample(sr, 16000)(wav)
                return wav.mean(0) if wav.shape[0] > 1 else wav.squeeze(0)
        except Exception as e:
            return None
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        wav = self._load_audio(self.audio_dir / s['filename'])
        if wav is None:
            wav = torch.zeros(self.max_len)
        wav = F.pad(wav, (0, max(0, self.max_len - wav.shape[0])))[:self.max_len]
        return {'input_values': wav, 'attention_mask': torch.ones(self.max_len, dtype=torch.long), 'task_label': torch.tensor(s['label'] if self.use_labels else 0, dtype=torch.long)}

# ============================================================================
# DOMAIN ADAPTATION LOSSES
# ============================================================================

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha): ctx.alpha = alpha; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad): return grad.neg() * ctx.alpha, None

def compute_mmd_loss(src, tgt, kernel='multiscale'):
    """MMD with multiple kernel options. Multiscale RBF is most robust."""
    def gauss(x, y, sigma): return torch.exp(-torch.cdist(x, y, p=2)**2 / (2 * sigma**2 + 1e-8))
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

# ============================================================================
# MODEL
# ============================================================================

class DomainAdaptationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wavlm = AutoModel.from_pretrained(config.model_name)
        self.wavlm.feature_extractor._freeze_parameters()
        
        if config.use_lora:
            self.wavlm = get_peft_model(self.wavlm, LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, r=config.lora_r,
                lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                target_modules=list(config.lora_target_modules)))
            for n, p in self.wavlm.named_parameters():
                if "lora" not in n.lower(): p.requires_grad = False
        else:
            for p in self.wavlm.parameters(): p.requires_grad = False
        
        self.classifier = nn.Sequential(nn.Linear(config.hidden_size, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 2))
        self.domain_disc = nn.Sequential(nn.Linear(config.hidden_size, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 2)) if config.adaptation_method == 'dann' else None
        self.grl_alpha = 0.0
    
    def forward(self, input_values, attention_mask=None):
        backbone = self.wavlm.base_model if hasattr(self.wavlm, "base_model") else self.wavlm
        features = backbone(input_values, attention_mask=attention_mask).last_hidden_state.mean(1)
        result = {"task_logits": self.classifier(features), "features": features}
        if self.domain_disc: result["domain_logits"] = self.domain_disc(GradientReversal.apply(features, self.grl_alpha))
        return result

# ============================================================================
# LOSS & TRAINER
# ============================================================================

class DALoss(nn.Module):
    def __init__(self, config):
        super().__init__(); self.config = config; self.ce = nn.CrossEntropyLoss()
    
    def forward(self, out, task_labels, domain_labels=None, src_feat=None, tgt_feat=None):
        task_loss = self.ce(out["task_logits"][:task_labels.size(0)], task_labels)
        adapt_loss = torch.tensor(0.0, device=task_loss.device)
        if self.config.adaptation_method == 'dann' and domain_labels is not None:
            adapt_loss = self.ce(out["domain_logits"], domain_labels)
        elif self.config.adaptation_method == 'mmd' and src_feat is not None:
            adapt_loss = compute_mmd_loss(src_feat, tgt_feat, self.config.mmd_kernel)
        elif self.config.adaptation_method == 'coral' and src_feat is not None:
            adapt_loss = compute_coral_loss(src_feat, tgt_feat)
        total = task_loss + self.config.domain_loss_weight * adapt_loss
        return total, {"task_loss": task_loss.item(), "adapt_loss": adapt_loss.item(), "total": total.item()}

class DATrainer:
    def __init__(self, model, config, source_loader, target_loader, eval_loader, resume_from=None):
        self.model, self.config = model, config
        self.source_loader, self.target_loader, self.eval_loader = source_loader, target_loader, eval_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_dir = config.get_run_dir(); os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, "config.json"), 'w') as f: json.dump(asdict(config), f, indent=2, default=str)
        print(f"📁 {self.run_dir}")
        
        self.model.to(self.device); self.criterion = DALoss(config)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        total_steps = len(source_loader) * config.num_epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=config.learning_rate, total_steps=total_steps, pct_start=config.warmup_epochs / config.num_epochs)
        self.scaler = GradScaler() if config.use_amp else None
        self.best_eer, self.best_epoch, self.patience_counter, self.history = float('inf'), -1, 0, []
        self.start_epoch = 0
        
        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            print(f"📂 Resuming from {resume_from}")
            ckpt = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
            if 'optimizer' in ckpt: self.optimizer.load_state_dict(ckpt['optimizer'])
            self.start_epoch = ckpt.get('epoch', 0)
            self.best_eer = ckpt.get('best_eer', float('inf'))
            self.best_epoch = ckpt.get('best_epoch', -1)
            # Load history if exists
            hist_path = os.path.join(os.path.dirname(resume_from), "history.json")
            if os.path.exists(hist_path):
                with open(hist_path) as f: self.history = json.load(f)
            print(f"   Resumed at epoch {self.start_epoch}, best EER: {self.best_eer:.2f}%")
        
        print(f"🔧 Trainable: {sum(p.numel() for p in params):,} / {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, epoch):
        self.model.train(); acc = {"task_loss": 0, "adapt_loss": 0}; steps = 0
        if self.config.adaptation_method == 'dann': self.model.grl_alpha = min(1.0, epoch / self.config.grl_warmup_epochs)
        target_iter = iter(self.target_loader) if self.target_loader else None
        self.optimizer.zero_grad(set_to_none=True)
        
        for step, batch in enumerate(pbar := tqdm(self.source_loader, desc=f"Ep {epoch+1}")):
            s_inp = batch['input_values'].to(self.device, non_blocking=True)
            s_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            s_label = batch['task_label'].to(self.device, non_blocking=True)
            s_dom = torch.zeros(s_inp.size(0), dtype=torch.long, device=self.device)
            
            if target_iter and self.config.adaptation_method != 'none':
                try: t_batch = next(target_iter)
                except StopIteration: target_iter = iter(self.target_loader); t_batch = next(target_iter)
                t_inp = t_batch['input_values'].to(self.device, non_blocking=True)
                t_mask = t_batch['attention_mask'].to(self.device, non_blocking=True)
                c_inp, c_mask = torch.cat([s_inp, t_inp]), torch.cat([s_mask, t_mask])
                c_dom = torch.cat([s_dom, torch.ones(t_inp.size(0), dtype=torch.long, device=self.device)])
            else: c_inp, c_mask, c_dom = s_inp, s_mask, None
            
            with autocast('cuda', enabled=self.config.use_amp):
                out = self.model(c_inp, attention_mask=c_mask)
                src_f = out['features'][:s_inp.size(0)] if self.config.adaptation_method in ['mmd', 'coral'] else None
                tgt_f = out['features'][s_inp.size(0):] if src_f is not None else None
                loss, m = self.criterion(out, s_label, c_dom, src_f, tgt_f)
                loss = loss / self.config.grad_accum_steps
            
            (self.scaler.scale(loss) if self.scaler else loss).backward()
            if (step + 1) % self.config.grad_accum_steps == 0:
                if self.scaler: self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                (self.scaler.step if self.scaler else lambda x: x)(self.optimizer); 
                if self.scaler: self.scaler.update()
                else: self.optimizer.step()
                self.scheduler.step(); self.optimizer.zero_grad(set_to_none=True)
            
            for k in acc: acc[k] += m.get(k, 0)
            steps += 1; pbar.set_postfix({'task': f"{acc['task_loss']/steps:.3f}", 'adapt': f"{acc['adapt_loss']/steps:.4f}"})
        return {k: v/steps for k, v in acc.items()}
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval(); scores, labels = [], []
        for batch in self.eval_loader:
            with autocast('cuda', enabled=self.config.use_amp):
                out = self.model(batch['input_values'].to(self.device), batch['attention_mask'].to(self.device))
            scores.extend(F.softmax(out['task_logits'], dim=1)[:, 1].cpu().numpy())
            labels.extend(batch['task_label'].numpy())
        labels, scores = np.array(labels), np.array(scores)
        if len(np.unique(labels)) < 2: return {"eer": 50.0, "auc": 0.5}
        fpr, tpr, _ = roc_curve(labels, scores); fnr = 1 - tpr
        eer = (fpr[np.nanargmin(np.abs(fpr - fnr))] + fnr[np.nanargmin(np.abs(fpr - fnr))]) / 2 * 100
        return {"eer": eer, "auc": roc_auc_score(labels, scores)}
    
    def train(self):
        print(f"\n{'='*60}\n🚀 {self.config.experiment_name} | {self.config.adaptation_method.upper()} | LoRA={self.config.use_lora} | λ={self.config.domain_loss_weight}\n{'='*60}")
        if self.start_epoch > 0:
            print(f"   Resuming from epoch {self.start_epoch + 1}")
        start = time.time()
        for epoch in range(self.start_epoch, self.config.num_epochs):
            train_m, eval_m = self.train_epoch(epoch), self.evaluate()
            eer = eval_m["eer"]
            self.history.append({"epoch": epoch+1, "train": train_m, "eval": eval_m})
            with open(os.path.join(self.run_dir, "history.json"), 'w') as f: json.dump(self.history, f, indent=2)
            is_best = eer < self.best_eer
            if is_best: self.best_eer, self.best_epoch, self.patience_counter = eer, epoch+1, 0
            else: self.patience_counter += 1
            torch.save({'epoch': epoch+1, 'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'eer': eer, 'best_eer': self.best_eer, 'best_epoch': self.best_epoch}, os.path.join(self.run_dir, "best_model.pt" if is_best else f"ckpt_ep{epoch+1:02d}.pt"))
            alpha_s = f" α={self.model.grl_alpha:.2f}" if self.config.adaptation_method == 'dann' else ""
            print(f"  Ep {epoch+1}: task={train_m['task_loss']:.3f} adapt={train_m['adapt_loss']:.4f}{alpha_s} | EER={eer:.2f}%{' 🏆' if is_best else ''}")
            if self.config.early_stopping and self.patience_counter >= self.config.patience: print(f"⏹️ Early stop"); break
            torch.cuda.empty_cache()
        print(f"\n{'='*60}\n✅ Done {(time.time()-start)/60:.1f}min | Best: {self.best_eer:.2f}% (Ep {self.best_epoch})\n{'='*60}")
        return {"best_eer": self.best_eer, "best_epoch": self.best_epoch, "run_dir": self.run_dir, "history": self.history}

# ============================================================================
# HELPERS
# ============================================================================

def create_balanced_sampler(ds):
    labels = [s['label'] for s in ds.samples]; counts = [labels.count(0), labels.count(1)]
    return WeightedRandomSampler([1.0/(counts[l]+1e-5) for l in labels], len(labels))

def run_experiment(method, seed, src_loader, tgt_loader, eval_loader, use_lora=False, num_epochs=50, domain_loss_weight=0.1, resume_from=None, run_id=None, **kw):
    """Run single experiment with optional resume."""
    set_seed(seed)
    cfg = DAConfig(experiment_name=f"{method}_{'lora' if use_lora else 'frozen'}_seed{seed}", adaptation_method=method, use_lora=use_lora, num_epochs=num_epochs, domain_loss_weight=domain_loss_weight, seed=seed, **kw)
    if run_id: cfg.run_id = run_id  # Reuse existing run directory
    return DATrainer(DomainAdaptationModel(cfg), cfg, src_loader, tgt_loader, eval_loader, resume_from=resume_from).train()

def run_comparison(src_loader, tgt_loader, eval_loader, methods=['none', 'dann', 'mmd', 'coral'], seeds=[42, 123, 456], use_lora=False, num_epochs=50, skip_existing=True):
    """Run full comparison with skip completed + resume partial experiments."""
    results = []
    base_dir = "/content/drive/MyDrive/Jakob/Experiments"
    
    print(f"\n{'='*70}\nDOMAIN ADAPTATION COMPARISON | Methods: {methods} | Seeds: {seeds}\n{'='*70}")
    
    for m in methods:
        for s in seeds:
            exp_name = f"{m}_{'lora' if use_lora else 'frozen'}_seed{s}"
            exp_dir = os.path.join(base_dir, exp_name)
            
            # Check for existing runs
            if skip_existing and os.path.exists(exp_dir):
                runs = sorted([d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))], reverse=True)
                found_complete = False
                resume_path = None
                
                for run in runs:
                    hist_path = os.path.join(exp_dir, run, "history.json")
                    best_path = os.path.join(exp_dir, run, "best_model.pt")
                    
                    if os.path.exists(hist_path) and os.path.exists(best_path):
                        with open(hist_path) as f: hist = json.load(f)
                        ckpt = torch.load(best_path, map_location='cpu')
                        epochs_done = len(hist)
                        best_eer = ckpt.get('best_eer', hist[-1]['eval']['eer'])
                        best_epoch = ckpt.get('best_epoch', epochs_done)
                        
                        # Check if completed (early stopped or max epochs)
                        patience_triggered = epochs_done - best_epoch >= 10
                        max_reached = epochs_done >= num_epochs
                        
                        if patience_triggered or max_reached:
                            print(f"\n⏭️ SKIP {m.upper()} seed {s} - done: EER={best_eer:.2f}% (ep {best_epoch}/{epochs_done})")
                            results.append({'method': m, 'seed': s, 'best_eer': best_eer, 'best_epoch': best_epoch, 'run_dir': os.path.join(exp_dir, run), 'skipped': True})
                            found_complete = True
                            break
                        elif epochs_done > 0 and resume_path is None:
                            # Partial run - can resume
                            resume_path = best_path
                            print(f"\n🔄 RESUME {m.upper()} seed {s} from epoch {epochs_done} (EER={best_eer:.2f}%)")
                
                if not found_complete:
                    print(f"\n{'='*50}\n🧪 {m.upper()} | Seed {s}\n{'='*50}")
                    # Extract run_id from resume_path if resuming
                    resume_run_id = os.path.basename(os.path.dirname(resume_path)) if resume_path else None
                    r = run_experiment(m, s, src_loader, tgt_loader, eval_loader, use_lora, num_epochs, resume_from=resume_path, run_id=resume_run_id)
                    r.update({'method': m, 'seed': s, 'skipped': False}); results.append(r)
            else:
                print(f"\n{'='*50}\n🧪 {m.upper()} | Seed {s}\n{'='*50}")
                r = run_experiment(m, s, src_loader, tgt_loader, eval_loader, use_lora, num_epochs)
                r.update({'method': m, 'seed': s, 'skipped': False}); results.append(r)
    
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}\n{'Method':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}\n{'-'*50}")
    summary = {}
    for m in methods:
        eers = [r['best_eer'] for r in results if r['method'] == m]
        if eers:
            summary[m] = {'mean': np.mean(eers), 'std': np.std(eers), 'min': np.min(eers), 'max': np.max(eers)}
            print(f"{m:<10} {summary[m]['mean']:.2f}%{'':<4} ±{summary[m]['std']:.2f}%{'':<4} {summary[m]['min']:.2f}%{'':<4} {summary[m]['max']:.2f}%")
    print('='*70)
    return {'results': results, 'summary': summary}
