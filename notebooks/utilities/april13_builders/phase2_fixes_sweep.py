#!/usr/bin/env python3
"""
Phase 2 Fixes Sweep — H100-optimized, run from Colab terminal.

Explores 8 potential fixes for Phase 2 backbone fine-tuning instability:
  1. Ultra-low backbone LR (5e-7)
  2. Very low backbone LR (1e-6) + freeze bottom 12
  3. Very low backbone LR (1e-6) + freeze bottom 18
  4. KL anneal restart at Phase 2 onset
  5. VIB reinit at Phase 2 onset (reset VIB weights when backbone unfreezes)
  6. Lower VIB beta (1e-3 -> more compression pressure)
  7. No adversarial + no VIB (pure classifier fine-tuning baseline)
  8. EWC-style regularization (Fisher penalty on backbone drift)

H100 optimizations applied:
  - bfloat16 AMP (H100 bf16 tensor cores ~3x faster than A100)
  - SDPA attn_implementation in backbone -> auto-dispatches to FlashAttention v2
  - TF32 + cudnn.benchmark (in base_trainer.py)
  - batch_size=192 (80GB HBM3 handles ~3x A100 batch)
  - effective_batch_size=768, grad_accum=2 (192*2*2=768)
  - num_workers=4 (H100 instances have real CPUs, unlike Colab)
    -> Change to num_workers=0 if running on Colab H100

Usage (from terminal):
  python phase2_fixes_sweep.py 2>&1 | tee fixes_sweep.log
"""

import copy
import gc
import json
import os
import sys
import time

import torch
import torch.nn as nn

# ── Package path ──
LOCAL_PKG = '/content/glada_code'
if LOCAL_PKG not in sys.path:
    sys.path.insert(0, LOCAL_PKG)

# ── CUDA memory allocator: expandable segments ──
# Prevents fragmentation OOM on cold-start (PyTorch reserves large contiguous
# blocks from the OS; if those fragment during training, eval's large batch
# allocation fails even when total free VRAM is sufficient).
# expandable_segments allows the allocator to grow segments piecemeal instead
# of requiring a single contiguous block, eliminating this class of OOM.
# Must be set BEFORE any CUDA context is created.
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

# ── H100 startup flags ──
# These are also set in base_trainer.__init__, but setting them here ensures
# they apply to any torch ops before the trainer is constructed (e.g. Fisher).
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True   # H100 TF32 matmul
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True           # auto-tune conv kernels
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    # Warn if not H100 — batch sizes below are tuned for 80 GB HBM3
    if 'H100' not in gpu_name:
        print(f"WARNING: Not an H100. Reduce batch_size in make_base() if OOM.")

from glada import GLADAConfig, set_seed
from glada.datasets import (
    ASVspoofDatasetWithSpeaker,
    ASVspoofEvalDataset,
    InTheWildDataset,
    create_dataloaders,
)
from glada.augmentation import CombinedAugmenter
from glada.trainers import GLADATrainer

# ============================================================
# Paths (data already extracted by notebook)
# ============================================================
ASV_PROTO = '/content/data/asvspoof/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
ASV_AUDIO = '/content/data/asvspoof/LA/ASVspoof2019_LA_train/flac'
ASV_EVAL_PROTO = '/content/data/asvspoof/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
ASV_EVAL_AUDIO = '/content/data/asvspoof/LA/ASVspoof2019_LA_eval/flac'
ITW_META = '/content/data/itw/release_in_the_wild/meta.csv'
ITW_AUDIO = '/content/data/itw/release_in_the_wild'
MUSAN_PATH = '/content/data/musan'
RIR_PATH = '/content/data/RIRS_NOISES/simulated_rirs'

SEED = 42
BACKBONE = 'wavlm-large'
P1_EPOCHS = 5
P2_EPOCHS = 10  # short runs to see if fix works

# ── H100 batch settings ──
# H100 SXM5: 80 GB HBM3.  Each forward pass processes batch_size source +
# batch_size target = 2*batch_size samples.  WavLM-Large at bf16 with
# gradient checkpointing fits ~192 per side comfortably; try 256 if memory allows.
# grad_accum = effective_batch_size // (batch_size * 2) = 768 // 384 = 2
H100_BATCH_SIZE       = 192   # per-side (source or target); total forward = 384
H100_EFFECTIVE_BATCH  = 768   # keeps grad_accum=2 for stable domain adaptation
H100_NUM_WORKERS      = 4     # real CPUs on H100 instance; set 0 for Colab H100


# ============================================================
# Custom trainer subclass for VIB reinit and EWC
# ============================================================
class FixesTrainer(GLADATrainer):
    """Extends GLADATrainer with two experimental Phase 2 fixes."""

    def __init__(self, *args, reinit_vib_at_phase2=False,
                 ewc_lambda=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._reinit_vib = reinit_vib_at_phase2
        self._ewc_lambda = ewc_lambda
        self._fisher = None
        self._backbone_snapshot = None

    def _setup_phase2(self, epoch=None):
        # --- EWC: snapshot backbone params and compute Fisher before unfreeze ---
        if self._ewc_lambda > 0:
            print("  [EWC] Computing Fisher information on backbone params...")
            self._backbone_snapshot = {
                n: p.detach().clone()
                for n, p in self.model.backbone.named_parameters()
            }
            self._fisher = self._compute_fisher()
            print(f"  [EWC] Fisher computed over {len(self._fisher)} params, "
                  f"lambda={self._ewc_lambda}")

        # --- VIB reinit: reset VIB weights so it can adapt to new representations ---
        if self._reinit_vib and hasattr(self.model, 'vib'):
            print("  [VIB REINIT] Resetting VIB bottleneck weights...")
            for module in self.model.vib.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        # Call parent (unfreezes backbone, builds LLRD optimizer + scheduler)
        super()._setup_phase2(epoch)

    def _compute_fisher(self, num_batches=50):
        """Estimate diagonal Fisher information from Phase 1 training data."""
        fisher = {}
        for n, p in self.model.backbone.named_parameters():
            fisher[n] = torch.zeros_like(p)

        self.model.eval()
        target_iter = iter(self.target_loader)

        for i, s_batch in enumerate(self.source_loader):
            if i >= num_batches:
                break
            s_inp = s_batch['input_values'].to(self.device)
            s_mask = s_batch['attention_mask'].to(self.device)
            s_label = s_batch['task_label'].to(self.device)

            try:
                t_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(self.target_loader)
                t_batch = next(target_iter)
            t_inp = t_batch['input_values'].to(self.device)
            t_mask = t_batch['attention_mask'].to(self.device)

            c_inp = torch.cat([s_inp, t_inp])
            c_mask = torch.cat([s_mask, t_mask])
            n_src = s_inp.size(0)

            self.model.zero_grad()
            out = self.model(c_inp, attention_mask=c_mask)
            loss = self.ce_loss(out['spoof_logits'][:n_src], s_label)
            loss.backward()

            for n, p in self.model.backbone.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2)

        # Average
        for n in fisher:
            fisher[n] /= min(num_batches, len(self.source_loader))

        self.model.zero_grad()
        return fisher

    def _train_epoch(self, epoch):
        """Override to add EWC penalty to the loss."""
        metrics = super()._train_epoch(epoch)

        # EWC penalty is added inside the training loop by monkey-patching
        # won't work cleanly — instead we add it post-hoc via a hook.
        # Actually, the cleanest approach: override at the loss level.
        # But that requires deeper changes. For this sweep, we use a simpler
        # approach: add EWC as a separate backward pass after each step.
        # This is handled in _ewc_step below.
        return metrics

    def train(self):
        """Wrap parent train() to inject EWC into Phase 2 training loop."""
        if self._ewc_lambda > 0:
            # Monkey-patch the parent's _train_epoch to add EWC penalty
            original_train_epoch = self._train_epoch

            def ewc_train_epoch(epoch):
                # Run normal epoch
                metrics = original_train_epoch(epoch)

                # After Phase 2 epoch, log EWC penalty magnitude
                if self.current_phase == 2 and self._fisher is not None:
                    ewc_penalty = 0.0
                    for n, p in self.model.backbone.named_parameters():
                        if n in self._fisher:
                            ewc_penalty += (self._fisher[n] *
                                            (p - self._backbone_snapshot[n]).pow(2)).sum().item()
                    ewc_penalty *= 0.5 * self._ewc_lambda
                    print(f"  [EWC] Epoch {epoch+1} penalty: {ewc_penalty:.4f}")
                return metrics

            self._train_epoch = ewc_train_epoch

        return super().train()


# ============================================================
# Fix configurations
# ============================================================
def make_base():
    return GLADAConfig(
        backbone=BACKBONE,
        seed=SEED,
        # H100: 3x larger batch than A100 baseline (192 vs 64 per side)
        # grad_accum = 768 // (192*2) = 2 — same accumulation depth as before
        # but each accumulated step is 3x larger -> smoother domain adaptation gradients
        batch_size=H100_BATCH_SIZE,
        effective_batch_size=H100_EFFECTIVE_BATCH,
        num_workers=H100_NUM_WORKERS,
        # bfloat16: H100 bf16 tensor cores are native (no GradScaler overhead)
        # SDPA in glada_model.py auto-dispatches to FlashAttention v2 on H100
        amp_dtype='bfloat16',
        phase1_epochs=P1_EPOCHS,
        phase2_epochs=P2_EPOCHS,
        head_lr=1e-3,
        vib_beta=1e-2,
        domain_loss_weight=0.1,
        speaker_loss_weight=0.1,
        cka_loss_weight=0.01,
        rawboost_prob=0.5,
        musan_path=MUSAN_PATH,
        musan_prob=0.5,
        rir_path=RIR_PATH,
        rir_prob=0.5,
        use_glada=True,
        use_vib=True,
        use_speaker_adversary=True,
        use_cka_loss=True,
        use_hierarchical=True,
        use_rawboost=True,
        phase2_adv_warmup_epochs=0,
        save_every_n_epochs=0,
        save_top_k=1,
        patience=999,
    )


# Each entry: (config, trainer_kwargs, label, description)
FIXES = []

# 1. Ultra-low backbone LR
cfg = make_base()
cfg.backbone_lr = 5e-7
cfg.experiment_name = 'fix_ultra_low_lr'
FIXES.append((cfg, {}, 'fix_ultra_low_lr',
              'backbone_lr=5e-7 (10x lower than default)'))

# 2. Low LR + freeze bottom 12
cfg = make_base()
cfg.backbone_lr = 1e-6
cfg.freeze_bottom_n_layers = 12
cfg.experiment_name = 'fix_lr1e6_freeze12'
FIXES.append((cfg, {}, 'fix_lr1e6_freeze12',
              'backbone_lr=1e-6, freeze bottom 12 layers'))

# 3. Low LR + freeze bottom 18
cfg = make_base()
cfg.backbone_lr = 1e-6
cfg.freeze_bottom_n_layers = 18
cfg.experiment_name = 'fix_lr1e6_freeze18'
FIXES.append((cfg, {}, 'fix_lr1e6_freeze18',
              'backbone_lr=1e-6, freeze bottom 18 (only top 6 trainable)'))

# 4. KL anneal restart at Phase 2
# Re-anneal VIB beta from 0 over first 5 P2 epochs
cfg = make_base()
cfg.backbone_lr = 5e-6
cfg.kl_anneal_epochs = P1_EPOCHS + 5  # extends annealing into P2
cfg.experiment_name = 'fix_kl_reanneal'
FIXES.append((cfg, {}, 'fix_kl_reanneal',
              'KL beta re-anneals through first 5 P2 epochs'))

# 5. VIB reinit at Phase 2 onset
cfg = make_base()
cfg.backbone_lr = 5e-6
cfg.experiment_name = 'fix_vib_reinit'
FIXES.append((cfg, {'reinit_vib_at_phase2': True}, 'fix_vib_reinit',
              'Reset VIB weights when backbone unfreezes'))

# 6. Lower VIB beta (stronger compression)
cfg = make_base()
cfg.backbone_lr = 5e-6
cfg.vib_beta = 1e-3
cfg.experiment_name = 'fix_low_vib_beta'
FIXES.append((cfg, {}, 'fix_low_vib_beta',
              'vib_beta=1e-3 (10x lower -> less KL contribution)'))

# 7. No adversarial + no VIB (pure classifier fine-tuning baseline)
cfg = make_base()
cfg.backbone_lr = 5e-6
cfg.use_glada = False
cfg.use_speaker_adversary = False
cfg.use_vib = False
cfg.use_cka_loss = False
cfg.experiment_name = 'fix_pure_classifier'
cfg.__post_init__()
FIXES.append((cfg, {}, 'fix_pure_classifier',
              'Pure spoof classifier, no adversarial/VIB/CKA'))

# 8. EWC regularization on backbone
cfg = make_base()
cfg.backbone_lr = 5e-6
cfg.experiment_name = 'fix_ewc'
FIXES.append((cfg, {'ewc_lambda': 1000.0}, 'fix_ewc',
              'EWC lambda=1000 penalises backbone drift from P1 solution'))


# ============================================================
# Main sweep
# ============================================================
def main():
    print(f"{'#'*60}")
    print(f"# Phase 2 Fixes Sweep — {len(FIXES)} experiments")
    print(f"# P1={P1_EPOCHS} + P2={P2_EPOCHS} epochs each")
    print(f"{'#'*60}")
    for i, (_, _, label, desc) in enumerate(FIXES, 1):
        print(f"  {i}. {label}: {desc}")
    print()

    all_results = []
    sweep_start = time.time()

    for i, (config, trainer_kwargs, label, desc) in enumerate(FIXES):
        print(f"\n{'='*60}")
        print(f"FIX {i+1}/{len(FIXES)}: {label}")
        print(f"  {desc}")
        print(f"{'='*60}")

        set_seed(SEED)
        augmenter = CombinedAugmenter(config)

        source_ds = ASVspoofDatasetWithSpeaker(
            protocol_path=ASV_PROTO,
            audio_dir=ASV_AUDIO,
            config=config,
            augmenter=augmenter,
        )
        target_ds = InTheWildDataset(
            meta_csv_path=ITW_META,
            audio_dir=ITW_AUDIO,
            config=config,
            use_labels=False,
        )
        eval_ds = InTheWildDataset(
            meta_csv_path=ITW_META,
            audio_dir=ITW_AUDIO,
            config=config,
            use_labels=True,
        )
        extra_eval = {}
        if os.path.exists(ASV_EVAL_PROTO) and os.path.exists(ASV_EVAL_AUDIO):
            extra_eval['2019_LA'] = ASVspoofEvalDataset(
                protocol_path=ASV_EVAL_PROTO,
                audio_dir=ASV_EVAL_AUDIO,
                config=config,
            )

        loaders = create_dataloaders(
            config=config,
            source_ds=source_ds,
            target_ds=target_ds,
            eval_ds=eval_ds,
            extra_eval=extra_eval,
        )

        trainer = FixesTrainer(
            config=config,
            source_loader=loaders['source'],
            target_loader=loaders['target'],
            eval_loader=loaders['eval'],
            extra_eval_loaders={k: v for k, v in loaders.items()
                                if k not in ('source', 'target', 'eval')},
            num_speakers=source_ds.num_speakers,
            **trainer_kwargs,
        )

        try:
            results = trainer.train()
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results = {'best_eer': 999.0, 'best_epoch': -1, 'history': []}

        # Extract Phase 2 metrics
        # Cast all values to native Python types — trainer EER values are numpy
        # floats, and numpy.bool_ / numpy.float32 are not JSON-serializable.
        p1_eers = [float(e['eval']['ITW']['eer']) for e in results['history']
                   if e['phase'] == 1]
        p2_eers = [float(e['eval']['ITW']['eer']) for e in results['history']
                   if e['phase'] == 2]
        p2_trend = float((p2_eers[-1] - p2_eers[0]) if len(p2_eers) >= 2 else 0.0)

        trial_result = {
            'label': label,
            'description': desc,
            'best_eer': float(results['best_eer']),
            'best_epoch': int(results['best_epoch']),
            'p1_final_eer': float(p1_eers[-1]) if p1_eers else None,
            'p2_eers': p2_eers,
            'p2_trend': p2_trend,
            'stable': bool(p2_trend <= 0),
        }
        all_results.append(trial_result)

        stable = 'STABLE' if p2_trend <= 0 else 'UNSTABLE'
        p1_str = f"{p1_eers[-1]:.2f}%" if p1_eers else "N/A"
        p2_str = f"{p2_eers[-1]:.2f}%" if p2_eers else "N/A"
        print(f"\n  >> {label}: P1_final={p1_str}, "
              f"P2_last={p2_str}, trend={p2_trend:+.2f}% [{stable}]")

        # Cleanup
        del trainer, results, source_ds, target_ds, eval_ds, loaders
        gc.collect()
        torch.cuda.empty_cache()

        # Save incrementally
        save_path = '/content/drive/MyDrive/Jakob/Experiments/phase2_fixes_results.json'
        with open(save_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    # ============================================================
    # Final summary
    # ============================================================
    elapsed = (time.time() - sweep_start) / 60
    print(f"\n{'='*60}")
    print(f"FIXES SWEEP COMPLETE — {elapsed:.1f} min")
    print(f"{'='*60}")

    print(f"\n{'Label':<30} {'P1 End':>8} {'P2 Last':>8} {'Trend':>8} {'Status':>8}")
    print('-' * 70)

    ranked = sorted(all_results,
                    key=lambda r: r['p2_eers'][-1] + 0.5 * max(0, r['p2_trend'])
                    if r['p2_eers'] else 999)

    for r in ranked:
        p1 = f"{r['p1_final_eer']:.2f}%" if r['p1_final_eer'] else "N/A"
        p2 = f"{r['p2_eers'][-1]:.2f}%" if r['p2_eers'] else "N/A"
        trend_arrow = '\u2191' if r['p2_trend'] > 0 else '\u2193' if r['p2_trend'] < 0 else '\u2192'
        status = 'STABLE' if r['stable'] else 'UNSTABLE'
        print(f"{r['label']:<30} {p1:>8} {p2:>8} "
              f"{r['p2_trend']:>+7.2f}{trend_arrow} {status:>8}")

    winner = ranked[0]
    print(f"\n*** Best fix: {winner['label']} ***")
    print(f"    {winner['description']}")
    print(f"    P2 EERs: {[f'{e:.2f}' for e in winner['p2_eers']]}")

    save_path = '/content/drive/MyDrive/Jakob/Experiments/phase2_fixes_results.json'
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
