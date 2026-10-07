#!/usr/bin/env python3
"""Build master_experiment.ipynb — single notebook: GPU config + Phase B sweep + Phases C-H."""
import json, pathlib

def md(source): return {"cell_type":"markdown","metadata":{},"source":source if isinstance(source,list) else [source]}
def code(source): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source if isinstance(source,list) else [source]}

cells = []

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Title
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""# GLADA Master Experiment Notebook
**Phase B** (8-fix sweep for Phase 2 instability) → **Phase C** (full training with winner) → **Phases D–H** (ablation / multi-seed / multi-dataset eval / diagnostics / figures)

Set the two booleans in the next cell before running anything else.
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — GPU selection
# ─────────────────────────────────────────────────────────────────────────────
cells.append(code("""\
# ══════════════════════════════════════════════════════════════════════════════
# GPU SELECTION — set exactly one to True
# ══════════════════════════════════════════════════════════════════════════════
USE_A100 = False   # ← True if on A100 (40 or 80 GB)
USE_H100 = True    # ← True if on H100 80 GB HBM3

assert USE_A100 ^ USE_H100, "Set exactly one of USE_A100 / USE_H100 to True"

# ── Derived batch / worker settings ──────────────────────────────────────────
# H100 HBM3 (80 GB): 3× A100 throughput; batch_size=192 per side, num_workers=4
# A100 SXM4 (40/80 GB): batch_size=64, num_workers=0 (Colab multiprocessing issues)
# grad_accum = effective_batch // (batch_size * 2):
#   H100: 768 // (192*2) = 2    A100: 384 // (64*2)  = 3
BATCH_SIZE       = 192 if USE_H100 else 64
EFFECTIVE_BATCH  = 768 if USE_H100 else 384
NUM_WORKERS      = 4   if USE_H100 else 0    # 0 required for Colab A100
EVAL_BATCH_SIZE  = 192 if USE_H100 else 48   # no gradients; H100 can handle larger
AMP_DTYPE        = 'bfloat16'                # native on both H100 and A100

# ── Phase B sweep epochs (short diagnostic runs) ─────────────────────────────
P1_SWEEP = 5
P2_SWEEP = 10   # short — just long enough to see if Phase 2 stays stable

# ── Full training epochs (Phases C – E) ──────────────────────────────────────
P1_FULL = 5
P2_FULL = 45

SEED     = 42
BACKBONE = 'wavlm-large'

gpu_label = 'H100' if USE_H100 else 'A100'
print(f"GPU profile : {gpu_label}")
print(f"batch_size  : {BATCH_SIZE} per side  ({BATCH_SIZE*2} total per step)")
print(f"eff_batch   : {EFFECTIVE_BATCH}  (grad_accum = {EFFECTIVE_BATCH//(BATCH_SIZE*2)})")
print(f"num_workers : {NUM_WORKERS}")
print(f"eval_batch  : {EVAL_BATCH_SIZE}")
print(f"amp_dtype   : {AMP_DTYPE}")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Install + mount
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## 0. One-Time Setup"))
cells.append(code("""\
# ── Install dependencies ──────────────────────────────────────────────────────
!pip install -q transformers soundfile librosa scikit-learn tqdm safetensors

# ── Mount Google Drive ────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ── CUDA allocator: expandable segments ──────────────────────────────────────
# Prevents contiguous-allocation OOM caused by memory fragmentation between
# experiments.  Must be set before any CUDA context is created.
import os
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

# ── GPU startup flags ─────────────────────────────────────────────────────────
import torch
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True   # TF32 matmul (both H100 & A100)
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True   # auto-tune conv kernels
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}  ({gpu_mem:.0f} GB)")
    if USE_H100 and 'H100' not in gpu_name:
        print("WARNING: USE_H100=True but GPU is not H100 — consider reducing BATCH_SIZE")
    if USE_A100 and 'A100' not in gpu_name:
        print("WARNING: USE_A100=True but GPU is not A100")
else:
    print("WARNING: No GPU detected")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Package copy
# ─────────────────────────────────────────────────────────────────────────────
cells.append(code("""\
# ── Copy GLADA package from Drive → local SSD ─────────────────────────────────
import shutil, sys

DRIVE_PKG = "/content/drive/Othercomputers/Jakob's MacBook Air/Thesis/298-deliverables/glada-code"
LOCAL_PKG  = '/content/glada_code'

if os.path.exists(LOCAL_PKG):
    shutil.rmtree(LOCAL_PKG)
shutil.copytree(DRIVE_PKG, LOCAL_PKG)

if LOCAL_PKG not in sys.path:
    sys.path.insert(0, LOCAL_PKG)

print(f'Package ready at {LOCAL_PKG}')
print(f'Contents: {os.listdir(LOCAL_PKG)}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Dataset extraction
# ─────────────────────────────────────────────────────────────────────────────
cells.append(code("""\
# ── Dataset extraction (one archive at a time to conserve SSD) ───────────────
import tarfile, zipfile

DRIVE_DATASETS = '/content/drive/MyDrive/Jakob/Datasets'
LOCAL_DATA     = '/content/data'

def extract_archive(drive_archive, extract_to, verify_path, name):
    if os.path.isdir(verify_path) and os.listdir(verify_path):
        print(f'  [{name}] Already present at {verify_path}'); return
    local_arc = f'/content/{os.path.basename(drive_archive)}'
    print(f'  [{name}] Copying to SSD...')
    shutil.copy2(drive_archive, local_arc)
    os.makedirs(extract_to, exist_ok=True)
    print(f'  [{name}] Extracting...')
    if local_arc.endswith('.zip'):
        with zipfile.ZipFile(local_arc) as z: z.extractall(extract_to)
    elif local_arc.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(local_arc, 'r:gz') as t: t.extractall(extract_to)
    elif local_arc.endswith('.tar'):
        with tarfile.open(local_arc, 'r:')  as t: t.extractall(extract_to)
    os.remove(local_arc)
    print(f'  [{name}] Done.')

for ds in [
    dict(drive_archive=f'{DRIVE_DATASETS}/LA.tar',
         extract_to=f'{LOCAL_DATA}/asvspoof',
         verify_path=f'{LOCAL_DATA}/asvspoof/LA', name='ASVspoof 2019 LA'),
    dict(drive_archive=f'{DRIVE_DATASETS}/release_in_the_wild.tar',
         extract_to=f'{LOCAL_DATA}/itw',
         verify_path=f'{LOCAL_DATA}/itw/release_in_the_wild', name='In-the-Wild'),
    dict(drive_archive=f'{DRIVE_DATASETS}/musan.tar.gz',
         extract_to=LOCAL_DATA,
         verify_path=f'{LOCAL_DATA}/musan', name='MUSAN'),
    dict(drive_archive=f'{DRIVE_DATASETS}/rirs_noises.zip',
         extract_to=LOCAL_DATA,
         verify_path=f'{LOCAL_DATA}/RIRS_NOISES', name='RIR'),
]:
    extract_archive(**ds)

# ── Shared path constants ─────────────────────────────────────────────────────
ASV_PROTO      = '/content/data/asvspoof/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
ASV_AUDIO      = '/content/data/asvspoof/LA/ASVspoof2019_LA_train/flac'
ASV_EVAL_PROTO = '/content/data/asvspoof/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
ASV_EVAL_AUDIO = '/content/data/asvspoof/LA/ASVspoof2019_LA_eval/flac'
ITW_META       = '/content/data/itw/release_in_the_wild/meta.csv'
ITW_AUDIO      = '/content/data/itw/release_in_the_wild'
MUSAN_PATH     = '/content/data/musan'
RIR_PATH       = '/content/data/RIRS_NOISES/simulated_rirs'

for p in [ASV_PROTO, ASV_AUDIO, ITW_META, MUSAN_PATH, RIR_PATH]:
    assert os.path.exists(p), f'Missing: {p}'
print('All dataset paths verified.')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Imports
# ─────────────────────────────────────────────────────────────────────────────
cells.append(code("""\
# ── Package imports (shared by all phases) ────────────────────────────────────
from glada import GLADAConfig, BACKBONE_REGISTRY, set_seed
from glada.datasets import (
    ASVspoofDatasetWithSpeaker, ASVspoofEvalDataset,
    InTheWildDataset, create_dataloaders,
)
from glada.augmentation import CombinedAugmenter
from glada.trainers import GLADATrainer
from glada.models import GLADAModel
from glada.evaluation import (
    MultiDatasetEvaluator, average_checkpoints, save_averaged_checkpoint,
)
import torch.nn as nn
import copy, gc, glob, json, time
import numpy as np
import matplotlib.pyplot as plt, matplotlib
import torch

matplotlib.rcParams.update({
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

print(f'GLADA package loaded | CUDA: {torch.cuda.is_available()}')
print(f'Backbones: {list(BACKBONE_REGISTRY.keys())}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# PHASE B
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
---
## Phase B: Phase 2 Instability Fix Sweep

8 short experiments (P1=5 + P2=10 epochs each) exploring fixes for Phase 2
backbone fine-tuning instability. Results saved to Drive after each experiment;
the winner is automatically loaded by Phase C.

**Root cause identified**: backbone fine-tuning shifts representations enough to
explode VIB KL (raw values 1000–9000+), even at low LR.  These 8 configs
systematically isolate the LR/freeze and VIB/regularisation axes.
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL B1 — FixesTrainer
# ─────────────────────────────────────────────────────────────────────────────
cells.append(code("""\
# ── FixesTrainer: extends GLADATrainer with VIB reinit and EWC ────────────────
class FixesTrainer(GLADATrainer):
    \"\"\"GLADATrainer extended with two experimental Phase-2 stabilisation fixes.

    reinit_vib_at_phase2 (bool): reset VIB linear layer weights at Phase 2
        onset so the bottleneck can adapt to the shifting backbone representations
        without being anchored to Phase-1 posterior parameters.

    ewc_lambda (float): Elastic Weight Consolidation penalty weight (Kirkpatrick
        et al., PNAS 2017).  Penalises backbone drift from the Phase-1 optimum
        via a Fisher-weighted L2 term: λ/2 · Σ_i F_i (θ_i − θ*_i)².
    \"\"\"

    def __init__(self, *args, reinit_vib_at_phase2=False, ewc_lambda=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._reinit_vib      = reinit_vib_at_phase2
        self._ewc_lambda      = ewc_lambda
        self._fisher          = None
        self._backbone_snapshot = None

    def _setup_phase2(self, epoch=None):
        if self._ewc_lambda > 0:
            print("  [EWC] Computing diagonal Fisher on backbone params...")
            self._backbone_snapshot = {
                n: p.detach().clone() for n, p in self.model.backbone.named_parameters()
            }
            self._fisher = self._compute_fisher()
            print(f"  [EWC] Fisher computed: {len(self._fisher)} params, λ={self._ewc_lambda}")

        if self._reinit_vib and hasattr(self.model, 'vib'):
            print("  [VIB REINIT] Resetting VIB bottleneck weights...")
            for m in self.model.vib.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None: nn.init.zeros_(m.bias)

        super()._setup_phase2(epoch)

    def _compute_fisher(self, num_batches=50):
        \"\"\"Estimate diagonal Fisher information matrix from spoof classification loss.\"\"\"
        fisher = {n: torch.zeros_like(p) for n, p in self.model.backbone.named_parameters()}
        self.model.eval()
        target_iter = iter(self.target_loader)

        for i, s_batch in enumerate(self.source_loader):
            if i >= num_batches: break
            s_inp   = s_batch['input_values'].to(self.device)
            s_mask  = s_batch['attention_mask'].to(self.device)
            s_label = s_batch['task_label'].to(self.device)
            try:
                t_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(self.target_loader); t_batch = next(target_iter)
            t_inp  = t_batch['input_values'].to(self.device)
            t_mask = t_batch['attention_mask'].to(self.device)

            self.model.zero_grad()
            out  = self.model(torch.cat([s_inp, t_inp]),
                              attention_mask=torch.cat([s_mask, t_mask]))
            loss = self.ce_loss(out['spoof_logits'][:s_inp.size(0)], s_label)
            loss.backward()
            for n, p in self.model.backbone.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2)

        n_batches = min(num_batches, len(self.source_loader))
        for n in fisher: fisher[n] /= n_batches
        self.model.zero_grad()
        return fisher

    def train(self):
        if self._ewc_lambda > 0:
            _orig = self._train_epoch
            def _ewc_epoch(epoch):
                metrics = _orig(epoch)
                if self.current_phase == 2 and self._fisher is not None:
                    penalty = sum(
                        (self._fisher[n] * (p - self._backbone_snapshot[n]).pow(2)).sum().item()
                        for n, p in self.model.backbone.named_parameters() if n in self._fisher
                    ) * 0.5 * self._ewc_lambda
                    print(f"  [EWC] Epoch {epoch+1} backbone drift penalty: {penalty:.4f}")
                return metrics
            self._train_epoch = _ewc_epoch
        return super().train()
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL B2 — Fix configs
# ─────────────────────────────────────────────────────────────────────────────
cells.append(code("""\
# ── Build the 8 fix configurations ───────────────────────────────────────────
def make_sweep_base():
    \"\"\"Base config for Phase B sweep — uses GPU-selected batch settings.\"\"\"
    return GLADAConfig(
        backbone=BACKBONE, seed=SEED,
        batch_size=BATCH_SIZE, effective_batch_size=EFFECTIVE_BATCH,
        num_workers=NUM_WORKERS, amp_dtype=AMP_DTYPE,
        phase1_epochs=P1_SWEEP, phase2_epochs=P2_SWEEP,
        head_lr=1e-3, vib_beta=1e-2,
        domain_loss_weight=0.1, speaker_loss_weight=0.1, cka_loss_weight=0.01,
        rawboost_prob=0.5,
        musan_path=MUSAN_PATH, musan_prob=0.5,
        rir_path=RIR_PATH,     rir_prob=0.5,
        use_glada=True, use_vib=True, use_speaker_adversary=True,
        use_cka_loss=True, use_hierarchical=True, use_rawboost=True,
        phase2_adv_warmup_epochs=0,
        save_every_n_epochs=0, save_top_k=1, patience=999,
    )

# Each entry: (config, trainer_kwargs, label, description)
FIXES = []

cfg = make_sweep_base(); cfg.backbone_lr = 5e-7;  cfg.experiment_name = 'fix_ultra_low_lr'
FIXES.append((cfg, {}, 'fix_ultra_low_lr', 'backbone_lr=5e-7 (10x lower than default)'))

cfg = make_sweep_base(); cfg.backbone_lr = 1e-6; cfg.freeze_bottom_n_layers = 12; cfg.experiment_name = 'fix_lr1e6_freeze12'
FIXES.append((cfg, {}, 'fix_lr1e6_freeze12', 'backbone_lr=1e-6, freeze bottom 12 layers'))

cfg = make_sweep_base(); cfg.backbone_lr = 1e-6; cfg.freeze_bottom_n_layers = 18; cfg.experiment_name = 'fix_lr1e6_freeze18'
FIXES.append((cfg, {}, 'fix_lr1e6_freeze18', 'backbone_lr=1e-6, freeze bottom 18 (only top 6 trainable)'))

cfg = make_sweep_base(); cfg.backbone_lr = 5e-6; cfg.kl_anneal_epochs = P1_SWEEP + 5; cfg.experiment_name = 'fix_kl_reanneal'
FIXES.append((cfg, {}, 'fix_kl_reanneal', 'KL beta re-anneals through first 5 P2 epochs'))

cfg = make_sweep_base(); cfg.backbone_lr = 5e-6; cfg.experiment_name = 'fix_vib_reinit'
FIXES.append((cfg, {'reinit_vib_at_phase2': True}, 'fix_vib_reinit', 'Reset VIB weights when backbone unfreezes'))

cfg = make_sweep_base(); cfg.backbone_lr = 5e-6; cfg.vib_beta = 1e-3; cfg.experiment_name = 'fix_low_vib_beta'
FIXES.append((cfg, {}, 'fix_low_vib_beta', 'vib_beta=1e-3 (10x lower -> less KL contribution)'))

cfg = make_sweep_base(); cfg.backbone_lr = 5e-6
cfg.use_glada = cfg.use_speaker_adversary = cfg.use_vib = cfg.use_cka_loss = False
cfg.experiment_name = 'fix_pure_classifier'; cfg.__post_init__()
FIXES.append((cfg, {}, 'fix_pure_classifier', 'Pure spoof classifier (no adversarial / VIB / CKA)'))

cfg = make_sweep_base(); cfg.backbone_lr = 5e-6; cfg.experiment_name = 'fix_ewc'
FIXES.append((cfg, {'ewc_lambda': 1000.0}, 'fix_ewc', 'EWC lambda=1000 — penalise backbone drift from P1'))

print(f'Phase B sweep: {len(FIXES)} experiments')
print(f'Epochs each: P1={P1_SWEEP} + P2={P2_SWEEP}')
for i, (_, _, lbl, desc) in enumerate(FIXES, 1):
    print(f'  {i}. {lbl}: {desc}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL B3 — Sweep loop
# ─────────────────────────────────────────────────────────────────────────────
cells.append(code("""\
# ── Phase B sweep loop ────────────────────────────────────────────────────────
# Results saved to Drive after each experiment so a disconnection only loses
# the currently-running experiment.
# To skip already-completed experiments: set SKIP_LABELS below.
SKIP_LABELS = set()   # e.g. {'fix_ultra_low_lr'} to skip fix 1

FIXES_RESULTS_PATH = '/content/drive/MyDrive/Jakob/Experiments/phase2_fixes_results.json'
sweep_results = []
sweep_start   = time.time()

for i, (config, trainer_kwargs, label, desc) in enumerate(FIXES):
    if label in SKIP_LABELS:
        print(f'  SKIP {i+1}/{len(FIXES)}: {label}'); continue

    print(f"\\n{'='*60}\\nFIX {i+1}/{len(FIXES)}: {label}\\n  {desc}\\n{'='*60}")

    set_seed(SEED)
    augmenter = CombinedAugmenter(config)

    source_ds = ASVspoofDatasetWithSpeaker(
        protocol_path=ASV_PROTO, audio_dir=ASV_AUDIO,
        config=config, augmenter=augmenter)
    target_ds = InTheWildDataset(
        meta_csv_path=ITW_META, audio_dir=ITW_AUDIO, config=config, use_labels=False)
    eval_ds = InTheWildDataset(
        meta_csv_path=ITW_META, audio_dir=ITW_AUDIO, config=config, use_labels=True)
    extra_eval = {}
    if os.path.exists(ASV_EVAL_PROTO) and os.path.exists(ASV_EVAL_AUDIO):
        extra_eval['2019_LA'] = ASVspoofEvalDataset(
            protocol_path=ASV_EVAL_PROTO, audio_dir=ASV_EVAL_AUDIO, config=config)

    loaders = create_dataloaders(config, source_ds, target_ds, eval_ds, extra_eval)
    trainer = FixesTrainer(
        config=config,
        source_loader=loaders['source'], target_loader=loaders['target'],
        eval_loader=loaders['eval'],
        extra_eval_loaders={k: v for k, v in loaders.items()
                            if k not in ('source','target','eval')},
        num_speakers=source_ds.num_speakers,
        **trainer_kwargs,
    )

    try:
        results = trainer.train()
    except Exception as e:
        import traceback; traceback.print_exc()
        results = {'best_eer': 999.0, 'best_epoch': -1, 'history': []}
        print(f'  FAILED: {e}')

    # Extract metrics — cast to native Python types (trainer returns numpy scalars)
    p1_eers = [float(h['eval']['ITW']['eer']) for h in results['history'] if h['phase']==1]
    p2_eers = [float(h['eval']['ITW']['eer']) for h in results['history'] if h['phase']==2]
    p2_trend = float((p2_eers[-1] - p2_eers[0]) if len(p2_eers) >= 2 else 0.0)

    rec = {
        'label': label, 'description': desc,
        'best_eer': float(results['best_eer']), 'best_epoch': int(results['best_epoch']),
        'p1_final_eer': float(p1_eers[-1]) if p1_eers else None,
        'p2_eers': p2_eers, 'p2_trend': p2_trend,
        'stable': bool(p2_trend <= 0),
    }
    sweep_results.append(rec)
    status = 'STABLE' if p2_trend <= 0 else 'UNSTABLE'
    p1_str = f'{p1_eers[-1]:.2f}%' if p1_eers else 'N/A'
    p2_str = f'{p2_eers[-1]:.2f}%' if p2_eers else 'N/A'
    print(f'\\n  >> {label}: P1={p1_str}  P2_last={p2_str}  trend={p2_trend:+.2f}% [{status}]')

    del trainer, results, source_ds, target_ds, eval_ds, loaders
    gc.collect(); torch.cuda.empty_cache()

    with open(FIXES_RESULTS_PATH, 'w') as f:
        json.dump(sweep_results, f, indent=2)

elapsed = (time.time() - sweep_start) / 60
print(f"\\nPhase B complete — {elapsed:.1f} min")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL B4 — Sweep summary + winner
# ─────────────────────────────────────────────────────────────────────────────
cells.append(code("""\
# ── Phase B summary + identify winner ────────────────────────────────────────
# Load from file in case notebook was restarted after sweep completed
if not sweep_results and os.path.exists(FIXES_RESULTS_PATH):
    with open(FIXES_RESULTS_PATH) as f:
        sweep_results = json.load(f)
    print(f'Loaded {len(sweep_results)} results from {FIXES_RESULTS_PATH}')

# Rank by: P2 last EER + 0.5 * penalty for instability (positive trend)
ranked = sorted(sweep_results,
    key=lambda r: (r['p2_eers'][-1] if r['p2_eers'] else 999)
                  + 0.5 * max(0, r['p2_trend']))

print(f"{'Label':<30} {'P1 End':>8} {'P2 Last':>8} {'Trend':>9} {'Status'}")
print('-' * 70)
for r in ranked:
    p1  = f"{r['p1_final_eer']:.2f}%" if r['p1_final_eer'] else 'N/A'
    p2  = f"{r['p2_eers'][-1]:.2f}%" if r['p2_eers'] else 'N/A'
    arr = '↑' if r['p2_trend'] > 0 else '↓' if r['p2_trend'] < 0 else '→'
    print(f"  {r['label']:<28} {p1:>8} {p2:>8} {r['p2_trend']:>+7.2f}%{arr} {'STABLE' if r['stable'] else 'UNSTABLE'}")

SWEEP_WINNER = ranked[0]
print(f"\\n*** Winner: {SWEEP_WINNER['label']} ***")
print(f"    {SWEEP_WINNER['description']}")
print(f"    P2 EERs: {[f'{e:.2f}' for e in SWEEP_WINNER['p2_eers']]}")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# PHASE C
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
---
## Phase C: Full Training with Phase B Winner

Full-budget training (5 + 45 = 50 epochs) using the winning fix configuration.
Auto-loads `SWEEP_WINNER` from the cell above.  Override `C_BACKBONE_LR` and
`C_FREEZE_N` manually if running after a disconnection.
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL C1 — Config
# ─────────────────────────────────────────────────────────────────────────────
cells.append(code("""\
# ── Phase C: build config from sweep winner ───────────────────────────────────
# The sweep tested backbone_lr and freeze_bottom_n_layers.
# We carry the winner's values forward into full training.

# Auto-read from SWEEP_WINNER (populated above); override here if needed:
C_BACKBONE_LR = SWEEP_WINNER.get('backbone_lr', 5e-6) if 'SWEEP_WINNER' in dir() else 5e-6
C_FREEZE_N    = SWEEP_WINNER.get('freeze_bottom_n', 0) if 'SWEEP_WINNER' in dir() else 0
# For fixes that toggle flags (pure_classifier, ewc, vib_reinit), replicate here:
C_VIB_BETA    = 1e-3 if SWEEP_WINNER.get('label','').endswith('low_vib_beta') else 1e-2
C_USE_VIB     = 'pure_classifier' not in SWEEP_WINNER.get('label','')
C_USE_GLADA   = 'pure_classifier' not in SWEEP_WINNER.get('label','')
C_USE_SPK     = 'pure_classifier' not in SWEEP_WINNER.get('label','')
C_USE_CKA     = 'pure_classifier' not in SWEEP_WINNER.get('label','')

config_c = GLADAConfig(
    experiment_name=f'glada_C_{BACKBONE}_seed{SEED}',
    backbone=BACKBONE, seed=SEED,
    backbone_lr=C_BACKBONE_LR,
    freeze_bottom_n_layers=C_FREEZE_N,
    phase1_epochs=P1_FULL, phase2_epochs=P2_FULL,
    batch_size=BATCH_SIZE, effective_batch_size=EFFECTIVE_BATCH,
    num_workers=NUM_WORKERS, amp_dtype=AMP_DTYPE,
    head_lr=1e-3, vib_beta=C_VIB_BETA,
    domain_loss_weight=0.1, speaker_loss_weight=0.1, cka_loss_weight=0.01,
    rawboost_prob=0.5,
    musan_path=MUSAN_PATH, musan_prob=0.5,
    rir_path=RIR_PATH,     rir_prob=0.5,
    use_glada=C_USE_GLADA, use_vib=C_USE_VIB,
    use_speaker_adversary=C_USE_SPK, use_cka_loss=C_USE_CKA,
    use_hierarchical=True, use_rawboost=True,
    save_every_n_epochs=1, save_top_k=10,
)

print(f'Phase C config:')
print(f'  backbone_lr={config_c.backbone_lr}  freeze_bottom={config_c.freeze_bottom_n_layers}')
print(f'  batch={config_c.batch_size}  eff={config_c.effective_batch_size}  grad_accum={config_c.grad_accum_steps}')
print(f'  epochs: P1={config_c.phase1_epochs} + P2={config_c.phase2_epochs} = {config_c.total_epochs}')
print(f'  vib={config_c.use_vib}  glada={config_c.use_glada}  speaker={config_c.use_speaker_adversary}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL C2 — Train
# ─────────────────────────────────────────────────────────────────────────────
cells.append(code("""\
# ── Phase C: build datasets + train ──────────────────────────────────────────
set_seed(SEED)
augmenter_c = CombinedAugmenter(config_c)

source_ds_c = ASVspoofDatasetWithSpeaker(
    protocol_path=ASV_PROTO, audio_dir=ASV_AUDIO,
    config=config_c, augmenter=augmenter_c)
target_ds_c = InTheWildDataset(
    meta_csv_path=ITW_META, audio_dir=ITW_AUDIO, config=config_c, use_labels=False)
eval_ds_c   = InTheWildDataset(
    meta_csv_path=ITW_META, audio_dir=ITW_AUDIO, config=config_c, use_labels=True)
extra_eval_c = {}
if os.path.exists(ASV_EVAL_PROTO):
    extra_eval_c['2019_LA'] = ASVspoofEvalDataset(
        protocol_path=ASV_EVAL_PROTO, audio_dir=ASV_EVAL_AUDIO, config=config_c)

loaders_c = create_dataloaders(config_c, source_ds_c, target_ds_c, eval_ds_c, extra_eval_c)

# Set RESUME_FROM to a checkpoint path to continue after disconnection:
# RESUME_FROM = '/content/drive/MyDrive/Jakob/Experiments/glada_C_.../epoch_12.pt'
RESUME_FROM = None

trainer_c = GLADATrainer(
    config=config_c,
    source_loader=loaders_c['source'], target_loader=loaders_c['target'],
    eval_loader=loaders_c['eval'],
    extra_eval_loaders={k: v for k, v in loaders_c.items()
                        if k not in ('source','target','eval')},
    num_speakers=source_ds_c.num_speakers,
    resume_from=RESUME_FROM,
)
results_c = trainer_c.train()
print(f'Phase C done.  Best ITW EER: {results_c["best_eer"]:.2f}% (epoch {results_c["best_epoch"]})')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL C3 — Eval + ckpt avg
# ─────────────────────────────────────────────────────────────────────────────
cells.append(code("""\
# ── Phase C: evaluation + checkpoint averaging ────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_path = os.path.join(results_c['run_dir'], 'best_model.pt')
model_c   = GLADAModel(config_c)
ckpt      = torch.load(best_path, map_location='cpu', weights_only=False)
model_c.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
model_c.to(device).eval()

eval_dict = {'ITW': loaders_c['eval']}
eval_dict.update({k: v for k, v in loaders_c.items() if k not in ('source','target','eval')})
ev_c      = MultiDatasetEvaluator(model_c, config_c, device, eval_dict)
results_c_eval = ev_c.evaluate_all()
print(ev_c.results_to_markdown(results_c_eval))

# Top-5 checkpoint averaging (Polyak & Juditsky, 1992)
try:
    avg_model = average_checkpoints(
        run_dir=results_c['run_dir'], model_class=GLADAModel,
        config=config_c, top_k=5, metric='eer', device=str(device))
    ev_avg = MultiDatasetEvaluator(avg_model, config_c, device, eval_dict)
    avg_results = ev_avg.evaluate_all()
    avg_itw = avg_results['ITW']['eer']
    delta   = results_c['best_eer'] - avg_itw
    print(f'Checkpoint avg (top-5): {avg_itw:.2f}% | Single best: {results_c["best_eer"]:.2f}% | Δ={delta:+.2f}%')
    save_averaged_checkpoint(avg_model, results_c['run_dir'], config_c, 5, source_epochs=[])
except Exception as e:
    print(f'Checkpoint averaging skipped: {e}')

with open(os.path.join(results_c['run_dir'], 'eval_results.json'), 'w') as f:
    json.dump({k: {kk: float(vv) if hasattr(vv,'item') else vv for kk,vv in v.items()}
               for k,v in results_c_eval.items()}, f, indent=2)

del trainer_c, source_ds_c, target_ds_c, eval_ds_c, loaders_c
gc.collect(); torch.cuda.empty_cache()
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL C4 — Training curves
# ─────────────────────────────────────────────────────────────────────────────
cells.append(code("""\
# ── Phase C: training curves ──────────────────────────────────────────────────
history = results_c['history']
epochs  = [h['epoch'] for h in history]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f'Phase C: {config_c.backbone} seed={config_c.seed}', fontsize=14)

ax = axes[0,0]
ax.plot(epochs, [h['train']['total_loss'] for h in history], 'b-')
ax.axvline(x=config_c.phase1_epochs, color='r', linestyle='--', alpha=0.5, label='P2 start')
ax.set(xlabel='Epoch', ylabel='Loss', title='Total Loss'); ax.legend()

ax = axes[0,1]
for key, lbl in [('spoof_loss','Task'),('glada_loss','GLADA'),('speaker_loss','Speaker')]:
    ax.plot(epochs, [h['train'][key] for h in history], label=lbl)
ax.axvline(x=config_c.phase1_epochs, color='r', linestyle='--', alpha=0.5)
ax.set(xlabel='Epoch', ylabel='Loss', title='Loss Components'); ax.legend()

ax = axes[0,2]
ax.plot(epochs, [h['train']['kl_loss'] for h in history], label='KL (clamped)')
ax.plot(epochs, [h['train'].get('kl_raw', h['train']['kl_loss']) for h in history],
        label='KL (raw)', alpha=0.5)
ax.axvline(x=config_c.phase1_epochs, color='r', linestyle='--', alpha=0.5)
ax.set(xlabel='Epoch', ylabel='KL', title='VIB KL Health'); ax.legend()

ds_names = list(history[0]['eval'].keys()) if history else []
ax = axes[1,0]
for ds in ds_names:
    eers = [h['eval'][ds]['eer'] for h in history if ds in h.get('eval',{})]
    ax.plot(epochs[:len(eers)], eers, label=ds)
ax.axvline(x=config_c.phase1_epochs, color='r', linestyle='--', alpha=0.5)
ax.set(xlabel='Epoch', ylabel='EER (%)', title='EER per Dataset'); ax.legend()

ax = axes[1,1]
for ds in ds_names:
    aucs = [h['eval'][ds]['auc'] for h in history if ds in h.get('eval',{})]
    ax.plot(epochs[:len(aucs)], aucs, label=ds)
ax.axvline(x=config_c.phase1_epochs, color='r', linestyle='--', alpha=0.5)
ax.set(xlabel='Epoch', ylabel='AUC', title='AUC per Dataset'); ax.legend()

ax = axes[1,2]
ax.plot(epochs, [h['train']['spoof_acc']*100   for h in history], label='Task Acc')
ax.plot(epochs, [h['train']['speaker_acc']*100 for h in history], label='Speaker Acc')
ax.axvline(x=config_c.phase1_epochs, color='r', linestyle='--', alpha=0.5)
ax.set(xlabel='Epoch', ylabel='Accuracy (%)', title='Accuracy'); ax.legend()

plt.tight_layout()
fig_path = os.path.join(results_c['run_dir'], 'training_curves.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight'); plt.show()
print(f'Saved {fig_path}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# PHASE D
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
---
## Phase D: Ablation Study

Leave-one-out ablations + baselines + augmentation + multi-backbone.
`USE_FULL_EPOCHS=True` — H100/A100 speed makes full-budget ablations feasible.
"""))

cells.append(code("""\
# ── Ablation config matrix ────────────────────────────────────────────────────
USE_FULL_EPOCHS = True
ABL_P1 = 5  if USE_FULL_EPOCHS else 3
ABL_P2 = 45 if USE_FULL_EPOCHS else 17

MUSAN_AVAIL = MUSAN_PATH if os.path.isdir(MUSAN_PATH) else ''
RIR_AVAIL   = RIR_PATH   if os.path.isdir(RIR_PATH)   else ''

ABLATION_CONFIGS = {
    'full_glada':  {},
    'no_glada':    {'use_glada': False, 'use_global_domain_disc': True},
    'no_vib':      {'use_vib': False},
    'no_speaker':  {'use_speaker_adversary': False},
    'no_cka':      {'use_cka_loss': False},
    'no_rawboost': {'use_rawboost': False, 'rawboost_prob': 0.0},
    'no_hier':     {'use_hierarchical': False},
}
BASELINE_CONFIGS = {
    'global_da_only':  {'use_glada': False, 'use_global_domain_disc': True, 'use_vib': True},
    'no_da':           {'use_glada': False, 'use_global_domain_disc': False, 'use_speaker_adversary': False},
    'backbone_only':   {'use_glada': False, 'use_global_domain_disc': False, 'use_vib': False,
                        'use_speaker_adversary': False, 'use_cka_loss': False,
                        'use_rawboost': False, 'rawboost_prob': 0.0},
}
AUG_CONFIGS = {
    'rawboost_only':      {'musan_path': '', 'rir_path': ''},
    'rawboost_musan_rir': {'musan_path': MUSAN_AVAIL, 'rir_path': RIR_AVAIL},
}
BACKBONES_ABL = ['xlsr-300m', 'data2vec-large', 'wavlm-base-plus']

def make_abl_cfg(name, backbone=BACKBONE, seed=42, overrides=None):
    cfg = GLADAConfig(
        experiment_name=f'ablation_{name}', backbone=backbone, seed=seed,
        phase1_epochs=ABL_P1, phase2_epochs=ABL_P2,
        batch_size=BATCH_SIZE, effective_batch_size=EFFECTIVE_BATCH,
        num_workers=NUM_WORKERS, amp_dtype=AMP_DTYPE,
        # Carry forward winner's backbone_lr and freeze settings
        backbone_lr=C_BACKBONE_LR, freeze_bottom_n_layers=C_FREEZE_N,
        head_lr=1e-3, vib_beta=1e-2,
        domain_loss_weight=0.1, speaker_loss_weight=0.1, cka_loss_weight=0.01,
        rawboost_prob=0.5, musan_path=MUSAN_PATH, musan_prob=0.5,
        rir_path=RIR_PATH, rir_prob=0.5,
        save_every_n_epochs=1, save_top_k=10,
    )
    for k, v in (overrides or {}).items():
        if hasattr(cfg, k): setattr(cfg, k, v)
    cfg.__post_init__(); return cfg

abl_experiments = []
for name, ov in {**ABLATION_CONFIGS, **BASELINE_CONFIGS}.items():
    abl_experiments.append(make_abl_cfg(name, overrides=ov))
for name, ov in AUG_CONFIGS.items():
    abl_experiments.append(make_abl_cfg(name, overrides=ov))
for bb in BACKBONES_ABL:
    abl_experiments.append(make_abl_cfg(f'full_{bb}', backbone=bb))

print(f'{len(abl_experiments)} ablation experiments | epochs: {ABL_P1}+{ABL_P2}={ABL_P1+ABL_P2}')
"""))

cells.append(code("""\
# ── Load shared datasets for ablation (once) ──────────────────────────────────
_base_cfg_d = GLADAConfig(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
source_ds_d = ASVspoofDatasetWithSpeaker(
    protocol_path=ASV_PROTO, audio_dir=ASV_AUDIO, config=_base_cfg_d, augmenter=None)
target_ds_d = InTheWildDataset(
    meta_csv_path=ITW_META, audio_dir=ITW_AUDIO, config=_base_cfg_d, use_labels=False)
eval_ds_d   = InTheWildDataset(
    meta_csv_path=ITW_META, audio_dir=ITW_AUDIO, config=_base_cfg_d, use_labels=True)
extra_eval_d = {}
if os.path.exists(ASV_EVAL_PROTO):
    extra_eval_d['2019_LA'] = ASVspoofEvalDataset(
        protocol_path=ASV_EVAL_PROTO, audio_dir=ASV_EVAL_AUDIO, config=_base_cfg_d)
NUM_SPEAKERS_D = source_ds_d.num_speakers
print(f'Source: {len(source_ds_d):,} | Target: {len(target_ds_d):,} | Eval: {len(eval_ds_d):,}')
print(f'Speakers: {NUM_SPEAKERS_D}')
"""))

cells.append(code("""\
# ── Ablation sweep loop ───────────────────────────────────────────────────────
abl_results = {}
abl_start   = time.time()
ABL_RESULTS_PATH = f'/content/drive/MyDrive/Jakob/Experiments/ablation_results.json'

for idx, cfg in enumerate(abl_experiments, 1):
    print(f"\\n{'='*60}\\nABL {idx}/{len(abl_experiments)}: {cfg.experiment_name}\\n{'='*60}")

    existing = glob.glob(os.path.join(cfg.drive_log_dir, cfg.experiment_name, '*', 'history.json'))
    if existing:
        print(f'  SKIPPING — already done: {existing[0]}')
        with open(existing[0]) as f: hist = json.load(f)
        abl_results[cfg.experiment_name] = {
            'best_eer': min((h.get('eval',{}).get('ITW',{}).get('eer',999) for h in hist), default=999),
            'status': 'loaded'}
        continue

    set_seed(cfg.seed)
    source_ds_d.augmenter = CombinedAugmenter(cfg)
    loaders_d = create_dataloaders(cfg, source_ds_d, target_ds_d, eval_ds_d, extra_eval_d)

    try:
        trainer_d = GLADATrainer(
            config=cfg,
            source_loader=loaders_d['source'], target_loader=loaders_d['target'],
            eval_loader=loaders_d['eval'],
            extra_eval_loaders={k: v for k, v in loaders_d.items()
                                if k not in ('source','target','eval')},
            num_speakers=NUM_SPEAKERS_D,
        )
        res = trainer_d.train()
        abl_results[cfg.experiment_name] = {
            'best_eer': float(res['best_eer']), 'best_epoch': int(res['best_epoch']),
            'run_dir': res['run_dir'], 'status': 'completed'}
        print(f'  Best ITW EER: {res["best_eer"]:.2f}% (ep {res["best_epoch"]})')
        del trainer_d
    except Exception as e:
        print(f'  FAILED: {e}')
        abl_results[cfg.experiment_name] = {'error': str(e), 'status': 'failed'}

    gc.collect(); torch.cuda.empty_cache()
    with open(ABL_RESULTS_PATH, 'w') as f:
        json.dump(abl_results, f, indent=2, default=str)

print(f'\\nAblation done — {(time.time()-abl_start)/60:.1f} min')
"""))

cells.append(code("""\
# ── Ablation results table ────────────────────────────────────────────────────
ref = abl_results.get('ablation_full_glada', {}).get('best_eer')
print(f"{'Config':<25} {'ITW EER':>9} {'Delta':>8}")
print('-' * 45)
for name in list(ABLATION_CONFIGS) + list(BASELINE_CONFIGS) + list(AUG_CONFIGS):
    exp  = f'ablation_{name}'
    r    = abl_results.get(exp, {})
    eer  = r.get('best_eer')
    delta = (eer - ref) if (eer and ref) else None
    eer_s   = f'{eer:.2f}%'   if eer   else 'N/A'
    delta_s = f'{delta:+.2f}' if delta is not None else '—'
    print(f'  {name:<23} {eer_s:>9} {delta_s:>8}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# PHASE E
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
---
## Phase E: Multi-Seed Verification

Full GLADA and no-GLADA baseline at seeds 123 and 2026 (seed 42 from Phase D).
Produces mean ± std EER for thesis reporting (Picard, 2021).
"""))

cells.append(code("""\
# ── Multi-seed runs + loop ────────────────────────────────────────────────────
multiseed_results = {}
ms_experiments = []
for seed in [123, 2026]:
    for name, ov in [('full_glada', {}),
                     ('no_glada', {'use_glada': False, 'use_global_domain_disc': True})]:
        ms_experiments.append(make_abl_cfg(f'{name}_s{seed}', seed=seed, overrides=ov))

for idx, cfg in enumerate(ms_experiments, 1):
    print(f"\\n{'='*55}\\nMS {idx}/{len(ms_experiments)}: {cfg.experiment_name}\\n{'='*55}")
    existing = glob.glob(os.path.join(cfg.drive_log_dir, cfg.experiment_name, '*', 'history.json'))
    if existing:
        print(f'  SKIPPING'); continue
    set_seed(cfg.seed)
    source_ds_d.augmenter = CombinedAugmenter(cfg)
    loaders_e = create_dataloaders(cfg, source_ds_d, target_ds_d, eval_ds_d, extra_eval_d)
    try:
        trainer_e = GLADATrainer(
            config=cfg,
            source_loader=loaders_e['source'], target_loader=loaders_e['target'],
            eval_loader=loaders_e['eval'],
            extra_eval_loaders={k: v for k, v in loaders_e.items()
                                if k not in ('source','target','eval')},
            num_speakers=NUM_SPEAKERS_D,
        )
        res = trainer_e.train()
        multiseed_results[cfg.experiment_name] = {
            'best_eer': float(res['best_eer']), 'seed': cfg.seed, 'status': 'completed'}
        print(f'  Best: {res["best_eer"]:.2f}%'); del trainer_e
    except Exception as e:
        print(f'  FAILED: {e}')
        multiseed_results[cfg.experiment_name] = {'error': str(e), 'status': 'failed'}
    gc.collect(); torch.cuda.empty_cache()

with open('/content/drive/MyDrive/Jakob/Experiments/multiseed_results.json', 'w') as f:
    json.dump(multiseed_results, f, indent=2, default=str)

# ── Summary: mean ± std ───────────────────────────────────────────────────────
all_ms = {**abl_results, **multiseed_results}
for grp, keys in [('Full GLADA', ['ablation_full_glada','ablation_full_glada_s123','ablation_full_glada_s2026']),
                  ('No GLADA',   ['ablation_no_glada',  'ablation_no_glada_s123',  'ablation_no_glada_s2026'])]:
    eers = [all_ms[k]['best_eer'] for k in keys if k in all_ms and isinstance(all_ms[k].get('best_eer'), float)]
    if eers:
        print(f'{grp}: {np.mean(eers):.2f} ± {np.std(eers):.2f}%  (n={len(eers)}, seeds={eers})')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# PHASE F
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
---
## Phase F: Multi-Dataset Evaluation

Evaluate best Phase C checkpoint on all available datasets.
Extracts one archive at a time to stay within SSD budget.
"""))

cells.append(code("""\
# ── Load Phase C best model ───────────────────────────────────────────────────
from glada.datasets.asvspoof2021 import ASVspoof2021Dataset
from glada.datasets.wavefake     import WaveFakeDataset
from torch.utils.data import DataLoader
import warnings, logging
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    BEST_RUN_DIR = results_c['run_dir']
    config_f     = config_c
    print(f'Using Phase C run: {BEST_RUN_DIR}')
except NameError:
    BEST_RUN_DIR = sorted(glob.glob('/content/drive/MyDrive/Jakob/Experiments/glada_C_*/*/'))[-1]
    with open(os.path.join(BEST_RUN_DIR, 'config.json')) as f: cfg_dict = json.load(f)
    config_f = GLADAConfig(**{k: v for k, v in cfg_dict.items() if hasattr(GLADAConfig, k)})
    config_f.__post_init__(); print(f'Loaded from {BEST_RUN_DIR}')

model_f = GLADAModel(config_f)
ckpt    = torch.load(os.path.join(BEST_RUN_DIR, 'best_model.pt'), map_location='cpu', weights_only=False)
model_f.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
model_f.to(device).eval()
print('Model loaded.')
"""))

cells.append(code("""\
# ── Sequential extract → eval → cleanup ──────────────────────────────────────
DRIVE_BASE = '/content/drive/MyDrive/Jakob/Datasets'
eval_summary = {}

def run_eval(loader, name):
    ev = MultiDatasetEvaluator(model_f, config_f, device, {name: loader})
    return ev.evaluate_all()[name]

# ITW (already on SSD)
print('--- ITW ---')
itw_ldr = DataLoader(InTheWildDataset(ITW_META, ITW_AUDIO, config=config_f, use_labels=True),
                     batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
eval_summary['ITW'] = run_eval(itw_ldr, 'ITW')
del itw_ldr; gc.collect()
print(f"  EER: {eval_summary['ITW']['eer']:.2f}%")

# 2019 LA eval (already on SSD)
print('--- ASVspoof 2019 LA eval ---')
asv19_ldr = DataLoader(ASVspoofEvalDataset(ASV_EVAL_PROTO, ASV_EVAL_AUDIO, config=config_f),
                       batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
eval_summary['2019_LA'] = run_eval(asv19_ldr, '2019_LA')
del asv19_ldr; gc.collect()
print(f"  EER: {eval_summary['2019_LA']['eer']:.2f}%")

# WaveFake
wf_arc = os.path.join(DRIVE_BASE, 'WaveFake.tar.gz')
wf_dir = '/content/data/wavefake'
if os.path.exists(wf_arc):
    print('--- WaveFake (extracting) ---')
    if not os.path.exists(wf_dir):
        with tarfile.open(wf_arc, 'r:gz') as t: t.extractall('/content/data')
    wf_ldr = DataLoader(WaveFakeDataset(wf_dir, max_len=config_f.max_audio_len),
                        batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    eval_summary['WaveFake'] = run_eval(wf_ldr, 'WaveFake')
    del wf_ldr; gc.collect(); shutil.rmtree(wf_dir)
    print(f"  EER: {eval_summary['WaveFake']['eer']:.2f}%")

with open(os.path.join(BEST_RUN_DIR, 'multidataset_eval.json'), 'w') as f:
    json.dump({k: {kk: float(vv) if hasattr(vv,'item') else vv for kk,vv in v.items()}
               for k,v in eval_summary.items()}, f, indent=2)

print('\\n=== Multi-Dataset Summary ===')
print(f"{'Dataset':<20} {'EER':>9} {'AUC':>9}")
print('-' * 40)
for ds, r in eval_summary.items():
    print(f'  {ds:<18} {r.get("eer",0):>8.2f}% {r.get("auc",0):>9.4f}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# PHASE G
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
---
## Phase G: Training Diagnostics

Post-hoc analysis of Phase C training dynamics. Reads history from Drive — no GPU needed.
"""))

cells.append(code("""\
# ── Load Phase C history ──────────────────────────────────────────────────────
EXP_DIR  = '/content/drive/MyDrive/Jakob/Experiments'
DIAG_DIR = os.path.join(EXP_DIR, 'diagnostics')
os.makedirs(DIAG_DIR, exist_ok=True)

history_g = None
try:
    history_g = results_c['history']; print(f'Using live history: {len(history_g)} epochs')
except NameError:
    for pat in [os.path.join(EXP_DIR, 'glada_C_*/*/history.json'),
                os.path.join(EXP_DIR, 'ablation_full_glada/*/history.json')]:
        matches = sorted(glob.glob(pat))
        if matches:
            with open(matches[-1]) as f: history_g = json.load(f)
            print(f'Loaded {len(history_g)} epochs from {matches[-1]}'); break

if history_g is None:
    print('No history found — run Phase C first or set history_g manually')
"""))

cells.append(code("""\
# ── G1: Loss balance + KL health ─────────────────────────────────────────────
if history_g:
    epochs_g = [h['epoch'] for h in history_g]
    spoof    = [h['train']['spoof_loss']   for h in history_g]
    glada    = [h['train']['glada_loss']   for h in history_g]
    spk      = [h['train']['speaker_loss'] for h in history_g]
    kl       = [h['train']['kl_loss']      for h in history_g]
    kl_raw   = [h['train'].get('kl_raw', h['train']['kl_loss']) for h in history_g]
    total    = [h['train']['total_loss']   for h in history_g]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase G — Loss Balance', fontsize=14)

    axes[0].stackplot(epochs_g, spoof, glada, spk, kl,
                      labels=['Task','GLADA','Speaker','KL'], alpha=0.7)
    axes[0].axvline(x=P1_FULL, color='r', ls='--', alpha=0.5, label='P2 start')
    axes[0].set(xlabel='Epoch', ylabel='Loss', title='Stacked Losses'); axes[0].legend()

    for vals, lbl in zip([spoof,glada,spk,kl],['Task','GLADA','Speaker','KL']):
        pct = [v/t*100 if t else 0 for v,t in zip(vals,total)]
        axes[1].plot(epochs_g, pct, label=lbl)
    axes[1].axvline(x=P1_FULL, color='r', ls='--', alpha=0.5)
    axes[1].set(xlabel='Epoch', ylabel='% of total', title='Relative Loss %'); axes[1].legend()

    axes[2].plot(epochs_g, kl_raw, label='KL raw')
    axes[2].plot(epochs_g, kl,     label='KL clamped')
    axes[2].axvline(x=P1_FULL, color='r', ls='--', alpha=0.5, label='P2 start')
    axes[2].set(xlabel='Epoch', ylabel='KL', title='VIB KL: raw vs clamped'); axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(DIAG_DIR, 'g1_loss_balance.png'), dpi=150); plt.show()
"""))

cells.append(code("""\
# ── G2: GRL alpha + EER trajectory ───────────────────────────────────────────
if history_g:
    d_alphas = [h.get('domain_alpha', h.get('grl_alpha')) for h in history_g]
    s_alphas = [h.get('speaker_alpha') for h in history_g]
    itw_eers = [h['eval']['ITW']['eer'] for h in history_g if 'ITW' in h.get('eval',{})]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Phase G — Adversarial Dynamics', fontsize=14)

    ax = axes[0]
    if any(a is not None for a in d_alphas):
        ax.plot(epochs_g, d_alphas, label='Domain α')
    if any(a is not None for a in s_alphas):
        ax.plot(epochs_g, s_alphas, label='Speaker α')
    ax.axvline(x=P1_FULL, color='r', ls='--', alpha=0.5, label='P2 start')
    ax.set(xlabel='Epoch', ylabel='GRL Alpha', title='GRL Schedule'); ax.legend()

    ax = axes[1]
    ax.plot(epochs_g[:len(itw_eers)], itw_eers, 'g-o', ms=4)
    ax.axvline(x=P1_FULL, color='r', ls='--', alpha=0.5, label='P2 start')
    ax.set(xlabel='Epoch', ylabel='ITW EER (%)', title='ITW EER Trajectory'); ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(DIAG_DIR, 'g2_adversarial.png'), dpi=150); plt.show()

    # Stability summary
    p1_eers = [h['eval']['ITW']['eer'] for h in history_g if h['phase']==1 and 'ITW' in h.get('eval',{})]
    p2_eers = [h['eval']['ITW']['eer'] for h in history_g if h['phase']==2 and 'ITW' in h.get('eval',{})]
    print(f'P1 EER: {p1_eers[0]:.2f}% → {p1_eers[-1]:.2f}%  (trend {p1_eers[-1]-p1_eers[0]:+.2f}%)')
    print(f'P2 EER: {p2_eers[0]:.2f}% → {p2_eers[-1]:.2f}%  (trend {p2_eers[-1]-p2_eers[0]:+.2f}%)')
    print(f'Phase 2 stable: {(p2_eers[-1]-p2_eers[0]) <= 0}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# PHASE H
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
---
## Phase H: Thesis Figures & Analysis

Generate all publication-ready figures and export summary JSON for LaTeX.
Pure analysis — no GPU needed.  Run after Phases C–F complete.
"""))

cells.append(code("""\
# ── Load all results ──────────────────────────────────────────────────────────
FIGURES_DIR = os.path.join(EXP_DIR, 'figures'); os.makedirs(FIGURES_DIR, exist_ok=True)

abl_path  = os.path.join(EXP_DIR, 'ablation_results.json')
if os.path.exists(abl_path):
    with open(abl_path) as f: abl_data = json.load(f)
    print(f'Ablation: {len(abl_data)} experiments')
else:
    abl_data = {}; print('No ablation results — run Phase D first')

eval_paths = sorted(glob.glob(os.path.join(EXP_DIR, 'glada_C_*/*/multidataset_eval.json')))
multidataset = {}
if eval_paths:
    with open(eval_paths[-1]) as f: multidataset = json.load(f)
    print(f'Multi-dataset: {list(multidataset.keys())}')
"""))

cells.append(code("""\
# ── H1: Ablation bar chart ────────────────────────────────────────────────────
abl_order  = ['full_glada','no_glada','global_da_only','no_vib','no_speaker',
               'no_cka','no_hier','no_rawboost','backbone_only']
abl_labels = ['Full GLADA','–GLADA','Global DA','–VIB','–Speaker',
               '–CKA','–Hier','–RawBoost','Backbone only']

eers_plot = [abl_data.get(f'ablation_{n}',{}).get('best_eer') for n in abl_order]
mask       = [e is not None for e in eers_plot]
labels_f   = [l for l,m in zip(abl_labels,mask) if m]
eers_f     = [e for e,m in zip(eers_plot,mask)  if m]
ref_eer    = abl_data.get('ablation_full_glada',{}).get('best_eer')

fig, ax = plt.subplots(figsize=(12, 5))
colors = ['#2196F3' if l=='Full GLADA' else '#F44336' if '–GLADA' in l else '#90A4AE' for l in labels_f]
bars = ax.barh(range(len(labels_f)), eers_f, color=colors, edgecolor='black', linewidth=0.5)
for bar, eer in zip(bars, eers_f):
    ax.text(eer+0.05, bar.get_y()+bar.get_height()/2, f'{eer:.2f}%', va='center', fontsize=9)
if ref_eer: ax.axvline(x=ref_eer, color='blue', ls='--', alpha=0.7, label=f'Full GLADA ({ref_eer:.2f}%)')
ax.set_yticks(range(len(labels_f))); ax.set_yticklabels(labels_f)
ax.set(xlabel='ITW EER (%)', title='Ablation Study (lower = better)'); ax.invert_yaxis(); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,'fig_ablation.png'), dpi=300); plt.show()
"""))

cells.append(code("""\
# ── H2: Multi-dataset bar + H3: SOTA comparison ──────────────────────────────
if multidataset:
    fig, ax = plt.subplots(figsize=(10, 4))
    ds_n = list(multidataset.keys()); ds_e = [multidataset[d].get('eer',0) for d in ds_n]
    bars = ax.bar(ds_n, ds_e, color='#2196F3', edgecolor='black', linewidth=0.5)
    for bar, eer in zip(bars, ds_e):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f'{eer:.2f}%', ha='center', fontsize=9)
    ax.set(ylabel='EER (%)', title='GLADA Multi-Dataset Generalization')
    ax.grid(True, alpha=0.3, axis='y'); plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR,'fig_multidataset.png'), dpi=300); plt.show()

# SOTA comparison (ITW EER)
sota = [
    ('Wav2Vec2-Large + LCNN',  'Kang et al., 2024',   9.34),
    ('RawGAT-ST',              'Tak et al., 2021',    7.67),
    ('HierCon (WavLM-BP)',     'Liang et al., 2026',  6.87),
    ('RawBMamba',              'Xie et al., 2024',    5.83),
    ('IB-CAAN (WavLM-Large)',  'Huang et al., 2025',  4.61),
    ('GLADA (ours)',           'This work',
     multidataset.get('ITW',{}).get('eer') or abl_data.get('ablation_full_glada',{}).get('best_eer',999)),
]
print('\\n=== SOTA (ITW EER) ===')
for model, ref, eer in sota:
    flag = ' <<' if 'ours' in model else ''
    print(f'  {model:<35} {ref:<25} {eer:.2f}%{flag}')

fig, ax = plt.subplots(figsize=(11, 4))
colors_s = ['#F44336' if 'ours' in s[0] else '#90A4AE' for s in sota]
bars = ax.bar(range(len(sota)), [s[2] for s in sota], color=colors_s, edgecolor='black', linewidth=0.5)
for bar, s in zip(bars, sota):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f'{s[2]:.2f}%', ha='center', fontsize=9)
ax.set_xticks(range(len(sota))); ax.set_xticklabels([s[0] for s in sota], rotation=20, ha='right', fontsize=9)
ax.set(ylabel='ITW EER (%)', title='State-of-the-Art Comparison (lower = better)')
ax.grid(True, alpha=0.3, axis='y'); plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,'fig_sota.png'), dpi=300); plt.show()
"""))

cells.append(code("""\
# ── H4: Export summary JSON + zip for download ───────────────────────────────
summary = {
    'full_glada_itw_eer': abl_data.get('ablation_full_glada',{}).get('best_eer'),
    'no_glada_itw_eer':   abl_data.get('ablation_no_glada',{}).get('best_eer'),
    'multidataset': multidataset,
    'sota': [{'model': s[0], 'ref': s[1], 'eer': s[2]} for s in sota],
}
with open(os.path.join(EXP_DIR,'thesis_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print('thesis_summary.json saved')

zip_path = shutil.make_archive('/content/thesis_figures', 'zip', FIGURES_DIR)
print(f'Figures zipped: {zip_path}')
try:
    from google.colab import files; files.download(zip_path)
except Exception: print(f'Download: {zip_path}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# Write notebook
# ─────────────────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

out = pathlib.Path(__file__).parent / 'master_experiment.ipynb'
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f'Written {len(cells)} cells → {out}')
print(f'Size: {out.stat().st_size / 1024:.1f} KB')
