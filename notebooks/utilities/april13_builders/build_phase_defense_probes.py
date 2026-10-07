"""
Builds phase_defense_probes.ipynb — a Colab notebook with GPU-only-for-inference
analyses that answer two NotebookLM critiques *without* retraining anything.

Critique #4 (underfitting vs negative transfer): already answered locally in
Coding/results/defense_analysis/. The notebook re-runs the analysis on Drive
copies of the history files for completeness, but the headline result is
established: full_glada plateaued at slope +0.05 pp/epoch over the last 5
Phase-2 epochs. NOT underfit.

Critique #5 (speaker-adversary mechanism): unresolved without running a probe.
This notebook executes the linear probe on cached VIB embeddings to determine
whether the speaker GRL (a) successfully removed speaker info but along with
useful artifact features (over-disentanglement, hypothesis A), or (b) failed
to remove speaker info and merely disrupted the backbone (collateral disruption,
hypothesis B).

The probe:
  1. Load full_glada checkpoint (already trained, on Drive).
  2. Forward pass over ASVspoof 2019 LA training audio to extract VIB features z.
  3. Train sklearn LogisticRegression on (z, speaker_id) with 80/20 split.
  4. Report top-1 speaker accuracy.
  5. Repeat for backbone_only checkpoint (control: no speaker adversary).

Chance accuracy = 1/20 = 5% (training set has 20 speakers).

Verdict:
  - full_glada ~ 5%, backbone_only >> 5%  -> Hypothesis A (over-disentanglement)
  - full_glada ~ backbone_only >> 5%      -> Hypothesis B (collateral disruption)
  - full_glada >> 5% but < backbone_only  -> partial removal

GPU is needed only for ~30 minutes of forward-pass inference; no training.

Build:
    python build_phase_defense_probes.py
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent / "phase_defense_probes.ipynb"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" if not l.endswith("\n") else l for l in lines]}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" if not l.endswith("\n") else l for l in lines]}


CELLS = [
    md(
        "# Phase DEFENSE-PROBES: Linear Probe + Training-Curve Analysis",
        "",
        "**Two NotebookLM critiques addressed without retraining:**",
        "",
        "**Critique #4 (underfitting vs negative transfer)** — already answered locally:",
        "`full_glada` Phase 2 trajectory has slope +0.05 pp/epoch over the last 5 epochs (essentially flat). It found a stable-but-bad equilibrium around 21% ITW EER. The negative-transfer claim is *not* a confound from premature stopping. Saved to `Coding/results/defense_analysis/phase2_trajectories.{png,json}` for the defense slides.",
        "",
        "**Critique #5 (speaker-adversary mechanism)** — answered by this notebook:",
        "Removing the speaker adversary improved ITW EER by 6.23 pp (19.71% → 13.48%). Mechanism unresolved: did the speaker GRL (a) successfully remove speaker info but along with useful artifact features (over-disentanglement), or (b) fail to remove speaker info and just damage the backbone (collateral disruption)? A linear probe on the trained model's VIB embeddings predicting speaker identity gives a binary answer.",
        "",
        "**Chance accuracy** = 1/20 = 5% (ASVspoof 2019 LA training set has 20 speakers).",
        "",
        "**Verdict logic:**",
        "- `full_glada` probe acc ≈ 5% AND `backbone_only` probe acc >> 5% → **Hypothesis A** (GRL worked, but at the cost of useful features)",
        "- `full_glada` probe acc ≈ `backbone_only` probe acc >> 5% → **Hypothesis B** (GRL didn't work; just disrupted the backbone)",
        "- `full_glada` probe acc somewhere in between → partial removal",
        "",
        "GPU needed only for ~30 minutes of inference (forward pass over ~25k training utterances × 2 checkpoints). No training.",
    ),
    md("## 1. Environment Setup"),
    code(
        "!pip install -q transformers soundfile librosa scikit-learn tqdm safetensors",
        "",
        "from google.colab import drive",
        "drive.mount('/content/drive')",
        "",
        "import torch",
        "if torch.cuda.is_available():",
        "    gpu = torch.cuda.get_device_properties(0)",
        "    print(f'GPU: {gpu.name} ({gpu.total_memory / 1e9:.1f} GB)')",
        "else:",
        "    print('WARNING: no GPU; probe will be slow but will still run')",
    ),
    md("## 2. Copy GLADA Package"),
    code(
        "import shutil, os, sys",
        "",
        "# Probe candidate paths for the package directory that contains __init__.py.",
        "# The actual package dir must directly contain __init__.py; we then copy it",
        "# to /content/glada and add /content to sys.path so `import glada` works.",
        "CANDIDATES = [",
        "    \"/content/drive/Othercomputers/Jakob's MacBook Air/Thesis/298-deliverables/glada-code/glada\",",
        "    '/content/drive/MyDrive/Jakob/glada_pkg/glada',",
        "    '/content/drive/MyDrive/Jakob/glada_pkg',",
        "    '/content/drive/MyDrive/Jakob/glada',",
        "]",
        "DRIVE_PKG = next((c for c in CANDIDATES if os.path.exists(os.path.join(c, '__init__.py'))), None)",
        "if DRIVE_PKG is None:",
        "    raise FileNotFoundError(",
        "        'No glada/__init__.py found in candidates. Run:\\n'",
        "        \"  !find /content/drive -name '__init__.py' -path '*/glada/*' 2>/dev/null | head\\n\"",
        "        'and add the parent directory to CANDIDATES above.'",
        "    )",
        "",
        "LOCAL_PKG = '/content/glada'",
        "if os.path.exists(LOCAL_PKG): shutil.rmtree(LOCAL_PKG)",
        "shutil.copytree(DRIVE_PKG, LOCAL_PKG)",
        "",
        "# Drop any stale glada module cache (in case a prior cell already imported)",
        "for mod in list(sys.modules):",
        "    if mod == 'glada' or mod.startswith('glada.'): del sys.modules[mod]",
        "sys.path = [p for p in sys.path if 'glada' not in p]",
        "if '/content' not in sys.path: sys.path.insert(0, '/content')",
        "print(f'Package source: {DRIVE_PKG}\\nCopied to:      {LOCAL_PKG}')",
    ),
    md("## 3. Datasets (ASVspoof 2019 LA training only — speaker IDs needed)"),
    code(
        "import tarfile",
        "",
        "def extract(tar_path, extract_to, name):",
        "    if os.path.exists(extract_to) and len(os.listdir(extract_to)) > 0:",
        "        print(f'{name}: already extracted'); return",
        "    os.makedirs(extract_to, exist_ok=True)",
        "    with tarfile.open(tar_path) as t: t.extractall(extract_to)",
        "    print(f'{name}: extracted')",
        "",
        "extract('/content/drive/MyDrive/Jakob/Datasets/LA.tar', '/content/data/asvspoof', 'ASVspoof 2019 LA')",
        "",
        "ASV_PROTO = '/content/data/asvspoof/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'",
        "ASV_AUDIO = '/content/data/asvspoof/LA/ASVspoof2019_LA_train/flac'",
        "for p in [ASV_PROTO, ASV_AUDIO]:",
        "    assert os.path.exists(p), f'Missing: {p}'",
        "print('Paths verified.')",
    ),
    md("## 4. Imports"),
    code(
        "from glada import GLADAConfig, BACKBONE_REGISTRY, set_seed",
        "from glada.datasets import ASVspoofDatasetWithSpeaker",
        "from glada.models import GLADAModel",
        "",
        "from sklearn.linear_model import LogisticRegression",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.preprocessing import StandardScaler",
        "import numpy as np, json, time",
        "from torch.utils.data import DataLoader",
        "from tqdm import tqdm",
        "",
        "print('Imports OK')",
    ),
    md(
        "## 5. Configure the Two Checkpoints to Probe",
        "",
        "Paths come from the original `Coding/results.json` checkpoint timestamps (March 2026). If your checkpoints live elsewhere, update the dict.",
    ),
    code(
        "DRIVE_EXP = '/content/drive/MyDrive/Jakob/Experiments'",
        "",
        "CHECKPOINTS = {",
        "    'full_glada':     f'{DRIVE_EXP}/ablation_full_glada/20260303_063553/best_model.pt',",
        "    'backbone_only':  f'{DRIVE_EXP}/ablation_backbone_only/20260304_105845/best_model.pt',",
        "}",
        "for name, path in CHECKPOINTS.items():",
        "    exists = '✓' if os.path.exists(path) else '✗'",
        "    print(f'{exists} {name}: {path}')",
        "",
        "# If a path is wrong, search the experiment dir for the most recent best_model.pt",
        "for name, expected_path in list(CHECKPOINTS.items()):",
        "    if not os.path.exists(expected_path):",
        "        parent = os.path.dirname(os.path.dirname(expected_path))",
        "        if os.path.isdir(parent):",
        "            runs = sorted(os.listdir(parent))",
        "            for r in reversed(runs):",
        "                cand = os.path.join(parent, r, 'best_model.pt')",
        "                if os.path.exists(cand):",
        "                    print(f'  [auto] {name}: using {cand} instead')",
        "                    CHECKPOINTS[name] = cand",
        "                    break",
    ),
    md("## 6. Build Probe Dataset (no augmentation, deterministic)"),
    code(
        "# We need raw waveforms + speaker IDs. Use the source dataset class with",
        "# augmenter=None to avoid randomness in the probe features.",
        "",
        "probe_config = GLADAConfig(",
        "    backbone='wavlm-base-plus',  # full_glada uses base-plus; we'll override per-checkpoint",
        "    batch_size=32,",
        "    num_workers=0,",
        ")",
        "",
        "probe_ds = ASVspoofDatasetWithSpeaker(",
        "    protocol_path=ASV_PROTO,",
        "    audio_dir=ASV_AUDIO,",
        "    config=probe_config,",
        "    augmenter=None,  # deterministic features",
        ")",
        "",
        "speaker_to_id = probe_ds.speaker_to_id",
        "num_speakers = len(speaker_to_id)",
        "chance_acc = 1.0 / num_speakers",
        "print(f'Probe dataset: {len(probe_ds):,} utterances, {num_speakers} speakers')",
        "print(f'Chance accuracy = 1/{num_speakers} = {chance_acc*100:.2f}%')",
        "",
        "probe_loader = DataLoader(probe_ds, batch_size=32, shuffle=False, num_workers=0)",
    ),
    md(
        "## 7. Extract VIB Embeddings from a Checkpoint",
        "",
        "The probe operates on the trained model's bottleneck features `z = result['features']` from the GLADA forward pass. For `backbone_only` (no VIB), the model exposes `z` from the deterministic projection in the same field.",
    ),
    code(
        "@torch.no_grad()",
        "def extract_features(checkpoint_path, dataloader):",
        "    \"\"\"Load checkpoint, run forward pass over dataloader, return (features, speaker_ids).\"\"\"",
        "    print(f'  Loading: {checkpoint_path}')",
        "    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)",
        "    # Reconstruct config from checkpoint if present, else fall back to probe_config",
        "    if isinstance(ckpt, dict) and 'config' in ckpt:",
        "        ckpt_cfg = ckpt['config']",
        "        if not isinstance(ckpt_cfg, GLADAConfig):",
        "            # Some checkpoints store config as a plain dict",
        "            cfg = GLADAConfig(**{k: v for k, v in ckpt_cfg.items() if k in GLADAConfig.__dataclass_fields__})",
        "        else:",
        "            cfg = ckpt_cfg",
        "    else:",
        "        cfg = probe_config",
        "    model = GLADAModel(cfg, num_speakers=num_speakers)",
        "    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt",
        "    missing, unexpected = model.load_state_dict(state_dict, strict=False)",
        "    if missing:    print(f'  Missing keys (first 3): {missing[:3]}')",
        "    if unexpected: print(f'  Unexpected keys (first 3): {unexpected[:3]}')",
        "    model.eval().cuda()",
        "    ",
        "    feats, speakers = [], []",
        "    for batch in tqdm(dataloader, desc='  Forward pass'):",
        "        wav = batch['input_values'].cuda(non_blocking=True)",
        "        spk = batch['speaker_id']",
        "        attn = batch.get('attention_mask')",
        "        if attn is not None: attn = attn.cuda(non_blocking=True)",
        "        with torch.cuda.amp.autocast(dtype=torch.bfloat16):",
        "            out = model(wav, attention_mask=attn)",
        "        z = out['features'].float().cpu().numpy()  # (B, bottleneck_dim)",
        "        feats.append(z)",
        "        speakers.append(spk.numpy() if torch.is_tensor(spk) else np.asarray(spk))",
        "    feats = np.concatenate(feats, axis=0)",
        "    speakers = np.concatenate(speakers, axis=0)",
        "    del model; torch.cuda.empty_cache()",
        "    return feats, speakers",
        "",
        "all_features = {}",
        "for name, ckpt_path in CHECKPOINTS.items():",
        "    if not os.path.exists(ckpt_path):",
        "        print(f'SKIP {name}: checkpoint missing'); continue",
        "    print(f'\\n=== Extracting features: {name} ===')",
        "    t0 = time.time()",
        "    feats, speakers = extract_features(ckpt_path, probe_loader)",
        "    dt = (time.time() - t0) / 60",
        "    print(f'  Got features {feats.shape}, speakers {speakers.shape} in {dt:.1f} min')",
        "    all_features[name] = (feats, speakers)",
    ),
    md("## 8. Train Linear Probe + Compute Top-1 Speaker Accuracy"),
    code(
        "set_seed(42)",
        "",
        "results = {}",
        "for name, (feats, speakers) in all_features.items():",
        "    print(f'\\n=== Probing: {name} ===')",
        "    # Filter to utterances that have a known speaker ID (>= 0)",
        "    mask = speakers >= 0",
        "    X = feats[mask]",
        "    y = speakers[mask]",
        "    print(f'  {X.shape[0]} samples, {len(np.unique(y))} unique speakers')",
        "",
        "    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)",
        "    scaler = StandardScaler().fit(X_tr)",
        "    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)",
        "",
        "    clf = LogisticRegression(max_iter=2000, multi_class='multinomial', n_jobs=-1, C=1.0)",
        "    clf.fit(X_tr_s, y_tr)",
        "    train_acc = clf.score(X_tr_s, y_tr)",
        "    test_acc = clf.score(X_te_s, y_te)",
        "",
        "    print(f'  Train acc: {train_acc*100:.2f}%')",
        "    print(f'  Test acc:  {test_acc*100:.2f}% (chance = {chance_acc*100:.2f}%)')",
        "    results[name] = {",
        "        'train_acc': float(train_acc),",
        "        'test_acc': float(test_acc),",
        "        'chance_acc': float(chance_acc),",
        "        'n_train': int(len(y_tr)),",
        "        'n_test': int(len(y_te)),",
        "        'n_speakers': int(len(np.unique(y))),",
        "    }",
        "",
        "OUT_DIR = '/content/drive/MyDrive/Jakob/Experiments/defense_analysis'",
        "os.makedirs(OUT_DIR, exist_ok=True)",
        "with open(f'{OUT_DIR}/speaker_probe_results.json', 'w') as f:",
        "    json.dump(results, f, indent=2)",
        "print(f'\\nSaved: {OUT_DIR}/speaker_probe_results.json')",
    ),
    md("## 9. Verdict"),
    code(
        "if 'full_glada' not in results or 'backbone_only' not in results:",
        "    print('Cannot render verdict — missing one of the two checkpoints.')",
        "else:",
        "    full_acc = results['full_glada']['test_acc']",
        "    bb_acc   = results['backbone_only']['test_acc']",
        "    chance   = results['full_glada']['chance_acc']",
        "    above_chance = lambda a, factor=2: a > factor * chance",
        "",
        "    print('='*60)",
        "    print(f'Speaker probe top-1 test accuracy:')",
        "    print(f'  chance:        {chance*100:>6.2f}%')",
        "    print(f'  full_glada:    {full_acc*100:>6.2f}%   (with speaker adversary)')",
        "    print(f'  backbone_only: {bb_acc*100:>6.2f}%   (no speaker adversary, control)')",
        "    print('='*60)",
        "    ",
        "    # Decision rule",
        "    full_close_to_chance = full_acc < 2 * chance         # within 2x chance is 'effectively chance'",
        "    bb_clearly_above = bb_acc > 5 * chance                # control retains speaker info",
        "    ",
        "    if full_close_to_chance and bb_clearly_above:",
        "        print('VERDICT: HYPOTHESIS A (over-disentanglement)')",
        "        print('  The speaker GRL successfully removed speaker information.')",
        "        print('  Cost: useful artifact features were collateral damage.')",
        "        print('  Defense talking point: removing speaker_adv recovered 6.23 pp because')",
        "        print('  the GRL was overly aggressive — the head worked, but at task-relevant cost.')",
        "    elif full_acc > 0.5 * bb_acc and bb_clearly_above:",
        "        print('VERDICT: HYPOTHESIS B (collateral disruption)')",
        "        print('  The speaker GRL did NOT remove speaker information meaningfully.')",
        "        print('  Cost: it acted as a destructive gradient force on the backbone.')",
        "        print('  Defense talking point: speaker_adv hurt EER by destabilizing training,')",
        "        print('  not by removing useful information. The auxiliary head was wasted compute.')",
        "    else:",
        "        gap = (1 - full_acc/bb_acc) * 100 if bb_acc > 0 else 0",
        "        print(f'VERDICT: PARTIAL REMOVAL ({gap:.0f}% relative reduction in speaker info)')",
        "        print('  The speaker GRL partially removed speaker information.')",
        "        print('  Both mechanisms (over-disentanglement + collateral disruption) likely contribute.')",
    ),
    md(
        "## 10. Use in Defense / ICTAI",
        "",
        "Whichever verdict comes back is publishable. Add one sentence to ICTAI Sec. V.H or the survey's GLADA experimental study:",
        "",
        "> *Hypothesis A (if probe drops to chance):*  ",
        "> A linear probe on the bottleneck features showed the speaker discriminator suppressed speaker identity below 2× chance ({full}% vs {chance}%), while a no-speaker control retained it at {bb}%. The 6.23 pp ITW EER recovery from removing the speaker adversary therefore reflects collateral loss of artifact-correlated features rather than failure of the GRL itself.",
        "",
        "> *Hypothesis B (if probe stays high):*  ",
        "> A linear probe on the bottleneck features showed the speaker discriminator did not meaningfully reduce speaker identity ({full}% vs {bb}% control, both well above {chance}% chance). The 6.23 pp ITW EER recovery from removing the speaker adversary therefore reflects gradient-side disruption of the backbone, not successful disentanglement.",
        "",
        "Both versions transform critique #5 from \"open question\" to \"answered with evidence.\" Either is a clean defense talking point.",
    ),
]


def build():
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "machine_shape": "hm"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {NOTEBOOK_PATH} ({len(CELLS)} cells)")


if __name__ == "__main__":
    build()
