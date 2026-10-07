"""Builds phase_c_probe.ipynb — Layer-probe analysis on the per_group_vib checkpoint.

Goal: characterize what each group of the trained per_group_vib model encodes,
so the ICTAI Section V.J discussion can claim *what* the groups represent
rather than just *that* they exist.

Method (no training, T4-friendly):
  1. Load the trained per_group_vib + WavLM-Base-Plus checkpoint (the 11.67% one).
  2. Run forward pass on the ASVspoof 2019 LA dev set; collect z_groups (post-VIB
     per-group features, dim 256 each) and labels (spoof, speaker_id).
  3. For each group k in {0,1,2}, fit two sklearn LogisticRegression probes:
       - bona-fide vs spoof (sanity check: every group should still carry it
         since the model was trained for that label)
       - speaker classification (multinomial; tells us how much speaker
         identity each group still encodes after compression + adversarial)
  4. Plot a 2-row grouped bar chart: group k on x-axis, accuracy on y-axis.

Expected pattern (per WavLM-Large layer-importance precedent, El Kheir 2025):
  - Spoof accuracy roughly equal across groups (all groups feed the classifier)
  - Speaker accuracy increases with k: late groups encode more speaker info
    (which the speaker adversary tries to suppress). If a group's speaker
    accuracy is very high, the adversary isn't fully working there; if very
    low, compression has been effective.

Build:
    python build_phase_c_probe.py
"""

from pathlib import Path
from _phase_common import (
    md, code,
    env_setup_cells, imports_cells, model_class_cells,
    write_notebook,
)

NOTEBOOK_PATH = Path(__file__).parent / "phase_c_probe.ipynb"


CELLS = [
    md(
        "# Phase C — Layer-Probe Analysis on the `per_group_vib` Checkpoint",
        "",
        "**Goal:** characterize what each layer group of the trained `per_group_vib` model encodes. The ICTAI Section V.J discussion currently says \"groups exist\" without saying *what* they represent. This notebook fills that gap with two cheap probes per group.",
        "",
        "**Method:**",
        "1. Load the trained `per_group_vib` + WavLM-Base-Plus checkpoint (the 11.67% one).",
        "2. Forward pass on ASVspoof 2019 LA *dev* set; collect `z_groups` (post-VIB compressed features, 256-dim per group) and labels (spoof, speaker_id).",
        "3. For each group k, fit two sklearn LogisticRegression probes:",
        "   - **bona-fide vs spoof** — sanity: every group should still carry it.",
        "   - **speaker classification** — tells us how much speaker identity each group still encodes after compression + adversarial training.",
        "4. Plot per-group accuracies in a grouped bar chart.",
        "",
        "**No training; runs on T4 in ~30 minutes.**",
        "",
        "Self-contained: Run All from scratch.",
    ),
    *env_setup_cells(),
    *imports_cells(),
    *model_class_cells(),
    md(
        "## 6. Locate Trained Checkpoint",
        "",
        "Edit `CHECKPOINT_DIR` below to point at the run directory of the trained `per_group_vib` Base-Plus model (the one that produced 11.67% ITW EER). The notebook auto-discovers the best checkpoint by reading the JSON results in that directory.",
    ),
    code(
        "# Path to the per_group_vib + Base-Plus run that produced 11.67% ITW EER.",
        "# Run dir is something like: /content/drive/MyDrive/Jakob/Experiments/vib_stagger/per_group_vib_<timestamp>/",
        "CHECKPOINT_DIR = '/content/drive/MyDrive/Jakob/Experiments/vib_stagger'",
        "",
        "# Auto-discover the per_group_vib subdirectory if there are multiple runs.",
        "import glob",
        "candidate_runs = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, 'per_group_vib*')))",
        "candidate_runs = [r for r in candidate_runs if os.path.isdir(r)]",
        "if not candidate_runs:",
        "    raise FileNotFoundError(",
        "        f'No per_group_vib* subdirectories under {CHECKPOINT_DIR}. '",
        "        f'Edit CHECKPOINT_DIR to the parent of the trained run.'",
        "    )",
        "RUN_DIR = candidate_runs[-1]  # latest by sort order (timestamps)",
        "print(f'Using run dir: {RUN_DIR}')",
        "",
        "# Best checkpoint is typically saved as 'best.pt' or 'checkpoint_best.pt'.",
        "ckpt_candidates = ['checkpoint_best.pt', 'best.pt', 'best_checkpoint.pt']",
        "CKPT_PATH = None",
        "for c in ckpt_candidates:",
        "    p = os.path.join(RUN_DIR, c)",
        "    if os.path.exists(p):",
        "        CKPT_PATH = p; break",
        "if CKPT_PATH is None:",
        "    pts = sorted(glob.glob(os.path.join(RUN_DIR, '*.pt')))",
        "    if not pts:",
        "        raise FileNotFoundError(f'No .pt checkpoints in {RUN_DIR}.')",
        "    CKPT_PATH = pts[-1]",
        "print(f'Loading checkpoint: {CKPT_PATH}')",
    ),
    md("## 7. Load Model & Data"),
    code(
        "# Build a fresh PerGroupVIBModel with the saved config, then load weights.",
        "import json as _json",
        "config_json = os.path.join(RUN_DIR, 'config.json')",
        "if os.path.exists(config_json):",
        "    with open(config_json) as f: saved_cfg = _json.load(f)",
        "    cfg = GLADAConfig(**{k: v for k, v in saved_cfg.items() if k in GLADAConfig.__dataclass_fields__})",
        "else:",
        "    print('No config.json in run dir; using defaults.')",
        "    cfg = GLADAConfig(backbone='wavlm-base-plus', batch_size=64, num_workers=0,",
        "                      phase1_epochs=3, phase2_epochs=17)",
        "set_seed(cfg.seed)",
        "",
        "model = PerGroupVIBModel(cfg).cuda()",
        "state = torch.load(CKPT_PATH, map_location='cuda')",
        "if 'model_state_dict' in state: state = state['model_state_dict']",
        "missing, unexpected = model.load_state_dict(state, strict=False)",
        "print(f'Loaded checkpoint. Missing keys: {len(missing)}, unexpected: {len(unexpected)}')",
        "model.eval()",
        "",
        "# Use the ASVspoof 2019 LA *dev* split for probing (train would overfit, eval is",
        "# what we report). Need labeled spoof + speaker_id, both available via",
        "# ASVspoofDatasetWithSpeaker on the dev protocol.",
        "DEV_PROTO = '/content/data/asvspoof/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'",
        "DEV_AUDIO = '/content/data/asvspoof/LA/ASVspoof2019_LA_dev/flac'",
        "assert os.path.exists(DEV_PROTO), DEV_PROTO",
        "",
        "probe_ds = ASVspoofDatasetWithSpeaker(",
        "    protocol_path=DEV_PROTO, audio_dir=DEV_AUDIO,",
        "    config=cfg, augmenter=None,  # no augmentation for probing",
        ")",
        "probe_loader = torch.utils.data.DataLoader(",
        "    probe_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0,",
        ")",
        "print(f'Probe set: {len(probe_ds)} utterances, {len(probe_ds.speaker_to_id)} speakers')",
    ),
    md(
        "## 8. Extract Per-Group Features",
        "",
        "Run forward, capture `result['group_outputs']` (which is `z_groups` in PerGroupVIBModel), and stack across the dev split.",
    ),
    code(
        "from tqdm.auto import tqdm",
        "",
        "all_z_groups = None  # will become list of K arrays of shape (N, bd)",
        "all_spoof_labels = []",
        "all_speaker_labels = []",
        "",
        "with torch.no_grad():",
        "    for batch in tqdm(probe_loader, desc='extracting features'):",
        "        x = batch['input_values'].cuda(non_blocking=True)",
        "        am = batch.get('attention_mask')",
        "        if am is not None: am = am.cuda(non_blocking=True)",
        "        out = model(x, attention_mask=am)",
        "        z_groups = out['group_outputs']  # list of K tensors, each (B, bd)",
        "        if all_z_groups is None:",
        "            all_z_groups = [[] for _ in z_groups]",
        "        for k, z in enumerate(z_groups):",
        "            all_z_groups[k].append(z.float().cpu().numpy())",
        "        all_spoof_labels.append(batch['label'].numpy())",
        "        all_speaker_labels.append(batch['speaker_id'].numpy())",
        "",
        "import numpy as np",
        "all_z_groups = [np.concatenate(z_list, axis=0) for z_list in all_z_groups]",
        "all_spoof_labels = np.concatenate(all_spoof_labels)",
        "all_speaker_labels = np.concatenate(all_speaker_labels)",
        "K = len(all_z_groups)",
        "print(f'Extracted features: K={K} groups, N={all_z_groups[0].shape[0]}, bd={all_z_groups[0].shape[1]}')",
        "print(f'Labels: spoof={all_spoof_labels.shape}, speaker={all_speaker_labels.shape} '",
        "      f'({len(np.unique(all_speaker_labels))} unique speakers)')",
    ),
    md(
        "## 9. Per-Group Linear Probes",
        "",
        "Train a `LogisticRegression` per group on (a) spoof labels and (b) speaker labels. 80/20 split, balanced class weight, max_iter=1000. Report accuracy per group + chance baseline.",
    ),
    code(
        "from sklearn.linear_model import LogisticRegression",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.metrics import accuracy_score, balanced_accuracy_score",
        "",
        "spoof_results = {}",
        "speaker_results = {}",
        "",
        "for k in range(K):",
        "    z = all_z_groups[k]",
        "    # spoof probe (binary)",
        "    z_tr, z_te, y_tr, y_te = train_test_split(z, all_spoof_labels, test_size=0.2, random_state=42, stratify=all_spoof_labels)",
        "    clf = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1)",
        "    clf.fit(z_tr, y_tr)",
        "    spoof_results[k] = balanced_accuracy_score(y_te, clf.predict(z_te))",
        "    # speaker probe (multinomial)",
        "    z_tr, z_te, y_tr, y_te = train_test_split(z, all_speaker_labels, test_size=0.2, random_state=42, stratify=all_speaker_labels)",
        "    clf = LogisticRegression(max_iter=1000, multi_class='multinomial', class_weight='balanced', n_jobs=-1)",
        "    clf.fit(z_tr, y_tr)",
        "    speaker_results[k] = accuracy_score(y_te, clf.predict(z_te))",
        "    print(f'  group {k}: spoof bal-acc={spoof_results[k]:.3f}, speaker acc={speaker_results[k]:.3f}')",
        "",
        "n_speakers = len(np.unique(all_speaker_labels))",
        "speaker_chance = 1.0 / n_speakers",
        "print(f'\\nSpeaker chance: {speaker_chance:.4f} ({n_speakers} speakers)')",
        "print(f'Spoof chance:   0.500 (balanced binary)')",
    ),
    md(
        "## 10. Plot",
        "",
        "Two-row grouped bar chart: spoof accuracy and speaker accuracy per group.",
    ),
    code(
        "import matplotlib.pyplot as plt",
        "",
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))",
        "groups = list(range(K))",
        "",
        "axes[0].bar(groups, [spoof_results[k] for k in groups], color='C0')",
        "axes[0].axhline(0.5, ls='--', color='gray', label='chance')",
        "axes[0].set_xticks(groups); axes[0].set_xlabel('Group index')",
        "axes[0].set_ylabel('Spoof balanced accuracy')",
        "axes[0].set_title('Spoof signal per group')",
        "axes[0].legend()",
        "",
        "axes[1].bar(groups, [speaker_results[k] for k in groups], color='C1')",
        "axes[1].axhline(speaker_chance, ls='--', color='gray', label=f'chance ({speaker_chance:.3f})')",
        "axes[1].set_xticks(groups); axes[1].set_xlabel('Group index')",
        "axes[1].set_ylabel('Speaker accuracy')",
        "axes[1].set_title('Speaker identity retained per group')",
        "axes[1].legend()",
        "",
        "plt.tight_layout()",
        "fig_path = os.path.join(RUN_DIR, 'phase_c_layer_probe.png')",
        "plt.savefig(fig_path, dpi=150)",
        "print(f'Saved: {fig_path}')",
        "plt.show()",
        "",
        "# Save numerics to JSON for the paper.",
        "out = {",
        "    'checkpoint': CKPT_PATH,",
        "    'n_speakers': int(n_speakers),",
        "    'speaker_chance': float(speaker_chance),",
        "    'per_group': {",
        "        str(k): {",
        "            'spoof_balanced_accuracy': float(spoof_results[k]),",
        "            'speaker_accuracy': float(speaker_results[k]),",
        "        } for k in groups",
        "    },",
        "}",
        "json_path = os.path.join(RUN_DIR, 'phase_c_layer_probe.json')",
        "with open(json_path, 'w') as f: _json.dump(out, f, indent=2)",
        "print(f'Saved: {json_path}')",
    ),
    md(
        "## 11. Interpretation Template (for ICTAI Section V.J)",
        "",
        "Replace the bracketed values with the printed numbers above. The narrative anchors what the paper claims about the groups.",
        "",
        "> The trained per_group_vib model retains spoof-discriminative signal across all three groups (balanced accuracy [SPOOF_G0 / SPOOF_G1 / SPOOF_G2] for groups 0/1/2 on the dev split), confirming that the per-group bottleneck does not destroy task signal at any depth. Speaker-identity probes show a [PATTERN: increasing / decreasing / flat] trend across groups (accuracy [SPK_G0 / SPK_G1 / SPK_G2] vs [CHANCE]), consistent with WavLM's known late-layer concentration of speaker information. Group K is the [LATE / EARLY / MIDDLE] group and carries the most [LINGUISTIC / ACOUSTIC] signal under our compression budget.",
    ),
]


def build():
    write_notebook(CELLS, str(NOTEBOOK_PATH))


if __name__ == "__main__":
    build()
