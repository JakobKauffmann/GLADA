"""Builds phase_pgvib_monolithic.ipynb — Monolithic VIB baseline + β sweep at g0.

Two phases queued back-to-back on Colab Runtime 1:

Phase 1 (monolithic VIB baseline, 3 seeds): the comparator for the
"monolithic VIB across SSL layer groups is suboptimal" claim. Stock
`GLADAModel` — single VIBBottleneck applied AFTER hierarchical pooling
(see `glada_model.py`, Step 4 of forward). All other knobs match the
11.67% per-group VIB reference.

Phase 2 (β sweep at g0, 3 values): probes the response curve of the
g0 bottleneck strength. PerGroupVIBModel with g1/g2 VIB disabled
(deterministic, kl=0). config.vib_beta swept across {0.005, 0.05, 0.10}
at the same seed (42). Shows whether stronger compression at g0 helps
further or whether the default 0.01 is near-optimal.

Variants:
    monolithic_vib_s42 / s123 / s456                   (Phase 1, ~9 hr)
    g0_only_beta_x0p5 / x5 / x10  (vib_beta multiplier) (Phase 2, ~9 hr)

Total wall time: ~18 hr (within Colab Pro 24 hr session limit).

The two phases share a single `VARIANTS` dict for sweep-loop simplicity.
After the notebook is run with outputs, this can be split back into two
notebooks (monolithic + beta sweep) post-hoc by selecting variants whose
name starts with `monolithic_` or `g0_only_beta_`.

Build:
    python build_phase_pgvib_monolithic.py
"""

from pathlib import Path
from _phase_common import (
    md, code,
    env_setup_cells, imports_cells, model_class_cells,
    make_config_cell, datasets_cells, variant_sweep_cells,
    reference_table_cells, write_notebook,
)

NOTEBOOK_PATH = Path(__file__).parent / "phase_pgvib_monolithic.ipynb"


CELLS = [
    md(
        "# Phase PGVIB — Monolithic VIB Baseline + β-sweep at g0 (Runtime 1)",
        "",
        "Two phases queued back-to-back on Colab Runtime 1, sharing one env-setup pass.",
        "",
        "## Phase 1 — Monolithic VIB baseline (3 seeds)",
        "",
        "**Goal:** establish the monolithic-VIB comparator that the per-group VIB contribution is claiming is suboptimal. A single-seed reference (`full_glada` ≈ 19.71% ITW) exists, but the paper needs a 3-seed multi-seed baseline matched on every other knob to the 11.67% per-group VIB run.",
        "",
        "**Architecture:** stock `GLADAModel` — VIBBottleneck applied AFTER hierarchical pooling (one z, not K z_k). Per-group discriminators receive the raw `group_outputs` (pre-VIB g_k). All other knobs match the per-group VIB reference: RawBoost on, GLADA disc on, CKA loss on, speaker adversary on, batch_size=64, Phase 1=3 ep, Phase 2=17 ep.",
        "",
        "**Variants:** seeds 42, 123, 456.",
        "",
        "## Phase 2 — β-sweep at g0 (3 values, single seed)",
        "",
        "**Goal:** probe the response curve of the g0 bottleneck strength. If g0-only VIB is the new headline, knowing how sensitive it is to β tells us whether the default 0.01 is near-optimal or if a different strength helps further.",
        "",
        "**Architecture:** `PerGroupVIBModel(vib_disabled_groups=(1, 2))` — only g0 has an active stochastic bottleneck. `config.vib_beta` is swept across {0.005, 0.05, 0.10} at seed 42. Effective β at g0 is `config.vib_beta / 3` due to the uniform-mean KL aggregation across groups (g1, g2 contribute 0 since disabled).",
        "",
        "**Variants:** `g0_only_beta_x0p5` (β=0.005), `g0_only_beta_x5` (β=0.05), `g0_only_beta_x10` (β=0.10). Naming is relative to the default β=0.01.",
        "",
        "## Combined sweep",
        "",
        "Single `VARIANTS` dict with 6 entries. Wall time ≈ 18 hr (within Colab Pro 24 hr session limit). After execution, results split cleanly into two notebooks by name prefix (`monolithic_` vs `g0_only_beta_`).",
        "",
        "Self-contained: Run All from scratch.",
    ),
    *env_setup_cells(),
    *imports_cells(),
    *model_class_cells(),  # PerGroupVIBModel inline def is unused here but kept for parity
    *make_config_cell(default_backbone="wavlm-base-plus", drive_subdir="phase_pgvib_monolithic"),
    *datasets_cells(use_rawboost=True),
    md(
        "## 8b. Variant Definitions (Phase 1 + Phase 2)",
        "",
        "Six runs total. Phase 1 = three monolithic VIB seeds (`GLADAModel`). Phase 2 = three β values for g0-only VIB (`PerGroupVIBModel(vib_disabled_groups=(1, 2))`) at seed 42.",
    ),
    code(
        "RAWBOOST_DEFAULT = True  # matches per-group VIB 11.67% reference",
        "RESULTS_FILENAME = 'phase_pgvib_monolithic_results.json'",
        "",
        "VARIANTS = {",
        "    # === Phase 1: Monolithic VIB baseline (3 seeds) ===",
        "    'monolithic_vib_s42': {",
        "        'model_class': 'GLADAModel',",
        "        'model_kwargs': {},",
        "        'config_overrides': {'seed': 42},",
        "    },",
        "    'monolithic_vib_s123': {",
        "        'model_class': 'GLADAModel',",
        "        'model_kwargs': {},",
        "        'config_overrides': {'seed': 123},",
        "    },",
        "    'monolithic_vib_s456': {",
        "        'model_class': 'GLADAModel',",
        "        'model_kwargs': {},",
        "        'config_overrides': {'seed': 456},",
        "    },",
        "    # === Phase 2: β sweep at g0 (3 values, seed 42) ===",
        "    'g0_only_beta_x0p5': {",
        "        'model_class': 'PerGroupVIBModel',",
        "        'model_kwargs': {'vib_disabled_groups': (1, 2)},",
        "        'config_overrides': {'seed': 42, 'vib_beta': 0.005},",
        "    },",
        "    'g0_only_beta_x5': {",
        "        'model_class': 'PerGroupVIBModel',",
        "        'model_kwargs': {'vib_disabled_groups': (1, 2)},",
        "        'config_overrides': {'seed': 42, 'vib_beta': 0.05},",
        "    },",
        "    'g0_only_beta_x10': {",
        "        'model_class': 'PerGroupVIBModel',",
        "        'model_kwargs': {'vib_disabled_groups': (1, 2)},",
        "        'config_overrides': {'seed': 42, 'vib_beta': 0.10},",
        "    },",
        "}",
        "for name, spec in VARIANTS.items():",
        "    ovr = spec.get('config_overrides', {})",
        "    print(f'  {name:<24s} model={spec[\"model_class\"]:<18s} '",
        "          f'seed={ovr.get(\"seed\")}, vib_beta={ovr.get(\"vib_beta\", \"default\")}, '",
        "          f'kwargs={spec.get(\"model_kwargs\", {})}')",
    ),
    md(
        "## 8c. Sanity Probe",
        "",
        "One-batch source-loader probe. Confirms label and speaker_id ranges before launching 9 hours of training. Cheap and catches the most common GPU-assert root causes CPU-side.",
    ),
    code(
        "print('Sanity-checking source loader (3 batches)...')",
        "sanity_loader = torch.utils.data.DataLoader(source_ds, batch_size=4, shuffle=True, num_workers=0)",
        "for i, batch in enumerate(sanity_loader):",
        "    if i >= 3: break",
        "    spk = batch.get('speaker_id')",
        "    lbl = batch.get('task_label')",
        "    assert lbl is not None, f'No task_label in batch keys: {list(batch.keys())}'",
        "    assert lbl.min() >= 0 and lbl.max() <= 1, f'task_label out of range: [{int(lbl.min())}, {int(lbl.max())}]'",
        "    if spk is not None:",
        "        assert spk.min() >= 0 and spk.max() < num_speakers, \\",
        "            f'speaker_id out of range: [{int(spk.min())}, {int(spk.max())}] vs num_speakers={num_speakers}'",
        "    shape_summary = [(k, tuple(v.shape)) for k, v in batch.items() if hasattr(v, 'shape')]",
        "    print(f'  batch {i}: shapes={shape_summary}, task_labels={lbl.unique().tolist()}')",
        "print('Source loader sanity OK.')",
    ),
    *variant_sweep_cells(),
    *reference_table_cells([
        "    'per_group_vib (BP, s42)':   11.67,  # reference per-group VIB",
        "    'drop_vib_g0 (BP, s42)':     17.61,  # leave-g0-out (worst LOO)",
        "    'drop_vib_g1 (BP, s42)':     13.80,  # leave-g1-out",
        "    'drop_vib_g2 (BP, s42)':     11.99,  # leave-g2-out (within noise of per-group)",
        "    'full_glada (BP, historical)': 19.71,  # historical single-seed monolithic",
    ], extra_intro="A meaningful positive delta vs per_group_vib (BP, s42) means monolithic VIB is materially worse — direct support for the contribution. The historical full_glada reference (19.71%) is single-seed and used different hyperparameters; this notebook is the apples-to-apples 3-seed comparator."),
    md(
        "## 11. Verdict",
        "",
        "Two summaries: Phase 1 (monolithic seed mean ± std vs the per-group VIB seed-42 reference of 11.67%) and Phase 2 (β response curve at g0).",
    ),
    code(
        "import statistics",
        "valid = {n: r for n, r in all_results.items() if 'error' not in r and r.get('best_eer') is not None}",
        "if not valid:",
        "    print('All variants failed; cannot summarize.'); raise SystemExit(0)",
        "",
        "# === Phase 1: monolithic seeds ===",
        "monolithic = {n: r for n, r in valid.items() if n.startswith('monolithic_vib_')}",
        "if monolithic:",
        "    eers = [r['best_eer'] for r in monolithic.values()]",
        "    mean_eer = statistics.mean(eers)",
        "    stdev_eer = statistics.stdev(eers) if len(eers) > 1 else 0.0",
        "    print('=' * 70)",
        "    print(f'PHASE 1 — Monolithic VIB ({len(eers)} seeds): mean = {mean_eer:.2f}%, std = {stdev_eer:.2f}%')",
        "    print(f'  Per-group VIB (s42 ref):           11.67%')",
        "    print(f'  Gap (monolithic - per-group s42):  {mean_eer - 11.67:+.2f} pp')",
        "    for name, r in monolithic.items():",
        "        print(f'    {name}: {r[\"best_eer\"]:.2f}%')",
        "    print('=' * 70)",
        "",
        "# === Phase 2: β response curve at g0 ===",
        "beta_sweep = {n: r for n, r in valid.items() if n.startswith('g0_only_beta_')}",
        "if beta_sweep:",
        "    print()",
        "    print('=' * 70)",
        "    print('PHASE 2 — β response curve at g0 (PerGroupVIBModel, g1/g2 disabled, seed 42):')",
        "    label_to_beta = {'g0_only_beta_x0p5': 0.005, 'g0_only_beta_x5': 0.05, 'g0_only_beta_x10': 0.10}",
        "    sweep_pts = sorted(",
        "        [(label_to_beta[n], r['best_eer'], n) for n, r in beta_sweep.items() if n in label_to_beta],",
        "    )",
        "    print(f'  {\"vib_beta\":>10s}  {\"ITW EER\":>10s}  {\"variant\":<24s}')",
        "    for beta, eer, name in sweep_pts:",
        "        print(f'  {beta:>10.4f}  {eer:>10.2f}  {name:<24s}')",
        "    print(f'  default 0.0100        ?    g0_only_vib_s42 (in phase_pgvib_g0only)')",
        "    print('=' * 70)",
    ),
]


def build():
    write_notebook(CELLS, str(NOTEBOOK_PATH))


if __name__ == "__main__":
    build()
