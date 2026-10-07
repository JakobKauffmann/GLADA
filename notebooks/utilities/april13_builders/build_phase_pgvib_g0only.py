"""Builds phase_pgvib_g0only.ipynb — g0-only VIB at 3 seeds + WavLM-Large g0-only.

Two phases queued back-to-back on Colab Runtime 2, sharing one env-setup pass.

Phase 1 (g0-only VIB on Base-Plus, 3 seeds): tests the proposed sufficient
configuration — VIB applied only to the lowest layer group (g0), with g1
and g2 left unbottled (PerGroupVIBModel(vib_disabled_groups=(1, 2))). The
LOO ablation suggests g0 is the only group whose bottleneck does
meaningful cross-domain work; this run quantifies whether g0-only matches
or beats full per-group across seeds.

Phase 2 (WavLM-Large g0-only, 1 run): cross-backbone validation — argues
the finding is not specific to WavLM-Base-Plus. Large has 4 SSL layer
groups, so g0-only means vib_disabled_groups=(1, 2, 3). batch_size=24 to
fit on A100 (Large is 316M params vs Base-Plus 95M); grad_accum_steps is
auto-recomputed by the patched variant_sweep. This is a directional
single-seed check, not apples-to-apples on every knob.

Variants:
    g0_only_vib_s42 / s123 / s456    (Phase 1, ~9 hr)
    g0_only_large_s42                (Phase 2, ~6-8 hr)

Total wall time: ~15-17 hr (within Colab Pro 24 hr session limit).

Build:
    python build_phase_pgvib_g0only.py
"""

from pathlib import Path
from _phase_common import (
    md, code,
    env_setup_cells, imports_cells, model_class_cells,
    make_config_cell, datasets_cells, variant_sweep_cells,
    reference_table_cells, write_notebook,
)

NOTEBOOK_PATH = Path(__file__).parent / "phase_pgvib_g0only.ipynb"


CELLS = [
    md(
        "# Phase PGVIB — g0-only VIB (Base-Plus 3 seeds + WavLM-Large) (Runtime 2)",
        "",
        "Two phases queued back-to-back on Colab Runtime 2, sharing one env-setup pass.",
        "",
        "## Phase 1 — g0-only VIB on WavLM-Base-Plus (3 seeds)",
        "",
        "**Goal:** test whether VIB applied ONLY to the lowest layer group (g0) recovers the full per-group VIB performance. The leave-one-out ablation showed g0 is the only group whose bottleneck materially helps cross-domain generalization — drop_vib_g0 collapses to 17.61% ITW while drop_vib_g2 sits at 11.99% (within noise of full per-group at 11.67%). If g0-only VIB matches or beats full per-group across seeds, the contribution refines to: 'low-layer-targeted VIB is sufficient and interpretable.'",
        "",
        "**Architecture:** `PerGroupVIBModel(vib_disabled_groups=(1, 2))`. The VIB encoder still runs on g1 and g2 (so dimensions match downstream), but `z_k = mu_k` (deterministic) and `kl_k = 0` for k ∈ {1, 2}. Only g0 has a stochastic bottleneck and KL pressure.",
        "",
        "**Variants:** seeds 42, 123, 456.",
        "",
        "## Phase 2 — Cross-backbone (WavLM-Large g0-only, 1 run)",
        "",
        "**Goal:** argue the g0-only finding is not specific to WavLM-Base-Plus. WavLM-Large has 4 SSL layer groups, so g0-only means `vib_disabled_groups=(1, 2, 3)`. `batch_size=24` to fit Large (316M params) on Colab A100; `grad_accum_steps` is auto-recomputed by the patched variant sweep.",
        "",
        "**Caveat:** this is a single-seed directional check, not apples-to-apples on every knob (batch size differs from the Base-Plus runs). The argument is 'g0-only VIB also helps on a different / larger backbone,' not 'g0-only Large beats g0-only Base-Plus by X.'",
        "",
        "## Combined sweep",
        "",
        "Single `VARIANTS` dict with 4 entries. Wall time ≈ 15-17 hr (within Colab Pro 24 hr session limit). Splittable later by name prefix (`g0_only_vib_` vs `g0_only_large_`).",
        "",
        "Self-contained: Run All from scratch.",
    ),
    *env_setup_cells(),
    *imports_cells(),
    *model_class_cells(),
    *make_config_cell(default_backbone="wavlm-base-plus", drive_subdir="phase_pgvib_g0only"),
    *datasets_cells(use_rawboost=True),
    md(
        "## 8b. Variant Definitions (Phase 1 + Phase 2)",
        "",
        "Four runs total. Phase 1 = three g0-only seeds on WavLM-Base-Plus. Phase 2 = one g0-only run on WavLM-Large with `batch_size=24` (4 groups → `vib_disabled_groups=(1, 2, 3)`).",
    ),
    code(
        "RAWBOOST_DEFAULT = True  # matches per-group VIB 11.67% reference",
        "RESULTS_FILENAME = 'phase_pgvib_g0only_results.json'",
        "",
        "VARIANTS = {",
        "    # === Phase 1: g0-only VIB on WavLM-Base-Plus (3 seeds) ===",
        "    'g0_only_vib_s42': {",
        "        'model_class': 'PerGroupVIBModel',",
        "        'model_kwargs': {'vib_disabled_groups': (1, 2)},",
        "        'config_overrides': {'seed': 42},",
        "    },",
        "    'g0_only_vib_s123': {",
        "        'model_class': 'PerGroupVIBModel',",
        "        'model_kwargs': {'vib_disabled_groups': (1, 2)},",
        "        'config_overrides': {'seed': 123},",
        "    },",
        "    'g0_only_vib_s456': {",
        "        'model_class': 'PerGroupVIBModel',",
        "        'model_kwargs': {'vib_disabled_groups': (1, 2)},",
        "        'config_overrides': {'seed': 456},",
        "    },",
        "    # === Phase 2: WavLM-Large g0-only (cross-backbone, 1 run, seed 42) ===",
        "    'g0_only_large_s42': {",
        "        'model_class': 'PerGroupVIBModel',",
        "        'backbone': 'wavlm-large',",
        "        'model_kwargs': {'vib_disabled_groups': (1, 2, 3)},  # Large has 4 groups",
        "        'config_overrides': {'seed': 42, 'batch_size': 24},",
        "    },",
        "}",
        "for name, spec in VARIANTS.items():",
        "    bb = spec.get('backbone', 'wavlm-base-plus (default)')",
        "    print(f'  {name:<24s} backbone={bb:<20s} '",
        "          f'seed={spec[\"config_overrides\"][\"seed\"]}, kwargs={spec[\"model_kwargs\"]}')",
    ),
    md(
        "## 8c. Sanity Probe",
        "",
        "One-batch source-loader probe before launching 9 hours of training. Confirms label and speaker_id ranges so a CPU-side bug surfaces before the GPU assert.",
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
        "    'per_group_vib (BP, s42)':   11.67,  # full per-group VIB reference",
        "    'drop_vib_g0 (BP, s42)':     17.61,  # leave-g0-out — worst LOO",
        "    'drop_vib_g1 (BP, s42)':     13.80,",
        "    'drop_vib_g2 (BP, s42)':     11.99,  # leave-g2-out (within noise)",
    ], extra_intro="If g0-only VIB lands within seed noise of per_group_vib (BP, s42) = 11.67%, the contribution refines to 'low-layer-targeted VIB is sufficient.' If it materially beats per-group, even better — the bottleneck on g1/g2 is actively hurting and the cleaner config is also the better one."),
    md(
        "## 11. Verdict",
        "",
        "Two summaries: Phase 1 (Base-Plus g0-only seeds) and Phase 2 (cross-backbone WavLM-Large g0-only).",
    ),
    code(
        "import statistics",
        "valid = {n: r for n, r in all_results.items() if 'error' not in r and r.get('best_eer') is not None}",
        "if not valid:",
        "    print('All variants failed; cannot summarize.'); raise SystemExit(0)",
        "",
        "# === Phase 1: g0-only Base-Plus seeds ===",
        "phase1 = {n: r for n, r in valid.items() if n.startswith('g0_only_vib_')}",
        "if phase1:",
        "    eers = [r['best_eer'] for r in phase1.values()]",
        "    mean_eer = statistics.mean(eers)",
        "    stdev_eer = statistics.stdev(eers) if len(eers) > 1 else 0.0",
        "    print('=' * 70)",
        "    print(f'PHASE 1 — g0-only VIB on WavLM-Base-Plus ({len(eers)} seeds):')",
        "    print(f'  mean = {mean_eer:.2f}%, std = {stdev_eer:.2f}%')",
        "    print(f'  Per-group VIB (s42 ref):       11.67%')",
        "    print(f'  Gap (g0-only - per-group, s42): {mean_eer - 11.67:+.2f} pp')",
        "    for name, r in phase1.items():",
        "        print(f'    {name}: {r[\"best_eer\"]:.2f}%')",
        "    print('=' * 70)",
        "",
        "# === Phase 2: WavLM-Large g0-only ===",
        "phase2 = {n: r for n, r in valid.items() if n.startswith('g0_only_large_')}",
        "if phase2:",
        "    print()",
        "    print('=' * 70)",
        "    print('PHASE 2 — Cross-backbone (WavLM-Large g0-only, seed 42, batch_size=24):')",
        "    for name, r in phase2.items():",
        "        print(f'  {name}: {r[\"best_eer\"]:.2f}% ITW EER')",
        "    print('  Reference: per_group_vib (Large, RB on, L1) = 13.52%')",
        "    print('  Reference: bb_wavlm-large_s42                = 12.64%')",
        "    print('  If g0_only_large lands near these, the g0-only finding generalizes.')",
        "    print('=' * 70)",
    ),
]


def build():
    write_notebook(CELLS, str(NOTEBOOK_PATH))


if __name__ == "__main__":
    build()
