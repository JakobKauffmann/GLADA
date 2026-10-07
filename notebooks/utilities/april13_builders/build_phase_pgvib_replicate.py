"""Builds phase_pgvib_replicate.ipynb — extra seeds for full per-group VIB
and drop_vib_g0, plus xlsr-300m g0-only cross-backbone.

Two phases queued back-to-back on Colab Runtime 3, sharing one env-setup pass.

Phase 1 (multi-seed replication, 4 runs): turns the existing seed-42
single-point per-group VIB (11.67%) and drop_vib_g0 (17.61%) results into
3-seed mean ± std measurements by adding seeds 123 and 456 for both. The
gap between per-group mean and drop_vib_g0 mean is the key number for
the LOO finding's robustness.

Phase 2 (xlsr-300m g0-only, 1 run): cross-backbone validation with a
different SSL family (wav2vec2-based, multilingual, distinct from WavLM).
xlsr-300m has 4 groups, so g0-only means vib_disabled_groups=(1, 2, 3).
batch_size=24 for memory; grad_accum_steps auto-recomputed.

Variants:
    pergroup_vib_s123 / s456                 (Phase 1, ~6 hr)
    drop_vib_g0_s123 / s456                  (Phase 1, ~6 hr)
    g0_only_xlsr_s42                         (Phase 2, ~6-8 hr)

Total wall time: ~18-20 hr (within Colab Pro 24 hr session limit).

Build:
    python build_phase_pgvib_replicate.py
"""

from pathlib import Path
from _phase_common import (
    md, code,
    env_setup_cells, imports_cells, model_class_cells,
    make_config_cell, datasets_cells, variant_sweep_cells,
    reference_table_cells, write_notebook,
)

NOTEBOOK_PATH = Path(__file__).parent / "phase_pgvib_replicate.ipynb"


CELLS = [
    md(
        "# Phase PGVIB — Multi-seed Replication + xlsr-300m g0-only (Runtime 3)",
        "",
        "Two phases queued back-to-back on Colab Runtime 3, sharing one env-setup pass.",
        "",
        "## Phase 1 — Multi-seed replication (per-group VIB + drop-g0)",
        "",
        "**Goal:** turn the existing seed-42 per-group VIB and drop_vib_g0 single-point results into 3-seed mean ± std measurements. Seed 42 is already done (per-group: 11.67%, drop-g0: 17.61%). This phase adds seeds 123 and 456 for both.",
        "",
        "The contribution claim — that monolithic VIB is suboptimal and g0 carries the bottleneck signal — depends on the per-group ↔ drop-g0 gap being stable across seeds. A 6-point gap at one seed is suggestive; the same gap with mean ± std across three seeds is evidence.",
        "",
        "**Variants (4 runs):**",
        "1. `pergroup_vib_s123` — full per-group VIB, seed 123",
        "2. `pergroup_vib_s456` — full per-group VIB, seed 456",
        "3. `drop_vib_g0_s123` — VIB disabled on g0, seed 123",
        "4. `drop_vib_g0_s456` — VIB disabled on g0, seed 456",
        "",
        "All other knobs match the existing seed-42 reference.",
        "",
        "## Phase 2 — Cross-backbone (xlsr-300m g0-only, 1 run)",
        "",
        "**Goal:** show the g0-only finding generalizes beyond the WavLM family. xlsr-300m is a wav2vec2-lineage multilingual SSL model — distinct training objective and data from WavLM. 4 groups, so `vib_disabled_groups=(1, 2, 3)` for g0-only. batch_size=24 to fit on A100; grad_accum_steps auto-recomputed.",
        "",
        "**Caveat:** single-seed directional check, not apples-to-apples on every knob.",
        "",
        "## Combined sweep",
        "",
        "Single `VARIANTS` dict with 5 entries. Wall time ≈ 18-20 hr (within Colab Pro 24 hr session limit). Splittable later by name prefix (`pergroup_vib_` / `drop_vib_g0_` / `g0_only_xlsr_`).",
        "",
        "Self-contained: Run All from scratch.",
    ),
    *env_setup_cells(),
    *imports_cells(),
    *model_class_cells(),
    *make_config_cell(default_backbone="wavlm-base-plus", drive_subdir="phase_pgvib_replicate"),
    *datasets_cells(use_rawboost=True),
    md(
        "## 8b. Variant Definitions (Phase 1 + Phase 2)",
        "",
        "Five runs total. Phase 1 = 4 replication runs on Base-Plus. Phase 2 = 1 cross-backbone run on xlsr-300m (4 groups).",
    ),
    code(
        "RAWBOOST_DEFAULT = True  # matches existing seed-42 references",
        "RESULTS_FILENAME = 'phase_pgvib_replicate_results.json'",
        "",
        "VARIANTS = {",
        "    # === Phase 1: replication (Base-Plus, 2 extra seeds for each condition) ===",
        "    'pergroup_vib_s123': {",
        "        'model_class': 'PerGroupVIBModel',",
        "        'model_kwargs': {},",
        "        'config_overrides': {'seed': 123},",
        "    },",
        "    'pergroup_vib_s456': {",
        "        'model_class': 'PerGroupVIBModel',",
        "        'model_kwargs': {},",
        "        'config_overrides': {'seed': 456},",
        "    },",
        "    'drop_vib_g0_s123': {",
        "        'model_class': 'PerGroupVIBModel',",
        "        'model_kwargs': {'vib_disabled_groups': (0,)},",
        "        'config_overrides': {'seed': 123},",
        "    },",
        "    'drop_vib_g0_s456': {",
        "        'model_class': 'PerGroupVIBModel',",
        "        'model_kwargs': {'vib_disabled_groups': (0,)},",
        "        'config_overrides': {'seed': 456},",
        "    },",
        "    # === Phase 2: xlsr-300m g0-only (cross-backbone, 1 run, seed 42) ===",
        "    'g0_only_xlsr_s42': {",
        "        'model_class': 'PerGroupVIBModel',",
        "        'backbone': 'xlsr-300m',",
        "        'model_kwargs': {'vib_disabled_groups': (1, 2, 3)},  # xlsr-300m has 4 groups",
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
        "One-batch source-loader probe before launching 12 hours of training.",
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
        "    'per_group_vib (BP, s42)':   11.67,  # existing seed-42 per-group VIB",
        "    'drop_vib_g0 (BP, s42)':     17.61,  # existing seed-42 drop-g0",
        "    'drop_vib_g1 (BP, s42)':     13.80,",
        "    'drop_vib_g2 (BP, s42)':     11.99,",
    ], extra_intro="The four runs in this notebook combined with the existing seed-42 results give 3-seed measurements for both per_group_vib and drop_vib_g0. The headline comparison is per-group mean vs drop-g0 mean — the LOO finding requires this gap to persist across seeds."),
    md(
        "## 11. Verdict & Multi-seed Summary",
        "",
        "Two summaries: Phase 1 (3-seed mean ± std for per-group VIB and drop-g0, combining the existing seed-42 numbers with the two new seeds), and Phase 2 (xlsr-300m g0-only cross-backbone result).",
    ),
    code(
        "import statistics",
        "valid = {n: r for n, r in all_results.items() if 'error' not in r and r.get('best_eer') is not None}",
        "if not valid:",
        "    print('All variants failed; cannot summarize.'); raise SystemExit(0)",
        "",
        "# Existing seed-42 results (from phase_a_leaveout)",
        "S42_PERGROUP = 11.67",
        "S42_DROP_G0  = 17.61",
        "",
        "# === Phase 1: multi-seed replication summary ===",
        "pergroup_eers = [S42_PERGROUP] + [valid[n]['best_eer'] for n in ('pergroup_vib_s123', 'pergroup_vib_s456') if n in valid]",
        "drop_g0_eers  = [S42_DROP_G0]  + [valid[n]['best_eer'] for n in ('drop_vib_g0_s123', 'drop_vib_g0_s456') if n in valid]",
        "",
        "def stats(name, eers):",
        "    if len(eers) < 2:",
        "        print(f'  {name}: insufficient seeds, have {eers}'); return None",
        "    m, s = statistics.mean(eers), statistics.stdev(eers)",
        "    print(f'  {name}: mean={m:.2f}%, std={s:.2f}% (n={len(eers)}, vals={[f\"{v:.2f}\" for v in eers]})')",
        "    return m",
        "",
        "print('=' * 70)",
        "print('PHASE 1 — Multi-seed replication (combining existing s42 + this notebook):')",
        "m_pg = stats('per_group_vib', pergroup_eers)",
        "m_g0 = stats('drop_vib_g0   ', drop_g0_eers)",
        "if m_pg is not None and m_g0 is not None:",
        "    print(f'  Gap (drop-g0 mean - per-group mean): {m_g0 - m_pg:+.2f} pp')",
        "    print('  -> If gap is positive and >> seed-noise, LOO finding is robust.')",
        "print('=' * 70)",
        "",
        "# === Phase 2: xlsr-300m cross-backbone ===",
        "phase2 = {n: r for n, r in valid.items() if n.startswith('g0_only_xlsr_')}",
        "if phase2:",
        "    print()",
        "    print('=' * 70)",
        "    print('PHASE 2 — Cross-backbone (xlsr-300m g0-only, seed 42, batch_size=24):')",
        "    for name, r in phase2.items():",
        "        print(f'  {name}: {r[\"best_eer\"]:.2f}% ITW EER')",
        "    print('  Different SSL family (wav2vec2 lineage, multilingual) vs WavLM.')",
        "    print('  If g0-only on xlsr also helps, finding generalizes across SSL families.')",
        "    print('=' * 70)",
    ),
]


def build():
    write_notebook(CELLS, str(NOTEBOOK_PATH))


if __name__ == "__main__":
    build()
