"""Builds phase_a_leaveout.ipynb — Phase A leave-one-VIB-out on WavLM-Base-Plus.

Three variants, all using PerGroupVIBModel + RawBoost on (matching the 11.67%
reference), differing only in which group's VIB is disabled:

  A2: drop_vib_g0  — disable VIB on Group 0 (early/acoustic layers)
  A3: drop_vib_g1  — disable VIB on Group 1 (middle/phonetic layers)
  A4: drop_vib_g2  — disable VIB on Group 2 (late/linguistic layers)

A "disabled" group still runs its VIB encoder so dimensions match downstream,
but z_k = mu_k (deterministic, no reparam sampling) and kl_k = 0 (no KL
pressure on that group). The encoder still learns a 256-dim projection; it
just isn't regularized as a stochastic bottleneck.

References (no retraining):
  full_glada (Base-Plus)              19.71% ITW EER
  no_glada                            14.89%
  no_cka                              12.98%
  per_group_vib (Base-Plus, RB on)    11.67%  <-- our 11.67% reference
  bb_wavlm-large_s42                  12.64%
  per_group_vib (Large, RB on, L1)    13.52%

Decision: any drop_vib_gK that beats 11.67% means that group's VIB was
hurting and we have a refined Base-Plus best. The winner informs Phase B
(scaled per-group beta) and the WavLM-Large transfer in B2.

Build:
    python build_phase_a_leaveout.py
"""

from pathlib import Path
from _phase_common import (
    md, code,
    env_setup_cells, imports_cells, model_class_cells,
    make_config_cell, datasets_cells, variant_sweep_cells,
    reference_table_cells, write_notebook,
)

NOTEBOOK_PATH = Path(__file__).parent / "phase_a_leaveout.ipynb"


CELLS = [
    md(
        "# Phase A — Leave-One-VIB-Out on WavLM-Base-Plus",
        "",
        "**Goal:** determine which group's VIB carries the gain in `per_group_vib` (11.67% ITW EER on Base-Plus). Three variants, each disabling exactly one group's VIB while keeping the other two active. RawBoost on (matching the 11.67% reference).",
        "",
        "**Variants:**",
        "1. `drop_vib_g0` — disable VIB on Group 0 (early/acoustic layers, ~layers 0-3 of WavLM-Base-Plus)",
        "2. `drop_vib_g1` — disable VIB on Group 1 (middle/phonetic layers, ~4-7)",
        "3. `drop_vib_g2` — disable VIB on Group 2 (late/linguistic layers, ~8-11)",
        "",
        "**'Disabled' semantics:** the VIB encoder still runs (so the 256-dim projection feeds the downstream concat normally), but `z_k = mu_k` (deterministic — no reparam noise at training time) and `kl_k = 0` (no KL pressure on that group). The other groups' VIBs run normally.",
        "",
        "**Decision rule:**",
        "- Any drop_vib_gK at < 11.67% → that group's VIB was hurting; the winner becomes the Base-Plus best.",
        "- All three drop_vib_gK at > 11.67% → every group's VIB contributes; per_group_vib stays as-is.",
        "- One drop_vib_gK ≪ 11.67% (e.g., < 10%) → strong leave-one-out winner; informs Phase B scaled-beta config.",
        "",
        "Self-contained: Run All from scratch. Does not depend on phase_vib_stagger or other notebooks.",
    ),
    *env_setup_cells(),
    *imports_cells(),
    *model_class_cells(),
    *make_config_cell(default_backbone="wavlm-base-plus", drive_subdir="phase_a_leaveout"),
    *datasets_cells(use_rawboost=True),
    md(
        "## 8b. Variant Definitions",
        "",
        "Each variant is `per_group_vib` with `vib_disabled_groups=(k,)` for k in {0,1,2}. Other model knobs (config_overrides) are unchanged from the 11.67% reference.",
    ),
    code(
        "RAWBOOST_DEFAULT = True  # all variants use RawBoost (matches 11.67% reference)",
        "RESULTS_FILENAME = 'phase_a_leaveout_results.json'",
        "",
        "# DIAGNOSTIC RUN ORDER: 'sanity_no_disable' first to confirm the new flag plumbing",
        "# didn't break anything (it should reproduce the 11.67% reference). Only after that",
        "# trains cleanly should drop_vib_g1 and drop_vib_g2 be uncommented and run.",
        "VARIANTS = {",
        "    'sanity_no_disable': {  # control: matches the 11.67% per_group_vib reference",
        "        'model_class': 'PerGroupVIBModel',",
        "        'model_kwargs': {},",
        "    },",
        "    'drop_vib_g0': {",
        "        'model_class': 'PerGroupVIBModel',",
        "        'model_kwargs': {'vib_disabled_groups': (0,)},",
        "    },",
        "    # 'drop_vib_g1': {",
        "    #     'model_class': 'PerGroupVIBModel',",
        "    #     'model_kwargs': {'vib_disabled_groups': (1,)},",
        "    # },",
        "    # 'drop_vib_g2': {",
        "    #     'model_class': 'PerGroupVIBModel',",
        "    #     'model_kwargs': {'vib_disabled_groups': (2,)},",
        "    # },",
        "}",
        "for name, spec in VARIANTS.items():",
        "    print(f'  {name:<20s} kwargs={spec[\"model_kwargs\"]}')",
    ),
    md(
        "## 8c. Diagnostic Pre-flight (DELETE ONCE STABLE)",
        "",
        "Two safeguards to make CUDA crashes legible. Remove these cells once the sweep runs cleanly:",
        "",
        "1. **Synchronous CUDA + anomaly detection** so kernel asserts surface as a real Python traceback at the exact failing line. Runs ~20% slower; only keep while debugging.",
        "2. **One-batch sanity probe** on the source loader to catch label-range bugs CPU-side before the GPU assert fires. Verifies `speaker_id ∈ [0, num_speakers)` and `label ∈ {0, 1}`.",
    ),
    code(
        "# Diagnostic env vars: ALL OMITTED to match the L1 run (proven 11.67% on Base-Plus).",
        "# Re-enable individually only if you need to debug a crash:",
        "#   CUDA_LAUNCH_BLOCKING=1   -- synchronous CUDA, gives Python traceback at failing op",
        "#   TORCH_USE_CUDA_DSA=1     -- device-side assertion file:line info",
        "#   set_detect_anomaly(True) -- DO NOT enable, OOMs on unfrozen backbone",
        "print('Diagnostic env disabled -- matching L1 (proven 11.67%) configuration')",
        "",
        "# 2) One-batch sanity probe.",
        "print('\\nSanity-checking source loader (3 batches)...')",
        "sanity_loader = torch.utils.data.DataLoader(source_ds, batch_size=4, shuffle=True, num_workers=0)",
        "for i, batch in enumerate(sanity_loader):",
        "    if i >= 3: break",
        "    spk = batch.get('speaker_id')",
        "    lbl = batch.get('task_label')  # ASVspoofDatasetWithSpeaker returns task_label, not label",
        "    assert lbl is not None, f'No task_label in batch keys: {list(batch.keys())}'",
        "    assert lbl.min() >= 0 and lbl.max() <= 1, f'task_label out of range: [{int(lbl.min())}, {int(lbl.max())}]'",
        "    if spk is not None:",
        "        assert spk.min() >= 0 and spk.max() < num_speakers, \\",
        "            f'speaker_id out of range: [{int(spk.min())}, {int(spk.max())}] vs num_speakers={num_speakers}'",
        "    shape_summary = [(k, tuple(v.shape)) for k, v in batch.items() if hasattr(v, 'shape')]",
        "    print(f'  batch {i}: shapes={shape_summary}, task_labels={lbl.unique().tolist()}, '",
        "          f'speakers={spk.unique().tolist() if spk is not None else \"n/a\"}')",
        "print('Source loader sanity OK.')",
    ),
    *variant_sweep_cells(),
    *reference_table_cells([
        "    'full_glada':              19.71,",
        "    'no_glada':                14.89,",
        "    'no_cka':                  12.98,",
        "    'per_group_vib (BP)':      11.67,  # the 11.67% reference",
        "    'bb_wavlm-large_s42':      12.64,",
        "    'per_group_vib (Large)':   13.52,",
    ], extra_intro="Negative delta vs `per_group_vib (BP)` (11.67%) means the leave-one-out variant **beats** the reference. The smallest (most negative) delta picks the winner."),
    md(
        "## 11. Verdict & Next Step",
        "",
        "If a variant beats 11.67% by more than 0.5 pp (significantly outside the bootstrap CI of [11.25, 12.10] for the reference): that group's VIB was actively hurting. The winner config (with that group's VIB disabled) becomes the Base-Plus best, and Phase B's scaled-beta variant is built around it (set the disabled group's β to 0 or near-0, redistribute to the other groups).",
        "",
        "If all three variants are ≥ 11.67%: every group's VIB contributes; per_group_vib stays as-is, and Phase B focuses on per-group β tuning (e.g., scale up the most informative group's β) rather than dropping anything.",
        "",
        "If results are mixed and noisy (within ±0.3 pp): re-run with seed 123 to disambiguate before authoring Phase B.",
    ),
    code(
        "valid = {n: r for n, r in all_results.items() if 'error' not in r and r.get('best_eer') is not None}",
        "if not valid:",
        "    print('All variants failed; cannot pick a winner.'); raise SystemExit(0)",
        "BP_REFERENCE = 11.67",
        "best_name = min(valid, key=lambda n: valid[n]['best_eer'])",
        "best_eer = valid[best_name]['best_eer']",
        "delta = best_eer - BP_REFERENCE",
        "print()",
        "print('=' * 70)",
        "print(f'Winner: {best_name} at {best_eer:.2f}% (delta vs 11.67% = {delta:+.2f} pp)')",
        "if delta < -0.5:",
        "    print('  -> SIGNIFICANT improvement; this group VIB was hurting.')",
        "    print(f'  -> Phase B: build scaled-beta variant with vib_disabled_groups={best_name[-2:]} or beta_k = 0.')",
        "elif delta < 0:",
        "    print('  -> Marginal improvement; check seed sensitivity before claiming.')",
        "else:",
        "    print('  -> No leave-one-out beat 11.67%; every group VIB contributes.')",
        "    print('  -> Phase B: focus on per-group beta tuning instead of disabling.')",
        "print('=' * 70)",
    ),
]


def build():
    write_notebook(CELLS, str(NOTEBOOK_PATH))


if __name__ == "__main__":
    build()
