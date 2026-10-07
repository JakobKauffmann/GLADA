"""Builds phase_a_norb.ipynb — Phase A RawBoost on/off probe on WavLM-Base-Plus.

Single variant: per_group_vib + Base-Plus + RawBoost OFF. Compares against
the 11.67% reference (RawBoost ON) to determine whether RawBoost helps in
the per_group_vib architecture.

Background: in the original ablation, no_rawboost (16.66%) BEAT full_glada
(19.71%), meaning RawBoost was hurting the multi-loss stack. With per_group_vib's
compression/GRL conflict resolved, RawBoost is more likely to help cleanly,
but it's untested. This notebook tests it.

Decision rule:
  If no-RB run < 11.67% → RawBoost was still hurting; drop RawBoost going forward.
  If no-RB run > 11.67% → RawBoost helps in per_group_vib; keep it.

Build:
    python build_phase_a_norb.py
"""

from pathlib import Path
from _phase_common import (
    md, code,
    env_setup_cells, imports_cells, model_class_cells,
    make_config_cell, datasets_cells, variant_sweep_cells,
    reference_table_cells, write_notebook,
)

NOTEBOOK_PATH = Path(__file__).parent / "phase_a_norb.ipynb"


CELLS = [
    md(
        "# Phase A — RawBoost On/Off Probe (per_group_vib, WavLM-Base-Plus)",
        "",
        "**Goal:** determine whether RawBoost helps `per_group_vib` (currently 11.67% with RawBoost ON). The original ablation showed RawBoost was *hurting* `full_glada` (no_rawboost beat full_glada by 3 pp), but we don't yet know if it still hurts now that the compression / per-group GRL conflict has been resolved.",
        "",
        "**Single variant:** `per_group_vib` + Base-Plus + **RawBoost OFF** (augmenter=None). Same seed, same epochs, same model class as the 11.67% reference; only the source-domain augmentation changes.",
        "",
        "**Decision:**",
        "- no-RB ≪ 11.67% (e.g., < 11.0%): RawBoost was still hurting; drop it for Phase B.",
        "- no-RB ≈ 11.67%: RawBoost is neutral; keep it (it provides robustness on other datasets).",
        "- no-RB ≫ 11.67% (e.g., > 12.5%): RawBoost helps; keep it for Phase B.",
        "",
        "Self-contained: Run All from scratch. Independent of other notebooks.",
    ),
    *env_setup_cells(),
    *imports_cells(),
    *model_class_cells(),
    *make_config_cell(default_backbone="wavlm-base-plus", drive_subdir="phase_a_norb"),
    *datasets_cells(use_rawboost=False),  # NO RawBoost in source dataset
    md(
        "## 8b. Variant Definition",
        "",
        "One variant: `per_group_vib_no_rb` is `per_group_vib` with `augmenter=None` on the source dataset (handled in section 8 above). All other settings match the 11.67% reference.",
    ),
    code(
        "RAWBOOST_DEFAULT = False  # this notebook builds source_ds without RawBoost",
        "RESULTS_FILENAME = 'phase_a_norb_results.json'",
        "",
        "VARIANTS = {",
        "    'per_group_vib_no_rb': {",
        "        'model_class': 'PerGroupVIBModel',",
        "        'model_kwargs': {},  # no leave-one-out, no per-group beta",
        "        'use_rawboost': False,  # explicit, matches the source_ds construction",
        "    },",
        "}",
        "for name, spec in VARIANTS.items():",
        "    print(f'  {name:<24s} model={spec[\"model_class\"]} rawboost={spec[\"use_rawboost\"]}')",
    ),
    *variant_sweep_cells(),
    *reference_table_cells([
        "    'full_glada':              19.71,",
        "    'no_glada':                14.89,",
        "    'no_rawboost (full_glada)':16.66,",
        "    'no_cka':                  12.98,",
        "    'per_group_vib (BP, RB)':  11.67,  # the 11.67% reference",
    ], extra_intro="The interesting comparison is **no-RB vs `per_group_vib (BP, RB)` = 11.67%**. The other rows give context for whether RawBoost still has the historical no-RB-beats-RB sign in this architecture."),
    md(
        "## 11. Verdict & Next Step",
        "",
        "Reads the no-RB EER and prints the recommendation for Phase B's RawBoost setting.",
    ),
    code(
        "valid = {n: r for n, r in all_results.items() if 'error' not in r and r.get('best_eer') is not None}",
        "if not valid:",
        "    print('No-RB run failed; cannot make a recommendation.'); raise SystemExit(0)",
        "BP_REFERENCE_RB = 11.67",
        "no_rb_eer = valid['per_group_vib_no_rb']['best_eer']",
        "delta = no_rb_eer - BP_REFERENCE_RB",
        "print()",
        "print('=' * 70)",
        "print(f'no-RB EER: {no_rb_eer:.2f}%, vs RB-on 11.67%: {delta:+.2f} pp')",
        "if delta < -0.5:",
        "    print('  -> RawBoost was STILL hurting per_group_vib. Drop it for Phase B.')",
        "elif delta > 0.5:",
        "    print('  -> RawBoost helps per_group_vib. Keep RB on for Phase B.')",
        "else:",
        "    print('  -> RawBoost is roughly neutral. Keep it (robustness on other datasets is unchanged).')",
        "print('=' * 70)",
    ),
]


def build():
    write_notebook(CELLS, str(NOTEBOOK_PATH))


if __name__ == "__main__":
    build()
