"""Builds phase_b_confirm.ipynb — Phase B confirmation runs after Phase A.

This notebook has TWO placeholder variants whose configs you fill in based
on Phase A results:

  B1: Base-Plus, scaled per-group beta (or leave-one-out winner) + best
      RawBoost setting from phase_a_norb. Refines the 11.67% Base-Plus best.

  B2: WavLM-Large, applying Phase A learnings (leave-one-out group dropped,
      RawBoost setting). Tests whether the Base-Plus fix transfers to Large
      (the L1 result on Large was 13.52% — worse than full-GLADA Large 12.64%).

Configs are TODO until Phase A finishes. Edit the VARIANTS dict in section
8b before running.

Build:
    python build_phase_b_confirm.py
"""

from pathlib import Path
from _phase_common import (
    md, code,
    env_setup_cells, imports_cells, model_class_cells,
    make_config_cell, datasets_cells, variant_sweep_cells,
    reference_table_cells, write_notebook,
)

NOTEBOOK_PATH = Path(__file__).parent / "phase_b_confirm.ipynb"


CELLS = [
    md(
        "# Phase B — Confirmation Runs (Base-Plus refinement + WavLM-Large transfer)",
        "",
        "**Prerequisite:** run `phase_a_norb.ipynb` and `phase_a_leaveout.ipynb` first; their results inform the configs below.",
        "",
        "**Two variants** (fill in section 8b based on Phase A outcomes):",
        "1. `B1_bp_refined` — Base-Plus best refinement (scaled per-group β based on leave-one-out winner), with the better RawBoost setting from phase_a_norb.",
        "2. `B2_large_transfer` — WavLM-Large with the Phase A learnings transferred (leave-one-out group dropped, RawBoost setting). Tests whether the Base-Plus 11.67% fix generalizes to 316M params.",
        "",
        "**Skip B2 if** phase_a_axes (or a future targeted Large run) already closes the gap. Skip B1 if Phase A leave-one-out shows no group's VIB hurts (then per_group_vib at 11.67% is already the Base-Plus best).",
        "",
        "Self-contained: Run All from scratch. RawBoost is set per-variant via `use_rawboost` in the spec, so source_ds is rebuilt as needed.",
    ),
    *env_setup_cells(),
    *imports_cells(),
    *model_class_cells(),
    *make_config_cell(default_backbone="wavlm-base-plus", drive_subdir="phase_b_confirm"),
    # Build source_ds with RawBoost ON by default; the sweep loop will rebuild it
    # without RawBoost for any variant that sets use_rawboost=False.
    *datasets_cells(use_rawboost=True),
    md(
        "## 8b. Variant Definitions — FILL IN AFTER PHASE A",
        "",
        "Below are TODO placeholders. Edit `VARIANTS` based on:",
        "- **phase_a_norb** result: was no-RB better, worse, or neutral vs 11.67%?",
        "- **phase_a_leaveout** result: which group's VIB was hurting (if any)?",
        "",
        "**B1 templates** (pick whichever matches Phase A):",
        "- Phase A leave-one-out winner is `drop_vib_gK`: B1 sets `vib_disabled_groups=(K,)` plus `vib_betas` redistributing β to the other groups (e.g., (0, 0.5, 0.5) if K=0).",
        "- Phase A leave-one-out has no winner: B1 sets `vib_betas` to scale up the most informative group (TBD per probe results).",
        "",
        "**B2 templates** (pick whichever matches the Base-Plus winner):",
        "- Apply the same `vib_disabled_groups` to Large. Note Large has G=4 groups, so the index meanings are slightly different (Group K on Large covers ~6 layers vs ~4 on Base-Plus).",
        "- If `vib_betas` is the Base-Plus winner, extend it to length 4 by either uniform redistribution or by mapping the Base-Plus β profile onto the Large layer ranges.",
    ),
    code(
        "RAWBOOST_DEFAULT = True  # the source_ds built in section 8 has RB on; spec.use_rawboost overrides per-variant",
        "RESULTS_FILENAME = 'phase_b_confirm_results.json'",
        "",
        "# ============================================================================",
        "# TODO: edit these configs based on Phase A results before running.",
        "# Each commented block shows a likely config for one Phase A outcome.",
        "# ============================================================================",
        "",
        "VARIANTS = {",
        "    # --- B1: Base-Plus refinement ---",
        "    # Example A: Phase A says drop_vib_g0 wins -> disable group 0 + scale up groups 1,2:",
        "    # 'B1_bp_refined': {",
        "    #     'model_class': 'PerGroupVIBModel',",
        "    #     'model_kwargs': {",
        "    #         'vib_disabled_groups': (0,),",
        "    #         'vib_betas': (0.0, 0.5, 0.5),  # set group 0 to zero, redistribute",
        "    #     },",
        "    #     'use_rawboost': True,  # OR False if phase_a_norb said RB hurts",
        "    # },",
        "",
        "    # Example B: Phase A says no leave-one-out winner -> just per-group beta sweep",
        "    # 'B1_bp_refined': {",
        "    #     'model_class': 'PerGroupVIBModel',",
        "    #     'model_kwargs': {",
        "    #         'vib_betas': (0.5, 1.0, 0.5),  # weight middle group highest",
        "    #     },",
        "    #     'use_rawboost': True,",
        "    # },",
        "",
        "    # --- B2: WavLM-Large transfer ---",
        "    # Example: transfer Base-Plus drop_vib_g0 to Large with G=4",
        "    # 'B2_large_transfer': {",
        "    #     'model_class': 'PerGroupVIBModel',",
        "    #     'backbone': 'wavlm-large',",
        "    #     'model_kwargs': {",
        "    #         'vib_disabled_groups': (0,),  # also drop group 0 on Large",
        "    #     },",
        "    #     'use_rawboost': True,  # match Base-Plus winner",
        "    # },",
        "}",
        "",
        "if not VARIANTS:",
        "    raise RuntimeError(",
        "        'VARIANTS is empty. Uncomment one of the example configs above '",
        "        'and adjust based on Phase A results.'",
        "    )",
        "",
        "for name, spec in VARIANTS.items():",
        "    print(f'  {name:<22s} backbone={spec.get(\"backbone\", \"wavlm-base-plus\")} '",
        "          f'rawboost={spec.get(\"use_rawboost\", RAWBOOST_DEFAULT)} '",
        "          f'kwargs={spec[\"model_kwargs\"]}')",
    ),
    *variant_sweep_cells(),
    *reference_table_cells([
        "    'full_glada (BP)':         19.71,",
        "    'no_glada':                14.89,",
        "    'no_cka':                  12.98,",
        "    'per_group_vib (BP, RB)':  11.67,",
        "    'bb_wavlm-large_s42':      12.64,",
        "    'per_group_vib (Large)':   13.52,",
    ]),
]


def build():
    write_notebook(CELLS, str(NOTEBOOK_PATH))


if __name__ == "__main__":
    build()
