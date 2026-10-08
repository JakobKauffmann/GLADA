# 03 Phase 2 diagnosis and per-group VIB (April to early May 2026)

Phases 01 and 02 showed ITW EER drifting upward once the backbone was
unfrozen. This phase looked for the cause and led to per-group VIB.

| Notebook | What it does | Result |
|---|---|---|
| `phase_a_mask_validation.ipynb` | One-epoch run with mask debugging on, to confirm padding masks reach the pooling layer correctly. | Masks verified; run `phase_a_mask_validation`. |
| `phase_b_sweep.ipynb` | Phase 2 diagnostic sweep on WavLM-Large: control, no adversarial ramp, per-group DA only, speaker adversary only. | `P2_diag_*` runs. Every configuration drifted upward in phase 2, which pointed to gradient shock when the backbone unfreezes rather than one component. |
| `phase_vib_stagger.ipynb` | Tests two fixes: one VIB per layer group (per-group VIB), and switching auxiliary losses on one at a time. Also runs per-group VIB on WavLM-Large. | Per-group VIB reached 11.67% (seed 42), against 19.71% for full GLADA with one VIB. Staggering alone: 12.06%. Both combined: 14.24%. WavLM-Large per-group VIB: 13.52%. |
| `phase_aug_validation.ipynb` | Retrains three configurations with RawBoost + MUSAN + RIR. | Full GLADA 14.18% (vs 19.71% without), no per-group DA 13.55% (vs 14.89%), WavLM-Large 12.03% (vs 12.64%). Values are from each run's `history.json`. |

The `fix_*` runs come from `utilities/april13_builders/phase2_fixes_sweep.py`.

Results: [`experiment_results/03_phase2_diagnosis`](../../experiment_results/03_phase2_diagnosis).
