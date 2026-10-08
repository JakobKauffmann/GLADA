# 02 Ablation and backbone comparison (March 2026)

All ablations use WavLM-Base-Plus, seed 42. Thesis numbers are full
evaluation-set EERs from `comprehensive_eval/results.json`.

| Notebook | What it does | Result |
|---|---|---|
| `glada_ablation.ipynb` | Ablation grid over the 11 configurations. Its saved outputs cover full GLADA, no per-group DA, backbone only, hierarchical only, and no DA. | Full GLADA was the worst configuration on ITW (19.71%); every single-component removal did better. |
| `glada_ablation_phase_2.ipynb` | Same notebook, second Colab session: the backbone runs (Wav2Vec2-Robust, XLS-R 300M, WavLM-Large seeds 123 and 2026). | WavLM-Large was the best backbone (12.64% at seed 42, 15.93% at seed 123); Wav2Vec2-Robust the worst (21.98%). |
| `glada_multiseed.ipynb` | Seed 456 for full GLADA, no per-group DA, and no CKA. | 19.92%, 19.78%, 16.15%. Full GLADA's poor ITW result held at a second seed. |
| `glada_sweep_experiment_diagnostics.ipynb` | Combined notebook: phase 2 learning-rate / layer-freezing sweep, a full run with the winner, evaluation with checkpoint averaging, and diagnostics. `glada_experiment` and `glada_diagnostics` in phase 01 are copies of its later sections. | Sweep runs `sweep_lr*` (in phase 03 results) and `sweep_results_A/B.json`. No setting stopped the phase 2 drift. |
| `glada_comprehensive_eval.ipynb` | Scores every ablation and backbone checkpoint on all six datasets with bootstrap CIs. | `comprehensive_eval/results.json`, the source of the thesis ablation and backbone tables. Best ablation on ITW: no CKA (12.98%). |
| `glada_comprehensive_eval_round2.ipynb` | Fills in model/dataset pairs that round 1 missed and merges them into the same `results.json`. | Completes the table. |
| `glada_eval_incremental.ipynb` | Adds one dataset at a time (2021 LA, 2021 DF, WaveFake, CodecFake+) as each became available. | Same `results.json`. |

Results: [`experiment_results/02_ablation_multibackbone`](../../experiment_results/02_ablation_multibackbone).
