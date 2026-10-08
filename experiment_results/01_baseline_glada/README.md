# 01 Baseline GLADA results

| Folder | Produced by | Contents |
|---|---|---|
| `glada_wavlm-large_seed42/` | `notebooks/01_baseline_glada/glada_experiment.ipynb` (and the combined notebook in phase 02) | Full GLADA on WavLM-Large, seed 42. Repeated runs between Feb 17 and Apr 8 (Colab restarts and later re-runs); best-epoch ITW EER ranged from 11.0% to 15.4%. |
| `glada_final_run/` | Full-length (34 epoch) training run used by `glada_diagnostics.ipynb` | Best ITW EER 17.04%. `train_summary.json` summarises the run. |
