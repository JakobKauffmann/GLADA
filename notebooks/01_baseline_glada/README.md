# 01 Baseline GLADA (February 2026)

Full GLADA (single post-pool VIB) on WavLM-Large, seed 42.

| Notebook | What it does | Result |
|---|---|---|
| `glada_experiment.ipynb` | Trains full GLADA end-to-end with the two-phase schedule. | Run `glada_wavlm-large_seed42/20260302_*`: 15.07% ITW EER. The model trains end-to-end, but ITW gets worse after the backbone is unfrozen, which set up phase 03. |
| `glada_diagnostics.ipynb` | Checks the implementation on finished runs: loss-term balance, GRL schedules, gradient flow into each group's discriminator, ablation configs match their intent, training stability, and grouped vs global domain adaptation. | Diagnostic plots and tables only; uses `glada_final_run` (34 epochs, 17.04%) and the phase 02 ablation runs. |

Results: [`experiment_results/01_baseline_glada`](../../experiment_results/01_baseline_glada).
