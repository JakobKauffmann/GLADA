# 06 Analysis and figures

| Notebook | What it does | Result |
|---|---|---|
| `glada_acoustic_analysis.ipynb` | On the best per-group VIB checkpoint: linear probes per group (spoof, domain, codec), CKA against acoustic features, and t-SNE. | Codec probe accuracy falls with depth (g0 0.857, g1 0.774, g2 0.563) while spoof stays recoverable in every group. Supports putting the bottleneck on g0. |
| `colab_b_ablation_analysis.ipynb` | The phase 02 ablation notebook run in a parallel Colab session (its saved outputs include the WavLM-Large full GLADA run, 12.59%), followed by top-checkpoint averaging and the ablation tables. | `ablation/averaged_results.json`. |
| `glada_analysis.ipynb` | Training curves, backbone and ablation charts, multi-seed statistics, and cross-dataset heatmaps for the thesis. | `aggregate_figures/experiment_summary.json` and the thesis figures. |
| `ieee_figures.ipynb` | Regenerates the thesis and paper figures in IEEE format (vector PDF plus 600 DPI PNG). | Final figures. |

Results: [`experiment_results/06_analysis_and_figures`](../../experiment_results/06_analysis_and_figures).
