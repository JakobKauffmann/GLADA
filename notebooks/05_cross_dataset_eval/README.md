# 05 Cross-dataset evaluation (May 2026)

| Notebook | What it does | Result |
|---|---|---|
| `cross_dataset_eval_pgvib_part_a.ipynb` | Scores every phase 04 checkpoint on ASVspoof 2021 LA and DF. | `cross_dataset_eval_pgvib_results_part_a.csv` |
| `cross_dataset_eval_pgvib_part_b.ipynb` | Same checkpoints on WaveFake and CodecFake+. | `cross_dataset_eval_pgvib_results_part_b.csv` |
| `cross_dataset_eval_pgvib_aggregate.ipynb` | Merges both parts into mean ± std tables per condition and dataset. | `*_summary.csv` and `*_pivot.csv`. g0-only VIB had the lowest mean EER on all four held-out sets (2021 LA 27.94%, 2021 DF 43.89%, WaveFake 19.98%, CodecFake+ 36.46%). |
| `colab_a_eval.ipynb` | Earlier evaluation harness: scores the WavLM-Large GLADA checkpoint on every dataset with bootstrap CIs and compares against published systems. | `eval_wavlm-large_results.json` (ITW 11.88%, 95% CI 11.55 to 12.27) and `sota_comparison.json`. |

Results: [`experiment_results/05_cross_dataset_eval`](../../experiment_results/05_cross_dataset_eval).
