# glada.evaluation

| File | Purpose |
|---|---|
| `metrics.py` | `compute_eer`, `compute_accuracy_at_eer`, and `compute_eer_with_ci` (bootstrap 95% confidence interval). |
| `evaluator.py` | `MultiDatasetEvaluator`: scores one checkpoint on several datasets and returns EER, AUC, accuracy, and CIs per dataset. |
| `checkpoint_avg.py` | `average_checkpoints` and `save_averaged_checkpoint`: averages the weights of the top-k checkpoints of a run. |
| `__init__.py` | Re-exports the functions above. |
