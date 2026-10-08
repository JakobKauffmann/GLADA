# experiment_results

Lightweight outputs of every training and evaluation run. Folder numbers
match [`../notebooks/`](../notebooks); each folder's README says which
notebook produced each subfolder. Checkpoints are not included.

Each training run is a folder named by start time (`YYYYMMDD_HHMMSS`) inside
a folder named after the configuration. Some configurations have several
timestamps because a Colab session was restarted.

| File | Contents |
|---|---|
| `config.json` | Every `GLADAConfig` value used for the run. |
| `history.json` | One entry per epoch: phase, training losses (spoof, domain, speaker, KL, CKA), evaluation EERs, GRL alphas, learning rates. |
| `eval_results.json` / `*_results.json` | Final or aggregated EERs written by the notebook. |

To find a run's best ITW EER, take the minimum ITW EER over the epochs in
`history.json`.
