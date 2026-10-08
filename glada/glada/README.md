# glada package source

| Path | Purpose |
|---|---|
| `__init__.py` | Exposes `GLADAConfig`, `BACKBONE_REGISTRY`, and `set_seed`. |
| `config.py` | `GLADAConfig`, the single dataclass holding every hyperparameter, ablation switch, and output path. `BACKBONE_REGISTRY` maps backbone names to HuggingFace IDs, hidden size, layer count, and number of layer groups. `set_seed` seeds Python, NumPy, and PyTorch. |
| [`models/`](models) | GLADA and per-group VIB models and their building blocks. |
| [`trainers/`](trainers) | Two-phase training loop, checkpointing, and sweep runners. |
| [`datasets/`](datasets) | Audio loading, dataset classes for all six corpora, and DataLoader factories. |
| [`augmentation/`](augmentation) | RawBoost, MUSAN, and RIR waveform augmentation. |
| [`optimizer/`](optimizer) | Layer-wise learning-rate decay and the phase 1 / phase 2 schedulers. |
| [`evaluation/`](evaluation) | EER metrics, multi-dataset evaluation, and checkpoint averaging. |
| [`analysis/`](analysis) | Post-hoc analysis of trained models: representation extraction, linear probes, CKA. |

Defaults in `GLADAConfig` that matter most: `vib_beta=0.01`,
`domain_loss_weight=0.1`, `speaker_loss_weight=0.1`, `cka_loss_weight=0.01`,
`effective_batch_size=384`. Each run's exact values are saved in its
`config.json` under `experiment_results/`.
