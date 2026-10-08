# glada.trainers

| File | Purpose |
|---|---|
| `base_trainer.py` | `BaseTrainer`: device and mixed-precision setup, gradient clipping, checkpoint saving and pruning, `history.json` / `config.json` logging. |
| `glada_trainer.py` | `GLADATrainer`: the two-phase loop. Phase 1 trains the heads with the backbone frozen; phase 2 unfreezes it with layer-wise LR decay. Interleaves ASVspoof (labelled) and ITW (unlabelled) batches, schedules GRL warm-ups and KL annealing, and combines all loss terms. |
| `sweep.py` | Generates and runs groups of configurations: the February backbone and group-count screens (`screen_A_*`, `screen_B_*`), the ablation grid, and the phase 2 diagnostic sweep (`P2_diag_*`). |
| `__init__.py` | Re-exports the trainers and sweep helpers. |
