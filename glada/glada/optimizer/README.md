# glada.optimizer

| File | Purpose |
|---|---|
| `llrd.py` | `build_llrd_optimizer`: AdamW with layer-wise learning-rate decay, so lower backbone layers move less than upper ones. |
| `scheduler.py` | `build_phase1_scheduler` and `build_phase2_scheduler`: OneCycle schedules for each training phase, with step counts corrected for gradient accumulation. |
| `__init__.py` | Re-exports the builders. |
