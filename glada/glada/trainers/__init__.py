"""
glada/trainers — Training loops, checkpointing, and sweep orchestration.

Components:
  ``BaseTrainer``              — Shared infrastructure: checkpoint save/load,
                                 training-history tracking, evaluation calls,
                                 and early stopping logic.
  ``GLADATrainer``             — Two-phase adversarial training loop for GLADA.
                                 Phase 1: frozen backbone, head warm-up.
                                 Phase 2: end-to-end fine-tuning with LLRD,
                                 GRL lambda scheduling, and AMP.
  ``run_sweep``                — Runs a list of GLADAConfigs sequentially,
                                 loading datasets once to avoid redundant I/O.
  ``run_two_phase_sweep``      — Like ``run_sweep`` but builds separate Phase 1
                                 and Phase 2 optimizers/schedulers per config.
  ``generate_ablation_configs``— Generates the standard ablation grid (Table 4
                                 in the thesis): no_vib, no_speaker, no_cka,
                                 no_rawboost, no_hier, no_glada, full_glada.
  ``generate_sweep_configs``   — Generates backbone comparison configs
                                 (wavlm-large, xlsr-300m, data2vec-large,
                                 wav2vec2-robust) for Table 3.
"""

from glada.trainers.base_trainer import BaseTrainer
from glada.trainers.glada_trainer import GLADATrainer
from glada.trainers.sweep import (
    run_sweep,
    run_two_phase_sweep,
    generate_ablation_configs,
    generate_sweep_configs,
)

__all__ = [
    "BaseTrainer",
    "GLADATrainer",
    "run_sweep",
    "run_two_phase_sweep",
    "generate_ablation_configs",
    "generate_sweep_configs",
]
