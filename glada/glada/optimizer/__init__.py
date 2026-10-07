"""
glada/optimizer — Optimizers and LR schedulers for two-phase GLADA training.

Components:
  ``build_llrd_optimizer``    — AdamW with Layer-wise Learning Rate Decay (LLRD).
                                Lower backbone layers receive exponentially smaller
                                LRs to preserve pre-trained acoustic representations
                                (Clark et al., EMNLP 2020; Howard & Ruder, ACL 2018).
  ``build_phase1_scheduler``  — OneCycleLR for Phase 1 (frozen backbone, heads only).
                                Single LR group; 10% warmup.
  ``build_phase2_scheduler``  — OneCycleLR for Phase 2 (unfrozen backbone + LLRD).
                                Passes a per-group max_lr list to preserve the LLRD
                                hierarchy (scalar max_lr would destroy it).

Typical usage in the training notebook::

    # Phase 1
    opt1 = torch.optim.AdamW(model.head_parameters(), lr=config.head_lr)
    sch1 = build_phase1_scheduler(opt1, config, steps_per_epoch)

    # Phase 2 (after unfreezing backbone)
    opt2 = build_llrd_optimizer(model, config)
    sch2 = build_phase2_scheduler(opt2, config, steps_per_epoch)
"""

from glada.optimizer.llrd import build_llrd_optimizer
from glada.optimizer.scheduler import build_phase1_scheduler, build_phase2_scheduler

__all__ = [
    "build_llrd_optimizer",
    "build_phase1_scheduler",
    "build_phase2_scheduler",
]
