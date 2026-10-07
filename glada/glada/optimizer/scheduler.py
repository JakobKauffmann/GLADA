"""
Learning rate scheduler builders for two-phase GLADA training.

Uses OneCycleLR (Smith & Topin, "Super-Convergence: Very Fast Training of Neural
Networks Using Large Learning Rates", 2019) with a critical fix for per-group LRs.

OneCycleLR schedule::

    Step 0 .. pct_start * total_steps  :  LR linearly ramps from ~0 to max_lr  (warmup)
    Step pct_start .. total_steps      :  LR follows cosine annealing back to ~0 (decay)

``total_steps`` must be the number of **optimizer steps**, not raw mini-batch steps.
With gradient accumulation, the relationship is::

    total_steps = (steps_per_epoch * num_epochs) // grad_accum_steps

Passing ``steps_per_epoch * num_epochs`` (raw batches) without dividing by
``grad_accum_steps`` would make the scheduler advance too quickly, exhausting the
cycle before training ends.

Phase 1 (frozen backbone, heads only):
  - Single LR group at head_lr (1e-3).
  - 10% warmup: heads start from random init; slow ramp prevents an early loss spike.
  - Purpose: stabilise classifier/discriminator before unfreezing the backbone.

Phase 2 (unfrozen backbone with LLRD):
  - Multiple LR groups with different peak LRs (assigned by build_llrd_optimizer).
  - 5% warmup: heads are already trained; backbone needs a gentler unfreeze ramp.
  - CRITICAL: pass a *list* of max_lr values, one per param group, to preserve the
    LLRD hierarchy.  A scalar max_lr would override all groups to the same peak LR,
    destroying the decay gradient (observed: EER spike from 14% to 33%).
"""

import torch
from glada.config import GLADAConfig

def build_phase1_scheduler(optimizer, config: GLADAConfig, steps_per_epoch: int):
    """
    Build OneCycleLR scheduler for Phase 1 (frozen backbone, heads only).

    In Phase 1 the backbone is frozen and there is exactly one parameter group
    (head parameters), so a scalar ``max_lr`` is safe and correct.

    The 10% warmup (``pct_start=0.1``) accounts for the fact that head parameters
    are randomly initialised: a too-rapid LR ramp causes an early loss spike before
    the classifier has learned a useful representation.

    Args:
        optimizer: AdamW optimizer containing only head parameter groups.
        config (GLADAConfig): Provides ``head_lr``, ``grad_accum_steps``,
            and ``phase1_epochs``.
        steps_per_epoch (int): Number of raw mini-batch steps per epoch
            (``len(dataloader)``).  This function divides by ``grad_accum_steps``
            internally to convert to optimizer steps.

    Returns:
        torch.optim.lr_scheduler.OneCycleLR
    """
    # Divide by grad_accum_steps: the optimizer only steps once every
    # grad_accum_steps mini-batches, so the scheduler must tick at that cadence.
    optimizer_steps_per_epoch = steps_per_epoch // config.grad_accum_steps
    total_steps = optimizer_steps_per_epoch * config.phase1_epochs

    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.head_lr,  # Scalar is fine here
        total_steps=max(total_steps, 1),
        pct_start=0.1,  # 10% warmup
    )


def build_phase2_scheduler(optimizer, config: GLADAConfig, steps_per_epoch: int):
    """
    Build OneCycleLR scheduler for Phase 2 (unfrozen backbone with LLRD).

    CRITICAL BUG FIX — per-group max_lr:
        ``OneCycleLR`` with a scalar ``max_lr`` silently overrides the learning rate
        of **all** parameter groups to the same peak value, destroying the LLRD
        hierarchy that ``build_llrd_optimizer`` carefully constructed.

        The original notebook used ``max_lr=config.head_lr`` (1e-3), forcing backbone
        layers — whose intended peak is ~1e-5 — up to 1e-3 (100x too high).  This
        caused immediate backbone destabilisation and model collapse
        (EER spike: 14% → 33% within the first phase-2 epoch).

        FIX: extract the current ``lr`` value from each param group (set by
        ``build_llrd_optimizer``) and pass the resulting list as ``max_lr``.
        PyTorch's OneCycleLR accepts a list and uses element ``i`` as the peak for
        group ``i``, preserving the full LLRD decay gradient.

    The 5% warmup (``pct_start=0.05``) is intentionally shorter than Phase 1
    because head parameters are already trained; the backbone just needs a brief
    gentle ramp to avoid gradient shock on unfreeze.

    Args:
        optimizer: AdamW optimizer built by ``build_llrd_optimizer`` (multiple
            LLRD parameter groups).
        config (GLADAConfig): Provides ``grad_accum_steps`` and ``phase2_epochs``.
        steps_per_epoch (int): Number of raw mini-batch steps per epoch.

    Returns:
        torch.optim.lr_scheduler.OneCycleLR
    """
    # See Phase 1 docstring for grad_accum_steps division rationale.
    optimizer_steps_per_epoch = steps_per_epoch // config.grad_accum_steps
    total_steps = optimizer_steps_per_epoch * config.phase2_epochs

    # Extract the LLRD-assigned peak LR for each group.
    # Using the current optimizer LR (not head_lr) preserves the per-layer decay.
    max_lrs = [pg['lr'] for pg in optimizer.param_groups]

    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,  # Critical fix: List of LRs
        total_steps=max(total_steps, 1),
        pct_start=0.05,  # 5% warmup (shorter because heads are already trained)
    )
