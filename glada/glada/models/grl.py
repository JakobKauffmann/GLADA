"""
Gradient Reversal Layer (GRL) for domain-adversarial training.

The GRL is the core mechanism enabling the minimax game between the feature extractor
and domain discriminator in DANN-style training:

  Feature extractor F:  min_F  L_task(F(x), y) - alpha * L_domain(D(F(x)), d)
  Domain discriminator D:  min_D  L_domain(D(F(x)), d)

In GLADA, separate GRL instances are inserted before each per-group domain discriminator
D_k, and before the speaker adversary head. The GRL is a parameter-free operation; its
only configurable quantity is alpha, the reversal strength.

GRL forward/backward:
  Forward:   GRL_alpha(x) = x                       (identity, no effect on activations)
  Backward:  dL/dx = -alpha * dL/d(GRL_alpha(x))   (negate and scale incoming gradient)

Alpha warmup schedule (linear, clamped at 1.0):
  alpha(t) = min(1, t / E_w)
  where t = global epoch index (NOT phase-relative),
        E_w = grl_warmup_epochs (default 5 for domain GRL, 10 for speaker GRL).

Rationale for warmup: at epoch 0 the discriminators are random. Immediately applying
full gradient reversal would push the feature extractor toward domains it cannot yet
distinguish, destabilizing the task loss. Warmup ramps pressure gradually so the
discriminators first become informative before the adversarial signal is trusted.

Note: GRL alpha is NOT reset at Phase 2. With default grl_warmup_epochs=5 and
phase1_epochs=5, domain alpha reaches 1.0 by Phase 2 start and stays there.
This is intentional — the discriminators are already informative by Phase 2,
and maintaining full reversal provides stable adversarial pressure during
backbone fine-tuning.

References:
  - Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation", ICML 2015.
  - Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016
    (Section 3.3 introduces the progressive schedule for alpha).
"""

import torch
import torch.nn as nn


class GradientReversal(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) implemented as a custom autograd Function.

    This is a stateless, parameter-free operation. Alpha is passed as a scalar
    argument at each forward call so that the reversal strength can be updated
    per-epoch without rebuilding the computation graph.

    Forward:   GRL_alpha(x) = x                      (identity)
    Backward:  dL/dx        = -alpha * dL/d(GRL(x))  (negate and scale)

    Usage:
        out = GradientReversal.apply(features, alpha)

    References:
      Ganin & Lempitsky (ICML 2015); Ganin et al. (JMLR 2016).
    """

    @staticmethod
    def forward(ctx, x, alpha):
        # Save alpha for use in backward; no tensor computation needed (identity).
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Flip and scale the gradient flowing back to the feature extractor.
        # The None return corresponds to the alpha argument, which has no gradient.
        # Negation causes the feature extractor to maximize D's loss (domain confusion).
        return grad_output.neg() * ctx.alpha, None
