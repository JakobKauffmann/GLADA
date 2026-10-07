"""Variational Information Bottleneck (VIB) for domain-invariant compression.

This module sits between the hierarchical attention pooling output and the
task classifier / adversarial heads.  It compresses the pooled representation
into a stochastic bottleneck that retains task-relevant information (spoof
vs. bonafide) while discarding domain-specific noise (microphone, codec,
recording environment).

Theoretical foundation
----------------------
Information Bottleneck (IB) principle (Tishby et al., 2000):
    min_Z  -I(Z; Y) + beta * I(Z; X)

    where:
      I(Z; Y) = mutual information between bottleneck Z and task label Y
                (maximized -- keep task-relevant information)
      I(Z; X) = mutual information between bottleneck Z and input X
                (minimized -- compress away domain-specific noise)
      beta > 0 controls the compression-prediction trade-off

Variational bound (Alemi et al., "Deep Variational Information Bottleneck",
ICLR 2017):
    I(Z; X) <= E_{x~p(x)}[ KL( q_phi(z|x) || p(z) ) ]

    where q_phi(z|x) = N(mu_phi(x), diag(sigma^2_phi(x))) is the learned
    encoder and p(z) = N(0, I) is a standard Gaussian prior.

Reparameterization trick (Kingma & Welling, ICLR 2014):
    z = mu_phi(x) + sigma_phi(x) * eps,   eps ~ N(0, I)

    Enables backpropagation through the stochastic sampling step by
    separating the randomness (eps) from the learned parameters (mu, sigma).

KL divergence closed-form (chap3.tex Section 3.4):
    KL(q || p) = -1/2 * sum_j [ 1 + log(sigma_j^2) - mu_j^2 - sigma_j^2 ]

Encoder architecture (two-layer MLPs with GELU, matching chap3.tex):
    mu_phi(x)        = W_{mu2} * GELU(W_{mu1} * x + b_{mu1}) + b_{mu2}
    log sigma^2_phi(x) = W_{sigma2} * GELU(W_{sigma1} * x + b_{sigma1}) + b_{sigma2}

Stability measures:
    - Log-variance clamped to [-20, 20] to prevent std overflow/underflow.
      Without clamping, logvar can drift to 100+ during early training, causing
      std = exp(50) ~ 5e21, which makes z and all downstream losses explode.
    - KL loss clamping (max 10.0) is handled in the trainer loop, not here,
      to allow monitoring of raw vs. clamped values.

Benefits for domain adaptation:
    - Compressed Z strips domain-specific artifacts (microphone, codec, noise)
    - Only task-relevant features survive the bottleneck
    - Validated empirically by Huang et al. (IB-CAAN, 2025, arxiv 2509.23618):
      VIB stabilizes adversarial DA and prevents mode collapse in deepfake
      detection

References:
    - Tishby et al. (2000) -- Information Bottleneck principle
    - Kingma & Welling (ICLR 2014) -- VAE and reparameterization trick
    - Alemi et al. (ICLR 2017) -- Deep Variational Information Bottleneck
    - Huang et al. (2025, arxiv 2509.23618) -- IB-CAAN: VIB for deepfake DA
"""

import torch
import torch.nn as nn

class VIBBottleneck(nn.Module):
    """Variational Information Bottleneck (Alemi et al., ICLR 2017).

    Compresses the pooled representation x into a stochastic bottleneck z by
    learning a Gaussian encoder q_phi(z|x) = N(mu_phi(x), diag(sigma^2_phi(x)))
    and minimizing KL(q_phi(z|x) || p(z)) where p(z) = N(0, I).

    The two-layer MLP architecture for mu and log-variance heads follows
    chap3.tex Section 3.4:
        mu_phi(x)           = W_{mu2} * GELU(W_{mu1} * x + b_{mu1}) + b_{mu2}
        log sigma^2_phi(x)  = W_{sigma2} * GELU(W_{sigma1} * x + b_{sigma1}) + b_{sigma2}

    Why GELU (not ReLU):
        GELU provides smooth gradients near zero, which helps the log-variance
        head learn stable variance estimates.  ReLU's hard cutoff at zero can
        cause dead neurons in the variance path, leading to collapsed posteriors.

    Args:
        input_dim: Dimension of the pooled input (D, from HierarchicalAttentionPooling).
        bottleneck_dim: Dimension of the bottleneck z.
    """

    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        # mu_phi(x) = W_{mu2} * GELU(W_{mu1} * x + b_{mu1}) + b_{mu2}
        self.mu_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, bottleneck_dim),
        )
        # log sigma^2_phi(x) = W_{sigma2} * GELU(W_{sigma1} * x + b_{sigma1}) + b_{sigma2}
        self.logvar_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, bottleneck_dim),
        )

    def forward(self, x, training=True):
        """Encode input to stochastic bottleneck and compute KL loss.

        Args:
            x: (B, D) pooled features from HierarchicalAttentionPooling.
            training: If True, sample z via reparameterization trick.
                If False, use deterministic mean (mu) for reproducible inference.

        Returns:
            z: (B, bottleneck_dim) -- sampled (training) or deterministic (eval)
                bottleneck representation.
            kl_loss: Scalar -- KL(q_phi(z|x) || p(z)), averaged over the batch.
                Note: additional clamping of the total KL loss (max 10.0) is
                handled in the trainer loop to allow monitoring raw vs. clamped.
        """
        mu = self.mu_head(x)
        logvar = self.logvar_head(x)

        # Clamp log-variance to [-10, 2] (chap3.tex Section 3.4).
        # This limits std to [exp(-5), exp(1)] ~ [0.007, 2.7], keeping raw KL
        # in a gradient-informative range where the model can actively learn
        # to compress the posterior toward N(0,I).
        #
        # History of this clamp:
        #   [-20, 20]: std up to exp(10) ~ 22,026 → KL exploded to 133,598
        #   [-10, 4]:  std up to exp(2) ~ 7.4 → KL still reached 94-130 per batch,
        #              permanently triggering the per-batch KL clamp (raw > 10),
        #              making VIB gradient signal flat and uninformative.
        #   [-10, 2]:  std up to exp(1) ~ 2.7 → KL stays ~2-3, no per-batch
        #              clamping needed, VIB actively learns to compress.
        logvar = torch.clamp(logvar, min=-10.0, max=2.0)

        if training:
            # Reparameterization trick (Kingma & Welling, ICLR 2014):
            #   z = mu_phi(x) + sigma_phi(x) * eps,  eps ~ N(0, I)
            # Separates stochasticity (eps) from learned parameters (mu, sigma)
            # so gradients flow through mu and sigma via the chain rule.
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            # Deterministic inference: use posterior mean for reproducibility
            z = mu

        # KL(q_phi(z|x) || p(z)) where q = N(mu, diag(sigma^2)), p = N(0, I)
        #
        # Closed-form derivation (Kingma & Welling, ICLR 2014; chap3.tex Eq. 3.X):
        #   KL(q||p) = integral q(z) log(q(z)/p(z)) dz
        #            = 0.5 * [tr(Sigma) + mu^T mu - d - log|Sigma|]
        #   For diagonal Sigma = diag(sigma^2):
        #            = 0.5 * sum_j [sigma_j^2 + mu_j^2 - 1 - log(sigma_j^2)]
        #            = -0.5 * sum_j [1 + log(sigma_j^2) - mu_j^2 - sigma_j^2]
        #
        # torch.mean averages over both batch and feature dimensions for stable
        # gradient scaling regardless of bottleneck_dim or batch size.
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return z, kl_loss
