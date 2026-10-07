"""
Grouped Layer Domain Discriminators for GLADA.

Standard DANN (Ganin et al., 2016) places a single domain discriminator on the final
pooled representation. GLADA extends this to operate at each SSL layer group, enforcing
domain invariance at every level of the representation hierarchy simultaneously.

Architectural motivation:
  El-Kheir et al. (NAACL 2025) demonstrate that different SSL transformer layers encode
  qualitatively different information and exhibit distinct cross-domain shift patterns:
    - Early layers  (e.g. layers 0-5):  low-level acoustic features; domain shift
                                         driven by recording channel and microphone type.
    - Middle layers (e.g. layers 6-11): phonetic/prosodic structure; domain shift
                                         driven by speaker demographics and accent.
    - Late layers   (e.g. layers 18-23): high-level semantic / artifact cues; domain
                                          shift from deepfake synthesis method distribution.
  A single discriminator on the CLS or mean-pooled representation is a weighted average
  over all these shifts and cannot separately adapt each level. Per-group discriminators
  give each level its own adversarial signal.

Per-group discriminator forward pass (group k):
  d_hat_k = D_k(GRL_alpha(g_k))
           = W_{k,2} * ReLU(W_{k,1} * GRL_alpha(g_k) + b_{k,1}) + b_{k,2}
  where:
    g_k     in R^{2D}    -- ASP output for group k (mean+std concatenated, so 2D)
    W_{k,1} in R^{128 x 2D}
    W_{k,2} in R^{2 x 128}
    Dropout(0.1) is applied after ReLU for regularization.

GLADA domain loss (averaged across K groups):
  L_GLADA = (1/K) * sum_{k=1}^{K} CE(D_k(GRL_alpha(g_k)), d)
  where d in {0=source, 1=target} is the domain label.

All K discriminators are fully independent (no shared parameters). This is intentional:
tying weights would couple the adaptation signals across groups and partially defeat the
purpose of group-specific alignment.

Speaker adversary (separate module, referenced here for completeness):
  3-layer MLP: d_z -> 512 -> 256 -> N_spk, with GRL (warmup 10 epochs, slower than
  the domain GRL warmup of 5 epochs). Disentangles speaker identity from the VIB
  bottleneck representation z so that domain shift attributed to speaker demographics
  does not leak into the spoof detection head.

Precedent in other modalities:
  - MLADA (Fang et al., Neurocomputing 2021): multi-layer adversarial domain adaptation
    for CNN-based image classification; separate discriminators per CNN stage.
  - DAA (Ngo Trung et al., ACL-IJCNLP 2021 Findings): per-layer domain-specific
    adversarial adapters on BERT for cross-lingual named entity recognition.

References:
  - Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016.
  - El-Kheir et al., "Layer-Wise Analysis of SSL for Downstream Tasks", NAACL 2025.
  - Fang et al., "MLADA: Multi-Level Adversarial Domain Adaptation", Neurocomputing 2021.
  - Ngo Trung et al., "DAA: Domain-Adversarial Adapters for Cross-Lingual NER",
    ACL-IJCNLP 2021 Findings.
"""

import torch
import torch.nn as nn
from glada.models.grl import GradientReversal


class GroupedLayerDomainDiscriminators(nn.Module):
    """
    Per-group domain discriminators for GLADA.

    Instantiates K independent MLP discriminators, one per SSL layer group. Each
    discriminator takes the ASP-pooled output of its group (shape B x 2D, where 2D
    comes from concatenating mean and std over time), passes it through a GRL, and
    predicts a binary domain label (source=0, target=1).

    Per-group forward:
      d_hat_k = W_{k,2} * ReLU(W_{k,1} * GRL_alpha(g_k) + b_{k,1}) + b_{k,2}

    Total GLADA loss (computed externally in GLADATrainer):
      L_GLADA = (1/K) * sum_{k=1}^{K} CE(d_hat_k, d)

    Args:
        num_groups (int): Number of SSL layer groups K.
        input_dim  (int): Dimensionality of each group's ASP output (= 2 * hidden_size
                          because ASP concatenates mean and standard deviation).
        hidden_dim (int): Hidden size of each discriminator MLP (default 128).

    References: see module docstring.
    """

    def __init__(self, num_groups, input_dim, hidden_dim=128):
        super().__init__()
        # Each discriminator is built independently — no shared parameters across groups.
        # W_{k,1}: input_dim -> hidden_dim (128); W_{k,2}: hidden_dim -> 2 (binary domain).
        self.discriminators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),          # Regularize: domain boundary is noisy
                nn.Linear(hidden_dim, 2),  # Binary domain logits (source / target)
            )
            for _ in range(num_groups)
        ])

    def forward(self, group_outputs, grl_alpha):
        """
        Apply GRL + discriminator to each group's pooled features.

        Args:
            group_outputs (list[Tensor]): K tensors of shape (B, 2*D), one per group,
                produced by the hierarchical attention pooling module's per-group ASP.
            grl_alpha (float): Reversal strength alpha, shared across all K groups.
                Ramped from 0 to 1 over grl_warmup_epochs by GLADATrainer.

        Returns:
            list[Tensor]: K tensors of shape (B, 2), raw domain logits per group.
                Cross-entropy against domain labels is computed in GLADATrainer.
        """
        logits = []
        for disc, feat in zip(self.discriminators, group_outputs):
            # GRL: identity in forward; negates gradient in backward so the backbone
            # is pushed to produce features that confuse this discriminator.
            feat_rev = GradientReversal.apply(feat, grl_alpha)
            logits.append(disc(feat_rev))
        return logits
