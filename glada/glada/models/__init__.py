"""
glada/models — Neural network components for GLADA.

Architecture (forward pass)::

    Raw waveform (16 kHz, 4 s, 64000 samples)
      -> SSL Backbone (HuggingFace AutoModel, output_hidden_states=True)
         returns (num_layers+1) hidden states of shape [B, T, D]
      -> Per-layer LayerNorm (one nn.LayerNorm per hidden state)
      -> HierarchicalAttentionPooling
           Groups layers into num_groups consecutive chunks.
           Within each group: attentive statistics pooling (ASP) over time.
           Across groups: learned weighted combination.
           CKA diversity loss penalises redundant group representations.
         -> pooled embedding z_raw  [B, D]
      -> VIBBottleneck (or DeterministicProjection if use_vib=False)
           VIB: reparameterise z_raw -> mu, log_var -> z ~ N(mu, sigma)
           KL = 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)  (Alemi et al., 2017)
         -> z  [B, bottleneck_dim]
      -> TaskClassifier (binary: bonafide vs spoof)
      -> SpeakerAdversary via GRL (Meng et al., Interspeech 2021)
      -> GroupedLayerDomainDiscriminators via GRL (per-group; primary contribution)

Components:
  ``GradientReversal``                   — GRL that multiplies gradients by -lambda
                                           during backprop (Ganin et al., JMLR 2016).
  ``HierarchicalAttentionPooling``       — Grouped layer attention pooling with CKA
                                           diversity regularisation.
  ``VIBBottleneck``                      — Variational Information Bottleneck
                                           (Alemi et al., ICLR 2017).
  ``GroupedLayerDomainDiscriminators``   — Separate domain discriminators for each
                                           SSL layer group; first such use in audio
                                           deepfake detection (this work).
  ``GLADAModel``                         — Full end-to-end model assembling all
                                           components above.
"""

from glada.models.grl import GradientReversal
from glada.models.attention import HierarchicalAttentionPooling
from glada.models.vib import VIBBottleneck
from glada.models.discriminators import GroupedLayerDomainDiscriminators
from glada.models.glada_model import GLADAModel, PerGroupVIBModel

__all__ = [
    "GLADAModel",
    "PerGroupVIBModel",
    "GradientReversal",
    "HierarchicalAttentionPooling",
    "VIBBottleneck",
    "GroupedLayerDomainDiscriminators",
]
