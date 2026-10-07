"""
GLADA: Grouped Layer Adversarial Domain Adaptation for Audio Deepfake Detection.

MS CS Thesis, SJSU — Jakob Kauffmann (Advisor: Dr. Faranak Abri, 2026).

GLADA trains an SSL backbone (WavLM, XLS-R, Data2Vec, Wav2Vec2) end-to-end with
three adversarial objectives applied simultaneously:

  1. Per-group domain discriminators (GRL) — one discriminator per SSL layer group,
     encouraging each group to produce domain-invariant representations.
     This is the primary architectural contribution of GLADA.
  2. Speaker adversary (GRL) — suppresses speaker identity leakage in the bottleneck.
  3. VIB bottleneck — information-theoretic regularisation that limits reliance on
     spurious cues (Alemi et al., ICLR 2017).

Package layout::

  config        — BACKBONE_REGISTRY, GLADAConfig, set_seed
  augmentation  — RawBoost (Tak et al., 2022), MUSAN (Snyder et al., 2015),
                   RIR (Ko et al., 2017), CombinedAugmenter pipeline
  datasets      — ASVspoof 2019 LA, In-the-Wild, WaveFake, CodecFake+, ASVspoof 2021
  backbones     — Namespace for SSL backbone utilities (models loaded via HuggingFace)
  models        — GLADAModel, GRL, VIB, HierarchicalAttentionPooling, discriminators
  optimizer     — LLRD optimizer (Clark et al., 2020) and OneCycleLR schedulers
  trainers      — Two-phase training loop, checkpointing, ablation sweep runner
  evaluation    — EER, bootstrap CI, multi-dataset evaluator, checkpoint averaging

Notebooks in ``Coding/notebooks/`` are thin orchestration layers that import from
this package — model and training logic must never be inlined in notebooks (Rule 13).
"""

from .config import BACKBONE_REGISTRY, GLADAConfig, set_seed

__all__ = ["BACKBONE_REGISTRY", "GLADAConfig", "set_seed"]
