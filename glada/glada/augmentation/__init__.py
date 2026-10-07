"""
glada/augmentation — Waveform augmentation for GLADA training.

Currently implemented:
  RawBoost — three-type raw waveform augmentation (Tak et al., ICASSP 2022).
  MusanAugmenter — additive noise from MUSAN corpus (Snyder et al., 2015).
  RIRAugmenter — room impulse response convolution (Ko et al., Interspeech 2017).
  CombinedAugmenter — chained augmentation pipeline for GLADA training.

Usage::

    from glada.augmentation import CombinedAugmenter
    augmenter = CombinedAugmenter(config)
    aug_waveform = augmenter.augment(waveform_np)
"""

from .rawboost import RawBoost
from .musan_rir import MusanAugmenter, RIRAugmenter, CombinedAugmenter

__all__ = ["RawBoost", "MusanAugmenter", "RIRAugmenter", "CombinedAugmenter"]
