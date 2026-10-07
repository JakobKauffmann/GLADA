"""
glada/evaluation — Metrics, multi-dataset evaluation, and checkpoint averaging.

Components:
  ``compute_eer``              — Equal Error Rate: the threshold at which FAR == FRR.
                                 Computed via linear interpolation on the DET curve.
                                 Lower EER = better detection performance.
  ``compute_eer_with_ci``      — EER with 95% bootstrap confidence intervals
                                 (n=1000 resamples).  Used for thesis results tables
                                 where statistical uncertainty must be reported.
  ``compute_accuracy_at_eer``  — Classification accuracy at the EER threshold;
                                 supplementary metric for sanity-checking.
  ``MultiDatasetEvaluator``    — Evaluates a trained model across multiple benchmarks
                                 (ASVspoof 2019 LA, ASVspoof 2021 LA/DF, In-the-Wild,
                                 WaveFake, CodecFake+) in a single pass, producing a
                                 results dict keyed by dataset name.
  ``average_checkpoints``      — Polyak-style weight averaging over the top-k
                                 checkpoints selected by validation EER
                                 (Polyak & Juditsky, 1992).  IB-CAAN (Huang et al.,
                                 2025) used top-5 averaging to achieve 4.61% ITW EER.
  ``save_averaged_checkpoint`` — Saves the averaged state dict with provenance
                                 metadata (which checkpoints were averaged and their
                                 individual EERs) for reproducibility.
"""

from glada.evaluation.metrics import compute_eer, compute_eer_with_ci, compute_accuracy_at_eer
from glada.evaluation.evaluator import MultiDatasetEvaluator
from glada.evaluation.checkpoint_avg import average_checkpoints, save_averaged_checkpoint

__all__ = [
    "compute_eer",
    "compute_eer_with_ci",
    "compute_accuracy_at_eer",
    "MultiDatasetEvaluator",
    "average_checkpoints",
    "save_averaged_checkpoint",
]
