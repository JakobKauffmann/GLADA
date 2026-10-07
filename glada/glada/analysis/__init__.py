"""
glada.analysis -- offline analysis tools for trained GLADA / PerGroupVIB models.

These modules are *not* used during training.  They consume saved checkpoints
and produce per-group representations, linear-probe results, and CKA-vs-
acoustic-feature matrices for the acoustic analysis section of the thesis
(Chapter 5) and the ICTAI paper.

Modules
-------
- ``extract_reps``  -- run a checkpoint forward pass over a list of audio files
                        and dump per-group pooled representations to disk.
- ``probing``       -- linear logistic-regression probes (Alain & Bengio,
                        ICLR-W 2017; Pasad et al., ASRU 2021) for label,
                        domain, and codec classification per group.
- ``cka_acoustic``  -- linear CKA (Kornblith et al., ICML 2019) between each
                        group's representations and classical acoustic
                        descriptors (MFCC, spectral centroid, ZCR).

These tools follow Rule 13 (logic in the package, not in notebooks) so the
orchestrator notebook ``glada_acoustic_analysis.ipynb`` is a thin import +
plot layer.
"""

from .extract_reps import extract_group_reps
from .probing import train_linear_probe, domain_probe, codec_probe
from .cka_acoustic import compute_cka, cka_vs_acoustic

__all__ = [
    "extract_group_reps",
    "train_linear_probe",
    "domain_probe",
    "codec_probe",
    "compute_cka",
    "cka_vs_acoustic",
]
