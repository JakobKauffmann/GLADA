"""
glada/datasets — Dataset classes, audio I/O, and DataLoader factories.

All audio loading uses ``soundfile.read()`` (not ``torchaudio.load()``), which
avoids deprecated backend calls and is more reliable in the Colab environment.

Components:
  ``load_audio``                  — Robust waveform loader: reads with soundfile,
                                    resamples to 16 kHz, mono-mixes, and zero-pads
                                    or truncates to ``config.max_audio_len`` samples.
  ``ASVspoofDatasetWithSpeaker``  — ASVspoof 2019 LA training set with speaker labels
                                    (needed for the speaker adversary head).
  ``ASVspoofEvalDataset``         — ASVspoof 2019 LA dev/eval split (no speaker labels).
  ``ASVspoof2021Dataset``         — ASVspoof 2021 LA and DF evaluation datasets.
  ``InTheWildDataset``            — In-the-Wild (ITW) deepfake benchmark
                                    (Müller et al., ICASSP 2022).
  ``WaveFakeDataset``             — WaveFake multi-vocoder benchmark
                                    (Frank & Schönherr, NeurIPS 2021).
  ``CodecFakeDataset``            — CodecFake+ codec-based deepfake benchmark.
  ``create_balanced_sampler``     — Weighted random sampler that balances real/spoof
                                    within each epoch.
  ``create_dataloaders``          — Convenience factory: wraps datasets into
                                    DataLoaders with ``num_workers=0`` (Colab).

DataLoaders use ``num_workers=0`` by default (``config.num_workers``); fork-based
multiprocessing is unreliable inside the Colab sandbox.
"""

from glada.datasets.audio import load_audio
from glada.datasets.asvspoof import ASVspoofDatasetWithSpeaker, ASVspoofEvalDataset
from glada.datasets.asvspoof2021 import ASVspoof2021Dataset
from glada.datasets.itw import InTheWildDataset
from glada.datasets.wavefake import WaveFakeDataset
from glada.datasets.codecfake import CodecFakeDataset
from glada.datasets.loader import create_balanced_sampler, create_dataloaders

__all__ = [
    "load_audio",
    "ASVspoofDatasetWithSpeaker",
    "ASVspoofEvalDataset",
    "ASVspoof2021Dataset",
    "InTheWildDataset",
    "WaveFakeDataset",
    "CodecFakeDataset",
    "create_balanced_sampler",
    "create_dataloaders",
]
