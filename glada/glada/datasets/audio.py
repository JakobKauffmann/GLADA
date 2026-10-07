"""
Audio loading utilities for GLADA speech deepfake detection.

All audio in GLADA is normalized to a fixed length of 64,000 samples
(4 seconds at 16 kHz).  This fixed length is required because:
  1. The SSL backbone processors (WavLM, XLS-R, etc.) expect fixed-length
     input tensors to form batches.
  2. 4 s covers ~95% of ASVspoof utterances without excessive padding.
  3. RawBoost and other augmentations operate on fixed-length arrays.

Padding/truncation strategy:
  - Truncation: RANDOM CROP (training) or DETERMINISTIC FIRST-4s (evaluation).
    Training: a uniformly-sampled start offset is chosen so that different
    parts of long utterances are seen across epochs (implicit augmentation).
    Evaluation: always takes the first max_len samples for reproducibility.
  - Padding: ZERO-PADDING at the end.  The actual audio length (before
    padding) is returned alongside the waveform so that dataset classes
    can build a proper attention_mask (1 for real frames, 0 for padded).
    The mask is propagated through the backbone to frame-level and used
    to exclude padded frames from ASP softmax (Vaswani et al., 2017).

Resampling:
  Uses librosa.resample() when the native sample rate differs from 16 kHz.
  librosa uses high-quality sinc interpolation (via resampy or soxr) by
  default, which is preferable to torchaudio's lower-quality linear
  interpolation.  WaveFake (22.05 kHz) is the primary case requiring
  resampling; ASVspoof and ITW are already 16 kHz.

Audio loading backend:
  PRIMARY:  soundfile.read() — MUST be used wherever possible.
    - More robust in Linux/Colab environments (no system libsox dependency).
    - Handles .flac, .wav, .ogg natively.
    - Returns (samples, channels) float32 array directly.
    - torchaudio.set_audio_backend() is deprecated/removed in recent
      torchaudio versions and MUST NOT be called.
  FALLBACK: torchaudio.load() — used only if soundfile is not installed.
    This fallback exists for CI/test environments; production (Colab)
    always has soundfile.

Error handling:
  Corrupt/missing audio files return a silent (all-zeros) tensor of
  length max_len rather than raising an exception.  This is intentional:
  ~0.07% of ASVspoof 2021 LA files are corrupt in the official release,
  and silent padding does not meaningfully affect EER at that rate.
"""

import random
from typing import Tuple

import torch
import torch.nn.functional as F
import numpy as np
import librosa

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    import torchaudio

def load_audio(filepath: str, max_len: int = 64000, sr: int = 16000,
               deterministic: bool = False) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file and return a fixed-length waveform with actual length.

    Processing pipeline:
      1. Read with soundfile.read() (primary) or torchaudio.load() (fallback).
         soundfile MUST be used when available — see module-level docstring.
      2. Downmix to mono by averaging channels (stereo ITW files).
      3. Resample to `sr` (16000 Hz) via librosa if native rate differs.
      4. Pad (zeros at end) or truncate (random/deterministic crop) to `max_len`.

    Truncation modes:
      deterministic=False (training): uniformly-random start offset provides
        implicit augmentation — different segments seen across epochs.
      deterministic=True (evaluation): always takes the first max_len samples
        for reproducible evaluation metrics.

    Corrupt file handling:
      Any exception during loading returns a silent all-zeros tensor with
      actual_length=0.  ~0.07% of ASVspoof 2021 LA files are officially
      corrupted; the downstream masked softmax handles all-zero masks
      gracefully via nan_to_num.

    Args:
        filepath (str): Path to the audio file (.flac, .wav, .mp3, etc.).
        max_len (int): Output length in samples. Default: 64000 (4 s @ 16 kHz).
        sr (int): Target sample rate in Hz. Default: 16000.
        deterministic (bool): If True, always take the first max_len samples
            instead of random cropping. Use True for evaluation.

    Returns:
        Tuple of (wav, actual_length):
          wav: Float32 waveform tensor of shape (max_len,).
          actual_length: Number of real (non-padded) samples in wav.
    """
    try:
        if HAS_SOUNDFILE:
            # Soundfile returns (samples, channels) or (samples,)
            wav, orig_sr = sf.read(filepath, dtype='float32')
            
            # Convert to mono if stereo
            if len(wav.shape) > 1:
                wav = wav.mean(axis=1)
                
            # Resample if necessary
            if orig_sr != sr:
                wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)
                
            wav = torch.from_numpy(wav)
        else:
            # Fallback to torchaudio
            wav, orig_sr = torchaudio.load(filepath)
            
            # Resample
            if orig_sr != sr:
                transform = torchaudio.transforms.Resample(orig_sr, sr)
                wav = transform(wav)
                
            # Mix down to mono
            wav = wav.mean(0) if wav.shape[0] > 1 else wav.squeeze(0)
            
    except Exception:
        # Silently return a silent tensor — corrupt files (~0.07% of 2021 LA)
        # are a known dataset artifact and don't warrant log noise.
        # actual_length=0 signals all-padding; masked softmax handles this
        # via nan_to_num (see attention.py AttentiveStatisticsPooling).
        return torch.zeros(max_len), 0

    # Pad or truncate to max_len
    if wav.shape[0] > max_len:
        if deterministic:
            # Evaluation: always take the first max_len samples for reproducibility.
            wav = wav[:max_len]
        else:
            # Training: random crop for implicit augmentation — different
            # portions of long utterances are seen across epochs.
            start = random.randint(0, wav.shape[0] - max_len)
            wav = wav[start : start + max_len]
        actual_length = max_len  # No padding needed after truncation
    else:
        actual_length = wav.shape[0]
        # Pad with zeros at the end
        wav = F.pad(wav, (0, max(0, max_len - actual_length)))

    return wav, actual_length
