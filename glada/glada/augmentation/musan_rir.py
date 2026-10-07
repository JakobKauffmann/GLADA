"""
glada/augmentation/musan_rir.py — MUSAN noise and RIR convolution augmentation.

References:
  MUSAN corpus:
    Snyder et al., "MUSAN: A Music, Speech, and Noise Corpus", arXiv 2015.
    Available at https://openslr.org/17/
    Three categories: noise (ambient/environmental), speech (babble), music.

  Room Impulse Response (RIR):
    Ko et al., "A Study on Data Augmentation of Reverberant Speech for
    Robust Speech Recognition", Interspeech 2017.
    Simulated RIRs available at https://openslr.org/28/

  Combined augmentation pipeline:
    Huang et al. (IB-CAAN, 2025) used RawBoost + MUSAN + RIR to achieve
    their best 4.61% ITW EER, demonstrating that diverse augmentation
    strategies synergize with adversarial domain adaptation.

  SNR ranges:
    Default per-category SNR ranges follow the speaker verification
    augmentation protocol from Snyder et al. (2015) and the VoxCeleb
    augmentation strategy (Chung et al., Interspeech 2020):
      noise:  [0, 15] dB  — environmental noise is typically loud
      speech: [13, 20] dB — babble at moderate levels
      music:  [5, 15] dB  — background music at moderate-to-loud levels

  Additive noise mixing at a target SNR:
    Given clean signal x and noise n, the mixing formula is:
      x_aug = x + alpha * n
    where alpha = sqrt(P_x / (P_n * 10^(SNR/10)))
    and P_x = (1/T) * sum(x^2), P_n = (1/T) * sum(n^2) are signal powers.
    This scales the noise to achieve the desired SNR before addition.

  RIR convolution:
    The reverberant signal is computed as:
      x_rev = x * h  (convolution)
    where h is the room impulse response.  We use scipy.signal.fftconvolve
    for O(N log N) efficiency (vs. O(N^2) for direct convolution).
    The output is truncated to the original signal length and peak-normalized
    to prevent clipping.

Implementation notes:
  - All audio loading uses soundfile.read() (NEVER torchaudio.load()).
  - All operations are on numpy float32 arrays (consistent with RawBoost).
  - Noise files are loaded lazily (on first access) to avoid memory explosion
    when MUSAN contains thousands of files.
  - Missing directories print a warning and disable that augmenter category,
    allowing partial MUSAN installations to work.
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from .rawboost import RawBoost


# =============================================================================
# MusanAugmenter
# Additive noise augmentation from the MUSAN corpus (Snyder et al., 2015).
# Three noise categories (noise, speech, music) with per-category SNR ranges.
# =============================================================================
class MusanAugmenter:
    """
    Additive noise augmentation using the MUSAN corpus.

    MUSAN provides three categories of noise, each modelling a different
    type of real-world acoustic degradation:

      noise  — environmental/ambient noise (air conditioning, traffic, wind).
               Mixed at [0, 15] dB SNR (louder) because environmental noise
               is typically the dominant interferer in real recordings.

      speech — babble noise from overlapping speakers.
               Mixed at [13, 20] dB SNR (quieter) because babble at lower
               SNRs would make the target speaker unintelligible and is
               unrealistic for anti-spoofing scenarios.

      music  — background music tracks.
               Mixed at [5, 15] dB SNR (moderate) because music is common
               in broadcast and social media deepfake scenarios.

    For audio deepfake detection, additive noise augmentation forces the
    SSL backbone to learn noise-invariant features.  This is especially
    important for cross-domain generalization (e.g., training on clean
    ASVspoof data, testing on noisy ITW data), where background noise is
    a major source of domain shift (Muller et al., 2022).

    Args:
        musan_path: Root directory of the MUSAN corpus.  Expected structure::

            musan_path/
                noise/   *.wav
                speech/  *.wav
                music/   *.wav

        snr_ranges: Per-category SNR ranges as {category: (min_snr, max_snr)}.
            Defaults follow Snyder et al. (2015) / Chung et al. (2020).
    """

    # Default SNR ranges per noise category (dB).
    # Source: Snyder et al. (2015), widely adopted in speaker verification.
    DEFAULT_SNR_RANGES: Dict[str, Tuple[float, float]] = {
        "noise":  (0.0, 15.0),
        "speech": (13.0, 20.0),
        "music":  (5.0, 15.0),
    }

    def __init__(
        self,
        musan_path: str,
        snr_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.musan_path = musan_path
        self.snr_ranges = snr_ranges or self.DEFAULT_SNR_RANGES

        # Scan for WAV files in each category subdirectory.
        # Categories with missing directories are disabled (not fatal) to
        # support partial MUSAN installations.
        self._file_lists: Dict[str, List[str]] = {}
        self._available_categories: List[str] = []

        for category in ["noise", "speech", "music"]:
            cat_dir = os.path.join(musan_path, category)
            if not os.path.isdir(cat_dir):
                warnings.warn(
                    f"MUSAN category directory not found: {cat_dir}. "
                    f"'{category}' augmentation will be disabled."
                )
                continue

            # Recursively find all .wav files under this category.
            wav_files = []
            for root, _dirs, files in os.walk(cat_dir):
                for f in files:
                    if f.lower().endswith(".wav"):
                        wav_files.append(os.path.join(root, f))

            if len(wav_files) == 0:
                warnings.warn(
                    f"No WAV files found in {cat_dir}. "
                    f"'{category}' augmentation will be disabled."
                )
                continue

            self._file_lists[category] = wav_files
            self._available_categories.append(category)

        if len(self._available_categories) == 0:
            warnings.warn(
                f"No valid MUSAN categories found at {musan_path}. "
                f"MusanAugmenter will be a no-op."
            )

        # Lazy-loaded noise cache: {filepath -> np.ndarray}.
        # We do NOT preload all files (MUSAN has ~1000 files totaling ~11 GB).
        # Instead, files are loaded on first access and cached in memory.
        # For very large corpora, consider an LRU cache — but for typical
        # MUSAN sizes (930 files), full caching uses ~1-2 GB which is fine.
        self._cache: Dict[str, np.ndarray] = {}

    @property
    def is_active(self) -> bool:
        """True if at least one noise category is available."""
        return len(self._available_categories) > 0

    def _load_noise(self, filepath: str) -> np.ndarray:
        """
        Load a noise file lazily (on first access).

        Uses soundfile.read() (NEVER torchaudio).
        Returns a 1-D float32 numpy array.  Multi-channel files are
        downmixed to mono by averaging channels.
        """
        if filepath not in self._cache:
            audio, sr = sf.read(filepath, dtype="float32")
            # Downmix to mono if multi-channel
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            # Handle empty files gracefully
            if len(audio) == 0:
                warnings.warn(f"Empty audio file: {filepath}")
                audio = np.zeros(1, dtype=np.float32)
            self._cache[filepath] = audio
        return self._cache[filepath]

    def _fit_noise_to_length(self, noise: np.ndarray, target_len: int) -> np.ndarray:
        """
        Pad (by looping) or truncate noise to match the target signal length.

        Looping (rather than zero-padding) preserves the statistical properties
        of the noise, which is important for maintaining a consistent SNR
        across the entire signal duration.
        """
        if len(noise) >= target_len:
            # Random offset crop to avoid always using the start of the file
            max_start = len(noise) - target_len
            start = np.random.randint(0, max_start + 1)
            return noise[start : start + target_len].copy()
        else:
            # Loop the noise to fill the target length
            repeats = (target_len // len(noise)) + 1
            looped = np.tile(noise, repeats)
            return looped[:target_len].copy()

    def augment(self, x: np.ndarray) -> np.ndarray:
        """
        Add noise from a random MUSAN category at a random SNR.

        The mixing formula scales the noise to achieve a target SNR:
          alpha = sqrt(P_x / (P_n * 10^(SNR/10)))
          x_aug = x + alpha * n

        where P_x, P_n are the mean squared powers of the clean signal
        and noise respectively.

        Args:
            x: Input waveform (float32, 1-D numpy array, 16 kHz).

        Returns:
            Augmented waveform (float32, same length as x).
            Returns x unchanged if no MUSAN categories are available.
        """
        if not self.is_active:
            return x

        # Pick a random noise category
        category = np.random.choice(self._available_categories)
        snr_min, snr_max = self.snr_ranges[category]

        # Pick a random noise file from that category
        noise_file = np.random.choice(self._file_lists[category])
        noise = self._load_noise(noise_file)
        noise = self._fit_noise_to_length(noise, len(x))

        # Compute target SNR (uniformly sampled in dB)
        target_snr = np.random.uniform(snr_min, snr_max)

        # Compute scaling factor for the noise.
        # SNR(dB) = 10 * log10(P_x / P_n_scaled)
        # => P_n_scaled = P_x / 10^(SNR/10)
        # => alpha = sqrt(P_x / (P_n * 10^(SNR/10)))
        signal_power = np.mean(x ** 2) + 1e-10    # avoid division by zero
        noise_power = np.mean(noise ** 2) + 1e-10  # avoid division by zero
        alpha = np.sqrt(signal_power / (noise_power * (10.0 ** (target_snr / 10.0))))

        x_aug = x + alpha * noise
        return x_aug.astype(np.float32)


# =============================================================================
# RIRAugmenter
# Room impulse response convolution (Ko et al., Interspeech 2017).
# Simulates reverberation by convolving the input with a randomly selected RIR.
# =============================================================================
class RIRAugmenter:
    """
    Room impulse response (RIR) convolution augmentation.

    Convolves the input waveform with a randomly selected room impulse
    response to simulate reverberation.  This models the acoustic channel
    variability encountered in real-world deployments where deepfakes may
    be played through speakers in rooms of varying sizes and materials.

    The reverberant signal is:
      x_rev(t) = (x * h)(t) = sum_{k=0}^{K-1} x(t-k) * h(k)

    where h is the RIR of length K.  We use scipy.signal.fftconvolve for
    O(N log N) computation via the overlap-add method, which is substantially
    faster than direct convolution for typical RIR lengths (K ~ 8000-16000
    samples at 16 kHz, corresponding to 0.5-1.0 s reverberation).

    After convolution, the output is:
      1. Truncated to the original signal length (removing the convolution tail).
      2. Peak-normalized to prevent clipping (max(|x_rev|) scaled to max(|x|)).

    Reference:
      Ko et al., "A Study on Data Augmentation of Reverberant Speech for
      Robust Speech Recognition", Interspeech 2017.

    Args:
        rir_path: Root directory containing RIR WAV files (searched recursively).
    """

    def __init__(self, rir_path: str):
        self.rir_path = rir_path
        self._rir_files: List[str] = []
        self._cache: Dict[str, np.ndarray] = {}

        if not os.path.isdir(rir_path):
            warnings.warn(
                f"RIR directory not found: {rir_path}. "
                f"RIRAugmenter will be a no-op."
            )
            return

        # Recursively scan for WAV files
        for root, _dirs, files in os.walk(rir_path):
            for f in files:
                if f.lower().endswith(".wav"):
                    self._rir_files.append(os.path.join(root, f))

        if len(self._rir_files) == 0:
            warnings.warn(
                f"No WAV files found in {rir_path}. "
                f"RIRAugmenter will be a no-op."
            )

    @property
    def is_active(self) -> bool:
        """True if at least one RIR file is available."""
        return len(self._rir_files) > 0

    def _load_rir(self, filepath: str) -> np.ndarray:
        """
        Load an RIR file lazily (on first access).

        Uses soundfile.read() (NEVER torchaudio).
        Returns a 1-D float32 numpy array, peak-normalized so that
        max(|h|) = 1.  Multi-channel RIRs are downmixed to mono.

        Very short RIRs (< 10 samples) are treated as degenerate and
        replaced with a unit impulse (i.e., no reverberation), since
        they would produce negligible acoustic effect.
        """
        if filepath not in self._cache:
            audio, sr = sf.read(filepath, dtype="float32")
            # Downmix to mono if multi-channel
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            # Handle degenerate RIRs (empty or extremely short)
            if len(audio) < 10:
                warnings.warn(
                    f"Degenerate RIR (length={len(audio)}): {filepath}. "
                    f"Replacing with unit impulse."
                )
                audio = np.array([1.0], dtype=np.float32)
            else:
                # Peak-normalize the RIR for consistent energy scaling
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = audio / peak
            self._cache[filepath] = audio
        return self._cache[filepath]

    def augment(self, x: np.ndarray) -> np.ndarray:
        """
        Convolve input with a randomly selected RIR.

        Pipeline:
          1. Select a random RIR h from the corpus.
          2. Convolve: x_rev = fftconvolve(x, h, mode='full').
          3. Truncate to original length: x_rev = x_rev[:len(x)].
          4. Peak-normalize to preserve the original signal's amplitude range:
             x_rev *= max(|x|) / max(|x_rev|)

        Args:
            x: Input waveform (float32, 1-D numpy array, 16 kHz).

        Returns:
            Reverberant waveform (float32, same length as x).
            Returns x unchanged if no RIR files are available.
        """
        if not self.is_active:
            return x

        rir_file = np.random.choice(self._rir_files)
        rir = self._load_rir(rir_file)

        # FFT-based convolution: O(N log N) via overlap-add.
        # mode='full' returns the full convolution of length len(x) + len(rir) - 1.
        x_rev = fftconvolve(x, rir, mode="full")

        # Truncate to original length (discard the reverb tail)
        x_rev = x_rev[: len(x)]

        # Peak-normalize to preserve original amplitude range.
        # This prevents clipping and ensures consistent volume levels
        # across augmented and non-augmented samples.
        orig_peak = np.max(np.abs(x)) + 1e-10
        rev_peak = np.max(np.abs(x_rev)) + 1e-10
        x_rev = x_rev * (orig_peak / rev_peak)

        return x_rev.astype(np.float32)


# =============================================================================
# CombinedAugmenter
# Chained augmentation pipeline: RawBoost -> MUSAN -> RIR.
# Each stage fires independently with a configurable probability.
#
# Motivation (Huang et al., IB-CAAN 2025):
#   Combining multiple augmentation strategies (RawBoost for codec/channel
#   artifacts, MUSAN for additive noise, RIR for reverberation) produces
#   more diverse training conditions than any single strategy alone.
#   This diversity forces the model to learn representations invariant to
#   a wider range of acoustic distortions, directly supporting the domain
#   adaptation objective of GLADA.
#
# Pipeline order: RawBoost -> MUSAN -> RIR
#   RawBoost first: it models codec/channel effects that occur before
#   environmental noise or room reverberation in the recording chain.
#   MUSAN second: additive noise occurs in the acoustic environment.
#   RIR last: reverberation affects the entire signal (clean + noise).
#   This order mirrors the physical signal chain: source -> channel ->
#   noise -> room acoustics.
# =============================================================================
class CombinedAugmenter:
    """
    Chained augmentation pipeline for GLADA training.

    Applies up to three augmentation stages in sequence, each with an
    independent Bernoulli probability of activation:

      1. RawBoost (Tak et al., ICASSP 2022) — codec/channel artifacts
      2. MUSAN (Snyder et al., 2015) — additive environmental noise
      3. RIR (Ko et al., Interspeech 2017) — room reverberation

    The pipeline order mirrors the physical recording chain:
      source -> channel effects (RawBoost) -> environmental noise (MUSAN)
      -> room acoustics (RIR)

    Each stage is independent: any combination (including none, or all three)
    can fire on a given sample.  This produces 2^3 = 8 possible augmentation
    states per sample, maximizing training diversity.

    Huang et al. (IB-CAAN, 2025) showed that combining RawBoost + MUSAN + RIR
    achieved their best 4.61% ITW EER, compared to 4.93% with RawBoost alone.

    Args:
        config: GLADAConfig object.  Uses the following fields:
            - use_rawboost (bool): Whether RawBoost is enabled.
            - rawboost_prob (float): Probability of applying RawBoost.
            - musan_path (str, optional): Path to MUSAN corpus root.
              If not set or empty, MUSAN augmentation is disabled.
            - musan_prob (float, optional): Probability of applying MUSAN.
              Defaults to 0.5 if not set on config.
            - rir_path (str, optional): Path to RIR corpus root.
              If not set or empty, RIR augmentation is disabled.
            - rir_prob (float, optional): Probability of applying RIR.
              Defaults to 0.5 if not set on config.
    """

    def __init__(self, config):
        self.config = config

        # --- Stage 1: RawBoost ---
        self.use_rawboost = getattr(config, "use_rawboost", True)
        self.rawboost_prob = getattr(config, "rawboost_prob", 0.5)

        # --- Stage 2: MUSAN noise ---
        musan_path = getattr(config, "musan_path", "")
        self.musan_prob = getattr(config, "musan_prob", 0.5)
        self.musan_augmenter: Optional[MusanAugmenter] = None
        if musan_path and os.path.isdir(musan_path):
            snr_ranges = getattr(config, "musan_snr_ranges", None)
            self.musan_augmenter = MusanAugmenter(musan_path, snr_ranges)
            if not self.musan_augmenter.is_active:
                self.musan_augmenter = None

        # --- Stage 3: RIR convolution ---
        rir_path = getattr(config, "rir_path", "")
        self.rir_prob = getattr(config, "rir_prob", 0.5)
        self.rir_augmenter: Optional[RIRAugmenter] = None
        if rir_path and os.path.isdir(rir_path):
            self.rir_augmenter = RIRAugmenter(rir_path)
            if not self.rir_augmenter.is_active:
                self.rir_augmenter = None

    def augment(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the chained augmentation pipeline.

        Each stage fires independently with its configured probability.
        Pipeline: RawBoost -> MUSAN -> RIR (mirrors physical signal chain).

        Args:
            x: Input waveform (float32, 1-D numpy array, 16 kHz).

        Returns:
            Augmented waveform (float32, same length as x).
        """
        # Stage 1: RawBoost — codec/channel artifacts
        # (Tak et al., ICASSP 2022)
        if self.use_rawboost and np.random.rand() < self.rawboost_prob:
            x = RawBoost.augment(x)

        # Stage 2: MUSAN additive noise — environmental degradation
        # (Snyder et al., 2015)
        if self.musan_augmenter is not None and np.random.rand() < self.musan_prob:
            x = self.musan_augmenter.augment(x)

        # Stage 3: RIR convolution — room reverberation
        # (Ko et al., Interspeech 2017)
        if self.rir_augmenter is not None and np.random.rand() < self.rir_prob:
            x = self.rir_augmenter.augment(x)

        return x

    def __repr__(self) -> str:
        stages = []
        if self.use_rawboost:
            stages.append(f"RawBoost(p={self.rawboost_prob})")
        if self.musan_augmenter is not None:
            n_files = sum(len(v) for v in self.musan_augmenter._file_lists.values())
            cats = ", ".join(self.musan_augmenter._available_categories)
            stages.append(f"MUSAN(p={self.musan_prob}, files={n_files}, cats=[{cats}])")
        if self.rir_augmenter is not None:
            stages.append(
                f"RIR(p={self.rir_prob}, files={len(self.rir_augmenter._rir_files)})"
            )
        if not stages:
            stages.append("(no augmentation active)")
        return f"CombinedAugmenter({' -> '.join(stages)})"
