"""
glada/augmentation/rawboost.py — RawBoost waveform augmentation.

Reference:
  Tak et al., 'RawBoost: A Raw Data Boosting and Augmentation Method
  Applied to Automatic Speaker Verification Anti-Spoofing', ICASSP 2022.

RawBoost operates entirely in the raw waveform domain — no spectral features
are extracted.  Three complementary noise types model distinct real-world
degradation patterns:

  Type 1 — Linear and non-linear convolutive noise:
    Spectral shaping via a random filter bank (frequency domain), followed by
    multiplicative Gaussian noise (non-linear component).  Models room acoustics
    and transmission channel distortion.

  Type 2 — Impulsive signal-dependent additive noise:
    Sparse Bernoulli-sampled transient bursts whose amplitude scales with the
    local signal level.  Models codec artefacts and click noise.

  Type 3 — Stationary signal-independent additive noise:
    Gaussian white noise at a uniformly-sampled SNR in [15, 30] dB.  Models
    background noise and microphone self-noise.

In GLADA, one type is chosen uniformly at random per sample during training
when torch.rand(1) < GLADAConfig.rawboost_prob (default 0.5).  This forces
the SSL backbone to learn noise-invariant representations, complementing the
adversarial domain adaptation objective.

Note: All operations use numpy.  Audio must be loaded with soundfile.read()
before passing to RawBoost (never torchaudio.load()).
"""

import numpy as np


# =============================================================================
# RawBoost
# All methods are @staticmethod: pure functional transforms, no internal state.
# Input/output: 1-D float32 numpy array (waveform at 16 kHz, any length).
# =============================================================================
class RawBoost:
    """
    RawBoost: three types of raw waveform augmentation for anti-spoofing.

    Reference: Tak et al., ICASSP 2022.

    All three types accept a 1-D float32 numpy waveform and return a
    float32 numpy waveform of identical length.

    Typical usage in a Dataset.__getitem__::

        if self.augment and np.random.rand() < self.cfg.rawboost_prob:
            waveform = RawBoost.augment(waveform)
    """

    @staticmethod
    def _lnl_convolutive(
        x: np.ndarray,
        N_f: int = 5,
        nBands: int = 5,
        minF: float = 20,
        maxF: float = 8000,
        minBW: float = 100,
        maxBW: float = 1000,
        minG: float = -12,
        maxG: float = 12,
        sr: int = 16000,
    ) -> np.ndarray:
        """
        Type 1: Linear and non-linear convolutive noise.

        Constructs a random spectral envelope from nBands Gaussian notch filters
        in the frequency domain (linear component), then applies N_f rounds of
        multiplicative Gaussian noise (non-linear component).

        Args:
            x:      Input waveform (float32, 16 kHz).
            N_f:    Number of non-linear iterations.
            nBands: Number of spectral notch filters.
            minF/maxF: Centre frequency range (Hz).
            minBW/maxBW: Bandwidth range (Hz).
            minG/maxG: Gain range (dB).
            sr:     Sample rate (Hz).

        Returns:
            Augmented waveform (float32, same length as x).
        """
        length = len(x)
        freqs = np.random.uniform(minF, maxF, nBands)
        bws = np.random.uniform(minBW, maxBW, nBands)
        gains = np.random.uniform(minG, maxG, nBands)

        f_axis = np.fft.rfftfreq(length, d=1.0 / sr)
        H = np.ones(len(f_axis))
        for f0, bw, g in zip(freqs, bws, gains):
            # Gaussian notch centred at f0 with bandwidth bw; gain in dB
            notch = 1.0 - np.exp(-0.5 * ((f_axis - f0) / (bw / 2.0 + 1e-8)) ** 2)
            H *= 1.0 + (10 ** (g / 20.0) - 1.0) * (1.0 - notch)

        # Apply spectral envelope in frequency domain (linear component)
        X = np.fft.rfft(x)
        X_filtered = X * H
        y = np.fft.irfft(X_filtered, n=length)

        # Non-linear component: N_f rounds of multiplicative Gaussian noise
        for _ in range(N_f):
            y = y + 0.01 * y * np.random.randn(length)

        return y.astype(np.float32)

    @staticmethod
    def _impulsive_noise(
        x: np.ndarray,
        prob: float = 0.05,
        g_sd: float = 0.1,
    ) -> np.ndarray:
        """
        Type 2: Impulsive signal-dependent additive noise.

        Adds sparse (Bernoulli-sampled with probability `prob`) transient
        bursts whose amplitude scales with the local signal value, modelling
        codec artefacts and click noise.

        Args:
            x:    Input waveform (float32, 16 kHz).
            prob: Per-sample impulse probability.
            g_sd: Standard deviation of the impulse gain.

        Returns:
            Augmented waveform (float32, same length as x).
        """
        length = len(x)
        mask = np.random.binomial(1, prob, length)           # sparse {0,1} impulse locations
        noise = mask * x * np.random.normal(0, g_sd, length) # signal-dependent amplitude
        return (x + noise).astype(np.float32)

    @staticmethod
    def _stationary_noise(
        x: np.ndarray,
        snr_min: float = 15,
        snr_max: float = 30,
    ) -> np.ndarray:
        """
        Type 3: Stationary signal-independent additive noise.

        Adds Gaussian white noise at a uniformly-sampled SNR in
        [snr_min, snr_max] dB, modelling background noise and microphone
        self-noise in the 15–30 dB range typical of real recordings.

        Args:
            x:       Input waveform (float32, 16 kHz).
            snr_min: Minimum SNR (dB).
            snr_max: Maximum SNR (dB).

        Returns:
            Augmented waveform (float32, same length as x).
        """
        sig_power = np.mean(x ** 2) + 1e-10             # avoid divide-by-zero
        snr = np.random.uniform(snr_min, snr_max)
        noise_power = sig_power / (10 ** (snr / 10.0))  # convert dB to linear
        noise = np.random.normal(0, np.sqrt(noise_power), len(x))
        return (x + noise).astype(np.float32)

    @staticmethod
    def augment(x: np.ndarray) -> np.ndarray:
        """
        Apply a random RawBoost augmentation type (1, 2, or 3).

        Selects uniformly among the three noise types.  Called per-sample
        during DataLoader iteration when use_rawboost=True and a Bernoulli
        trial passes with probability GLADAConfig.rawboost_prob.

        Args:
            x: Input waveform (float32, 16 kHz).

        Returns:
            Augmented waveform (float32, same shape as x).
        """
        aug_type = np.random.randint(1, 4)  # uniform choice: 1, 2, or 3
        if aug_type == 1:
            return RawBoost._lnl_convolutive(x)
        elif aug_type == 2:
            return RawBoost._impulsive_noise(x)
        else:
            return RawBoost._stationary_noise(x)
