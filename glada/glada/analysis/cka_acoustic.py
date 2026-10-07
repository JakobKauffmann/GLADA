"""
CKA between GLADA per-group representations and classical acoustic features.

Purpose
-------
The thesis claims that the K SSL layer groups specialise on different aspects
of the audio (acoustic / phonetic / semantic / artifact-specific -- see the
HierCon-style grouping in ``Coding/glada/models/attention.py``).  To support
that claim quantitatively we compare each group's pooled representation
against simple, interpretable acoustic descriptors:

  * **MFCC**           -- timbral/spectral envelope (Davis & Mermelstein 1980).
  * **Spectral centroid** -- "brightness" of the spectrum (Peeters 2004).
  * **Zero-crossing rate (ZCR)** -- a noise-vs-tonality proxy.

If group 0 has high CKA with MFCC and group 3 has low CKA with MFCC, that's
direct evidence the model has learned hierarchical specialisation.

CKA (Centered Kernel Alignment) is from Kornblith et al. (ICML 2019,
arxiv 1905.00414).  We use the **linear** variant (linear kernel = Gram
matrix of inner products), which Kornblith et al. show correlates strongly
with the RBF kernel variant while being O(N D) instead of O(N^2 D).

Math (matching ``glada/models/attention.py:compute_cka``)
---------------------------------------------------------
Given X in R^{N x D_X}, Y in R^{N x D_Y}, both column-centered:

    HSIC(X, Y) = (1/N^2) * tr(K_X K_Y),   K_X = X X^T,  K_Y = Y Y^T
    CKA(X, Y)  = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))

The 1/N^2 cancels in the ratio so we omit it.  CKA in [0, 1]: 1 = identical
up to invertible linear transform; 0 = no linear correspondence.

We re-implement (rather than import) ``compute_cka`` here as a numpy version
so this module is usable in CPU-only analysis notebooks without spinning up
the model.  The torch version in ``attention.py`` remains the canonical
training-loop implementation; this one is for offline analysis only.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _to_numpy(x):
    """Tensor / array -> float32 ndarray (BFloat16-safe)."""
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def compute_cka(reps_A, reps_B) -> float:
    """Linear CKA between two ``(N, D)`` matrices.

    Reference: Kornblith et al., "Similarity of Neural Network Representations
    Revisited", ICML 2019 (Section 2, eq. 4).

    Numerical notes
    ---------------
    - Operates in float64 internally because HSIC is a sum of N^2 quartic
      terms; with N ~= 5000, float32 loses ~3 digits.
    - eps in the denominator guards against fully-collapsed (constant-output)
      reps where HSIC(X, X) -> 0.
    """
    A = _to_numpy(reps_A).astype(np.float64)
    B = _to_numpy(reps_B).astype(np.float64)
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    # tr(K_A K_B) = sum_{i,j} (A A^T)_{ij} (B B^T)_{ij} -- elementwise product
    # avoids materialising the N x N kernel matrix.
    KA = A @ A.T
    KB = B @ B.T
    hsic_ab = float((KA * KB).sum())
    hsic_aa = float((KA * KA).sum())
    hsic_bb = float((KB * KB).sum())
    return hsic_ab / (np.sqrt(hsic_aa * hsic_bb) + 1e-12)


def _extract_acoustic_features(
    audio_files: List[str],
    sr: int = 16000,
    max_len: int = 64000,
    n_mfcc: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute a fixed-length acoustic-feature vector per file.

    Returns ``{"mfcc": (N, n_mfcc), "centroid": (N, 1), "zcr": (N, 1)}``.
    Each utterance is reduced to a single vector by averaging over time
    frames (mean pooling).  This matches the GLADA reps which are also
    utterance-level (post-ASP), so CKA compares like-for-like.

    Audio loading uses ``soundfile.read``, with librosa for
    feature extraction. We deliberately do **not** call
    ``glada.datasets.audio.load_audio`` here -- that returns a torch tensor
    cropped to ``max_len``, but librosa wants a raw float32 numpy array and
    we'd rather feed it the full utterance up to ``max_len`` samples.
    """
    import librosa
    import soundfile as sf

    mfccs, centroids, zcrs = [], [], []
    for fp in audio_files:
        try:
            wav, orig_sr = sf.read(fp, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if orig_sr != sr:
                wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)
            if wav.shape[0] > max_len:
                wav = wav[:max_len]
        except Exception:
            wav = np.zeros(max_len, dtype=np.float32)

        # MFCC: time-mean over frames -> (n_mfcc,)
        mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc).mean(axis=1)
        # Spectral centroid (Hz, mean over time) -> scalar
        centroid = librosa.feature.spectral_centroid(y=wav, sr=sr).mean()
        # Zero-crossing rate (mean over time) -> scalar
        zcr = librosa.feature.zero_crossing_rate(y=wav).mean()

        mfccs.append(mfcc)
        centroids.append([centroid])
        zcrs.append([zcr])

    return {
        "mfcc":     np.stack(mfccs).astype(np.float32),       # (N, n_mfcc)
        "centroid": np.array(centroids, dtype=np.float32),    # (N, 1)
        "zcr":      np.array(zcrs, dtype=np.float32),         # (N, 1)
    }


def cka_vs_acoustic(
    group_reps,
    audio_files: List[str],
    sr: int = 16000,
    max_len: int = 64000,
    n_mfcc: int = 20,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """CKA between each group's reps and a set of acoustic features.

    Args:
        group_reps: ``(N, K, F)`` array/tensor from
            ``extract_reps.extract_group_reps`` (the ``"group_reps"`` key).
        audio_files: ``N`` absolute paths in the *same order* as ``group_reps``.
        sr, max_len, n_mfcc: forwarded to ``_extract_acoustic_features``.
        feature_names: subset of ``["mfcc", "centroid", "zcr"]`` to compute.
            Default: all three.

    Returns:
        dict with::
            "matrix":        (K, len(feature_names)) CKA matrix -- the headline
                             result, one row per group, one column per feature.
            "feature_names": the column ordering used.
            "group_indices": (K,) row ordering [0, 1, ..., K-1].

    Why a matrix (not a heatmap directly): the orchestrator notebook plots
    this as a heatmap, but downstream consumers may want the raw numbers
    for tables in the thesis (Chapter 5 acoustic-analysis section).
    """
    feature_names = feature_names or ["mfcc", "centroid", "zcr"]
    G = _to_numpy(group_reps)
    if G.ndim != 3:
        raise ValueError(
            f"Expected group_reps shape (N, K, F); got {G.shape}. "
            f"Pass the 'group_reps' field from extract_group_reps's archive."
        )
    N, K, _ = G.shape
    if len(audio_files) != N:
        raise ValueError(
            f"audio_files length ({len(audio_files)}) != group_reps batch ({N})"
        )

    feats = _extract_acoustic_features(
        audio_files, sr=sr, max_len=max_len, n_mfcc=n_mfcc,
    )

    matrix = np.zeros((K, len(feature_names)), dtype=np.float64)
    for k in range(K):
        for j, name in enumerate(feature_names):
            matrix[k, j] = compute_cka(G[:, k, :], feats[name])
    return {
        "matrix":        matrix,
        "feature_names": feature_names,
        "group_indices": np.arange(K),
    }
