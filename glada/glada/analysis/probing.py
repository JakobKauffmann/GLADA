"""
Linear probing of GLADA per-group representations.

Purpose
-------
For the acoustic-feature analysis we ask: *what information does each group
g_k retain about (a) the spoof/bonafide label, (b) the recording domain,
(c) the codec / synthesis family?*  A standard tool for this is **linear
probing** -- train a linear classifier on top of frozen features and report
its accuracy.  If a probe trained on group k's features predicts domain at
~100% accuracy, the per-group GRL is failing to remove that nuisance signal.

The technique is standard in the SSL representation literature:
    - Alain & Bengio, ICLR-W 2017, "Understanding intermediate layers using
      linear classifier probes" (probe definition).
    - Kornblith et al., ICML 2019 (Section 4 uses linear probes alongside CKA).
    - Pasad et al., ASRU 2021, "Layer-wise analysis of a self-supervised
      speech representation model" (per-layer probing of wav2vec2 / HuBERT).

Probe choice
------------
Logistic regression with stratified k-fold CV.  We use sklearn rather than a
torch ``nn.Linear`` because:
  - Datasets here are small (~5k examples), so sklearn's L-BFGS solver is
    fast and avoids GPU/optimizer plumbing.
  - L2 penalty is well-tuned (Pasad et al. use C=1.0; we expose it as an arg).
  - Per-class precision/recall fall out for free via ``classification_report``.

Math (one-vs-rest logistic regression):
    For class c:
        P(y = c | x) = exp(w_c^T x + b_c) / sum_{c'} exp(w_{c'}^T x + b_{c'})
        L = -sum_i log P(y_i | x_i) + (1 / 2C) * sum_c ||w_c||^2

Cross-validation prevents leakage: each fold trains on (k-1)/k of the data
and reports accuracy on the held-out 1/k fold.  Mean +/- std across folds
is the headline number.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _to_numpy(x):
    """Permissive tensor->ndarray converter (BFloat16-safe).

    BFloat16 -> numpy is not supported directly (see MEMORY.md note on the
    evaluator's float() fix).  We cast to float32 first.
    """
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def train_linear_probe(
    reps,
    labels,
    num_classes: int,
    n_splits: int = 5,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> dict:
    """Train an L2-regularised logistic regression probe with stratified k-fold CV.

    Args:
        reps: ``(N, F)`` feature matrix (Tensor or ndarray).  Flattened first
            if more than 2 dims (e.g. ``(N, K, F)`` -> ``(N, K*F)`` -- but
            the per-group probe wrappers below pass each group's slice
            separately, so this code path is only hit for already-2D input).
        labels: ``(N,)`` integer class labels in ``[0, num_classes)``.
        num_classes: Number of classes; used to build per-class metric arrays.
        n_splits: Number of stratified CV folds (Pasad et al. use 5; default).
        C: Inverse L2 strength (sklearn convention).  Larger C = weaker
            regularisation.  1.0 is the sklearn default and Pasad et al.'s value.
        max_iter: L-BFGS iterations.  Bumped from sklearn's 100 because some
            high-dim group reps converge slowly otherwise.
        random_state: Seed for fold splitting (reproducibility).

    Returns:
        dict with keys::
            "accuracy_mean": float                 (mean across folds)
            "accuracy_std":  float                 (std across folds)
            "per_class_precision_mean": (C,) array (mean over folds, per class)
            "per_class_recall_mean":    (C,) array
            "per_fold_accuracy":        (n_splits,) array

    Why these metrics: accuracy alone is misleading under class imbalance
    (e.g. 90% bonafide -> always-bonafide probe scores 90%).  Per-class
    precision/recall surfaces probes that "cheat" by ignoring the minority
    class.
    """
    # Local imports keep this module loadable without sklearn at import time
    # (e.g. for static analysis), and isolate sklearn version differences.
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import precision_score, recall_score, accuracy_score

    X = _to_numpy(reps)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    y = _to_numpy(labels).astype(np.int64).ravel()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_acc, fold_prec, fold_rec = [], [], []
    classes = np.arange(num_classes)
    for train_idx, test_idx in skf.split(X, y):
        # multi_class='auto' picks 'multinomial' for >2 classes (softmax),
        # 'ovr' for binary -- both yield identical results in the binary case.
        clf = LogisticRegression(
            C=C, max_iter=max_iter, solver="lbfgs",
            multi_class="auto", random_state=random_state,
        )
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        fold_acc.append(accuracy_score(y[test_idx], y_pred))
        # zero_division=0 silences warnings for classes absent from a fold's
        # test set; the corresponding entry will be 0 and pulled down by averaging.
        fold_prec.append(precision_score(
            y[test_idx], y_pred, labels=classes, average=None, zero_division=0,
        ))
        fold_rec.append(recall_score(
            y[test_idx], y_pred, labels=classes, average=None, zero_division=0,
        ))

    return {
        "accuracy_mean":            float(np.mean(fold_acc)),
        "accuracy_std":             float(np.std(fold_acc)),
        "per_class_precision_mean": np.mean(fold_prec, axis=0),
        "per_class_recall_mean":    np.mean(fold_rec, axis=0),
        "per_fold_accuracy":        np.array(fold_acc),
    }


def _per_group_probe(
    group_reps_dict: Dict[int, "ndarray"],
    labels,
    num_classes: int,
    **probe_kwargs,
) -> Dict[int, dict]:
    """Run ``train_linear_probe`` separately for each group's reps.

    ``group_reps_dict`` maps group index k -> ``(N, F_k)`` feature matrix.
    Caller is responsible for slicing the ``(N, K, F)`` archive from
    ``extract_group_reps`` into one entry per group.  We do not concatenate
    across groups: the whole point is to localise where information lives.
    """
    return {
        k: train_linear_probe(reps, labels, num_classes=num_classes, **probe_kwargs)
        for k, reps in group_reps_dict.items()
    }


def domain_probe(
    group_reps_dict: Dict[int, "ndarray"],
    domain_labels,
    num_classes: Optional[int] = None,
    **probe_kwargs,
) -> Dict[int, dict]:
    """Probe each group's representations for *recording-domain* identity.

    Background: GLADA's per-group GRL is meant to *remove* domain information
    so the spoof classifier generalises across recording conditions
    (Ganin et al., JMLR 2016).  A successful per-group GRL drives this probe
    accuracy down toward chance (1 / num_classes).  Anything substantially
    above chance is leakage that the discriminator at that group failed to
    suppress.

    Args:
        group_reps_dict: ``{k: (N, F_k) array}`` -- per-group feature matrix.
        domain_labels: ``(N,)`` integer domain index (e.g. 0 = ASVspoof,
            1 = ITW, 2 = WaveFake, ...).
        num_classes: Inferred from labels if None.
        **probe_kwargs: Forwarded to ``train_linear_probe`` (n_splits, C, ...).

    Returns:
        ``{k: probe_result_dict}`` -- one entry per group.
    """
    y = _to_numpy(domain_labels).astype(np.int64).ravel()
    K = num_classes if num_classes is not None else int(y.max()) + 1
    return _per_group_probe(group_reps_dict, y, num_classes=K, **probe_kwargs)


def codec_probe(
    group_reps_dict: Dict[int, "ndarray"],
    codec_labels,
    num_classes: Optional[int] = None,
    **probe_kwargs,
) -> Dict[int, dict]:
    """Probe each group's representations for *codec / synthesis family* identity.

    Background: CodecFake+ (Wu et al., 2024) annotates each spoof with the
    codec or vocoder used.  Probing tells us which group(s) carry codec-
    specific artifacts, complementing the domain probe.  Earlier groups
    (acoustic) might leak codec; later groups (semantic) ideally should not.

    Same shape as ``domain_probe``; the split exists for clarity at call
    sites and to allow per-task probe hyperparameters in the future.
    """
    y = _to_numpy(codec_labels).astype(np.int64).ravel()
    K = num_classes if num_classes is not None else int(y.max()) + 1
    return _per_group_probe(group_reps_dict, y, num_classes=K, **probe_kwargs)
