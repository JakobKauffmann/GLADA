"""
Evaluation metrics for audio deepfake detection.

Purpose
-------
Implements the primary evaluation metrics used in the ASVspoof challenge series
and the biometric presentation attack detection literature:

  1. Equal Error Rate (EER) — primary metric
  2. EER with Bootstrap Confidence Intervals — required for thesis defense
  3. Accuracy at the EER operating point — supplementary metric

Equal Error Rate (EER)
----------------------
EER is the threshold-independent operating point where the False Acceptance
Rate (FAR) equals the False Rejection Rate (FRR):

    FAR(tau) = P(score >= tau | spoof)    # fraction of spoofs accepted as bonafide
    FRR(tau) = P(score <  tau | bonafide) # fraction of bonafide rejected

    tau* = argmin |FAR(tau) - FRR(tau)|
    EER  = (FAR(tau*) + FRR(tau*)) / 2   [reported as percentage, 0-100%]

Lower EER is better; EER = 50% corresponds to random guessing.

EER is defined in ISO/IEC 30107-3 (biometric presentation attack detection)
and has been the standard metric for the ASVspoof challenge series since 2015
(Wu et al., 2015; Kinnunen et al., 2017; Yamagishi et al., 2021).

t-DCF (Tandem Detection Cost Function)
---------------------------------------
t-DCF (Kinnunen et al., 2020) is an alternative metric that accounts for the
cost of spoofing attacks on a downstream ASV system and models the tandem
countermeasure (CM) + ASV pipeline. It requires ASV system scores and is
therefore omitted here; we use EER as the primary metric following recent
SOTA papers (IB-CAAN: Huang et al., 2025; HierCon: arxiv 2602.01032).

Reference:
    Kinnunen et al. (2020), "t-DCF: a Detection Cost Function for the Tandem
    Assessment of Spoofing Countermeasures and Automatic Speaker Verification",
    IEEE/ACM TASLP 29:2495-2506.

Bootstrap Confidence Intervals
--------------------------------
Single-point EER estimates have high variance on small test sets. Bootstrap
resampling (Efron & Tibshirani, 1993) provides a distribution over EER values
by repeatedly resampling the test set with replacement:

    For b in 1..B:
        (labels_b, scores_b) = resample(labels, scores, n=N, replace=True)
        eer_b = compute_eer(labels_b, scores_b)
    CI = [percentile(eer_samples, alpha/2), percentile(eer_samples, 1-alpha/2)]

Standard in the ASVspoof community (B=1000, alpha=5% for 95% CI).
Required by Rule 3 (committee-ready quality) for the thesis defense.
"""

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER) as a percentage.

    Implementation uses sklearn's ``roc_curve`` to sweep decision thresholds,
    then finds the crossing point of FAR (= FPR) and FRR (= 1 - TPR):

        eer_idx = argmin |FPR(tau) - FNR(tau)|
        EER = (FPR(eer_idx) + FNR(eer_idx)) / 2 * 100

    The average of FPR and FNR at the crossing index (rather than just one of
    them) is used to reduce quantization error when the curves do not cross
    exactly — this is the convention in the ASVspoof scoring scripts.

    Args:
        labels: Ground truth integer labels. Convention: 0 = spoof, 1 = bonafide.
                Must contain at least one sample of each class.
        scores: Continuous model output scores. Higher score = more likely bonafide.
                Typically the bonafide softmax probability P(class=1 | x).

    Returns:
        float: EER as a percentage in [0, 100]. Returns 50.0 if only one class
               is present in labels (undefined case; treated as chance level).
    """
    if len(np.unique(labels)) < 2:
        # EER is undefined with only one class; return chance-level 50%
        return 50.0

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr  # FRR = 1 - TPR

    # Find the threshold index where FPR is closest to FNR (equal error point)
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2 * 100

    return eer


def compute_accuracy_at_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """
    Compute classification accuracy at the EER operating point.

    Finds the decision threshold tau* where FAR ≈ FRR (the EER point), then
    evaluates hard binary predictions at that threshold:

        preds = (scores >= tau*)
        accuracy = mean(preds == labels) * 100

    This is a supplementary metric. Unlike EER (which is threshold-independent),
    accuracy at EER is threshold-dependent and therefore less commonly used in
    the ASVspoof community — but it provides an intuitive performance measure
    for committee presentations.

    Args:
        labels: Ground truth labels (0 = spoof, 1 = bonafide).
        scores: Model bonafide probability scores.

    Returns:
        (accuracy_percent, threshold): Accuracy as a percentage and the EER
        decision threshold value. Returns (0.0, 0.0) if only one class present.
    """
    if len(np.unique(labels)) < 2:
        return 0.0, 0.0

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    threshold = thresholds[eer_idx]

    preds = (scores >= threshold).astype(int)
    acc = (preds == labels).mean() * 100

    return acc, threshold


def compute_eer_with_ci(labels: np.ndarray, scores: np.ndarray,
                        n_bootstrap: int = 1000, ci: int = 95) -> dict:
    """
    Compute EER with a percentile Bootstrap Confidence Interval.

    Single-point EER estimates have high variance on finite test sets and are
    insufficient for publication-quality comparisons. Bootstrap resampling
    (Efron & Tibshirani, 1993) provides a non-parametric CI without distributional
    assumptions:

        For b = 1 .. B:
            indices_b ~ Uniform({0, ..., N-1}) with replacement, |indices_b| = N
            eer_b = compute_eer(labels[indices_b], scores[indices_b])
        alpha = (100 - ci) / 2
        ci_low  = percentile(eer_samples, alpha)
        ci_high = percentile(eer_samples, 100 - alpha)

    The random seed is fixed at 42 for reproducibility — the same CI will be
    reported across runs as long as the test set does not change.

    Note: bootstrap resamples that happen to contain only one class are skipped
    (EER is undefined in that case). This rarely occurs on large test sets.

    Required by Rule 3 (committee-ready quality) and Rule 14 (documentation).

    Args:
        labels: Ground truth integer labels (0 = spoof, 1 = bonafide).
        scores: Continuous bonafide probability scores from the model.
        n_bootstrap: Number of bootstrap resamples. Default 1000 gives stable
                     CI estimates. Efron & Tibshirani recommend >= 1000.
        ci: Confidence interval width as a percentage (default 95 for 95% CI).

    Returns:
        dict with keys:
            "eer"     (float): Point estimate of EER as a percentage.
            "ci_low"  (float): Lower bound of the bootstrap CI.
            "ci_high" (float): Upper bound of the bootstrap CI.
            "auc"     (float): Area Under ROC Curve (0.5 = random, 1.0 = perfect).
            "acc"     (float): Accuracy at the EER threshold (%).

    References:
        Efron & Tibshirani (1993), "An Introduction to the Bootstrap", Chapman & Hall.
    """
    # 1. Compute point estimates
    eer = compute_eer(labels, scores)
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.5
    acc, _ = compute_accuracy_at_eer(labels, scores)

    # 2. Bootstrap resampling for CI
    eer_samples = []
    # Fixed seed ensures reproducible CI bounds across notebook re-runs
    rng = np.random.RandomState(42)

    for _ in range(n_bootstrap):
        # Sample N indices with replacement (standard bootstrap)
        indices = rng.choice(len(labels), len(labels), replace=True)
        b_labels = labels[indices]
        b_scores = scores[indices]

        # Skip resamples that contain only one class (EER undefined)
        if len(np.unique(b_labels)) < 2:
            continue

        b_eer = compute_eer(b_labels, b_scores)
        eer_samples.append(b_eer)

    # 3. Compute percentile CI
    if eer_samples:
        alpha = (100 - ci) / 2.0  # e.g., 2.5 for 95% CI
        ci_low = np.percentile(eer_samples, alpha)
        ci_high = np.percentile(eer_samples, 100 - alpha)
    else:
        # Degenerate case: all bootstrap resamples had only one class
        # (should not occur on real test sets with both classes present)
        ci_low = eer
        ci_high = eer

    return {
        "eer": eer,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "auc": auc,
        "acc": acc
    }
