"""Evaluation metrics for probabilistic football predictions — uses numpy only."""

import numpy as np


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Overall Brier score (multi-class)."""
    n_samples, n_classes = y_prob.shape
    y_onehot = np.zeros_like(y_prob)
    y_onehot[np.arange(n_samples), y_true] = 1.0
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def brier_score_decomposition(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict:
    """Decompose Brier score into refinement, calibration, and uncertainty."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges, right=True) - 1

    refinement = 0.0
    calibration = 0.0
    uncertainty = 0.0

    overall_freq = float(y_true.mean())
    uncertainty = overall_freq * (1 - overall_freq)

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        bin_prob = float(y_prob[mask].mean())
        bin_freq = float(y_true[mask].mean())
        refinement += float((mask * (y_prob - bin_prob) ** 2).sum())
        calibration += n_bin * (bin_prob - bin_freq) ** 2

    refinement /= len(y_true)
    calibration /= len(y_true)

    return {
        "brier": refinement + calibration + uncertainty,
        "refinement": refinement,
        "calibration": calibration,
        "uncertainty": uncertainty,
    }


def ranked_probability_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Ranked Probability Score for ordered outcomes (H/D/A coded 0/1/2)."""
    n = len(y_true)
    rps = 0.0
    for i in range(n):
        cum_true = np.zeros(3)
        cum_prob = np.zeros(3)
        cum_true[: int(y_true[i]) + 1] = 1.0
        cum_prob = np.cumsum(y_prob[i])
        rps += float(np.sum((cum_prob - cum_true) ** 2))
    return rps / (2 * n)


def calibration_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple:
    """Return calibration curve (bin_probs, bin_freqs)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges, right=True) - 1
    bin_probs = []
    bin_freqs = []
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if mask.sum() > 0:
            bin_probs.append(float(y_prob[mask].mean()))
            bin_freqs.append(float(y_true[mask].mean()))
    return np.array(bin_probs), np.array(bin_freqs)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Expected calibration error (mass-weighted |pred - freq| over bins)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_prob, bin_edges, right=True) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(y_prob)
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        w = mask.sum() / n
        ece += w * abs(float(y_prob[mask].mean()) - float(y_true[mask].mean()))
    return float(ece)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """Compute all evaluation metrics."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    metrics = {}

    if y_prob.shape[1] > 2:
        metrics["rps"] = ranked_probability_score(y_true, y_prob)
        metrics["brier"] = brier_score(y_true, y_prob)

        for i, name in enumerate(class_names or ["H", "D", "A"]):
            y_bin = (y_true == i).astype(int)
            metrics[f"brier_{name}"] = float(brier_score_loss(y_bin, y_prob[:, i]))
            try:
                metrics[f"auc_{name}"] = float(roc_auc_score(y_bin, y_prob[:, i]))
            except ValueError:
                metrics[f"auc_{name}"] = 0.5
    else:
        metrics["brier"] = float(brier_score_loss(y_true, y_prob[:, 1]))
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        except ValueError:
            metrics["auc"] = 0.5

    metrics["log_loss"] = float(log_loss(y_true, y_prob))

    y_pred = y_prob.argmax(axis=1)
    metrics["accuracy"] = float((y_pred == y_true).mean())

    return metrics
