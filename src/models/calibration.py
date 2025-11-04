"""
Calibration and confidence interval utilities.

Provides:
- Reliability curve (bin-wise accuracy vs confidence)
- Expected Calibration Error (ECE) and Brier score
- Bootstrap confidence intervals for multilabel metrics based on backtest results
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import logging

from .validation import AccuracyMetrics

logger = logging.getLogger(__name__)


def compute_reliability_curve(
    confidences: List[float],
    labels: List[int],
    n_bins: int = 10
) -> Dict[str, List[float]]:
    """Compute reliability curve statistics.
    
    Args:
        confidences: List of confidence scores (0-1)
        labels: List of binary outcomes (1 if correct, 0 otherwise)
        n_bins: Number of bins for calibration
    
    Returns:
        Dictionary with per-bin mean_confidence, accuracy, and counts
    """
    if not confidences or not labels or len(confidences) != len(labels):
        logger.warning("Empty or mismatched inputs for reliability curve")
        return {
            'bin_centers': [], 'mean_confidence': [], 'accuracy': [], 'count': []
        }
    conf = np.clip(np.asarray(confidences, dtype=float), 0.0, 1.0)
    y = np.asarray(labels, dtype=int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1

    mean_conf_per_bin: List[float] = []
    acc_per_bin: List[float] = []
    count_per_bin: List[int] = []
    centers: List[float] = []

    for b in range(n_bins):
        mask = bin_ids == b
        count = int(np.sum(mask))
        count_per_bin.append(count)
        if count == 0:
            mean_conf_per_bin.append(0.0)
            acc_per_bin.append(0.0)
        else:
            mean_conf_per_bin.append(float(np.mean(conf[mask])))
            acc_per_bin.append(float(np.mean(y[mask])))
        centers.append(float((bins[b] + bins[b + 1]) / 2.0))

    return {
        'bin_centers': centers,
        'mean_confidence': mean_conf_per_bin,
        'accuracy': acc_per_bin,
        'count': count_per_bin,
    }


def compute_ece(
    mean_confidence: List[float],
    accuracy: List[float],
    count: List[int]
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    total = int(np.sum(count))
    if total == 0:
        return 0.0
    ece = 0.0
    for mc, acc, c in zip(mean_confidence, accuracy, count):
        ece += abs(acc - mc) * (c / total)
    return float(ece)


def compute_brier_score(confidences: List[float], labels: List[int]) -> float:
    """Compute Brier score for probabilistic predictions."""
    if not confidences or not labels or len(confidences) != len(labels):
        return 0.0
    conf = np.clip(np.asarray(confidences, dtype=float), 0.0, 1.0)
    y = np.asarray(labels, dtype=float)
    return float(np.mean((conf - y) ** 2))


def bootstrap_multilabel_confidence_intervals(
    predictions: List[List[str]],
    actuals: List[List[str]],
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """Compute bootstrap CIs for multilabel precision/recall/F1.
    
    Args:
        predictions: Per-sample predicted label lists
        actuals: Per-sample actual label lists
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level (e.g., 0.05 for 95% CI)
        random_state: RNG seed
    
    Returns:
        Dict with keys 'precision', 'recall', 'f1_score' each having 'low' and 'high'
    """
    rng = np.random.default_rng(random_state)
    n = len(predictions)
    if n == 0 or n != len(actuals):
        logger.warning("Cannot bootstrap CIs: empty or mismatched inputs")
        return {
            'precision': {'low': 0.0, 'high': 0.0},
            'recall': {'low': 0.0, 'high': 0.0},
            'f1_score': {'low': 0.0, 'high': 0.0},
        }

    prec_samples: List[float] = []
    rec_samples: List[float] = []
    f1_samples: List[float] = []

    idx = np.arange(n)
    for _ in range(n_bootstrap):
        sample_idx = rng.choice(idx, size=n, replace=True)
        p_sample = [predictions[i] for i in sample_idx]
        a_sample = [actuals[i] for i in sample_idx]
        metrics = AccuracyMetrics.calculate_multilabel_metrics(p_sample, a_sample)
        prec_samples.append(metrics['precision'])
        rec_samples.append(metrics['recall'])
        f1_samples.append(metrics['f1_score'])

    def quantiles(values: List[float]) -> Tuple[float, float]:
        low_q = float(np.quantile(values, alpha / 2.0))
        high_q = float(np.quantile(values, 1.0 - alpha / 2.0))
        return low_q, high_q

    p_low, p_high = quantiles(prec_samples)
    r_low, r_high = quantiles(rec_samples)
    f_low, f_high = quantiles(f1_samples)

    return {
        'precision': {'low': p_low, 'high': p_high},
        'recall': {'low': r_low, 'high': r_high},
        'f1_score': {'low': f_low, 'high': f_high},
    }


