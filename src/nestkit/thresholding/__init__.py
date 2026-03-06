"""Decision-threshold optimization for binary classification.

Provides five built-in criteria (Youden's J, F-beta, cost-sensitive,
balanced accuracy, precision-at-recall) and two optimization strategies
(pooled and fold-specific) for selecting an optimal decision boundary
from out-of-fold predicted probabilities.
"""

from nestkit.thresholding.criteria import (
    balanced_accuracy_criterion,
    cost_sensitive,
    f_beta_criterion,
    precision_at_recall,
    youden_j,
)
from nestkit.thresholding.results import ThresholdResult
from nestkit.thresholding.strategies import compute_threshold_sensitivity, optimize_threshold

__all__ = [
    "ThresholdResult",
    "balanced_accuracy_criterion",
    "compute_threshold_sensitivity",
    "cost_sensitive",
    "f_beta_criterion",
    "optimize_threshold",
    "precision_at_recall",
    "youden_j",
]
