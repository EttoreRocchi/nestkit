"""Result containers for nested cross-validation.

Provides :class:`ClassifierResults` and :class:`RegressorResults` for
storing, summarizing, and exporting nested CV outcomes, including
per-fold metrics, predictions, confusion matrices, calibration
diagnostics, and prediction intervals.
"""

from nestkit.results._base import _BaseNestedCVResults
from nestkit.results.classifier_results import ClassifierOuterFoldResult, ClassifierResults
from nestkit.results.regressor_results import RegressorOuterFoldResult, RegressorResults

__all__ = [
    "ClassifierOuterFoldResult",
    "ClassifierResults",
    "RegressorOuterFoldResult",
    "RegressorResults",
    "_BaseNestedCVResults",
]
