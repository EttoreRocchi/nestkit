"""Result containers for nested cross-validation.

Provides :class:`ClassifierResults`, :class:`RegressorResults`, and
:class:`SurvivalResults` for storing, summarizing, and exporting
nested CV outcomes, including per-fold metrics, predictions,
confusion matrices, calibration diagnostics, prediction intervals,
and coefficient stability analysis.
"""

from nestkit.results._base import _BaseNestedCVResults
from nestkit.results.classifier_results import ClassifierOuterFoldResult, ClassifierResults
from nestkit.results.regressor_results import RegressorOuterFoldResult, RegressorResults
from nestkit.results.survival_results import SurvivalOuterFoldResult, SurvivalResults

__all__ = [
    "ClassifierOuterFoldResult",
    "ClassifierResults",
    "RegressorOuterFoldResult",
    "RegressorResults",
    "SurvivalOuterFoldResult",
    "SurvivalResults",
    "_BaseNestedCVResults",
]
