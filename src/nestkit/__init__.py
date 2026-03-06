"""nestkit  -  A rigorous nested cross-validation toolkit for scikit-learn.

Core estimators
---------------
- :class:`NestedCVClassifier`  -  Classification with optional calibration
  and threshold optimization.
- :class:`NestedCVRegressor`  -  Regression with optional residual-based
  prediction intervals.

Results
-------
- :class:`ClassifierResults` / :class:`RegressorResults`  -  Rich result
  containers with summary statistics, export methods, and plots.

Submodules
----------
- :mod:`nestkit.calibration`  -  Post-hoc probability calibration and
  calibration diagnostics.
- :mod:`nestkit.callbacks`  -  Fold-level monitoring (progress, logging,
  checkpointing).
- :mod:`nestkit.comparison`  -  Statistical model comparison.
- :mod:`nestkit.diagnostics`  -  Hyperparameter stability analysis.
- :mod:`nestkit.importance`  -  Cross-fold feature importance aggregation.
- :mod:`nestkit.inner`  -  Inner CV tuning reports.
- :mod:`nestkit.plotting`  -  25+ plotting functions for nested CV results.
- :mod:`nestkit.thresholding`  -  Decision-threshold optimization criteria.
"""

from __future__ import annotations

import logging

from nestkit.classifier import NestedCVClassifier
from nestkit.regressor import NestedCVRegressor
from nestkit.results.classifier_results import ClassifierResults
from nestkit.results.regressor_results import RegressorResults

__version__ = "0.1.0"

logger = logging.getLogger("nestkit")
logger.addHandler(logging.NullHandler())

__all__ = [
    "ClassifierResults",
    "NestedCVClassifier",
    "NestedCVRegressor",
    "RegressorResults",
    "__version__",
]
