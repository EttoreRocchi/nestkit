.. _api-thresholding:

============
Thresholding
============

Utilities for optimising classification decision thresholds, including
multiple built-in criteria and strategies for computing thresholds across
folds.

Result container
----------------

.. autoclass:: nestkit.thresholding.results.ThresholdResult
   :members:
   :show-inheritance:

Criteria functions
------------------

Pre-built objective functions that can be passed to the threshold optimiser.

.. autofunction:: nestkit.thresholding.criteria.youden_j

.. autofunction:: nestkit.thresholding.criteria.f_beta_criterion

.. autofunction:: nestkit.thresholding.criteria.cost_sensitive

.. autofunction:: nestkit.thresholding.criteria.balanced_accuracy_criterion

.. autofunction:: nestkit.thresholding.criteria.precision_at_recall

