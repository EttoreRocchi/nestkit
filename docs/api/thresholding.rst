.. _api-thresholding:

============
Thresholding
============

Utilities for optimising classification decision thresholds, including
multiple built-in criteria and strategies for computing thresholds across
folds.

Result container
----------------

.. autoclass:: nestkit.thresholding.ThresholdResult
   :show-inheritance:
   :no-members:

Criteria functions
------------------

Pre-built objective functions that can be passed to the threshold optimiser.

.. autofunction:: nestkit.thresholding.youden_j

.. autofunction:: nestkit.thresholding.f_beta_criterion

.. autofunction:: nestkit.thresholding.cost_sensitive

.. autofunction:: nestkit.thresholding.balanced_accuracy_criterion

.. autofunction:: nestkit.thresholding.precision_at_recall

