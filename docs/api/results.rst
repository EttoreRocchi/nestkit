.. _api-results:

=======
Results
=======

Result containers that store the outputs of a nested cross-validation run,
including per-fold metrics, predictions, and fitted models.

Classifier results
------------------

.. autoclass:: nestkit.results.ClassifierOuterFoldResult
   :members:
   :show-inheritance:

.. autoclass:: nestkit.ClassifierResults
   :members:
   :show-inheritance:

Regressor results
-----------------

.. autoclass:: nestkit.results.RegressorOuterFoldResult
   :members:
   :show-inheritance:

.. autoclass:: nestkit.RegressorResults
   :members:
   :show-inheritance:
