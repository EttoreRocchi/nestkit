.. _api-results:

=======
Results
=======

Result containers that store the outputs of a nested cross-validation run,
including per-fold metrics, predictions, and fitted models.

Classifier results
------------------

.. autoclass:: nestkit.results.classifier_results.ClassifierOuterFoldResult
   :members:
   :show-inheritance:

.. autoclass:: nestkit.results.classifier_results.ClassifierResults
   :members:
   :show-inheritance:

Regressor results
-----------------

.. autoclass:: nestkit.results.regressor_results.RegressorOuterFoldResult
   :members:
   :show-inheritance:

.. autoclass:: nestkit.results.regressor_results.RegressorResults
   :members:
   :show-inheritance:
