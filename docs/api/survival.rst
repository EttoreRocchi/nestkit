.. _api-survival:

=================
Survival Analysis
=================

Nested cross-validation for survival analysis using Cox proportional hazards
models from lifelines.  Requires the ``[survival]`` optional dependency
(``pip install nestkit[survival]``).

Core estimator
--------------

.. autoclass:: nestkit.NestedCVSurvival
   :members:
   :show-inheritance:
   :no-index:

CoxPH wrapper
-------------

.. autoclass:: nestkit.survival.CoxPHWrapper
   :members:
   :show-inheritance:

Target helpers
--------------

.. autofunction:: nestkit.survival.make_survival_target

Scoring
-------

.. autofunction:: nestkit.survival._scoring.concordance_index_scorer

.. autofunction:: nestkit.survival._scoring.uno_c_index_scorer
