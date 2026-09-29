.. _api-reference:

=============
API Reference
=============

This section provides detailed documentation for every public class and function
in **nestkit**.  Use the table below to jump to a specific component, or browse
the sub-pages organised by topic.

.. currentmodule:: nestkit

Core estimators
---------------

.. autosummary::
   :nosignatures:

   NestedCVClassifier
   NestedCVRegressor
   NestedCVSurvival

Result containers
-----------------

.. autosummary::
   :nosignatures:

   ClassifierResults
   RegressorResults
   SurvivalResults
   results.ClassifierOuterFoldResult
   results.RegressorOuterFoldResult
   results.SurvivalOuterFoldResult

Survival analysis
-----------------

.. autosummary::
   :nosignatures:

   survival.CoxPHWrapper
   survival.make_survival_target

Calibration
-----------

.. autosummary::
   :nosignatures:

   calibration.PostHocCalibrator
   calibration.CalibrationDiagnostics

Thresholding
-------------

.. autosummary::
   :nosignatures:

   thresholding.ThresholdResult

Conformal prediction
--------------------

.. autosummary::
   :nosignatures:

   conformal.MondrianClassifierConformal
   conformal.MondrianRegressorConformal
   conformal.ClassifierConformalResult
   conformal.RegressorConformalResult

Model comparison
----------------

.. autosummary::
   :nosignatures:

   comparison.NestedCVComparator

Diagnostics
-----------

.. autosummary::
   :nosignatures:

   diagnostics.HyperparameterStability

Feature importance
------------------

.. autosummary::
   :nosignatures:

   importance.FeatureImportanceAggregator

Inner CV
--------

.. autosummary::
   :nosignatures:

   inner.InnerCVReport

Callbacks
---------

.. autosummary::
   :nosignatures:

   callbacks.FoldCallback
   callbacks.ProgressCallback
   callbacks.CheckpointCallback
   callbacks.LoggingCallback

Sub-pages
---------

.. toctree::
   :maxdepth: 1

   core
   results
   survival
   calibration
   thresholding
   conformal
   comparison
   diagnostics
   importance
   inner
   callbacks
   plotting
