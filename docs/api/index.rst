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

Result containers
-----------------

.. autosummary::
   :nosignatures:

   ClassifierResults
   RegressorResults
   results.ClassifierOuterFoldResult
   results.RegressorOuterFoldResult

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
   calibration
   thresholding
   comparison
   diagnostics
   importance
   inner
   callbacks
   plotting
