.. _api-plotting:

========
Plotting
========

Visualization functions for inspecting nested cross-validation results.
All functions return a Matplotlib ``Axes`` and can be imported directly
from :mod:`nestkit.plotting`.

Fold-level plots
----------------

.. autofunction:: nestkit.plotting.folds.plot_outer_scores

.. autofunction:: nestkit.plotting.folds.plot_inner_cv_heatmap

.. autofunction:: nestkit.plotting.folds.plot_score_stability

Inner tuning plots
------------------

.. autofunction:: nestkit.plotting.tuning.plot_inner_tuning_curve

.. autofunction:: nestkit.plotting.tuning.plot_param_selection

Calibration plots
-----------------

.. autofunction:: nestkit.plotting.calibration.plot_calibration_curves

.. autofunction:: nestkit.plotting.calibration.plot_calibration_improvement

Threshold plots
---------------

.. autofunction:: nestkit.plotting.threshold.plot_threshold_sensitivity

.. autofunction:: nestkit.plotting.threshold.plot_threshold_distribution

.. autofunction:: nestkit.plotting.threshold.plot_threshold_comparison

Comparison plots
----------------

.. autofunction:: nestkit.plotting.comparison.plot_comparison

.. autofunction:: nestkit.plotting.comparison.plot_score_differences

.. autofunction:: nestkit.plotting.comparison.plot_critical_difference

.. autofunction:: nestkit.plotting.comparison.plot_bayesian_posterior

Importance plots
----------------

.. autofunction:: nestkit.plotting.importance.plot_importance

.. autofunction:: nestkit.plotting.importance.plot_selection_frequency

.. autofunction:: nestkit.plotting.importance.plot_rank_stability_features

.. autofunction:: nestkit.plotting.importance.plot_shap_summary

Summary plots
-------------

.. autofunction:: nestkit.plotting.summary.plot_roc_curves

.. autofunction:: nestkit.plotting.summary.plot_precision_recall_curves

.. autofunction:: nestkit.plotting.summary.plot_confusion_matrices

.. autofunction:: nestkit.plotting.summary.plot_predicted_vs_actual

.. autofunction:: nestkit.plotting.summary.plot_residuals

.. autofunction:: nestkit.plotting.summary.plot_residual_qq

.. autofunction:: nestkit.plotting.summary.plot_prediction_intervals

.. autofunction:: nestkit.plotting.summary.plot_rank_stability
