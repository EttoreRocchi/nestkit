"""Plotting functions for nested CV results.

All functions require matplotlib and seaborn (install with
``pip install nestkit[plotting]``). Each function accepts an optional
``ax`` parameter; if ``None``, a new figure is created automatically.
"""

from nestkit.plotting.calibration import plot_calibration_curves, plot_calibration_improvement
from nestkit.plotting.comparison import (
    plot_bayesian_posterior,
    plot_comparison,
    plot_critical_difference,
    plot_score_differences,
)
from nestkit.plotting.folds import plot_inner_cv_heatmap, plot_outer_scores, plot_score_stability
from nestkit.plotting.importance import (
    plot_importance,
    plot_rank_stability_features,
    plot_selection_frequency,
    plot_shap_summary,
)
from nestkit.plotting.summary import (
    plot_confusion_matrices,
    plot_precision_recall_curves,
    plot_predicted_vs_actual,
    plot_prediction_intervals,
    plot_rank_stability,
    plot_residual_qq,
    plot_residuals,
    plot_roc_curves,
)
from nestkit.plotting.threshold import (
    plot_threshold_comparison,
    plot_threshold_distribution,
    plot_threshold_sensitivity,
)
from nestkit.plotting.tuning import plot_inner_tuning_curve, plot_param_selection

__all__ = [
    "plot_bayesian_posterior",
    "plot_calibration_curves",
    "plot_calibration_improvement",
    "plot_comparison",
    "plot_confusion_matrices",
    "plot_critical_difference",
    "plot_importance",
    "plot_inner_cv_heatmap",
    "plot_inner_tuning_curve",
    "plot_outer_scores",
    "plot_param_selection",
    "plot_precision_recall_curves",
    "plot_predicted_vs_actual",
    "plot_prediction_intervals",
    "plot_rank_stability",
    "plot_rank_stability_features",
    "plot_residual_qq",
    "plot_residuals",
    "plot_roc_curves",
    "plot_score_differences",
    "plot_score_stability",
    "plot_selection_frequency",
    "plot_shap_summary",
    "plot_threshold_comparison",
    "plot_threshold_distribution",
    "plot_threshold_sensitivity",
]
