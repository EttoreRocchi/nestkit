"""Aggregate result plots  -  confusion matrices, ROC, precision-recall, residuals, and more."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nestkit.plotting._style import _apply_axis_limits, _get_ax

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_UNIT = (0.0, 1.0)


def plot_confusion_matrices(
    results,
    threshold: str = "default",
    normalize: str | None = None,
    cmap: str = "Blues",
    fontsize: int = 10,
    ax=None,
    **kwargs,
) -> Axes:
    """Per-fold and aggregate confusion matrices.

    Parameters
    ----------
    results : ClassifierResults
        Fitted nested CV classification results object.
    threshold : {'default', 'optimized'}, optional
        Which threshold's confusion matrices to display.
    normalize : {None, 'true', 'pred', 'all'}, optional
        Normalization mode. ``'true'`` normalizes by row (true label),
        ``'pred'`` by column (predicted label), ``'all'`` by total count.
        ``None`` displays raw counts.
    cmap : str, optional
        Colormap for the heatmaps.
    fontsize : int, optional
        Font size for cell values.
    ax : matplotlib.axes.Axes or None, optional
        Ignored (subplots are always created). Kept for API consistency.
    **kwargs
        Additional keyword arguments passed to ``imshow``.

    Returns
    -------
    matplotlib.axes.Axes
        The first axes of the created subplots.
    """
    import matplotlib.pyplot as plt

    if threshold == "optimized" and hasattr(results, "confusion_matrices_optimized_"):
        cms = results.confusion_matrices_optimized_
        agg = results.confusion_matrix_aggregate_optimized_
    else:
        cms = results.confusion_matrices_default_
        agg = results.confusion_matrix_aggregate_default_

    def _normalize_cm(cm):
        cm = cm.astype(float)
        if normalize == "true":
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            return cm / row_sums
        if normalize == "pred":
            col_sums = cm.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1
            return cm / col_sums
        if normalize == "all":
            total = cm.sum()
            return cm / total if total > 0 else cm
        return cm

    n = len(cms) + 1
    _fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]

    labels = getattr(results, "classes_", None)
    n_classes = cms[0].shape[0]
    tick_labels = labels if labels is not None else list(range(n_classes))

    for i, cm in enumerate(cms):
        display = _normalize_cm(cm)
        axes[i].imshow(display, cmap=cmap, **kwargs)
        axes[i].set_title(f"Fold {i}")
        axes[i].set_xticks(range(n_classes))
        axes[i].set_xticklabels(tick_labels)
        axes[i].set_yticks(range(n_classes))
        axes[i].set_yticklabels(tick_labels)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")
        for r in range(display.shape[0]):
            for c in range(display.shape[1]):
                val = f"{display[r, c]:.2f}" if normalize else str(int(cm[r, c]))
                axes[i].text(c, r, val, ha="center", va="center", fontsize=fontsize)

    display_agg = _normalize_cm(agg)
    axes[-1].imshow(display_agg, cmap=cmap, **kwargs)
    axes[-1].set_title("Aggregate")
    axes[-1].set_xticks(range(n_classes))
    axes[-1].set_xticklabels(tick_labels)
    axes[-1].set_yticks(range(n_classes))
    axes[-1].set_yticklabels(tick_labels)
    axes[-1].set_xlabel("Predicted")
    axes[-1].set_ylabel("True")
    for r in range(display_agg.shape[0]):
        for c in range(display_agg.shape[1]):
            val = f"{display_agg[r, c]:.2f}" if normalize else str(int(agg[r, c]))
            axes[-1].text(c, r, val, ha="center", va="center", fontsize=fontsize)

    plt.tight_layout()
    return axes[0]


def plot_roc_curves(
    results,
    fold_alpha: float = 0.4,
    mean_color: str = "b",
    mean_lw: float = 2,
    band_alpha: float = 0.2,
    full_range: bool = False,
    ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    ax=None,
    **kwargs,
) -> Axes:
    """Per-fold ROC curves with mean and confidence-interval band.

    Parameters
    ----------
    results : ClassifierResults
        Fitted nested CV classification results object.
    fold_alpha : float, optional
        Opacity of individual fold curves.
    mean_color : str, optional
        Color of the mean ROC curve.
    mean_lw : float, optional
        Line width of the mean ROC curve.
    band_alpha : float, optional
        Opacity of the +/- 1 std band.
    full_range : bool, optional
        If ``True``, set both axes to [0, 1].
    ylim, xlim : tuple of float or None, optional
        Explicit axis limits (override *full_range*).
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If ``None``, a new figure is created.
    **kwargs
        Additional keyword arguments passed to the underlying matplotlib call.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    from sklearn.metrics import roc_curve

    ax = _get_ax(ax)

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for fr in results.fold_results_:
        proba = fr.y_proba_calibrated if fr.y_proba_calibrated is not None else fr.y_proba_raw
        if proba.ndim == 2:
            proba = proba[:, 1]
        fpr, tpr, _ = roc_curve(fr.y_true, proba)
        ax.plot(fpr, tpr, alpha=fold_alpha, lw=1)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)

    ax.plot(mean_fpr, mean_tpr, color=mean_color, lw=mean_lw, label="Mean ROC")
    ax.fill_between(
        mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=band_alpha, color=mean_color
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    _apply_axis_limits(
        ax, xlim=xlim, ylim=ylim, full_range=full_range, natural_xlim=_UNIT, natural_ylim=_UNIT
    )
    return ax


def plot_precision_recall_curves(
    results,
    fold_alpha: float = 0.5,
    full_range: bool = False,
    ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    ax=None,
    **kwargs,
) -> Axes:
    """Per-fold precision-recall curves.

    Parameters
    ----------
    results : ClassifierResults
        Fitted nested CV classification results object.
    fold_alpha : float, optional
        Opacity of individual fold curves.
    full_range : bool, optional
        If ``True``, set both axes to [0, 1].
    ylim, xlim : tuple of float or None, optional
        Explicit axis limits (override *full_range*).
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If ``None``, a new figure is created.
    **kwargs
        Additional keyword arguments passed to the underlying matplotlib call.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    from sklearn.metrics import precision_recall_curve

    ax = _get_ax(ax)

    for i, fr in enumerate(results.fold_results_):
        proba = fr.y_proba_calibrated if fr.y_proba_calibrated is not None else fr.y_proba_raw
        if proba.ndim == 2:
            proba = proba[:, 1]
        precision, recall, _ = precision_recall_curve(fr.y_true, proba)
        ax.plot(recall, precision, alpha=fold_alpha, label=f"Fold {i}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(fontsize=7)
    _apply_axis_limits(
        ax, xlim=xlim, ylim=ylim, full_range=full_range, natural_xlim=_UNIT, natural_ylim=_UNIT
    )
    return ax


def plot_rank_stability(
    results,
    top_k: int = 5,
    fold_alpha: float = 0.6,
    ylim: tuple[float, float] | None = None,
    ax=None,
    **kwargs,
) -> Axes:
    """Configuration rank stability across folds.

    Parameters
    ----------
    results : ClassifierResults or RegressorResults
        Fitted nested CV results object.
    top_k : int, optional
        Number of top configurations to display per fold.
    fold_alpha : float, optional
        Opacity of fold lines.
    ylim : tuple of float or None, optional
        Explicit y-axis limits.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If ``None``, a new figure is created.
    **kwargs
        Additional keyword arguments passed to the underlying matplotlib call.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    ax = _get_ax(ax)

    for i, report in enumerate(results.inner_reports_):
        df = report.ranking()
        if "mean_test_score" in df.columns:
            ax.plot(
                range(min(top_k, len(df))),
                df["mean_test_score"].values[:top_k],
                marker="o",
                alpha=fold_alpha,
                label=f"Fold {i}",
            )

    ax.set_xlabel("Configuration Rank")
    ax.set_ylabel("Mean Test Score")
    ax.set_title("Inner CV Rank Stability")
    ax.legend(fontsize=7)
    _apply_axis_limits(ax, ylim=ylim)
    return ax


def plot_residuals(
    results,
    fold_idx: int | list[int] | None = None,
    bins: int = 30,
    fold_alpha: float = 0.5,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax=None,
    **kwargs,
) -> Axes:
    """Residual distributions per fold.

    Parameters
    ----------
    results : RegressorResults
        Fitted nested CV regression results object.
    fold_idx : int, list of int, or None, optional
        Outer fold index or indices to plot.  If ``None`` (default), all
        folds are shown.
    bins : int, optional
        Number of histogram bins.
    fold_alpha : float, optional
        Opacity of fold histograms.
    xlim, ylim : tuple of float or None, optional
        Explicit axis limits.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If ``None``, a new figure is created.
    **kwargs
        Additional keyword arguments passed to the underlying matplotlib call.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    ax = _get_ax(ax)

    if fold_idx is None:
        indices = list(range(len(results.fold_results_)))
    elif isinstance(fold_idx, int):
        indices = [fold_idx]
    else:
        indices = list(fold_idx)

    for i in indices:
        fr = results.fold_results_[i]
        ax.hist(fr.residuals, bins=bins, alpha=fold_alpha, label=f"Fold {i}")

    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distributions")
    ax.legend(fontsize=7)
    _apply_axis_limits(ax, xlim=xlim, ylim=ylim, full_range=False)
    return ax


def plot_predicted_vs_actual(
    results,
    point_alpha: float = 0.4,
    point_size: float = 12,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax=None,
    **kwargs,
) -> Axes:
    """Scatter of predicted vs actual values with identity line.

    Parameters
    ----------
    results : RegressorResults
        Fitted nested CV regression results object.
    point_alpha : float, optional
        Opacity of scatter points.
    point_size : float, optional
        Size of scatter points.
    xlim, ylim : tuple of float or None, optional
        Explicit axis limits.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If ``None``, a new figure is created.
    **kwargs
        Additional keyword arguments passed to the underlying matplotlib call.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    ax = _get_ax(ax)

    preds = results.predictions_
    ax.scatter(preds["y_true"], preds["y_pred"], alpha=point_alpha, s=point_size)

    lims = [
        min(preds["y_true"].min(), preds["y_pred"].min()),
        max(preds["y_true"].max(), preds["y_pred"].max()),
    ]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    _apply_axis_limits(ax, xlim=xlim, ylim=ylim, full_range=False)
    return ax


def plot_prediction_intervals(
    results,
    band_alpha: float = 0.25,
    point_size: float = 8,
    ylim: tuple[float, float] | None = None,
    ax=None,
    **kwargs,
) -> Axes:
    """Predictions with interval bands.

    Parameters
    ----------
    results : RegressorResults
        Fitted nested CV regression results object.
    band_alpha : float, optional
        Opacity of the prediction interval band.
    point_size : float, optional
        Size of scatter points.
    ylim : tuple of float or None, optional
        Explicit y-axis limits.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If ``None``, a new figure is created.
    **kwargs
        Additional keyword arguments passed to the underlying matplotlib call.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    ax = _get_ax(ax)

    preds = results.predictions_
    if "pi_lower" not in preds.columns:
        ax.text(
            0.5, 0.5, "No prediction intervals", ha="center", va="center", transform=ax.transAxes
        )
        return ax

    sorted_idx = preds["y_true"].argsort()
    x = np.arange(len(preds))

    ax.fill_between(
        x,
        preds["pi_lower"].values[sorted_idx],
        preds["pi_upper"].values[sorted_idx],
        alpha=band_alpha,
        color="blue",
        label="Prediction Interval",
    )
    ax.scatter(x, preds["y_true"].values[sorted_idx], s=point_size, color="red", label="Actual")
    ax.scatter(
        x, preds["y_pred"].values[sorted_idx], s=point_size, color="blue", label="Predicted"
    )
    ax.set_xlabel("Sample (sorted by actual)")
    ax.set_ylabel("Value")
    ax.set_title("Prediction Intervals")
    ax.legend(fontsize=7)
    _apply_axis_limits(ax, ylim=ylim, full_range=False)
    return ax


def plot_residual_qq(
    results,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax=None,
    **kwargs,
) -> Axes:
    """QQ plot of pooled residuals.

    Parameters
    ----------
    results : RegressorResults
        Fitted nested CV regression results object.
    xlim, ylim : tuple of float or None, optional
        Explicit axis limits.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If ``None``, a new figure is created.
    **kwargs
        Additional keyword arguments passed to the underlying matplotlib call.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    from scipy import stats

    ax = _get_ax(ax)

    all_residuals = np.concatenate([fr.residuals for fr in results.fold_results_])
    stats.probplot(all_residuals, dist="norm", plot=ax)
    ax.set_title("Residual QQ Plot")
    _apply_axis_limits(ax, xlim=xlim, ylim=ylim, full_range=False)
    return ax
