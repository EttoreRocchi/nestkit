"""Threshold optimization visualizations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nestkit.plotting._style import _apply_axis_limits, _get_ax

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_UNIT = (0.0, 1.0)


def plot_threshold_sensitivity(
    results,
    fold_idx: int = 0,
    line_alpha: float = 0.8,
    full_range: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax=None,
    **kwargs,
) -> Axes:
    """Metrics as a function of decision threshold for a single fold.

    Parameters
    ----------
    results : ClassifierResults
        Fitted nested CV classification results object.
    fold_idx : int, optional
        Index of the outer fold to visualize.
    line_alpha : float, optional
        Opacity of metric curves.
    full_range : bool, optional
        If ``True``, set both axes to [0, 1].
    xlim, ylim : tuple of float or None, optional
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
    ax = _get_ax(ax)

    fr = results.fold_results_[fold_idx]
    if fr.threshold_result is None or fr.threshold_result.threshold_sensitivity.empty:
        ax.text(0.5, 0.5, "No threshold data", ha="center", va="center")
        return ax

    df = fr.threshold_result.threshold_sensitivity
    for col in ["sensitivity", "specificity", "precision", "recall", "f1"]:
        if col in df.columns:
            ax.plot(df["threshold"], df[col], label=col, alpha=line_alpha)

    if "criterion_value" in df.columns:
        ax.plot(
            df["threshold"],
            df["criterion_value"],
            label=fr.threshold_result.criterion_name or "criterion",
            color="black",
            linewidth=2,
            linestyle=":",
        )

    ax.axvline(fr.threshold_result.optimal_threshold, color="red", linestyle="--", label="Optimal")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_title(f"Threshold Sensitivity (Fold {fold_idx})")
    ax.legend(fontsize=7)
    _apply_axis_limits(
        ax, xlim=xlim, ylim=ylim, full_range=full_range, natural_xlim=_UNIT, natural_ylim=_UNIT
    )
    return ax


def plot_threshold_distribution(
    results,
    bar_alpha: float = 0.7,
    bins: int | None = None,
    full_range: bool = False,
    xlim: tuple[float, float] | None = None,
    ax=None,
    **kwargs,
) -> Axes:
    """Distribution of optimal thresholds across folds.

    Parameters
    ----------
    results : ClassifierResults
        Fitted nested CV classification results object.
    bar_alpha : float, optional
        Opacity of histogram bars.
    bins : int or None, optional
        Number of histogram bins. ``None`` uses a heuristic.
    full_range : bool, optional
        If ``True``, set x-axis to [0, 1].
    xlim : tuple of float or None, optional
        Explicit x-axis limits (override *full_range*).
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

    if not results.has_threshold_optimization:
        ax.text(0.5, 0.5, "No threshold data", ha="center", va="center")
        return ax

    thresholds = results.thresholds_per_fold_
    n_bins = bins if bins is not None else max(3, len(thresholds) // 2)
    ax.hist(thresholds, bins=n_bins, edgecolor="black", alpha=bar_alpha)
    ax.axvline(
        np.mean(thresholds), color="red", linestyle="--", label=f"Mean={np.mean(thresholds):.3f}"
    )
    ax.set_xlabel("Optimal threshold")
    ax.set_ylabel("Count")
    ax.set_title("Threshold Distribution Across Folds")
    ax.legend()
    _apply_axis_limits(ax, xlim=xlim, full_range=full_range, natural_xlim=_UNIT)
    return ax


def plot_threshold_comparison(
    results,
    full_range: bool = False,
    ylim: tuple[float, float] | None = None,
    ax=None,
    **kwargs,
) -> Axes:
    """Default vs optimized metrics side-by-side.

    Parameters
    ----------
    results : ClassifierResults
        Fitted nested CV classification results object.
    full_range : bool, optional
        If ``True``, set y-axis to [0, 1].
    ylim : tuple of float or None, optional
        Explicit y-axis limits (override *full_range*).
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

    if not results.has_threshold_optimization:
        ax.text(0.5, 0.5, "No threshold data", ha="center", va="center")
        return ax

    comp = results.threshold_comparison()
    x = np.arange(len(comp))
    width = 0.35

    ax.bar(
        x - width / 2, comp["mean_default"], width, label="Default (0.5)", yerr=comp["std_default"]
    )
    ax.bar(
        x + width / 2, comp["mean_optimized"], width, label="Optimized", yerr=comp["std_optimized"]
    )
    ax.set_xticks(x)
    ax.set_xticklabels(comp["metric"], rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Default vs Optimized Threshold")
    ax.legend()
    _apply_axis_limits(ax, ylim=ylim, full_range=full_range, natural_ylim=_UNIT)
    return ax
