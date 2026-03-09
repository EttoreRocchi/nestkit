"""Fold-level visualizations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nestkit.plotting._style import _apply_axis_limits, _get_ax

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_outer_scores(
    results,
    metric: str,
    point_alpha: float = 0.7,
    ylim: tuple[float, float] | None = None,
    ax=None,
    **kwargs,
) -> Axes:
    """Box-and-strip plot of outer fold scores.

    Parameters
    ----------
    results : ClassifierResults or RegressorResults
        Fitted nested CV results object.
    metric : str
        Name of the scoring metric to plot.
    point_alpha : float, optional
        Opacity of individual fold score markers.
    ylim : tuple of float or None, optional
        Explicit y-axis limits.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If ``None``, a new figure is created.
    **kwargs
        Additional keyword arguments passed to the underlying matplotlib
        ``boxplot`` call.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    ax = _get_ax(ax)
    scores = results.outer_scores_default_[metric].values
    ax.boxplot(scores, orientation="vertical", **kwargs)
    ax.scatter(np.ones(len(scores)), scores, alpha=point_alpha, zorder=3)
    ax.set_ylabel(metric)
    ax.set_title(f"Outer Fold Scores: {metric}")
    ax.set_xticks([])
    _apply_axis_limits(ax, ylim=ylim)
    return ax


def plot_inner_cv_heatmap(
    results,
    cmap: str = "YlOrRd",
    annot: bool = True,
    fmt: str = ".3f",
    ax=None,
    **kwargs,
) -> Axes:
    """Heatmap of mean inner CV scores across hyperparameter grid per outer fold.

    Each row is an outer fold, each column is a hyperparameter configuration
    from the inner search grid.  Color intensity reflects the inner CV score.
    This reveals whether the optimal region of the hyperparameter space is
    consistent across outer folds.

    Parameters
    ----------
    results : ClassifierResults or RegressorResults
        Fitted nested CV results object with ``inner_reports_`` attribute.
    cmap : str, optional
        Matplotlib colormap name.
    annot : bool, optional
        Whether to annotate cells with score values.
    fmt : str, optional
        Format string for annotations.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If ``None``, a new figure is created.
    **kwargs
        Additional keyword arguments passed to ``matplotlib.axes.Axes.imshow``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    import matplotlib.pyplot as plt

    rows = []
    param_labels = None
    for report in results.inner_reports_:
        df = report.to_dataframe()
        score_col = "mean_test_score"
        if score_col not in df.columns:
            raise ValueError(
                f"Column 'mean_test_score' not found in inner CV results. "
                f"Available columns: {list(df.columns)}"
            )
        scores = df[score_col].values
        rows.append(scores)

        if param_labels is None:
            param_cols = [c for c in df.columns if c.startswith("param_")]
            param_labels = [
                ", ".join(f"{c.replace('param_', '')}={v}" for c, v in zip(param_cols, row))
                for _, row in df[param_cols].iterrows()
            ]

    score_matrix = np.array(rows)
    n_folds, n_configs = score_matrix.shape

    ax = _get_ax(ax, figsize=(max(6, n_configs * 0.8), max(3, n_folds * 0.6)))
    im = ax.imshow(score_matrix, aspect="auto", cmap=cmap, **kwargs)

    if annot:
        for i in range(n_folds):
            for j in range(n_configs):
                ax.text(
                    j,
                    i,
                    format(score_matrix[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if score_matrix[i, j] > score_matrix.mean() else "black",
                    fontsize=8,
                )

    ax.set_xticks(range(n_configs))
    ax.set_xticklabels(param_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_folds))
    ax.set_yticklabels([f"Fold {i}" for i in range(n_folds)])
    scoring = getattr(results, "scoring_", None) or "score"
    ax.set_title(f"Inner CV Scores: {scoring}")

    plt.colorbar(im, ax=ax, label="Mean CV score")
    return ax


def plot_score_stability(
    results,
    metrics: list[str] | None = None,
    point_alpha: float = 0.7,
    xlim: tuple[float, float] | None = None,
    ax=None,
    **kwargs,
) -> Axes:
    """Strip plot of multiple metrics across outer folds.

    Shows the spread of each metric's outer fold scores on a single axis,
    giving a holistic view of result stability.  Each metric is a row
    with individual fold scores plotted as points.

    Parameters
    ----------
    results : ClassifierResults or RegressorResults
        Fitted nested CV results object.
    metrics : list of str or None, optional
        Metric names to include.  If ``None``, all metrics from
        ``outer_scores_default_`` are used.
    point_alpha : float, optional
        Opacity of individual fold score markers.
    xlim : tuple of float or None, optional
        Explicit x-axis limits (score axis).
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If ``None``, a new figure is created.
    **kwargs
        Additional keyword arguments passed to the underlying matplotlib
        ``scatter`` call.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    scores_df = results.outer_scores_default_
    if metrics is None:
        metrics = list(scores_df.columns)

    n_metrics = len(metrics)
    ax = _get_ax(ax, figsize=(8, max(2, n_metrics * 0.5)))

    for i, metric in enumerate(metrics):
        values = scores_df[metric].values
        mean_val = np.mean(values)
        ax.scatter(values, np.full(len(values), i), alpha=point_alpha, zorder=3, **kwargs)
        ax.plot(mean_val, i, "D", color="black", markersize=6, zorder=4)

    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels(metrics)
    ax.set_xlabel("Score")
    ax.set_title("Score Stability Across Folds")
    ax.invert_yaxis()
    _apply_axis_limits(ax, xlim=xlim)
    return ax
