"""Inner CV tuning visualizations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nestkit.plotting._style import _apply_axis_limits, _get_ax

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_param_selection(
    results,
    param: str,
    bar_color: str | None = None,
    ax=None,
    **kwargs,
) -> Axes:
    """Frequency of each hyperparameter value selected across outer folds.

    Shows a bar chart where each bar represents a unique value from the
    search grid, and its height is the number of outer folds that
    selected that value.

    Parameters
    ----------
    results : ClassifierResults or RegressorResults
        Fitted nested CV results object.
    param : str
        Name of the hyperparameter to visualize.
    bar_color : str or None, optional
        Color of the bars. ``None`` uses the default color cycle.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If ``None``, a new figure is created.
    **kwargs
        Additional keyword arguments passed to the underlying matplotlib
        ``bar`` call.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    from collections import Counter

    from matplotlib.ticker import MaxNLocator

    ax = _get_ax(ax)
    values = [bp.get(param) for bp in results.best_params_per_fold_]
    counts = Counter(str(v) for v in values)
    labels = list(counts.keys())
    try:
        labels.sort(key=float)
    except ValueError:
        labels.sort()
    freqs = [counts[label] for label in labels]

    bar_kw = dict(kwargs)
    if bar_color is not None:
        bar_kw["color"] = bar_color
    ax.bar(range(len(labels)), freqs, **bar_kw)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel(param)
    ax.set_ylabel("Folds selected")
    ax.set_title(f"Parameter selection: {param}")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return ax


def plot_inner_tuning_curve(
    inner_report,
    param: str,
    metric: str | None = None,
    band_alpha: float = 0.2,
    ylim: tuple[float, float] | None = None,
    ax=None,
    **kwargs,
) -> Axes:
    """Score vs hyperparameter curve for a single inner CV report.

    Parameters
    ----------
    inner_report : InnerReport
        A single inner CV report (e.g., ``results.inner_reports_[i]``).
    param : str
        Name of the hyperparameter to place on the x-axis.
    metric : str or None, optional
        Scoring metric to use. If ``None``, the default metric from the
        inner report is used.
    band_alpha : float, optional
        Opacity of the +/- 1 std band.
    ylim : tuple of float or None, optional
        Explicit y-axis limits.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If ``None``, a new figure is created.
    **kwargs
        Additional keyword arguments passed to the underlying matplotlib
        ``plot`` call.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    ax = _get_ax(ax)
    df = inner_report.score_distribution(param, metric)
    ax.plot(df[param], df["mean_score"], "o-", **kwargs)
    if "std_score" in df.columns:
        ax.fill_between(
            df[param],
            df["mean_score"] - df["std_score"],
            df["mean_score"] + df["std_score"],
            alpha=band_alpha,
        )
    ax.set_xlabel(param)
    ax.set_ylabel("Mean score")
    ax.set_title(f"Tuning curve: {param}")
    _apply_axis_limits(ax, ylim=ylim)
    return ax
