"""Model comparison visualizations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nestkit.plotting._style import _apply_axis_limits, _get_ax

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_comparison(
    comparator,
    metric: str,
    threshold: str = "default",
    point_alpha: float = 0.6,
    line_alpha: float = 0.15,
    ylim: tuple[float, float] | None = None,
    ax=None,
    **kwargs,
) -> Axes:
    """Paired box-and-strip plot of per-fold scores across models.

    Parameters
    ----------
    comparator : ModelComparator
        Fitted model comparator containing two or more result sets.
    metric : str
        Name of the scoring metric to compare.
    threshold : {'default', 'optimized'}, optional
        Which threshold's scores to use.
    point_alpha : float, optional
        Opacity of individual fold score markers.
    line_alpha : float, optional
        Opacity of lines connecting paired observations.
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

    models = list(comparator._results.keys())
    all_scores = []
    for name in models:
        scores = comparator._get_scores(name, metric, threshold)
        all_scores.append(scores)

    ax.boxplot(all_scores, labels=models)
    for i, scores in enumerate(all_scores):
        ax.scatter(np.full(len(scores), i + 1), scores, alpha=point_alpha, zorder=3)

    for j in range(len(all_scores[0])):
        xs = list(range(1, len(models) + 1))
        ys = [all_scores[i][j] for i in range(len(models))]
        ax.plot(xs, ys, "k-", alpha=line_alpha)

    ax.set_ylabel(metric)
    ax.set_title(f"Model Comparison: {metric}")
    _apply_axis_limits(ax, ylim=ylim)
    return ax


def plot_score_differences(
    comparator,
    metric: str,
    model_a: str,
    model_b: str,
    threshold: str = "default",
    bar_alpha: float = 0.7,
    bar_color: str | None = None,
    ylim: tuple[float, float] | None = None,
    ax=None,
    **kwargs,
) -> Axes:
    """Per-fold score differences between two models.

    Parameters
    ----------
    comparator : ModelComparator
        Fitted model comparator containing the two models.
    metric : str
        Name of the scoring metric to compare.
    model_a, model_b : str
        Names of the two models to compare.
    threshold : {'default', 'optimized'}, optional
        Which threshold's scores to use.
    bar_alpha : float, optional
        Opacity of the bars.
    bar_color : str or None, optional
        Color of the bars. ``None`` uses the default color cycle.
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

    scores_a = comparator._get_scores(model_a, metric, threshold)
    scores_b = comparator._get_scores(model_b, metric, threshold)
    diffs = scores_a - scores_b

    bar_kw = {"alpha": bar_alpha}
    if bar_color is not None:
        bar_kw["color"] = bar_color
    ax.bar(range(len(diffs)), diffs, **bar_kw)
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.axhline(np.mean(diffs), color="red", linestyle="--", label=f"Mean={np.mean(diffs):.4f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel(f"Score diff ({model_a} - {model_b})")
    ax.set_title(f"Score Differences: {metric}")
    ax.legend()
    _apply_axis_limits(ax, ylim=ylim, full_range=False)
    return ax


def plot_bayesian_posterior(
    comparator,
    metric: str,
    model_a: str,
    model_b: str,
    rope: float = 0.01,
    threshold: str = "default",
    color_a: str = "blue",
    color_b: str = "red",
    color_rope: str = "gray",
    fill_alpha: float = 0.3,
    ax=None,
    **kwargs,
) -> Axes:
    """Posterior distribution of score differences with ROPE.

    Parameters
    ----------
    comparator : ModelComparator
        Fitted model comparator containing the two models.
    metric : str
        Name of the scoring metric to compare.
    model_a, model_b : str
        Names of the two models to compare.
    rope : float, optional
        Half-width of the Region of Practical Equivalence.
    threshold : {'default', 'optimized'}, optional
        Which threshold's scores to use.
    color_a : str, optional
        Fill color for the "A is better" region.
    color_b : str, optional
        Fill color for the "B is better" region.
    color_rope : str, optional
        Fill color for the equivalence region.
    fill_alpha : float, optional
        Opacity of all filled regions.
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

    result = comparator.bayesian_comparison(metric, model_a, model_b, rope, threshold)
    scores_a = comparator._get_scores(model_a, metric, threshold)
    scores_b = comparator._get_scores(model_b, metric, threshold)
    diffs = scores_a - scores_b

    from scipy.stats import t as t_dist

    n = len(diffs)
    mean = np.mean(diffs)
    sd = np.std(diffs, ddof=1)
    ref_results = next(iter(comparator._results.values()))
    n_test = np.mean([len(fr.test_indices) for fr in ref_results.fold_results_])
    n_train = np.mean([len(fr.train_indices) for fr in ref_results.fold_results_])
    se = sd * np.sqrt((1.0 / n) + (n_test / n_train))

    x = np.linspace(mean - 4 * se, mean + 4 * se, 200)
    pdf = t_dist.pdf(x, df=n - 1, loc=mean, scale=se)

    ax.plot(x, pdf, "k-")
    ax.fill_between(
        x,
        pdf,
        where=(x > rope),
        alpha=fill_alpha,
        color=color_a,
        label=f"P(A>{model_a})={result['p_a_better']:.3f}",
    )
    ax.fill_between(
        x,
        pdf,
        where=(x < -rope),
        alpha=fill_alpha,
        color=color_b,
        label=f"P(B>{model_b})={result['p_b_better']:.3f}",
    )
    ax.fill_between(
        x,
        pdf,
        where=(np.abs(x) <= rope),
        alpha=fill_alpha,
        color=color_rope,
        label=f"P(equiv)={result['p_equivalent']:.3f}",
    )
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)

    ax.set_xlabel(f"Score difference ({model_a} - {model_b})")
    ax.set_ylabel("Density")
    ax.set_title(f"Bayesian Comparison: {metric}")
    ax.legend(fontsize=7)
    return ax


def plot_critical_difference(
    comparator,
    metric: str,
    threshold: str = "default",
    bar_alpha: float = 0.7,
    bar_color: str | None = None,
    ax=None,
    **kwargs,
) -> Axes:
    """Demsar critical difference diagram.

    Parameters
    ----------
    comparator : ModelComparator
        Fitted model comparator containing three or more result sets.
    metric : str
        Name of the scoring metric to rank.
    threshold : {'default', 'optimized'}, optional
        Which threshold's scores to use.
    bar_alpha : float, optional
        Opacity of the bars.
    bar_color : str or None, optional
        Color of the bars. ``None`` uses the default color cycle.
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

    models = list(comparator._results.keys())
    if len(models) < 3:
        ax.text(0.5, 0.5, "Need >= 3 models", ha="center", va="center")
        return ax

    n_folds = comparator._results[models[0]].n_outer_folds_
    ranks_per_model = {}
    for name in models:
        scores = comparator._get_scores(name, metric, threshold)
        ranks_per_model[name] = scores

    from scipy.stats import rankdata

    avg_ranks = {}
    for fold_idx in range(n_folds):
        fold_scores = [ranks_per_model[m][fold_idx] for m in models]
        fold_ranks = rankdata(-np.array(fold_scores))
        for i, m in enumerate(models):
            avg_ranks.setdefault(m, []).append(fold_ranks[i])

    mean_ranks = {m: np.mean(r) for m, r in avg_ranks.items()}
    sorted_models = sorted(mean_ranks, key=mean_ranks.get)

    y_pos = np.arange(len(sorted_models))
    bar_kw = {"alpha": bar_alpha}
    if bar_color is not None:
        bar_kw["color"] = bar_color
    ax.barh(y_pos, [mean_ranks[m] for m in sorted_models], **bar_kw)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_models)
    ax.set_xlabel("Average rank")
    ax.set_title(f"Critical Difference Diagram: {metric}")
    ax.invert_xaxis()
    return ax
