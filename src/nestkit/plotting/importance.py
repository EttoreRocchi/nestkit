"""Feature importance visualizations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nestkit.plotting._style import _apply_axis_limits, _get_ax

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_UNIT = (0.0, 1.0)


def plot_importance(
    aggregator,
    top_k: int = 20,
    show_folds: bool = True,
    bar_alpha: float = 0.7,
    fold_color: str = "red",
    fold_alpha: float = 0.5,
    fold_size: float = 18,
    ax=None,
    **kwargs,
) -> Axes:
    """Bar plot of mean feature importance with optional per-fold jitter.

    Parameters
    ----------
    aggregator : ImportanceAggregator
        Fitted importance aggregator with per-fold importance data.
    top_k : int, optional
        Number of top features to display.
    show_folds : bool, optional
        Whether to overlay per-fold importance values.
    bar_alpha : float, optional
        Opacity of the bars.
    fold_color : str, optional
        Color of per-fold scatter markers.
    fold_alpha : float, optional
        Opacity of per-fold scatter markers.
    fold_size : float, optional
        Size of per-fold scatter markers.
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

    df = aggregator.summary_.head(top_k)
    y = np.arange(len(df))

    ax.barh(y, df["mean_importance"], xerr=df["std_importance"], alpha=bar_alpha)

    if show_folds:
        features = df["feature"].values
        names = aggregator.feature_names or [
            f"feature_{i}" for i in range(aggregator.importances_matrix_.shape[1])
        ]
        for i, feat in enumerate(features):
            feat_idx = names.index(feat) if feat in names else i
            fold_vals = aggregator.importances_matrix_[:, feat_idx]
            ax.scatter(
                fold_vals,
                np.full(len(fold_vals), i),
                alpha=fold_alpha,
                s=fold_size,
                color=fold_color,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"])
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance (top {top_k})")
    ax.invert_yaxis()
    return ax


def plot_rank_stability_features(
    aggregator,
    top_k: int = 20,
    cmap: str = "YlOrRd",
    label_fontsize: int = 7,
    ax=None,
    **kwargs,
) -> Axes:
    """Feature rank stability heatmap across folds.

    Parameters
    ----------
    aggregator : ImportanceAggregator
        Fitted importance aggregator with per-fold rank data.
    top_k : int, optional
        Number of top features to display.
    cmap : str, optional
        Colormap for the heatmap.
    label_fontsize : int, optional
        Font size for y-axis feature labels.
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
    import matplotlib.pyplot as plt

    df = aggregator.summary_.head(top_k)
    features = df["feature"].values
    names = aggregator.feature_names or [
        f"feature_{i}" for i in range(aggregator.ranks_matrix_.shape[1])
    ]

    rank_data = []
    for feat in features:
        feat_idx = names.index(feat) if feat in names else 0
        rank_data.append(aggregator.ranks_matrix_[:, feat_idx])

    rank_data = np.array(rank_data)
    im = ax.imshow(rank_data, aspect="auto", cmap=cmap)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=label_fontsize)
    ax.set_xlabel("Fold")
    ax.set_title("Feature Rank Stability")
    plt.colorbar(im, ax=ax, label="Rank")
    return ax


def plot_shap_summary(aggregator, top_k: int = 20, ax=None, **kwargs) -> Axes:
    """Pooled SHAP beeswarm plot.

    Concatenates raw SHAP values from all outer test folds and renders
    a beeswarm summary plot using the ``shap`` package. Requires the
    aggregator to have been created with ``method='shap'``.

    Parameters
    ----------
    aggregator : ImportanceAggregator
        Fitted importance aggregator with raw SHAP values stored.
    top_k : int, optional
        Maximum number of features to display.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If ``None``, a new figure is created.
    **kwargs
        Additional keyword arguments passed to the underlying matplotlib call.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.

    Raises
    ------
    ValueError
        If raw SHAP values are not available in the aggregator.
    """
    ax = _get_ax(ax)

    if not aggregator.raw_importances_:
        raise ValueError("SHAP raw values not available. Use method='shap'.")

    try:
        import shap

        pooled = np.concatenate(aggregator.raw_importances_)
        shap.summary_plot(pooled, max_display=top_k, show=False)
    except ImportError:
        ax.text(0.5, 0.5, "shap package required", ha="center", va="center")

    return ax


def plot_selection_frequency(
    aggregator,
    top_k: int = 10,
    bar_alpha: float = 0.7,
    label_fontsize: int = 7,
    full_range: bool = False,
    xlim: tuple[float, float] | None = None,
    ax=None,
    **kwargs,
) -> Axes:
    """Feature selection frequency across folds.

    Parameters
    ----------
    aggregator : ImportanceAggregator
        Fitted importance aggregator with per-fold importance data.
    top_k : int, optional
        The top-*k* threshold used to count selection frequency.
    bar_alpha : float, optional
        Opacity of the bars.
    label_fontsize : int, optional
        Font size for y-axis feature labels.
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

    n_folds, n_features = aggregator.importances_matrix_.shape
    names = aggregator.feature_names or [f"feature_{i}" for i in range(n_features)]

    frequency = np.zeros(n_features)
    for i in range(n_folds):
        top_idx = np.argsort(-aggregator.importances_matrix_[i])[:top_k]
        frequency[top_idx] += 1
    frequency /= n_folds

    sorted_idx = np.argsort(-frequency)[: top_k * 2]
    y = np.arange(len(sorted_idx))

    ax.barh(y, frequency[sorted_idx], alpha=bar_alpha)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in sorted_idx], fontsize=label_fontsize)
    ax.set_xlabel(f"Frequency in top-{top_k}")
    ax.set_title("Feature Selection Frequency")
    ax.invert_yaxis()
    _apply_axis_limits(ax, xlim=xlim, full_range=full_range, natural_xlim=_UNIT)
    return ax
