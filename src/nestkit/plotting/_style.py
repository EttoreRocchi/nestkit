"""Shared helpers for plotting functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _get_ax(ax=None, figsize: tuple[float, float] | None = None) -> Axes:
    """Return *ax* or create a new figure and axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.  If ``None``, a new figure is created.
    figsize : tuple of float or None, optional
        Figure size ``(width, height)`` in inches, used only when
        creating a new figure.

    Returns
    -------
    matplotlib.axes.Axes
        The axes to draw on.
    """
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=figsize)
    return ax


def _apply_axis_limits(
    ax: Axes,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    full_range: bool = False,
    natural_xlim: tuple[float, float] | None = None,
    natural_ylim: tuple[float, float] | None = None,
) -> None:
    """Apply axis limits with priority: explicit > full_range > auto.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    xlim, ylim : tuple of float or None, optional
        Explicit axis limits; take highest priority.
    full_range : bool, optional
        If ``True`` and no explicit limit is given, use *natural_xlim*
        or *natural_ylim*.
    natural_xlim, natural_ylim : tuple of float or None, optional
        Default "full" limits applied when *full_range* is ``True``.
    """
    if xlim is not None:
        ax.set_xlim(xlim)
    elif full_range and natural_xlim is not None:
        ax.set_xlim(natural_xlim)

    if ylim is not None:
        ax.set_ylim(ylim)
    elif full_range and natural_ylim is not None:
        ax.set_ylim(natural_ylim)
