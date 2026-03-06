"""Shared helpers for plotting functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _get_ax(ax=None, figsize: tuple[float, float] | None = None) -> Axes:
    """Return *ax* or create a new figure and axes."""
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
    """Apply axis limits with priority: explicit > full_range > auto."""
    if xlim is not None:
        ax.set_xlim(xlim)
    elif full_range and natural_xlim is not None:
        ax.set_xlim(natural_xlim)

    if ylim is not None:
        ax.set_ylim(ylim)
    elif full_range and natural_ylim is not None:
        ax.set_ylim(natural_ylim)
