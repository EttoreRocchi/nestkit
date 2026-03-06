"""Calibration visualizations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nestkit._validation import extract_positive_proba
from nestkit.calibration.diagnostics import CalibrationDiagnostics
from nestkit.plotting._style import _apply_axis_limits, _get_ax

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_UNIT = (0.0, 1.0)


def plot_calibration_curves(
    results,
    fold_idx: int | list[int] | None = None,
    fold_alpha: float = 0.4,
    full_range: bool = False,
    ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    ax=None,
    **kwargs,
) -> Axes:
    """Reliability diagrams showing raw vs calibrated probabilities per fold.

    Parameters
    ----------
    results : ClassifierResults
        Fitted nested CV classification results object.
    fold_idx : int, list of int, or None, optional
        Outer fold index or indices to plot.  If ``None`` (default), all
        folds are shown.
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
    import matplotlib.pyplot as plt

    ax = _get_ax(ax)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")

    # Determine which folds to plot
    if fold_idx is None:
        selected = results.fold_results_
    elif isinstance(fold_idx, int):
        selected = [fr for fr in results.fold_results_ if fr.fold_idx == fold_idx]
    else:
        fold_set = set(fold_idx)
        selected = [fr for fr in results.fold_results_ if fr.fold_idx in fold_set]

    # Use same color per fold, different markers for raw vs calibrated
    cmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, fr in enumerate(selected):
        color = cmap[i % len(cmap)]
        p_raw = extract_positive_proba(fr.y_proba_raw)
        diag_data = CalibrationDiagnostics.reliability_diagram_data(fr.y_true, p_raw)
        valid = diag_data.dropna(subset=["fraction_positive"])
        ax.plot(
            valid["mean_predicted"],
            valid["fraction_positive"],
            "o-",
            color=color,
            alpha=fold_alpha,
            label=f"Raw fold {fr.fold_idx}",
        )
        if fr.y_proba_calibrated is not None:
            p_cal = extract_positive_proba(fr.y_proba_calibrated)
            diag_cal = CalibrationDiagnostics.reliability_diagram_data(fr.y_true, p_cal)
            valid_cal = diag_cal.dropna(subset=["fraction_positive"])
            ax.plot(
                valid_cal["mean_predicted"],
                valid_cal["fraction_positive"],
                "s--",
                color=color,
                alpha=fold_alpha,
                label=f"Cal fold {fr.fold_idx}",
            )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability Diagram")
    ax.legend(fontsize=7)
    _apply_axis_limits(
        ax, xlim=xlim, ylim=ylim, full_range=full_range, natural_xlim=_UNIT, natural_ylim=_UNIT
    )
    return ax


def plot_calibration_improvement(
    results,
    annot: bool = False,
    annot_fmt: str = ".3f",
    full_range: bool = False,
    ylim: tuple[float, float] | None = None,
    ax=None,
    **kwargs,
) -> Axes:
    """Paired bar plot of ECE before vs after calibration per fold.

    Shows raw and calibrated ECE side by side for each fold, with the
    gap (improvement) annotated above each pair.  Mean and standard
    deviation of the gap are reported in the legend.

    Parameters
    ----------
    results : ClassifierResults
        Fitted nested CV classification results object.
    annot : bool, optional
        If ``True``, annotate each bar with its ECE value and display
        the gap above each pair.
    annot_fmt : str, optional
        Format string for annotations.
    full_range : bool, optional
        If ``True``, set y-axis to [0, 1].
    ylim : tuple of float or None, optional
        Explicit y-axis limits (override *full_range*).
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If ``None``, a new figure is created.
    **kwargs
        Additional keyword arguments passed to the underlying matplotlib
        ``bar`` calls.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    import numpy as np

    ax = _get_ax(ax)
    if not results.has_calibration:
        ax.text(0.5, 0.5, "No calibration data", ha="center", va="center")
        return ax

    cal_df = results.calibration_summary_
    if "ece_raw" not in cal_df.columns:
        ax.text(0.5, 0.5, "No ECE data", ha="center", va="center")
        return ax

    folds = cal_df["fold_idx"].values
    ece_raw = cal_df["ece_raw"].values
    ece_cal = cal_df["ece_calibrated"].values
    gaps = ece_raw - ece_cal

    n = len(folds)
    x = np.arange(n)
    width = 0.35

    gap_mean = np.mean(gaps)
    gap_std = np.std(gaps, ddof=1) if n > 1 else 0.0

    ax.bar(
        x - width / 2,
        ece_raw,
        width,
        label="ECE raw",
        **kwargs,
    )
    ax.bar(
        x + width / 2,
        ece_cal,
        width,
        label="ECE calibrated",
        **kwargs,
    )

    # Invisible bar entry for gap stats in legend
    ax.bar([], [], width=0, label=f"Gap: {gap_mean:{annot_fmt}} \u00b1 {gap_std:{annot_fmt}}")

    if annot:
        for i in range(n):
            ax.text(
                x[i] - width / 2,
                ece_raw[i],
                format(ece_raw[i], annot_fmt),
                ha="center",
                va="bottom",
                fontsize=7,
            )
            ax.text(
                x[i] + width / 2,
                ece_cal[i],
                format(ece_cal[i], annot_fmt),
                ha="center",
                va="bottom",
                fontsize=7,
            )
            top = max(ece_raw[i], ece_cal[i])
            ax.text(
                x[i],
                top * 1.05,
                f"\u0394{format(gaps[i], annot_fmt)}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="gray",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {int(f)}" for f in folds])
    ax.set_ylabel("ECE")
    ax.set_title("Calibration Improvement")
    ax.legend(fontsize=8)
    _apply_axis_limits(ax, ylim=ylim, full_range=full_range, natural_ylim=_UNIT)
    return ax
