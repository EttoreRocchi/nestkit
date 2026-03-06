"""Calibration diagnostic metrics.

Provides ``CalibrationDiagnostics``, a collection of static methods for
evaluating the quality of predicted probabilities before and after
post-hoc calibration.  Includes Expected Calibration Error (ECE),
Maximum Calibration Error (MCE), Brier score and its decomposition, and
reliability diagram data.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd

from nestkit._validation import extract_positive_proba


def _make_bins(p: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    """Return bin edges for calibration diagnostics.

    Parameters
    ----------
    p : numpy.ndarray
        1-D array of predicted probabilities.
    n_bins : int
        Desired number of bins.
    strategy : {"quantile", "uniform"}
        ``"quantile"`` produces bins with approximately equal sample
        counts (adaptive binning).  ``"uniform"`` produces equal-width
        bins over [0, 1].

    Returns
    -------
    numpy.ndarray
        Sorted, unique bin edges.  For quantile bins the actual number
        of bins may be smaller than *n_bins* when many probabilities
        share the same value.
    """
    if strategy == "quantile":
        return np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if strategy == "uniform":
        return np.linspace(0, 1, n_bins + 1)
    raise ValueError(f"Unknown binning strategy: {strategy!r}. Use 'quantile' or 'uniform'.")


def _bin_mask(p: np.ndarray, lo: float, hi: float, *, is_last: bool) -> np.ndarray:
    """Return a boolean mask selecting samples in the bin ``[lo, hi)`` (or ``[lo, hi]`` for the last bin)."""
    if is_last:
        return (p >= lo) & (p <= hi)
    return (p >= lo) & (p < hi)


class CalibrationDiagnostics:
    """Assess calibration quality before and after post-hoc calibration.

    All methods are static and operate on arrays of true labels and
    predicted probabilities.  No fitting or state is required.

    See Also
    --------
    nestkit.calibration.calibrators.PostHocCalibrator :
        Apply post-hoc calibration to predicted probabilities.

    Examples
    --------
    >>> import numpy as np
    >>> from nestkit.calibration.diagnostics import CalibrationDiagnostics
    >>> y_true = np.array([0, 0, 1, 1, 1])
    >>> y_proba = np.array([0.1, 0.3, 0.6, 0.8, 0.9])
    >>> CalibrationDiagnostics.brier_score(y_true, y_proba)  # doctest: +SKIP
    0.042...
    """

    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
        strategy: str = "quantile",
    ) -> float:
        """Compute the Expected Calibration Error (ECE).

        ECE is the weighted average of the absolute difference between
        observed accuracy and mean predicted confidence within each
        probability bin.

        Parameters
        ----------
        y_true : numpy.ndarray of shape (n_samples,)
            True binary labels (0 or 1).
        y_proba : numpy.ndarray of shape (n_samples,) or (n_samples, 2)
            Predicted probabilities.  If 2-D, the positive-class column
            is extracted.
        n_bins : int, default 10
            Number of bins.
        strategy : {"quantile", "uniform"}, default "quantile"
            Binning strategy.  ``"quantile"`` produces bins with
            approximately equal sample counts; ``"uniform"`` produces
            equal-width bins over [0, 1].

        Returns
        -------
        float
            Expected Calibration Error in [0, 1].

        Notes
        -----
        ECE is defined as:

        .. math::

            \\text{ECE} = \\sum_{b=1}^{B} \\frac{n_b}{N}
            \\left| \\text{acc}(b) - \\text{conf}(b) \\right|

        where *B* is the number of bins, :math:`n_b` the number of
        samples in bin *b*, *N* the total number of samples,
        :math:`\\text{acc}(b)` the observed accuracy in bin *b*, and
        :math:`\\text{conf}(b)` the mean predicted probability in bin
        *b*.

        References
        ----------
        .. [1] Naeini, M.P., Cooper, G.F., and Hauskrecht, M. (2015).
           "Obtaining Well Calibrated Probabilities Using Bayesian
           Binning into Quantiles." *AAAI*.

        Examples
        --------
        >>> import numpy as np
        >>> from nestkit.calibration.diagnostics import CalibrationDiagnostics
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_proba = np.array([0.2, 0.3, 0.7, 0.8])
        >>> CalibrationDiagnostics.expected_calibration_error(
        ...     y_true, y_proba, n_bins=5
        ... )  # doctest: +SKIP
        0.0...
        """
        p = extract_positive_proba(y_proba)
        bins = _make_bins(p, n_bins, strategy)
        bin_pairs = list(pairwise(bins))
        ece = 0.0
        n_total = len(y_true)

        for i, (lo, hi) in enumerate(bin_pairs):
            mask = _bin_mask(p, lo, hi, is_last=(i == len(bin_pairs) - 1))
            if mask.sum() == 0:
                continue
            bin_acc = y_true[mask].mean()
            bin_conf = p[mask].mean()
            ece += mask.sum() / n_total * abs(bin_acc - bin_conf)

        return float(ece)

    @staticmethod
    def maximum_calibration_error(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
        strategy: str = "quantile",
    ) -> float:
        """Compute the Maximum Calibration Error (MCE).

        MCE is the worst-case (maximum) absolute difference between
        observed accuracy and mean predicted confidence across all bins.

        Parameters
        ----------
        y_true : numpy.ndarray of shape (n_samples,)
            True binary labels (0 or 1).
        y_proba : numpy.ndarray of shape (n_samples,) or (n_samples, 2)
            Predicted probabilities.
        n_bins : int, default 10
            Number of bins.
        strategy : {"quantile", "uniform"}, default "quantile"
            Binning strategy.  ``"quantile"`` produces bins with
            approximately equal sample counts; ``"uniform"`` produces
            equal-width bins over [0, 1].

        Returns
        -------
        float
            Maximum Calibration Error in [0, 1].

        Notes
        -----
        MCE is defined as:

        .. math::

            \\text{MCE} = \\max_{b \\in \\{1, \\ldots, B\\}}
            \\left| \\text{acc}(b) - \\text{conf}(b) \\right|

        This is useful for identifying the single worst-calibrated
        probability region.

        Examples
        --------
        >>> import numpy as np
        >>> from nestkit.calibration.diagnostics import CalibrationDiagnostics
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_proba = np.array([0.2, 0.3, 0.7, 0.8])
        >>> CalibrationDiagnostics.maximum_calibration_error(
        ...     y_true, y_proba, n_bins=5
        ... )  # doctest: +SKIP
        0.0...
        """
        p = extract_positive_proba(y_proba)
        bins = _make_bins(p, n_bins, strategy)
        bin_pairs = list(pairwise(bins))
        mce = 0.0

        for i, (lo, hi) in enumerate(bin_pairs):
            mask = _bin_mask(p, lo, hi, is_last=(i == len(bin_pairs) - 1))
            if mask.sum() == 0:
                continue
            bin_acc = y_true[mask].mean()
            bin_conf = p[mask].mean()
            mce = max(mce, abs(bin_acc - bin_conf))

        return float(mce)

    @staticmethod
    def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Compute the Brier score (mean squared error of probabilities).

        Parameters
        ----------
        y_true : numpy.ndarray of shape (n_samples,)
            True binary labels (0 or 1).
        y_proba : numpy.ndarray of shape (n_samples,) or (n_samples, 2)
            Predicted probabilities.

        Returns
        -------
        float
            Brier score in [0, 1].  Lower is better; 0 indicates
            perfect probabilistic predictions.

        Notes
        -----
        The Brier score is defined as:

        .. math::

            \\text{BS} = \\frac{1}{N} \\sum_{i=1}^{N} (p_i - y_i)^2

        It can be decomposed into reliability, resolution, and
        uncertainty (see :meth:`brier_decomposition`).

        References
        ----------
        .. [1] Brier, G.W. (1950). "Verification of Forecasts Expressed
           in Terms of Probability." *Monthly Weather Review*, 78(1),
           1--3.

        Examples
        --------
        >>> import numpy as np
        >>> from nestkit.calibration.diagnostics import CalibrationDiagnostics
        >>> CalibrationDiagnostics.brier_score(
        ...     np.array([0, 1]), np.array([0.0, 1.0])
        ... )
        0.0

        See Also
        --------
        brier_decomposition : Reliability--resolution--uncertainty
            decomposition of the Brier score.
        """
        p = extract_positive_proba(y_proba)
        return float(np.mean((p - y_true) ** 2))

    @staticmethod
    def brier_decomposition(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
        strategy: str = "quantile",
    ) -> dict:
        """Decompose the Brier score into reliability, resolution, and uncertainty.

        Parameters
        ----------
        y_true : numpy.ndarray of shape (n_samples,)
            True binary labels (0 or 1).
        y_proba : numpy.ndarray of shape (n_samples,) or (n_samples, 2)
            Predicted probabilities.
        n_bins : int, default 10
            Number of bins.
        strategy : {"quantile", "uniform"}, default "quantile"
            Binning strategy.  ``"quantile"`` produces bins with
            approximately equal sample counts; ``"uniform"`` produces
            equal-width bins over [0, 1].

        Returns
        -------
        dict
            Dictionary with keys:

            * ``"reliability"`` -- Calibration component (lower is
              better).  Measures how close the predicted probabilities
              are to the observed frequencies within each bin.
            * ``"resolution"`` -- Resolution component (higher is
              better).  Measures how much the per-bin frequencies deviate
              from the overall base rate.
            * ``"uncertainty"`` -- Uncertainty component.  Equal to
              ``p_bar * (1 - p_bar)`` where ``p_bar`` is the base rate.
              This is independent of the model.

        Notes
        -----
        The decomposition satisfies:

        .. math::

            \\text{BS} = \\text{Reliability} - \\text{Resolution}
            + \\text{Uncertainty}

        References
        ----------
        .. [1] Murphy, A.H. (1973). "A New Vector Partition of the
           Probability Score." *Journal of Applied Meteorology*, 12(4),
           595--600.

        Examples
        --------
        >>> import numpy as np
        >>> from nestkit.calibration.diagnostics import CalibrationDiagnostics
        >>> decomp = CalibrationDiagnostics.brier_decomposition(
        ...     np.array([0, 0, 1, 1]),
        ...     np.array([0.1, 0.2, 0.8, 0.9]),
        ... )  # doctest: +SKIP
        >>> decomp.keys()  # doctest: +SKIP
        dict_keys(['reliability', 'resolution', 'uncertainty'])

        See Also
        --------
        brier_score : The scalar Brier score.
        """
        p = extract_positive_proba(y_proba)
        n = len(y_true)
        climatology = y_true.mean()
        uncertainty = climatology * (1 - climatology)

        bins = _make_bins(p, n_bins, strategy)
        bin_pairs = list(pairwise(bins))
        reliability = 0.0
        resolution = 0.0

        for i, (lo, hi) in enumerate(bin_pairs):
            mask = _bin_mask(p, lo, hi, is_last=(i == len(bin_pairs) - 1))
            n_k = mask.sum()
            if n_k == 0:
                continue
            bin_acc = y_true[mask].mean()
            bin_conf = p[mask].mean()
            reliability += n_k / n * (bin_conf - bin_acc) ** 2
            resolution += n_k / n * (bin_acc - climatology) ** 2

        return {
            "reliability": float(reliability),
            "resolution": float(resolution),
            "uncertainty": float(uncertainty),
        }

    @staticmethod
    def reliability_diagram_data(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
        strategy: str = "quantile",
    ) -> pd.DataFrame:
        """Compute binned data for reliability (calibration) diagrams.

        Returns a DataFrame with one row per bin, suitable for plotting
        a reliability diagram (mean predicted probability vs. observed
        fraction of positives).

        Parameters
        ----------
        y_true : numpy.ndarray of shape (n_samples,)
            True binary labels (0 or 1).
        y_proba : numpy.ndarray of shape (n_samples,) or (n_samples, 2)
            Predicted probabilities.
        n_bins : int, default 10
            Number of bins.
        strategy : {"quantile", "uniform"}, default "quantile"
            Binning strategy.  ``"quantile"`` produces bins with
            approximately equal sample counts; ``"uniform"`` produces
            equal-width bins over [0, 1].

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns:

            * ``bin_lower`` -- Lower edge of the bin.
            * ``bin_upper`` -- Upper edge of the bin.
            * ``bin_mid`` -- Midpoint of the bin.
            * ``mean_predicted`` -- Mean predicted probability in the
              bin (NaN if the bin is empty).
            * ``fraction_positive`` -- Observed fraction of positive
              samples in the bin (NaN if empty).
            * ``count`` -- Number of samples in the bin.

        Examples
        --------
        >>> import numpy as np
        >>> from nestkit.calibration.diagnostics import CalibrationDiagnostics
        >>> df = CalibrationDiagnostics.reliability_diagram_data(
        ...     np.array([0, 0, 1, 1]),
        ...     np.array([0.1, 0.2, 0.8, 0.9]),
        ... )  # doctest: +SKIP
        >>> df.columns.tolist()  # doctest: +SKIP
        ['bin_lower', 'bin_upper', 'bin_mid', 'mean_predicted',
         'fraction_positive', 'count']
        """
        p = extract_positive_proba(y_proba)
        bins = _make_bins(p, n_bins, strategy)
        bin_pairs = list(pairwise(bins))
        rows = []

        for i, (lo, hi) in enumerate(bin_pairs):
            mask = _bin_mask(p, lo, hi, is_last=(i == len(bin_pairs) - 1))
            n_k = mask.sum()
            if n_k == 0:
                rows.append(
                    {
                        "bin_lower": lo,
                        "bin_upper": hi,
                        "bin_mid": (lo + hi) / 2,
                        "mean_predicted": np.nan,
                        "fraction_positive": np.nan,
                        "count": 0,
                    }
                )
            else:
                rows.append(
                    {
                        "bin_lower": lo,
                        "bin_upper": hi,
                        "bin_mid": (lo + hi) / 2,
                        "mean_predicted": float(p[mask].mean()),
                        "fraction_positive": float(y_true[mask].mean()),
                        "count": int(n_k),
                    }
                )

        return pd.DataFrame(rows)

    @staticmethod
    def compare_before_after(
        y_true: np.ndarray, raw_proba: np.ndarray, cal_proba: np.ndarray
    ) -> pd.DataFrame:
        """Side-by-side comparison of calibration metrics before and after calibration.

        Computes ECE, MCE, and Brier score for both the raw and
        calibrated predicted probabilities and returns them in a
        two-row DataFrame.

        Parameters
        ----------
        y_true : numpy.ndarray of shape (n_samples,)
            True binary labels (0 or 1).
        raw_proba : numpy.ndarray of shape (n_samples,) or (n_samples, 2)
            Raw (uncalibrated) predicted probabilities.
        cal_proba : numpy.ndarray of shape (n_samples,) or (n_samples, 2)
            Calibrated predicted probabilities.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns ``stage`` (``"raw"`` or
            ``"calibrated"``), ``ece``, ``mce``, ``brier``.

        Examples
        --------
        >>> import numpy as np
        >>> from nestkit.calibration.diagnostics import CalibrationDiagnostics
        >>> y_true = np.array([0, 0, 1, 1])
        >>> raw = np.array([0.3, 0.4, 0.6, 0.7])
        >>> cal = np.array([0.2, 0.3, 0.7, 0.8])
        >>> df = CalibrationDiagnostics.compare_before_after(
        ...     y_true, raw, cal
        ... )  # doctest: +SKIP
        >>> df["stage"].tolist()  # doctest: +SKIP
        ['raw', 'calibrated']

        See Also
        --------
        expected_calibration_error : ECE computation.
        maximum_calibration_error : MCE computation.
        brier_score : Brier score computation.
        """
        diag = CalibrationDiagnostics
        return pd.DataFrame(
            [
                {
                    "stage": "raw",
                    "ece": diag.expected_calibration_error(y_true, raw_proba),
                    "mce": diag.maximum_calibration_error(y_true, raw_proba),
                    "brier": diag.brier_score(y_true, raw_proba),
                },
                {
                    "stage": "calibrated",
                    "ece": diag.expected_calibration_error(y_true, cal_proba),
                    "mce": diag.maximum_calibration_error(y_true, cal_proba),
                    "brier": diag.brier_score(y_true, cal_proba),
                },
            ]
        )
