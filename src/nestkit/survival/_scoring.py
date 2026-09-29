"""Survival-specific scoring functions and metrics.

Provides concordance index scorers (Harrell and Uno), integrated
Brier score computation, and a Kaplan-Meier censoring estimator
used as a building block for IPCW-based metrics.
"""

from __future__ import annotations

import warnings

import numpy as np

_PAIR_CHUNK_BYTES = 16 * 1024 * 1024


def _kaplan_meier_censoring(event: np.ndarray, duration: np.ndarray):
    """Fit a Kaplan-Meier estimator for the censoring distribution.

    The censoring distribution treats *censoring* as the event of
    interest: ``G(t) = P(C > t)`` where ``C`` is the censoring time.

    Parameters
    ----------
    event : array-like of shape (n_samples,)
        Binary event indicator (1 = event, 0 = censored).
    duration : array-like of shape (n_samples,)
        Observed times.

    Returns
    -------
    km_func : callable
        A right-continuous step function ``G(t)`` returning the
        censoring survival probability at time ``t`` (scalar or
        array).  The exact product-limit value is returned, including
        ``0`` past the last censoring time; callers must therefore
        guard against division by zero.  The attribute
        ``km_func.support_`` gives the largest time at which ``G`` is
        still strictly positive (``inf`` if it never reaches zero),
        which is the horizon beyond which IPC weights are undefined.

    Notes
    -----
    Earlier versions clipped ``G(t)`` to a small positive ``eps``.
    That silently turned undefined IPC weights into weights of order
    ``1e10`` instead of excluding them, so the value is now returned
    unclipped and truncation is handled explicitly by the callers
    (see :func:`_truncate_ibs_eval_times`).
    """
    event = np.asarray(event, dtype=np.float64).ravel()
    duration = np.asarray(duration, dtype=np.float64).ravel()
    n_samples = duration.shape[0]

    censoring_event = (event == 0).astype(np.float64)

    unique_times, inverse = np.unique(duration, return_inverse=True)
    inverse = inverse.ravel()
    n_times = unique_times.shape[0]

    counts = np.bincount(inverse, minlength=n_times).astype(np.float64)
    events_at = np.bincount(inverse, weights=censoring_event, minlength=n_times)
    at_risk = n_samples - np.concatenate(([0.0], np.cumsum(counts)[:-1]))

    factors = np.ones(n_times, dtype=np.float64)
    positive = at_risk > 0
    factors[positive] = 1.0 - events_at[positive] / at_risk[positive]
    survival = np.concatenate(([1.0], np.cumprod(factors)))
    km_times = np.concatenate(([0.0], unique_times))

    def km_func(t):
        """Evaluate G(t) = P(C > t)."""
        t = np.asarray(t, dtype=np.float64)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)

        idx = np.searchsorted(km_times, t, side="right") - 1
        idx = np.clip(idx, 0, survival.shape[0] - 1)
        result = survival[idx]
        return float(result[0]) if scalar else result

    positive_idx = np.flatnonzero(survival > 0)
    if positive_idx.size == 0:
        km_func.support_ = 0.0
    elif positive_idx[-1] == survival.shape[0] - 1:
        km_func.support_ = float("inf")
    else:
        km_func.support_ = float(km_times[positive_idx[-1]])

    return km_func


def concordance_index_scorer(estimator, X, y) -> float:
    """Harrell's concordance index scorer for sklearn compatibility.

    Follows the ``(estimator, X, y) -> float`` scorer protocol
    expected by :class:`~sklearn.model_selection.GridSearchCV`.

    Parameters
    ----------
    estimator : estimator with ``predict`` method
        Fitted survival estimator returning risk scores.
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples, 2)
        Survival target ``[event, duration]``.

    Returns
    -------
    float
        Harrell's concordance index in ``[0, 1]``.
    """
    from nestkit.survival._wrapper import _check_lifelines

    ll = _check_lifelines()
    risk_scores = estimator.predict(X)
    y = np.asarray(y)
    event = y[:, 0].astype(bool)
    duration = y[:, 1]
    return ll.utils.concordance_index(duration, -risk_scores, event)


def _compute_uno_c_index(
    train_event: np.ndarray,
    train_duration: np.ndarray,
    test_event: np.ndarray,
    test_duration: np.ndarray,
    risk_scores: np.ndarray,
    tau: float | None = None,
) -> float:
    """Compute Uno's concordance index (IPCW-weighted).

    More robust to censoring than Harrell's C-index because it uses
    inverse-probability-of-censoring weights derived from the
    *training* data.

    Parameters
    ----------
    train_event : ndarray of shape (n_train,)
        Training event indicators.
    train_duration : ndarray of shape (n_train,)
        Training durations.
    test_event : ndarray of shape (n_test,)
        Test event indicators.
    test_duration : ndarray of shape (n_test,)
        Test durations.
    risk_scores : ndarray of shape (n_test,)
        Predicted risk scores (higher = higher risk).
    tau : float or None, default=None
        Truncation time.  Pairs with ``T_i > tau`` are excluded to
        avoid instability in the tail where ``G(t) -> 0``.  Defaults
        to the maximum uncensored event time in the training data.

    Returns
    -------
    float
        Uno's concordance index in ``[0, 1]``.  Returns ``0.5`` when
        no comparable pair carries a defined weight.

    Notes
    -----
    Comparable pairs are enumerated in a vectorized fashion over
    chunks of "case" rows, so peak memory is ``O(chunk * n_test)``
    rather than ``O(n_test ** 2)``.
    """
    train_event = np.asarray(train_event).ravel()
    train_duration = np.asarray(train_duration).ravel()
    test_event = np.asarray(test_event, dtype=bool).ravel()
    test_duration = np.asarray(test_duration, dtype=np.float64).ravel()
    risk_scores = np.asarray(risk_scores, dtype=np.float64).ravel()

    G = _kaplan_meier_censoring(train_event, train_duration)

    if tau is None:
        uncensored_times = train_duration[train_event.astype(bool)]
        tau = float(np.max(uncensored_times)) if len(uncensored_times) > 0 else np.inf

    case_mask = test_event & (test_duration <= tau)
    if not np.any(case_mask):
        return 0.5

    case_duration = test_duration[case_mask]
    case_risk = risk_scores[case_mask]
    g_case = np.asarray(G(case_duration), dtype=np.float64)

    defined = g_case > 0
    if not np.any(defined):
        return 0.5
    case_duration = case_duration[defined]
    case_risk = case_risk[defined]
    g_case = g_case[defined]

    weights = 1.0 / (g_case * g_case)

    n_test = test_duration.shape[0]
    n_cases = case_duration.shape[0]
    chunk = max(1, int(_PAIR_CHUNK_BYTES // (8 * max(n_test, 1))))

    numerator = 0.0
    denominator = 0.0
    for start in range(0, n_cases, chunk):
        stop = min(start + chunk, n_cases)
        d_i = case_duration[start:stop, None]
        r_i = case_risk[start:stop, None]
        w_i = weights[start:stop, None]

        comparable = test_duration[None, :] > d_i
        concordant = np.where(
            r_i > risk_scores[None, :],
            1.0,
            np.where(r_i == risk_scores[None, :], 0.5, 0.0),
        )
        weighted = w_i * comparable
        denominator += float(weighted.sum())
        numerator += float((weighted * concordant).sum())

    if denominator == 0.0:
        return 0.5

    return numerator / denominator


def uno_c_index_scorer(estimator, X, y) -> float:
    """Uno's concordance index scorer for sklearn compatibility.

    Within cross-validation, the censoring distribution is estimated
    from the validation fold (pragmatic approximation since the
    scorer protocol does not expose training data).

    Parameters
    ----------
    estimator : estimator with ``predict`` method
        Fitted survival estimator returning risk scores.
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples, 2)
        Survival target ``[event, duration]``.

    Returns
    -------
    float
        Uno's concordance index in ``[0, 1]``.
    """
    risk_scores = estimator.predict(X)
    y = np.asarray(y)
    event = y[:, 0]
    duration = y[:, 1]
    return _compute_uno_c_index(event, duration, event, duration, risk_scores)


def _truncate_ibs_eval_times(
    eval_times: np.ndarray,
    train_event: np.ndarray,
    train_duration: np.ndarray,
    *,
    warn: bool = True,
) -> np.ndarray:
    """Drop IBS evaluation times where IPC weights are undefined.

    The IPCW Brier score weights each contribution by ``1 / G(t)``.
    Past the horizon where the training censoring estimator reaches
    zero those weights are undefined, and evaluating there produces
    arbitrarily large scores rather than a value in ``[0, 1]``.

    Parameters
    ----------
    eval_times : array-like of shape (n_times,)
        Candidate evaluation times.
    train_event : ndarray of shape (n_train,)
        Training event indicators.
    train_duration : ndarray of shape (n_train,)
        Training durations.
    warn : bool, default=True
        Whether to emit a :class:`UserWarning` when times are dropped.

    Returns
    -------
    ndarray
        The subset of *eval_times* that lies within the horizon where
        the censoring estimator is strictly positive.  May be empty.
    """
    eval_times = np.asarray(eval_times, dtype=np.float64).ravel()
    G = _kaplan_meier_censoring(train_event, train_duration)

    keep = eval_times <= G.support_
    n_dropped = int(np.sum(~keep))
    if n_dropped and warn:
        warnings.warn(
            f"Dropped {n_dropped} of {eval_times.shape[0]} IBS evaluation time(s) "
            f"beyond t={G.support_:.6g}, where the censoring distribution estimated "
            "on the training data reaches zero and inverse-probability-of-censoring "
            "weights are undefined. The integrated Brier score is reported over the "
            "remaining times.",
            UserWarning,
            stacklevel=2,
        )
    return eval_times[keep]


def _compute_ibs(
    train_event: np.ndarray,
    train_duration: np.ndarray,
    test_event: np.ndarray,
    test_duration: np.ndarray,
    survival_fn_at_times: np.ndarray,
    eval_times: np.ndarray,
) -> float:
    """Compute the integrated Brier score (IBS) using IPCW.

    Parameters
    ----------
    train_event : ndarray of shape (n_train,)
        Training event indicators.
    train_duration : ndarray of shape (n_train,)
        Training durations.
    test_event : ndarray of shape (n_test,)
        Test event indicators.
    test_duration : ndarray of shape (n_test,)
        Test durations.
    survival_fn_at_times : ndarray of shape (n_times, n_test)
        Predicted survival probabilities.  Row ``k`` contains
        ``S(eval_times[k] | X_i)`` for each test sample ``i``.
    eval_times : ndarray of shape (n_times,)
        Time points at which to evaluate the Brier score.  Times past
        the horizon where the training censoring estimator vanishes
        should be removed first with :func:`_truncate_ibs_eval_times`.

    Returns
    -------
    float
        Integrated Brier score (lower is better), or ``nan`` if
        *eval_times* is empty.

    Notes
    -----
    Each time-specific Brier score is the IPCW-weighted sum of the
    squared errors divided by the **full** test-set size:

    .. math::

        BS(t) = \\frac{1}{n} \\sum_i \\left[
            \\frac{S(t \\mid x_i)^2 \\, \\mathbb{1}(T_i \\le t, \\delta_i = 1)}{G(T_i)}
          + \\frac{(1 - S(t \\mid x_i))^2 \\, \\mathbb{1}(T_i > t)}{G(t)}
        \\right]

    Samples censored before ``t`` contribute zero: the ``1 / G``
    weights already inflate the retained samples to account for them,
    so dividing by the number of retained samples instead of ``n``
    would apply the censoring correction twice and inflate the score.
    """
    train_event = np.asarray(train_event).ravel()
    train_duration = np.asarray(train_duration).ravel()
    test_event = np.asarray(test_event, dtype=bool).ravel()
    test_duration = np.asarray(test_duration, dtype=np.float64).ravel()
    eval_times = np.asarray(eval_times, dtype=np.float64).ravel()
    survival_fn_at_times = np.asarray(survival_fn_at_times, dtype=np.float64)

    if eval_times.shape[0] == 0:
        return float("nan")

    G = _kaplan_meier_censoring(train_event, train_duration)
    n_test = test_duration.shape[0]
    if n_test == 0:
        return float("nan")

    g_event = np.asarray(G(test_duration), dtype=np.float64)
    g_times = np.asarray(G(eval_times), dtype=np.float64)

    brier_scores = np.empty(eval_times.shape[0], dtype=np.float64)

    for k, t in enumerate(eval_times):
        s_t = survival_fn_at_times[k]
        contrib = np.zeros(n_test, dtype=np.float64)

        is_case = (test_duration <= t) & test_event & (g_event > 0)
        contrib[is_case] = s_t[is_case] ** 2 / g_event[is_case]

        if g_times[k] > 0:
            is_control = test_duration > t
            contrib[is_control] = (1.0 - s_t[is_control]) ** 2 / g_times[k]

        brier_scores[k] = contrib.sum() / n_test

    if eval_times.shape[0] < 2:
        return float(brier_scores[0])

    time_range = eval_times[-1] - eval_times[0]
    if time_range <= 0:
        return float(np.mean(brier_scores))

    # ``trapz`` was renamed to ``trapezoid`` in NumPy 2.0 and later removed
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(brier_scores, eval_times) / time_range)
