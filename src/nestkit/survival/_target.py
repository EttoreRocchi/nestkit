"""Survival target construction and normalization helpers.

Provides utilities for creating and converting survival targets
(event indicator + duration) into the canonical 2-column ndarray
format used internally by :class:`~nestkit.NestedCVSurvival`.
"""

from __future__ import annotations

import numpy as np


def make_survival_target(
    event: np.ndarray,
    duration: np.ndarray,
) -> np.ndarray:
    """Create a survival target array from event and duration arrays.

    Parameters
    ----------
    event : array-like of shape (n_samples,)
        Binary event indicator (1 = event observed, 0 = censored).
    duration : array-like of shape (n_samples,)
        Observed time-to-event or censoring time. Must be positive.

    Returns
    -------
    y : ndarray of shape (n_samples, 2)
        Survival target with columns ``[event, duration]``.

    Raises
    ------
    ValueError
        If ``event`` is not binary or ``duration`` contains
        non-positive values.

    Examples
    --------
    >>> import numpy as np
    >>> from nestkit.survival import make_survival_target
    >>> event = np.array([1, 0, 1, 1, 0])
    >>> duration = np.array([5.0, 10.0, 3.0, 7.0, 12.0])
    >>> y = make_survival_target(event, duration)
    >>> y.shape
    (5, 2)
    """
    event = np.asarray(event, dtype=np.float64).ravel()
    duration = np.asarray(duration, dtype=np.float64).ravel()

    if event.shape[0] != duration.shape[0]:
        raise ValueError(
            f"event and duration must have the same length, "
            f"got {event.shape[0]} and {duration.shape[0]}"
        )

    unique_events = np.unique(event)
    if not np.all(np.isin(unique_events, [0.0, 1.0])):
        raise ValueError(f"event must be binary (0 or 1), got unique values {unique_events}")

    if np.any(duration <= 0):
        raise ValueError("duration must contain only positive values")

    return np.column_stack([event, duration])


def _normalize_survival_target(y) -> np.ndarray:
    """Normalize a survival target to 2-column ndarray ``[event, duration]``.

    Accepts multiple input formats:

    1. **2-column ndarray**: passed through after validation.
    2. **Structured array** with fields ``'event'`` and ``'duration'``.
    3. **pandas DataFrame** with columns ``'event'`` and ``'duration'``.

    Parameters
    ----------
    y : array-like
        Survival target in any supported format.

    Returns
    -------
    ndarray of shape (n_samples, 2)
        Normalized target with columns ``[event, duration]``.

    Raises
    ------
    ValueError
        If the input format is not recognized or lacks required
        fields/columns.
    """
    # DataFrame with 'event' and 'duration' columns
    if hasattr(y, "columns"):
        missing = {"event", "duration"} - set(y.columns)
        if missing:
            raise ValueError(
                f"DataFrame target must have 'event' and 'duration' columns, missing: {missing}"
            )
        return np.column_stack(
            [
                y["event"].to_numpy(dtype=np.float64),
                y["duration"].to_numpy(dtype=np.float64),
            ]
        )

    y = np.asarray(y)

    # Structured array with 'event' and 'duration' fields
    if y.dtype.names is not None:
        missing = {"event", "duration"} - set(y.dtype.names)
        if missing:
            raise ValueError(
                f"Structured array target must have 'event' and 'duration' "
                f"fields, missing: {missing}"
            )
        return np.column_stack(
            [
                y["event"].astype(np.float64),
                y["duration"].astype(np.float64),
            ]
        )

    # 2-column ndarray
    if y.ndim == 2 and y.shape[1] == 2:
        return y.astype(np.float64)

    raise ValueError(
        "Survival target must be one of: "
        "(1) a 2-column ndarray [event, duration], "
        "(2) a structured array with 'event' and 'duration' fields, or "
        "(3) a DataFrame with 'event' and 'duration' columns. "
        f"Got array with shape {y.shape} and dtype {y.dtype}."
    )
