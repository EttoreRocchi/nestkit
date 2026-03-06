"""Built-in threshold optimization criteria.

All criterion functions follow the signature
``(y_true, y_proba, threshold) -> float`` and are designed to be
**maximised** by ``argmax``.  For cost-based criteria (where the goal is
minimisation), the returned value is negated so that ``argmax`` still
selects the optimum.

See Also
--------
nestkit.thresholding.strategies.optimize_threshold :
    Grid search that maximises a criterion over thresholds.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)


def youden_j(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> float:
    """Compute Youden's J statistic at the given threshold.

    Youden's J is defined as ``sensitivity + specificity - 1`` and
    ranges from -1 (complete misclassification) to +1 (perfect
    classification).  Maximising J yields the threshold that best
    separates the two classes.

    Parameters
    ----------
    y_true : numpy.ndarray of shape (n_samples,)
        True binary labels (0 or 1).
    y_proba : numpy.ndarray of shape (n_samples,)
        Predicted positive-class probabilities.
    threshold : float
        Decision threshold in [0, 1].

    Returns
    -------
    float
        Youden's J in [-1, 1].

    Notes
    -----
    .. math::

        J = \\text{sensitivity} + \\text{specificity} - 1
          = \\frac{TP}{TP + FN} + \\frac{TN}{TN + FP} - 1

    This is equivalent to the vertical distance between the ROC curve
    and the diagonal chance line.

    Examples
    --------
    >>> import numpy as np
    >>> from nestkit.thresholding.criteria import youden_j
    >>> youden_j(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]), 0.5)
    1.0
    """
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity + specificity - 1.0


def f_beta_criterion(beta: float = 1.0):
    """Create a criterion function that maximises the F-beta score.

    Parameters
    ----------
    beta : float, default 1.0
        The beta parameter of the F-beta score.  ``beta < 1`` weights
        precision higher; ``beta > 1`` weights recall higher.
        ``beta = 1`` gives the standard F1 score.

    Returns
    -------
    callable
        A criterion function with signature
        ``(y_true, y_proba, threshold) -> float``.

    Notes
    -----
    The F-beta score is defined as:

    .. math::

        F_\\beta = (1 + \\beta^2) \\cdot
        \\frac{\\text{precision} \\cdot \\text{recall}}
        {\\beta^2 \\cdot \\text{precision} + \\text{recall}}

    Examples
    --------
    >>> import numpy as np
    >>> from nestkit.thresholding.criteria import f_beta_criterion
    >>> f1_criterion = f_beta_criterion(beta=1.0)
    >>> f1_criterion(
    ...     np.array([0, 0, 1, 1]),
    ...     np.array([0.1, 0.4, 0.6, 0.9]),
    ...     0.5,
    ... )  # doctest: +SKIP
    1.0

    See Also
    --------
    youden_j : Alternative criterion based on sensitivity + specificity.
    """

    def _criterion(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> float:
        y_pred = (y_proba >= threshold).astype(int)
        return fbeta_score(y_true, y_pred, beta=beta, zero_division=0.0)

    _criterion.__name__ = f"f_{beta}"
    return _criterion


def cost_sensitive(cost_matrix):
    """Create a criterion that minimises expected misclassification cost.

    The returned function computes the *negative* total cost so that
    ``argmax`` corresponds to ``argmin`` of cost.

    Parameters
    ----------
    cost_matrix : array-like of shape (2, 2)
        Cost matrix ``[[C_TN, C_FP], [C_FN, C_TP]]`` where:

        * ``C_TN`` -- cost of a true negative (usually 0).
        * ``C_FP`` -- cost of a false positive.
        * ``C_FN`` -- cost of a false negative.
        * ``C_TP`` -- cost of a true positive (usually 0).

    Returns
    -------
    callable
        A criterion function with signature
        ``(y_true, y_proba, threshold) -> float`` returning negative
        total cost.

    Notes
    -----
    The total cost is:

    .. math::

        \\text{Cost} = C_{TN} \\cdot TN + C_{FP} \\cdot FP
        + C_{FN} \\cdot FN + C_{TP} \\cdot TP

    The function returns ``-\\text{Cost}`` so that maximisation via
    ``argmax`` yields the cost-minimising threshold.

    Examples
    --------
    >>> import numpy as np
    >>> from nestkit.thresholding.criteria import cost_sensitive
    >>> # FP costs 1, FN costs 5
    >>> criterion = cost_sensitive([[0, 1], [5, 0]])
    >>> criterion(
    ...     np.array([0, 0, 1, 1]),
    ...     np.array([0.1, 0.4, 0.6, 0.9]),
    ...     0.5,
    ... )  # doctest: +SKIP
    0

    See Also
    --------
    youden_j : Cost-agnostic criterion.
    """
    c_tn, c_fp = cost_matrix[0][0], cost_matrix[0][1]
    c_fn, c_tp = cost_matrix[1][0], cost_matrix[1][1]

    def _criterion(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> float:
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        total_cost = c_tn * tn + c_fp * fp + c_fn * fn + c_tp * tp
        return -total_cost

    _criterion.__name__ = "cost_sensitive"
    return _criterion


def balanced_accuracy_criterion(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> float:
    """Maximise balanced accuracy at the given threshold.

    Balanced accuracy is the arithmetic mean of sensitivity and
    specificity, equivalent to ``(Youden's J + 1) / 2``.

    Parameters
    ----------
    y_true : numpy.ndarray of shape (n_samples,)
        True binary labels (0 or 1).
    y_proba : numpy.ndarray of shape (n_samples,)
        Predicted positive-class probabilities.
    threshold : float
        Decision threshold in [0, 1].

    Returns
    -------
    float
        Balanced accuracy in [0, 1].

    Notes
    -----
    .. math::

        \\text{BA} = \\frac{\\text{sensitivity} + \\text{specificity}}{2}

    Examples
    --------
    >>> import numpy as np
    >>> from nestkit.thresholding.criteria import balanced_accuracy_criterion
    >>> balanced_accuracy_criterion(
    ...     np.array([0, 0, 1, 1]),
    ...     np.array([0.1, 0.4, 0.6, 0.9]),
    ...     0.5,
    ... )
    1.0

    See Also
    --------
    youden_j : Equivalent to ``2 * balanced_accuracy - 1``.
    """
    y_pred = (y_proba >= threshold).astype(int)
    return balanced_accuracy_score(y_true, y_pred)


def precision_at_recall(min_recall: float = 0.90):
    """Create a criterion that maximises precision subject to a minimum recall.

    Thresholds that produce a recall below ``min_recall`` receive a
    score of -1, effectively excluding them from selection.

    Parameters
    ----------
    min_recall : float, default 0.90
        Minimum acceptable recall.  Must be in (0, 1].

    Returns
    -------
    callable
        A criterion function with signature
        ``(y_true, y_proba, threshold) -> float``.  Returns
        ``precision`` when ``recall >= min_recall``, else ``-1``.

    Notes
    -----
    This implements a constrained optimisation: among all thresholds
    achieving at least ``min_recall``, select the one with the highest
    precision.  The penalty of -1 for violating the recall constraint
    ensures that ``argmax`` never selects an infeasible threshold.

    Examples
    --------
    >>> import numpy as np
    >>> from nestkit.thresholding.criteria import precision_at_recall
    >>> criterion = precision_at_recall(min_recall=0.80)
    >>> criterion(
    ...     np.array([0, 0, 1, 1, 1]),
    ...     np.array([0.1, 0.3, 0.6, 0.7, 0.9]),
    ...     0.5,
    ... )  # doctest: +SKIP
    1.0

    See Also
    --------
    f_beta_criterion : Unconstrained precision--recall trade-off.
    """

    def _criterion(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> float:
        y_pred = (y_proba >= threshold).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0.0)
        if rec < min_recall:
            return -1.0
        return precision_score(y_true, y_pred, zero_division=0.0)

    _criterion.__name__ = f"precision_at_recall_{min_recall}"
    return _criterion
