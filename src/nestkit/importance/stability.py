"""Feature selection stability metrics.

Provides the Nogueira et al. (2018) stability index for quantifying
the consistency of top-k feature selection across cross-validation
folds.

References
----------
.. [1] Nogueira, S., Sechidis, K., and Brown, G. (2018). "On the
       Stability of Feature Selection Algorithms." *JMLR*, 18(174),
       1--54.
"""

from __future__ import annotations

import numpy as np


def nogueira_stability_index(
    importances_matrix: np.ndarray,
    top_k: int = 10,
) -> float:
    """Compute the Nogueira et al. (2018) stability index for top-k feature selection.

    Converts continuous importance scores into binary selection vectors
    (1 if a feature is in the top-k, 0 otherwise) and then measures
    the agreement across folds using a chance-corrected consistency
    metric.

    Parameters
    ----------
    importances_matrix : numpy.ndarray
        Importance scores of shape ``(n_folds, n_features)``.
    top_k : int, default=10
        Number of top features to select per fold.

    Returns
    -------
    float
        Stability index in ``[-1, 1]``.  A value of **1** indicates
        perfect agreement (identical top-k sets across all folds),
        **0** corresponds to random selection, and negative values
        indicate less-than-chance agreement.

    Notes
    -----
    Let :math:`\\hat{p}_f` be the fraction of folds in which feature
    *f* is selected (selection frequency) and
    :math:`\\bar{p} = (1/d)\\sum_f \\hat{p}_f`.  The index is:

    .. math::

        \\text{Stability} = 1 -
        \\frac{
            \\frac{1}{d} \\sum_{f=1}^{d}
            \\frac{M}{M-1} \\hat{p}_f (1 - \\hat{p}_f)
        }{
            \\bar{p}\\,(1 - \\bar{p})
        }

    where *M* is the number of folds and *d* the number of features.

    When only a single fold is available the index is trivially 1.0.

    References
    ----------
    .. [1] Nogueira, S., Sechidis, K., and Brown, G. (2018). "On the
           Stability of Feature Selection Algorithms." *JMLR*, 18(174),
           1--54.

    See Also
    --------
    nestkit.importance.aggregator.FeatureImportanceAggregator.stability_index

    Examples
    --------
    >>> import numpy as np
    >>> M = np.array([[0.5, 0.3, 0.2], [0.4, 0.35, 0.25]])
    >>> nogueira_stability_index(M, top_k=2)  # doctest: +SKIP
    1.0
    """
    n_folds, n_features = importances_matrix.shape

    if n_folds < 2:
        return 1.0

    # Binary selection matrix
    selections = np.zeros((n_folds, n_features), dtype=int)
    for i in range(n_folds):
        top_indices = np.argsort(-importances_matrix[i])[:top_k]
        selections[i, top_indices] = 1

    # Selection frequency per feature
    pf = selections.mean(axis=0)
    pf_bar = pf.mean()

    if pf_bar == 0 or pf_bar == 1:
        return 1.0

    numerator = (1.0 / n_features) * np.sum((n_folds / (n_folds - 1)) * pf * (1 - pf))
    denominator = pf_bar * (1 - pf_bar)

    if denominator == 0:
        return 1.0

    return float(1.0 - numerator / denominator)
