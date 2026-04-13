"""CV+ Mondrian conformal prediction for classification."""

from __future__ import annotations

import warnings

import numpy as np

from nestkit.conformal.results import ClassifierConformalResult


class MondrianClassifierConformal:
    """Class-conditional (Mondrian) conformal prediction sets.

    Uses nonconformity score ``s(x, y) = 1 - p_hat(y | x)`` and computes
    a separate quantile threshold per class for class-conditional coverage.
    """

    @staticmethod
    def fit(
        oof_probas: np.ndarray,
        oof_y_true: np.ndarray,
        classes: np.ndarray,
        alpha: float = 0.1,
    ) -> ClassifierConformalResult:
        """Compute per-class q-hat from OOF nonconformity scores.

        Parameters
        ----------
        oof_probas : ndarray of shape (n_cal, n_classes)
            Out-of-fold predicted probabilities (calibrated or raw).
        oof_y_true : ndarray of shape (n_cal,)
            True labels for calibration samples.
        classes : ndarray of shape (n_classes,)
            Ordered class labels (matching columns of ``oof_probas``).
        alpha : float
            Significance level (default 0.1 for 90% target coverage).

        Returns
        -------
        ClassifierConformalResult
        """
        n_classes = len(classes)
        qhat = np.empty(n_classes)
        n_cal_per_class = np.empty(n_classes, dtype=int)

        for idx, cls in enumerate(classes):
            mask = oof_y_true == cls
            n_c = int(mask.sum())
            n_cal_per_class[idx] = n_c

            if n_c < 2:
                qhat[idx] = 1.0
                if n_c == 0:
                    warnings.warn(
                        f"Class {cls} has no calibration samples; "
                        f"setting q_hat=1.0 (always include).",
                        UserWarning,
                        stacklevel=2,
                    )
                continue

            # Nonconformity scores: s = 1 - p_hat(true_class | x)
            scores = 1.0 - oof_probas[mask, idx]

            # Finite-sample corrected quantile (exact order statistic)
            k = int(np.ceil((n_c + 1) * (1 - alpha)))
            if k > n_c:
                qhat[idx] = 1.0
            else:
                sorted_scores = np.sort(scores)
                qhat[idx] = float(sorted_scores[k - 1])

        return ClassifierConformalResult(
            alpha=alpha,
            qhat_per_class=qhat,
            n_calibration_per_class=n_cal_per_class,
        )

    @staticmethod
    def predict(
        probas: np.ndarray,
        conformal_result: ClassifierConformalResult,
        classes: np.ndarray | None = None,
    ) -> dict:
        """Generate prediction sets for test data.

        A class ``c`` is included in the prediction set for sample ``i``
        if ``1 - p_hat(c | x_i) <= q_hat[c]``.

        Parameters
        ----------
        probas : ndarray of shape (n_test, n_classes)
            Predicted probabilities (calibrated or raw, matching ``fit``).
        conformal_result : ClassifierConformalResult
            Result from :meth:`fit`.
        classes : ndarray of shape (n_classes,) or None, optional
            Ordered class labels matching the columns of ``probas``.
            When provided, prediction sets contain actual class labels
            instead of column indices.

        Returns
        -------
        dict
            ``prediction_sets``: list of lists (class labels if *classes*
            is provided, otherwise column indices).
            ``set_sizes``: ndarray of int.
            ``is_uncertain``: bool ndarray (True where ``set_size > 1``).
            ``is_empty``: bool ndarray (True where ``set_size == 0``).
        """
        qhat = conformal_result.qhat_per_class

        # Vectorised inclusion: (n_test, n_classes) bool matrix
        included = (1.0 - probas) <= qhat[np.newaxis, :]

        set_sizes = included.sum(axis=1).astype(int)

        if classes is not None:
            prediction_sets = [classes[included[i]].tolist() for i in range(len(probas))]
        else:
            prediction_sets = [np.where(included[i])[0].tolist() for i in range(len(probas))]

        return {
            "prediction_sets": prediction_sets,
            "set_sizes": set_sizes,
            "is_uncertain": set_sizes > 1,
            "is_empty": set_sizes == 0,
        }
