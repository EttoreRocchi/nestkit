"""Post-hoc probability calibration."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from nestkit._constants import _EPS
from nestkit._validation import extract_positive_proba

logger = logging.getLogger("nestkit")


class PostHocCalibrator:
    """Unified interface for post-hoc probability calibration.

    Binary-only. For multiclass, use one calibrator per class via OVR
    decomposition at the NestedCVClassifier level.

    Parameters
    ----------
    method : {"sigmoid", "isotonic", "beta", "venn_abers"}
        Calibration method.  ``"sigmoid"`` applies logistic recalibration
        on probability logits (sometimes called temperature scaling on
        probabilities).  This differs from classical Platt scaling, which
        operates on raw decision function scores rather than probabilities.
    """

    def __init__(self, method: str = "sigmoid"):
        self.method = method
        self._calibrator = None
        self._is_fitted = False

    def fit(self, y_proba: np.ndarray, y_true: np.ndarray) -> PostHocCalibrator:
        """Fit calibration mapping from uncalibrated probs to calibrated.

        Parameters
        ----------
        y_proba : array of shape (n_samples,) or (n_samples, 2)
            Uncalibrated predicted probabilities.
        y_true : array of shape (n_samples,)
            True binary labels.
        """
        p = extract_positive_proba(y_proba)

        if self.method == "sigmoid":
            self._fit_sigmoid(p, y_true)
        elif self.method == "isotonic":
            self._fit_isotonic(p, y_true)
        elif self.method == "beta":
            self._fit_beta(p, y_true)
        elif self.method == "venn_abers":
            self._fit_venn_abers(p, y_true)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self._is_fitted = True
        return self

    def predict_proba(self, y_proba: np.ndarray) -> np.ndarray:
        """Apply calibration mapping.

        Parameters
        ----------
        y_proba : array of shape (n_samples,) or (n_samples, 2) or (n_samples,)
            Uncalibrated probabilities.

        Returns
        -------
        calibrated_proba : array of shape (n_samples, 2)
        """
        if not self._is_fitted:
            raise RuntimeError("PostHocCalibrator is not fitted.")

        p = extract_positive_proba(y_proba)

        if self.method == "sigmoid":
            cal_p = self._predict_sigmoid(p)
        elif self.method == "isotonic":
            cal_p = self._predict_isotonic(p)
        elif self.method == "beta":
            cal_p = self._predict_beta(p)
        elif self.method == "venn_abers":
            cal_p = self._predict_venn_abers(p)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        cal_p = np.clip(cal_p, 0, 1)
        return np.column_stack([1 - cal_p, cal_p])

    # --- Sigmoid (logistic recalibration on probability logits) ---

    def _fit_sigmoid(self, p: np.ndarray, y: np.ndarray) -> None:
        p_clipped = np.clip(p, _EPS, 1 - _EPS)
        logits = np.log(p_clipped / (1 - p_clipped))
        self._calibrator = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        self._calibrator.fit(logits.reshape(-1, 1), y)

    def _predict_sigmoid(self, p: np.ndarray) -> np.ndarray:
        p_clipped = np.clip(p, _EPS, 1 - _EPS)
        logits = np.log(p_clipped / (1 - p_clipped))
        return self._calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]

    # --- Isotonic ---

    def _fit_isotonic(self, p: np.ndarray, y: np.ndarray) -> None:
        self._calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        self._calibrator.fit(p, y)

    def _predict_isotonic(self, p: np.ndarray) -> np.ndarray:
        return self._calibrator.predict(p)

    # --- Beta calibration (Kull et al., 2017) ---

    def _fit_beta(self, p: np.ndarray, y: np.ndarray) -> None:
        # 3-parameter beta calibration: logit(q) = a * log(p) + b * log(1-p) + c
        # Fit via logistic regression on log(p) and log(1-p)
        p_clipped = np.clip(p, _EPS, 1 - _EPS)
        features = np.column_stack([np.log(p_clipped), np.log(1 - p_clipped)])
        self._calibrator = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        self._calibrator.fit(features, y)

    def _predict_beta(self, p: np.ndarray) -> np.ndarray:
        p_clipped = np.clip(p, _EPS, 1 - _EPS)
        features = np.column_stack([np.log(p_clipped), np.log(1 - p_clipped)])
        return self._calibrator.predict_proba(features)[:, 1]

    # --- Venn-ABERS ---

    def _fit_venn_abers(self, p: np.ndarray, y: np.ndarray) -> None:
        # Store calibration data for Venn-ABERS inductive prediction
        sort_idx = np.argsort(p)
        self._va_scores = p[sort_idx]
        self._va_labels = y[sort_idx]

    def _predict_venn_abers(self, p: np.ndarray) -> np.ndarray:
        cal_probs = np.zeros(len(p))

        for i, score in enumerate(p):
            # Compute isotonic regression with score inserted as label=0 and label=1
            p0 = self._va_isotonic_with(score, 0)
            p1 = self._va_isotonic_with(score, 1)
            # Venn-ABERS midpoint
            cal_probs[i] = p1 / (1 - p0 + p1 + _EPS)

        return cal_probs

    def _va_isotonic_with(self, score: float, label: int) -> float:
        """Insert (score, label) into calibration set and return isotonic prediction."""
        scores = np.append(self._va_scores, score)
        labels = np.append(self._va_labels, label)
        sort_idx = np.argsort(scores)
        scores_sorted = scores[sort_idx]
        labels_sorted = labels[sort_idx]

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(scores_sorted, labels_sorted)
        return float(iso.predict([score])[0])
