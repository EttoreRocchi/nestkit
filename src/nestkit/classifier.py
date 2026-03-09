"""Nested cross-validation estimator for classification tasks.

Extends :class:`~nestkit._base._BaseNestedCV` with optional post-hoc
probability calibration (Platt scaling, isotonic regression, beta
calibration, Venn-ABERS) and decision-threshold optimization.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import check_cv

from nestkit._base import _BaseNestedCV
from nestkit._constants import _EPS
from nestkit._validation import (
    extract_positive_proba,
    validate_calibration_method,
    validate_threshold_params,
)
from nestkit.calibration.calibrators import PostHocCalibrator
from nestkit.calibration.diagnostics import CalibrationDiagnostics
from nestkit.results.classifier_results import ClassifierOuterFoldResult, ClassifierResults
from nestkit.thresholding.criteria import (
    balanced_accuracy_criterion,
    cost_sensitive,
    f_beta_criterion,
    precision_at_recall,
    youden_j,
)
from nestkit.thresholding.strategies import (
    FoldSpecificThreshold,
    PooledThreshold,
)

logger = logging.getLogger("nestkit")


class NestedCVClassifier(_BaseNestedCV):
    """Nested cross-validation for classification tasks.

    Supports binary and multiclass classification. Extends
    :class:`~nestkit._base._BaseNestedCV` with optional post-hoc
    probability calibration and decision-threshold optimization.
    Both features are disabled by default and must be explicitly enabled.

    When calibration is enabled, out-of-fold (OOF) predictions from the
    inner CV are used to fit a calibrator, which is then applied to the
    outer test-set probabilities. When threshold optimization is enabled,
    the optimal decision boundary is selected on the calibrated (or raw)
    OOF probabilities.

    Parameters
    ----------
    estimator : estimator object
        A scikit-learn compatible classifier that implements ``fit``
        and ``predict_proba``. Cloned for each outer fold.
    param_grid : dict or list of dict
        Hyperparameter search space. See
        :class:`~sklearn.model_selection.GridSearchCV`.
    search_strategy : {'grid', 'random', 'bayesian'}, default='grid'
        Inner hyperparameter search strategy.
    outer_cv : int, cross-validation generator, or iterable, default=5
        Outer cross-validation splitting strategy.
    inner_cv : int, cross-validation generator, or iterable, default=5
        Inner cross-validation splitting strategy.
    scoring : str, callable, list, tuple, or dict, default=None
        Scoring metric(s) for the inner search.
    refit : bool or str, default=True
        Whether to refit on the full outer training set.
    return_train_score : bool, default=False
        Whether to include training scores in inner CV results.
    return_estimator : bool, default=True
        Whether to store fitted estimators per outer fold.
    error_score : 'raise' or numeric, default='raise'
        Value assigned on inner CV fitting errors.
    n_jobs_outer : int or None, default=None
        Number of parallel jobs for outer folds.
    n_jobs_inner : int or None, default=None
        Number of parallel jobs for inner search.
    verbose : int, default=0
        Verbosity level.
    random_state : int, RandomState instance, or None, default=None
        Random state for reproducibility.
    callbacks : list of callback objects or None, default=None
        :class:`~nestkit.FoldCallback` instances for monitoring.
    pre_dispatch : int or str, default='2*n_jobs'
        Controls job dispatch for parallel execution.
    calibration_method : {'sigmoid', 'isotonic', 'beta', 'venn_abers'} or None, default=None
        Post-hoc calibration method. If ``None``, no calibration is
        applied. ``'sigmoid'`` corresponds to Platt scaling,
        ``'isotonic'`` to isotonic regression, ``'beta'`` to beta
        calibration, and ``'venn_abers'`` to Venn-ABERS prediction.
    threshold_strategy : {'pooled', 'fold_specific'} or None, default=None
        Threshold optimization strategy. If ``None``, no threshold
        optimization is performed. ``'pooled'`` selects a single
        threshold from all OOF predictions; ``'fold_specific'`` selects
        a per-fold threshold.
    threshold_criterion : str or callable, default='youden'
        Criterion for threshold selection. Built-in options:
        ``'youden'``, ``'f_beta'``, ``'cost'``, ``'balanced_accuracy'``,
        ``'precision_at_recall'``. A custom callable must accept
        ``(y_true, y_proba, threshold)`` and return a ``float`` to be
        maximised.
    threshold_beta : float, default=1.0
        Beta parameter for the F-beta criterion. Only used when
        ``threshold_criterion='f_beta'``.
    cost_matrix : array-like of shape (2, 2) or None, default=None
        Cost matrix ``[[TN_cost, FP_cost], [FN_cost, TP_cost]]`` for
        cost-sensitive threshold optimization. Required when
        ``threshold_criterion='cost'``.
    min_recall : float or None, default=None
        Minimum recall constraint for the ``'precision_at_recall'``
        criterion. Required when ``threshold_criterion='precision_at_recall'``.
    calibration_cv : int, cross-validation generator, or None, default=None
        CV strategy for generating OOF calibration predictions. If
        ``None``, uses the same ``inner_cv`` strategy.  Note that when
        ``inner_cv`` is an integer, a **new** splitter instance is
        created for the calibration OOF loop, which may produce
        different fold assignments than the inner hyperparameter search.

    Notes
    -----
    Enabling calibration and/or threshold optimization roughly doubles
    computation time per outer fold, as the inner CV folds must be re-run
    to produce OOF probability estimates for the calibrator and threshold
    optimizer.

    For multiclass tasks, calibration is applied independently per class
    using a one-vs-rest (OVR) decomposition.  After calibration the
    per-class probabilities are renormalized to sum to 1.  Because each
    calibrator is fitted on a marginal binary problem, the resulting
    multiclass probabilities may not be jointly well-calibrated -- this
    is a known limitation of OVR calibration approaches.

    Examples
    --------
    Basic classification:

    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from nestkit import NestedCVClassifier
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> ncv = NestedCVClassifier(
    ...     estimator=RandomForestClassifier(random_state=42),
    ...     param_grid={"n_estimators": [50, 100], "max_depth": [3, 5]},
    ...     outer_cv=5, inner_cv=3, random_state=42,
    ... )
    >>> ncv.fit(X, y)  # doctest: +SKIP
    >>> print(ncv.results_.summary_default_)  # doctest: +SKIP

    With calibration and threshold optimization:

    >>> ncv = NestedCVClassifier(
    ...     estimator=RandomForestClassifier(random_state=42),
    ...     param_grid={"n_estimators": [50, 100]},
    ...     outer_cv=5, inner_cv=3,
    ...     calibration_method="isotonic",
    ...     threshold_strategy="pooled",
    ...     threshold_criterion="youden",
    ...     random_state=42,
    ... )
    >>> ncv.fit(X, y)  # doctest: +SKIP

    See Also
    --------
    nestkit.NestedCVRegressor : Regression-specific nested CV.
    nestkit.calibration.PostHocCalibrator : Standalone calibrator.
    nestkit.thresholding.strategies.PooledThreshold : Pooled threshold strategy.
    """

    def __init__(
        self,
        estimator,
        param_grid,
        *,
        search_strategy="grid",
        outer_cv=5,
        inner_cv=5,
        scoring=None,
        refit=True,
        return_train_score=False,
        return_estimator=True,
        error_score="raise",
        n_jobs_outer=None,
        n_jobs_inner=None,
        verbose=0,
        random_state=None,
        callbacks=None,
        pre_dispatch="2*n_jobs",
        calibration_method=None,
        threshold_strategy=None,
        threshold_criterion="youden",
        threshold_beta=1.0,
        cost_matrix=None,
        min_recall=None,
        calibration_cv=None,
    ):
        super().__init__(
            estimator=estimator,
            param_grid=param_grid,
            search_strategy=search_strategy,
            outer_cv=outer_cv,
            inner_cv=inner_cv,
            scoring=scoring,
            refit=refit,
            return_train_score=return_train_score,
            return_estimator=return_estimator,
            error_score=error_score,
            n_jobs_outer=n_jobs_outer,
            n_jobs_inner=n_jobs_inner,
            verbose=verbose,
            random_state=random_state,
            callbacks=callbacks,
            pre_dispatch=pre_dispatch,
        )
        self.calibration_method = calibration_method
        self.threshold_strategy = threshold_strategy
        self.threshold_criterion = threshold_criterion
        self.threshold_beta = threshold_beta
        self.cost_matrix = cost_matrix
        self.min_recall = min_recall
        self.calibration_cv = calibration_cv

    def fit(self, X, y, groups=None, **fit_params):
        """Run nested cross-validation with optional calibration and thresholding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels.
        groups : array-like of shape (n_samples,) or None, default=None
            Group labels for group-aware CV splitters.
        **fit_params : dict
            Additional keyword arguments forwarded to the estimator's
            ``fit`` method.

        Returns
        -------
        self
            The fitted estimator. Results are accessible via
            :attr:`results_`.

        Raises
        ------
        ValueError
            If calibration or threshold parameters are invalid.
        """
        validate_calibration_method(self.calibration_method)
        validate_threshold_params(
            self.threshold_strategy,
            self.threshold_criterion,
            self.cost_matrix,
            self.min_recall,
        )
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        return super().fit(X, y, groups=groups, **fit_params)

    def _build_results_container(self) -> type:
        return ClassifierResults

    def _post_inner_processing(self, search, X_train, y_train, groups_train, **fit_params) -> dict:
        """Phase 2 + Phase 3: calibration and threshold optimization.

        Note: The OOF loop uses ``search.best_params_`` which were selected
        using all of ``X_train``. The OOF validation folds therefore
        influenced hyperparameter selection. This is a widely accepted
        approximation -- the alternative (triple-nested CV) is
        computationally prohibitive for most practical use cases.
        """
        artifacts: dict[str, Any] = {
            "calibrator": None,
            "calibrators_ovr": None,
            "optimal_threshold": 0.5,
            "optimal_thresholds_ovr": None,
            "fold_thresholds": None,
            "threshold_result": None,
            "oof_probas_raw": None,
            "oof_probas_calibrated": None,
            "oof_y_true": None,
            "oof_calibration_diagnostics": None,
        }

        # Fast path
        if self.calibration_method is None and self.threshold_strategy is None:
            return artifacts

        # Slow path: collect inner OOF predictions (always refit)
        cal_cv = check_cv(self.calibration_cv or self.inner_cv, y_train, classifier=True)
        best_params = search.best_params_
        base_estimator = clone(self.estimator).set_params(**best_params)

        oof_probas: list[np.ndarray] = []
        oof_y_true: list[np.ndarray] = []

        for inner_train_idx, inner_val_idx in cal_cv.split(X_train, y_train, groups_train):
            est_j = clone(base_estimator)
            est_j.fit(X_train[inner_train_idx], y_train[inner_train_idx], **fit_params)
            oof_probas.append(est_j.predict_proba(X_train[inner_val_idx]))
            oof_y_true.append(y_train[inner_val_idx])

        oof_probas_all = np.concatenate(oof_probas)
        oof_y_all = np.concatenate(oof_y_true)
        artifacts["oof_probas_raw"] = oof_probas_all
        artifacts["oof_y_true"] = oof_y_all

        n_classes = oof_probas_all.shape[1] if oof_probas_all.ndim == 2 else 2
        is_binary = n_classes == 2

        # --- Phase 2: Calibration ---
        if self.calibration_method is not None:
            if is_binary:
                calibrator = PostHocCalibrator(method=self.calibration_method)
                p_pos = extract_positive_proba(oof_probas_all)
                calibrator.fit(p_pos, oof_y_all)
                artifacts["calibrator"] = calibrator
                cal_probas_all = calibrator.predict_proba(p_pos)
                cal_probas_per_fold = [
                    calibrator.predict_proba(extract_positive_proba(p)) for p in oof_probas
                ]
            else:
                calibrators_ovr = []
                cal_probas_all = np.zeros_like(oof_probas_all)
                cal_probas_per_fold = [np.zeros_like(p) for p in oof_probas]

                for c in range(n_classes):
                    y_binary = (oof_y_all == self.classes_[c]).astype(int)
                    p_c = oof_probas_all[:, c]
                    cal_c = PostHocCalibrator(method=self.calibration_method)
                    cal_c.fit(p_c, y_binary)
                    calibrators_ovr.append(cal_c)
                    cal_probas_all[:, c] = cal_c.predict_proba(p_c)[:, 1]

                    for j, p_fold in enumerate(oof_probas):
                        cal_probas_per_fold[j][:, c] = cal_c.predict_proba(p_fold[:, c])[:, 1]

                # Renormalize
                row_sums = cal_probas_all.sum(axis=1, keepdims=True)
                cal_probas_all /= row_sums + _EPS
                for j in range(len(cal_probas_per_fold)):
                    rs = cal_probas_per_fold[j].sum(axis=1, keepdims=True)
                    cal_probas_per_fold[j] /= rs + _EPS

                artifacts["calibrators_ovr"] = calibrators_ovr

            # Calibration diagnostics (binary only for now)
            if is_binary:
                raw_p = extract_positive_proba(oof_probas_all)
                cal_p = extract_positive_proba(cal_probas_all)
                diag = CalibrationDiagnostics
                artifacts["oof_calibration_diagnostics"] = {
                    "ece_raw": diag.expected_calibration_error(oof_y_all, raw_p),
                    "ece_calibrated": diag.expected_calibration_error(oof_y_all, cal_p),
                    "mce_raw": diag.maximum_calibration_error(oof_y_all, raw_p),
                    "mce_calibrated": diag.maximum_calibration_error(oof_y_all, cal_p),
                    "brier_raw": diag.brier_score(oof_y_all, raw_p),
                    "brier_calibrated": diag.brier_score(oof_y_all, cal_p),
                }
        else:
            cal_probas_all = oof_probas_all
            cal_probas_per_fold = oof_probas

        artifacts["oof_probas_calibrated"] = cal_probas_all

        # --- Phase 3: Threshold optimization ---
        if self.threshold_strategy is not None:
            criterion_fn = self._resolve_criterion()
            criterion_name = (
                self.threshold_criterion
                if isinstance(self.threshold_criterion, str)
                else getattr(self.threshold_criterion, "__name__", "custom")
            )

            if is_binary:
                cal_p_pos_per_fold = [extract_positive_proba(p) for p in cal_probas_per_fold]
                if self.threshold_strategy == "fold_specific":
                    tr = FoldSpecificThreshold.optimize(
                        oof_y_true, cal_p_pos_per_fold, criterion_fn, criterion_name
                    )
                else:
                    tr = PooledThreshold.optimize(
                        oof_y_true, cal_p_pos_per_fold, criterion_fn, criterion_name
                    )
                artifacts["optimal_threshold"] = tr.optimal_threshold
                artifacts["threshold_result"] = tr
            else:
                # Multiclass OVR: apply threshold strategy per class
                thresholds_ovr = []
                for c in range(n_classes):
                    y_binary_per_fold = [(y == self.classes_[c]).astype(int) for y in oof_y_true]
                    p_c_per_fold = [p[:, c] for p in cal_probas_per_fold]
                    if self.threshold_strategy == "fold_specific":
                        tr_c = FoldSpecificThreshold.optimize(
                            y_binary_per_fold, p_c_per_fold, criterion_fn, criterion_name
                        )
                    else:
                        tr_c = PooledThreshold.optimize(
                            y_binary_per_fold, p_c_per_fold, criterion_fn, criterion_name
                        )
                    thresholds_ovr.append(tr_c.optimal_threshold)
                artifacts["optimal_thresholds_ovr"] = np.array(thresholds_ovr)

        return artifacts

    def _evaluate_outer_fold(self, estimator, X_test, y_test, artifacts) -> dict:
        """Evaluate best estimator on outer test fold."""
        raw_proba = estimator.predict_proba(X_test)
        n_classes = raw_proba.shape[1]
        is_binary = n_classes == 2

        # Apply calibration
        cal_proba = self._apply_calibration(raw_proba, artifacts)

        # Default predictions
        if is_binary:
            effective_proba = extract_positive_proba(cal_proba)
            y_pred_default = (effective_proba >= 0.5).astype(int)
        else:
            y_pred_default = np.argmax(cal_proba, axis=1)

        scores_default = self._compute_metrics(y_test, y_pred_default, cal_proba, is_binary)
        cm_default = confusion_matrix(y_test, y_pred_default)

        has_calibration = (
            artifacts["calibrator"] is not None or artifacts.get("calibrators_ovr") is not None
        )

        result = {
            "y_true": y_test,
            "y_proba_raw": raw_proba,
            "y_proba_calibrated": cal_proba if has_calibration else None,
            "y_pred_default": y_pred_default,
            "scores_default": scores_default,
            "confusion_matrix_default": cm_default,
            "y_pred_optimized": None,
            "scores_optimized": None,
            "confusion_matrix_optimized": None,
        }

        # Optimized predictions
        has_threshold = (
            artifacts.get("threshold_result") is not None
            or artifacts.get("optimal_thresholds_ovr") is not None
        )
        if has_threshold:
            if is_binary:
                threshold = artifacts["optimal_threshold"]
                y_pred_opt = (effective_proba >= threshold).astype(int)
            else:
                thresholds = artifacts["optimal_thresholds_ovr"]
                above = cal_proba >= thresholds[np.newaxis, :]
                n_above = above.sum(axis=1)
                y_pred_opt = np.where(
                    n_above == 1,
                    np.argmax(above, axis=1),
                    np.argmax(cal_proba, axis=1),
                )
            result["y_pred_optimized"] = y_pred_opt
            result["scores_optimized"] = self._compute_metrics(
                y_test, y_pred_opt, cal_proba, is_binary
            )
            result["confusion_matrix_optimized"] = confusion_matrix(y_test, y_pred_opt)

        return result

    def _apply_calibration(self, raw_proba: np.ndarray, artifacts: dict) -> np.ndarray:
        """Apply calibration to raw probabilities."""
        if artifacts["calibrator"] is not None:
            return artifacts["calibrator"].predict_proba(extract_positive_proba(raw_proba))
        if artifacts.get("calibrators_ovr") is not None:
            cal_proba = np.zeros_like(raw_proba)
            for c, cal_c in enumerate(artifacts["calibrators_ovr"]):
                cal_proba[:, c] = cal_c.predict_proba(raw_proba[:, c])[:, 1]
            row_sums = cal_proba.sum(axis=1, keepdims=True)
            return cal_proba / (row_sums + _EPS)
        return raw_proba

    def _compute_metrics(self, y_true, y_pred, y_proba, is_binary: bool) -> dict[str, float]:
        """Compute classification metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        }

        if is_binary:
            metrics["precision"] = precision_score(y_true, y_pred, zero_division=0.0)
            metrics["recall"] = recall_score(y_true, y_pred, zero_division=0.0)
            metrics["f1"] = f1_score(y_true, y_pred, zero_division=0.0)
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, extract_positive_proba(y_proba))
            except ValueError:
                metrics["roc_auc"] = float("nan")
        else:
            avg = "macro"
            metrics["precision"] = precision_score(y_true, y_pred, average=avg, zero_division=0.0)
            metrics["recall"] = recall_score(y_true, y_pred, average=avg, zero_division=0.0)
            metrics["f1"] = f1_score(y_true, y_pred, average=avg, zero_division=0.0)
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average=avg)
            except ValueError:
                metrics["roc_auc"] = float("nan")

        return metrics

    def _resolve_criterion(self):
        """Resolve threshold criterion to callable."""
        if callable(self.threshold_criterion):
            return self.threshold_criterion
        mapping = {
            "youden": lambda: youden_j,
            "f_beta": lambda: f_beta_criterion(self.threshold_beta),
            "cost": lambda: cost_sensitive(self.cost_matrix),
            "balanced_accuracy": lambda: balanced_accuracy_criterion,
            "precision_at_recall": lambda: precision_at_recall(self.min_recall),
        }
        return mapping[self.threshold_criterion]()

    def _build_fold_result(self, **kwargs) -> ClassifierOuterFoldResult:
        artifacts = kwargs.pop("artifacts")
        eval_result = kwargs.pop("eval_result")

        return ClassifierOuterFoldResult(
            fold_idx=kwargs["fold_idx"],
            train_indices=kwargs["train_idx"],
            test_indices=kwargs["test_idx"],
            best_params=kwargs["best_params"],
            best_inner_score=kwargs["best_inner_score"],
            inner_cv_results=kwargs["inner_cv_results"],
            fit_time=kwargs["fit_time"],
            score_time=kwargs["score_time"],
            fitted_estimator=kwargs["estimator"],
            y_true=eval_result["y_true"],
            y_proba_raw=eval_result["y_proba_raw"],
            y_pred_default=eval_result["y_pred_default"],
            outer_scores_default=eval_result["scores_default"],
            confusion_matrix_default=eval_result["confusion_matrix_default"],
            y_proba_calibrated=eval_result["y_proba_calibrated"],
            calibration_method=self.calibration_method,
            calibrator=artifacts.get("calibrator"),
            oof_calibration_diagnostics=artifacts.get("oof_calibration_diagnostics"),
            y_pred_optimized=eval_result["y_pred_optimized"],
            outer_scores_optimized=eval_result["scores_optimized"],
            confusion_matrix_optimized=eval_result["confusion_matrix_optimized"],
            threshold_result=artifacts.get("threshold_result"),
        )
