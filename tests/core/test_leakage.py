"""Data leakage tests  -  critical for nested CV correctness."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from nestkit.classifier import NestedCVClassifier


def test_outer_test_never_in_inner():
    """Inner CV training indices must never overlap with outer test indices."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=0)

    ncv = NestedCVClassifier(
        estimator=RandomForestClassifier(random_state=0, n_estimators=5),
        param_grid={"max_depth": [2, 3]},
        outer_cv=3,
        inner_cv=2,
        return_estimator=True,
    )

    captured_inner_X = []
    from sklearn.model_selection import GridSearchCV

    _original_fit = GridSearchCV.fit

    def _spy_fit(self, X_inner, y_inner, **kwargs):
        captured_inner_X.append(set(map(tuple, X_inner.tolist())))
        return _original_fit(self, X_inner, y_inner, **kwargs)

    with patch.object(GridSearchCV, "fit", _spy_fit):
        ncv.fit(X, y)

    outer_test_sets = []
    for fr in ncv.results_.fold_results_:
        test_rows = set(map(tuple, X[fr.test_indices].tolist()))
        outer_test_sets.append(test_rows)

    for fold_idx, (inner_rows, test_rows) in enumerate(zip(captured_inner_X, outer_test_sets)):
        overlap = inner_rows & test_rows
        assert len(overlap) == 0, (
            f"Fold {fold_idx}: {len(overlap)} outer-test samples leaked into inner CV"
        )


@pytest.mark.filterwarnings("ignore:The groups parameter is ignored by StratifiedKFold")
def test_groups_respected():
    """With GroupKFold, no group should appear in both train and test of the same outer fold."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=0)
    groups = np.repeat(np.arange(10), 10)

    ncv = NestedCVClassifier(
        estimator=RandomForestClassifier(random_state=0, n_estimators=5),
        param_grid={"max_depth": [2, 3]},
        outer_cv=GroupKFold(n_splits=5),
        inner_cv=2,
        return_estimator=True,
    )
    ncv.fit(X, y, groups=groups)

    for fr in ncv.results_.fold_results_:
        train_groups = set(groups[fr.train_indices])
        test_groups = set(groups[fr.test_indices])
        overlap = train_groups & test_groups
        assert len(overlap) == 0, (
            f"Fold {fr.fold_idx}: groups {overlap} appear in both train and test"
        )


def test_groups_respected_in_calibration_cv():
    """With GroupKFold, calibration OOF loop must also respect group boundaries."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=0)
    groups = np.repeat(np.arange(10), 10)

    ncv = NestedCVClassifier(
        estimator=RandomForestClassifier(random_state=0, n_estimators=5),
        param_grid={"max_depth": [2, 3]},
        outer_cv=GroupKFold(n_splits=5),
        inner_cv=GroupKFold(n_splits=2),
        calibration_method="isotonic",
        return_estimator=True,
    )

    captured_cal_splits = []

    original_group_split = GroupKFold.split

    def _spy_group_split(self, X, y=None, groups=None):
        captured_cal_splits.append(groups is not None)
        return original_group_split(self, X, y, groups)

    with patch.object(GroupKFold, "split", _spy_group_split):
        ncv.fit(X, y, groups=groups)

    assert any(captured_cal_splits), "Calibration CV loop did not pass groups to GroupKFold.split"
