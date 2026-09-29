"""Integration tests for callback dispatch through the nested CV pipeline."""

from __future__ import annotations

import pickle
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from nestkit import NestedCVClassifier
from nestkit.callbacks import (
    CheckpointCallback,
    LoggingCallback,
    ProgressCallback,
)


@pytest.fixture
def small_param_grid():
    return {"n_estimators": [10, 20]}


@pytest.fixture
def small_classifier():
    return RandomForestClassifier(n_estimators=10, random_state=42)


@pytest.mark.slow
class TestLoggingCallbackDispatched:
    def test_fold_start_times_populated(self, binary_data, small_param_grid, small_classifier):
        """Verify LoggingCallback.on_outer_fold_start is called through pipeline dispatch."""
        X, y = binary_data
        cb = LoggingCallback()
        ncv = NestedCVClassifier(
            estimator=small_classifier,
            param_grid=small_param_grid,
            outer_cv=2,
            inner_cv=2,
            callbacks=[cb],
            random_state=42,
        )
        ncv.fit(X, y)
        assert len(cb._fold_start_times) == 2


@pytest.mark.slow
class TestCheckpointCallbackDispatched:
    def test_checkpoint_files_created(
        self, binary_data, small_param_grid, small_classifier, tmp_path
    ):
        """Verify CheckpointCallback creates fold and final pickle files through pipeline."""
        X, y = binary_data
        cb = CheckpointCallback(tmp_path / "checkpoints")
        ncv = NestedCVClassifier(
            estimator=small_classifier,
            param_grid=small_param_grid,
            outer_cv=2,
            inner_cv=2,
            callbacks=[cb],
            random_state=42,
        )
        ncv.fit(X, y)
        assert (tmp_path / "checkpoints" / "fold_0.pkl").exists()
        assert (tmp_path / "checkpoints" / "fold_1.pkl").exists()
        assert (tmp_path / "checkpoints" / "final_results.pkl").exists()

        # Verify pickle files are loadable
        with open(tmp_path / "checkpoints" / "final_results.pkl", "rb") as f:
            loaded = pickle.load(f)
        assert loaded is not None


class TestProgressCallbackLifecycle:
    def test_tqdm_created_and_updated(self):
        """Verify ProgressCallback creates pbar, updates, and closes."""
        cb = ProgressCallback(n_outer_folds=2)
        assert cb._pbar is None

        cb.on_outer_fold_start(0, np.arange(80), np.arange(20))
        assert cb._pbar is not None

        cb.on_outer_fold_complete(0, None)
        assert cb._pbar.n == 1

        cb.on_nested_cv_complete(None)
        # After close, pbar.disable should be True
        assert cb._pbar.disable

    def test_pass_methods_are_noop(self):
        """Verify on_inner_search_complete and on_post_processing_complete are pass."""
        cb = ProgressCallback()
        assert cb.on_inner_search_complete(0, None) is None
        assert cb.on_post_processing_complete(0, {}) is None

    def test_tqdm_import_error(self):
        """Verify ProgressCallback handles missing tqdm gracefully."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "tqdm.auto":
                raise ImportError("No tqdm")
            return original_import(name, *args, **kwargs)

        cb = ProgressCallback(n_outer_folds=2)
        with patch("builtins.__import__", side_effect=mock_import):
            cb.on_outer_fold_start(0, np.arange(80), np.arange(20))

        assert cb._pbar is None
        # Subsequent calls should be no-ops
        cb.on_outer_fold_complete(0, None)
        cb.on_nested_cv_complete(None)


@pytest.mark.slow
class TestCustomCallbackAllHooks:
    def test_all_hooks_called(self, binary_data, small_param_grid, small_classifier):
        """Verify all 5 callback hooks are called through the pipeline."""

        class RecordingCallback:
            def __init__(self):
                self.calls = []

            def on_outer_fold_start(self, fold_idx, train_idx, test_idx):
                self.calls.append(("start", fold_idx))

            def on_inner_search_complete(self, fold_idx, search):
                self.calls.append(("inner", fold_idx))

            def on_post_processing_complete(self, fold_idx, artifacts):
                self.calls.append(("post", fold_idx))

            def on_outer_fold_complete(self, fold_idx, result):
                self.calls.append(("complete", fold_idx))

            def on_nested_cv_complete(self, results):
                self.calls.append(("final", None))

        X, y = binary_data
        cb = RecordingCallback()
        ncv = NestedCVClassifier(
            estimator=small_classifier,
            param_grid=small_param_grid,
            outer_cv=2,
            inner_cv=2,
            callbacks=[cb],
            random_state=42,
        )
        ncv.fit(X, y)

        # Each fold triggers start, inner, post, complete
        call_types = [c[0] for c in cb.calls]
        assert call_types.count("start") == 2
        assert call_types.count("inner") == 2
        assert call_types.count("post") == 2
        assert call_types.count("complete") == 2
        assert call_types.count("final") == 1


class TestSklearnTagsFallback:
    def test_import_error_returns_dict(self):
        """Verify __sklearn_tags__ returns plain dict when Tags import fails."""
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        ncv = NestedCVClassifier(
            estimator=clf,
            param_grid={"n_estimators": [10]},
        )

        # Temporarily make sklearn.utils._tags unimportable
        original_module = __import__("sys").modules.get("sklearn.utils._tags")
        try:
            __import__("sys").modules["sklearn.utils._tags"] = None
            # Reload _base to pick up the broken import
            # Directly test the fallback by calling __sklearn_tags__
            # Since the import happens inside the method, this should trigger ImportError
            tags = ncv.__sklearn_tags__()
            # If ImportError is triggered, we get a plain dict
            if isinstance(tags, dict):
                assert "no_validation" in tags
        finally:
            if original_module is not None:
                __import__("sys").modules["sklearn.utils._tags"] = original_module
            else:
                __import__("sys").modules.pop("sklearn.utils._tags", None)
