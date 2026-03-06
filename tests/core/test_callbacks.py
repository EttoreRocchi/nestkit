"""Tests for callback protocol and built-in callbacks."""

from __future__ import annotations

import pickle
import tempfile
import typing
from pathlib import Path

import numpy as np

from nestkit.callbacks import (
    CheckpointCallback,
    FoldCallback,
    LoggingCallback,
    ProgressCallback,
)


def test_fold_callback_protocol():
    """LoggingCallback, ProgressCallback, CheckpointCallback implement the FoldCallback protocol."""
    with tempfile.TemporaryDirectory() as tmpdir:
        instances = [
            LoggingCallback(),
            ProgressCallback(n_outer_folds=3),
            CheckpointCallback(path=tmpdir),
        ]
        for cb in instances:
            assert isinstance(cb, FoldCallback), (
                f"{type(cb).__name__} does not satisfy FoldCallback protocol"
            )


def test_logging_callback():
    """Calling every method on LoggingCallback must not raise."""
    cb = LoggingCallback()
    train_idx = np.arange(80)
    test_idx = np.arange(80, 100)

    cb.on_outer_fold_start(0, train_idx, test_idx)

    class _FakeSearch:
        best_params_: typing.ClassVar[dict] = {"n_estimators": 10}
        best_score_ = 0.9

    cb.on_inner_search_complete(0, _FakeSearch())
    cb.on_post_processing_complete(0, {})
    cb.on_outer_fold_complete(0, object())

    class _FakeResults:
        n_outer_folds_ = 3

    cb.on_nested_cv_complete(_FakeResults())


def test_checkpoint_callback():
    """CheckpointCallback saves pickle files to the target directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cb = CheckpointCallback(path=tmpdir)
        train_idx = np.arange(80)
        test_idx = np.arange(80, 100)

        cb.on_outer_fold_start(0, train_idx, test_idx)
        cb.on_inner_search_complete(0, object())
        cb.on_post_processing_complete(0, {})

        fold_result = {"fold_idx": 0, "score": 0.95}
        cb.on_outer_fold_complete(0, fold_result)

        fold_file = Path(tmpdir) / "fold_0.pkl"
        assert fold_file.exists(), "fold_0.pkl was not created"

        with open(fold_file, "rb") as f:
            loaded = pickle.load(f)
        assert loaded == fold_result

        cb.on_nested_cv_complete({"summary": "ok"})
        final_file = Path(tmpdir) / "final_results.pkl"
        assert final_file.exists(), "final_results.pkl was not created"
