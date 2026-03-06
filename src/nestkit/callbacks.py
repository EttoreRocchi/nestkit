"""Fold callback protocol and built-in callback implementations.

Defines the :class:`FoldCallback` runtime-checkable protocol that all
callbacks must satisfy, together with three ready-made implementations:

* :class:`ProgressCallback` -- tqdm progress bar.
* :class:`LoggingCallback` -- structured ``logging``-based messages.
* :class:`CheckpointCallback` -- pickle intermediate fold results.
Custom callbacks need only implement the five hook methods specified by
:class:`FoldCallback`.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger("nestkit")


@runtime_checkable
class FoldCallback(Protocol):
    """Runtime-checkable protocol for nested CV fold callbacks.

    Any object that implements the five hook methods below can be used
    as a callback.  The nested CV engine calls these hooks at well-
    defined points during execution.

    Methods
    -------
    on_outer_fold_start(fold_idx, train_idx, test_idx)
        Called before inner search begins for an outer fold.
    on_inner_search_complete(fold_idx, search)
        Called after the inner hyperparameter search finishes.
    on_post_processing_complete(fold_idx, artifacts)
        Called after any post-processing (e.g., threshold tuning).
    on_outer_fold_complete(fold_idx, result)
        Called after the outer fold evaluation is complete.
    on_nested_cv_complete(results)
        Called once after all outer folds have been processed.

    Examples
    --------
    >>> class MyCallback:  # doctest: +SKIP
    ...     def on_outer_fold_start(self, fold_idx, train_idx, test_idx):
    ...         print(f"Starting fold {fold_idx}")
    ...     def on_inner_search_complete(self, fold_idx, search): ...
    ...     def on_post_processing_complete(self, fold_idx, artifacts): ...
    ...     def on_outer_fold_complete(self, fold_idx, result): ...
    ...     def on_nested_cv_complete(self, results): ...

    See Also
    --------
    ProgressCallback, LoggingCallback, CheckpointCallback
    """

    def on_outer_fold_start(
        self, fold_idx: int, train_idx: np.ndarray, test_idx: np.ndarray
    ) -> None: ...
    def on_inner_search_complete(self, fold_idx: int, search: Any) -> None: ...
    def on_post_processing_complete(self, fold_idx: int, artifacts: dict) -> None: ...
    def on_outer_fold_complete(self, fold_idx: int, result: Any) -> None: ...
    def on_nested_cv_complete(self, results: Any) -> None: ...


class ProgressCallback:
    """Display a tqdm progress bar during nested cross-validation.

    The progress bar is created lazily on the first
    ``on_outer_fold_start`` call and advances by one step after each
    outer fold completes.  If ``tqdm`` is not installed the callback
    silently does nothing.

    Parameters
    ----------
    n_outer_folds : int or None, optional
        Total number of outer folds.  Passed as the ``total`` argument
        to ``tqdm``.  If ``None``, the progress bar will have
        indeterminate length.

    Examples
    --------
    >>> cb = ProgressCallback(n_outer_folds=5)  # doctest: +SKIP

    See Also
    --------
    LoggingCallback : Text-based logging alternative.
    """

    def __init__(self, n_outer_folds: int | None = None):
        self._n_folds = n_outer_folds
        self._pbar = None

    def on_outer_fold_start(self, fold_idx, train_idx, test_idx):
        if self._pbar is None:
            try:
                from tqdm.auto import tqdm

                self._pbar = tqdm(total=self._n_folds, desc="Outer folds")
            except ImportError:
                pass

    def on_inner_search_complete(self, fold_idx, search):
        pass

    def on_post_processing_complete(self, fold_idx, artifacts):
        pass

    def on_outer_fold_complete(self, fold_idx, result):
        if self._pbar is not None:
            self._pbar.update(1)

    def on_nested_cv_complete(self, results):
        if self._pbar is not None:
            self._pbar.close()


class CheckpointCallback:
    """Pickle intermediate fold results to disk after each outer fold.

    After every outer fold, the fold result is saved as
    ``fold_<idx>.pkl`` inside the given directory.  When the full
    nested CV completes, the final results object is saved as
    ``final_results.pkl``.

    Parameters
    ----------
    path : str or pathlib.Path
        Directory in which checkpoint files are written.  Created
        automatically (including parents) if it does not exist.

    Attributes
    ----------
    path : pathlib.Path
        Resolved checkpoint directory.

    Examples
    --------
    >>> cb = CheckpointCallback("/tmp/ncv_checkpoints")  # doctest: +SKIP

    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def on_outer_fold_start(self, fold_idx, train_idx, test_idx):
        pass

    def on_inner_search_complete(self, fold_idx, search):
        pass

    def on_post_processing_complete(self, fold_idx, artifacts):
        pass

    def on_outer_fold_complete(self, fold_idx, result):
        filepath = self.path / f"fold_{fold_idx}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(result, f)
        logger.info("Checkpointed fold %d to %s", fold_idx, filepath)

    def on_nested_cv_complete(self, results):
        filepath = self.path / "final_results.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(results, f)
        logger.info("Checkpointed final results to %s", filepath)


class LoggingCallback:
    """Emit structured log messages at each nested CV lifecycle event.

    Logs fold start (with train/test sizes), inner search completion
    (with best parameters and score), post-processing completion, fold
    completion (with elapsed time), and overall completion.

    Parameters
    ----------
    level : int, default=logging.INFO
        Python logging level for all emitted messages.

    Attributes
    ----------
    level : int
        Logging level.
    _fold_start_times : dict[int, float]
        Mapping from fold index to wall-clock start time, used to
        compute elapsed seconds.

    Examples
    --------
    >>> import logging
    >>> cb = LoggingCallback(level=logging.DEBUG)  # doctest: +SKIP

    See Also
    --------
    ProgressCallback : Visual progress bar alternative.
    """

    def __init__(self, level: int = logging.INFO):
        self.level = level
        self._fold_start_times: dict[int, float] = {}

    def on_outer_fold_start(self, fold_idx, train_idx, test_idx):
        self._fold_start_times[fold_idx] = time.time()
        logger.log(
            self.level,
            "Fold %d: started (train=%d, test=%d)",
            fold_idx,
            len(train_idx),
            len(test_idx),
        )

    def on_inner_search_complete(self, fold_idx, search):
        logger.log(
            self.level,
            "Fold %d: inner search complete, best_params=%s, best_score=%.4f",
            fold_idx,
            search.best_params_,
            search.best_score_,
        )

    def on_post_processing_complete(self, fold_idx, artifacts):
        logger.log(self.level, "Fold %d: post-processing complete", fold_idx)

    def on_outer_fold_complete(self, fold_idx, result):
        elapsed = time.time() - self._fold_start_times.get(fold_idx, time.time())
        logger.log(self.level, "Fold %d: complete (%.1fs)", fold_idx, elapsed)

    def on_nested_cv_complete(self, results):
        logger.log(self.level, "Nested CV complete: %d folds", results.n_outer_folds_)
