Architecture
============

Standard cross-validation conflates hyperparameter tuning with performance
estimation, producing optimistically biased scores. Nested cross-validation
fixes this by separating the two concerns into an outer loop (unbiased
evaluation) and an inner loop (tuning).


The 4-Phase Pipeline
--------------------

For every outer fold, nestkit executes:

.. list-table::
   :header-rows: 1
   :widths: 10 25 65

   * - Phase
     - Name
     - Description
   * - 1
     - Inner CV search
     - ``GridSearchCV`` or ``RandomizedSearchCV`` on the outer training fold
       to select the best hyperparameters.
   * - 2
     - Post-hoc calibration
     - *(Classification, opt-in.)* Fit a calibrator on inner out-of-fold
       predictions so that calibrated probabilities are leakage-free.
   * - 3
     - Threshold optimization
     - *(Classification, opt-in.)* Find the decision threshold that maximizes
       a chosen criterion on inner out-of-fold predictions.
   * - 4
     - Refit & evaluate
     - Refit with the best hyperparameters on the full outer training fold,
       then score on the held-out outer test fold.

The outer test fold is never used for tuning, calibration, or thresholding.


Probability Calibration
-----------------------

Many classifiers produce poorly calibrated probabilities. The predicted
confidence does not match the true positive rate. nestkit integrates post-hoc
calibration directly into Phase 2, fitting on inner out-of-fold predictions to
avoid leakage.

Supported methods: **Platt scaling** (``"sigmoid"``), **isotonic regression**
(``"isotonic"``), **beta calibration** (``"beta"``), and **Venn-ABERS**
(``"venn_abers"``).

.. code-block:: python

   ncv = NestedCVClassifier(
       estimator=GradientBoostingClassifier(),
       param_grid={...},
       calibration_method="isotonic",
       ...
   )
   ncv.fit(X, y)
   print(ncv.results_.calibration_summary_)


Threshold Optimization
----------------------

The default 0.5 threshold is rarely optimal for imbalanced classes or
asymmetric costs. nestkit searches for a better threshold on inner out-of-fold
predictions (Phase 3), keeping the outer test fold untouched.

Five built-in criteria: **Youden's J**, **F-beta**, **cost-sensitive**,
**balanced accuracy**, and **precision-at-recall**. Two strategies: ``"pooled"``
(single threshold) and ``"fold_specific"`` (averaged per-fold thresholds).

.. code-block:: python

   ncv = NestedCVClassifier(
       estimator=GradientBoostingClassifier(),
       param_grid={...},
       calibration_method="isotonic",
       threshold_strategy="pooled",
       threshold_criterion="youden",
       ...
   )
   ncv.fit(X, y)
   print(ncv.results_.summary_optimized_)


Model Comparison
----------------

Comparing models with a naive paired *t*-test inflates Type I error because
CV fold scores are correlated. nestkit's :class:`~nestkit.comparison.NestedCVComparator`
provides proper statistical tests.

- **Nadeau-Bengio corrected *t*-test**: adjusts variance for fold overlap.
- **Bayesian correlated *t*-test with ROPE**: posterior probabilities for
  "A better", "equivalent", or "B better".
- **Holm-Bonferroni correction**: controls family-wise error for multi-model
  comparisons.

.. code-block:: python

   from nestkit.comparison import NestedCVComparator

   comparator = NestedCVComparator()
   comparator.add("RF", rf_results)
   comparator.add("GBM", gbm_results)

   print(comparator.corrected_paired_ttest(metric="roc_auc", model_a="GBM", model_b="RF"))
   print(comparator.bayesian_comparison(metric="roc_auc", model_a="GBM", model_b="RF", rope=0.01))


Diagnostics
-----------

**Hyperparameter stability**: if the inner search selects different
hyperparameters across folds, the model may be too sensitive or the data
too small. :class:`~nestkit.diagnostics.HyperparameterStability` reports selection
frequency, entropy, agreement rate, and pairwise Jaccard similarity.

**Generalization gap**: ``results.generalization_gap_`` compares inner CV
scores to outer test scores per fold; a large gap signals overfitting during
tuning.

.. code-block:: python

   from nestkit.diagnostics import HyperparameterStability

   stab = HyperparameterStability(ncv.results_.best_params_per_fold_)
   print(stab.summary())


Feature Importance
------------------

Single-split importance scores are unreliable.
:class:`~nestkit.importance.FeatureImportanceAggregator` extracts importances from each
outer fold estimator (model-native or SHAP), aggregates them,
and reports the **Nogueira stability index** to measure top-*k* feature
consistency across folds.

.. code-block:: python

   from nestkit.importance import FeatureImportanceAggregator

   agg = FeatureImportanceAggregator(ncv.results_, method="auto", feature_names=names)
   agg.compute()
   print(agg.summary_)
   print(f"Stability (k=10): {agg.stability_index(top_k=10):.3f}")


Callbacks
---------

The callback system hooks into the pipeline lifecycle for progress tracking,
checkpointing, and logging.

Built-in callbacks: :class:`~nestkit.callbacks.ProgressCallback`,
:class:`~nestkit.callbacks.CheckpointCallback`,
:class:`~nestkit.callbacks.LoggingCallback`.

.. code-block:: python

   from nestkit.callbacks import ProgressCallback, CheckpointCallback

   ncv = NestedCVClassifier(
       ...,
       callbacks=[ProgressCallback(n_outer_folds=5), CheckpointCallback(path="./ckpt")],
   )


Plotting
--------

nestkit includes 25+ plotting functions. All accept an
optional ``ax`` parameter and return a matplotlib ``Axes``.

Categories: fold scores, ROC/PR curves, confusion matrices, residuals,
calibration diagrams, threshold sensitivity, comparison plots, critical
difference diagrams, feature importance, and SHAP summaries.

.. code-block:: python

   from nestkit.plotting import plot_roc_curves, plot_calibration_curves

   plot_roc_curves(ncv.results_)
   plot_calibration_curves(ncv.results_)

Install plotting support with ``pip install nestkit[plotting]``.
See the :ref:`API Reference <api-reference>` for the full function list.
