.. _architecture:

Architecture
============

This page describes what nestkit does beyond the textbook nested
cross-validation loop: where each post-hoc step is fitted, which
probabilities and residuals every reported number comes from, and the
corrections applied along the way.


Per-fold procedure
------------------

For every outer fold:

1. **Inner search** on the outer training rows (``search_strategy``:
   ``GridSearchCV``, ``RandomizedSearchCV`` or ``BayesSearchCV``).
2. **Out-of-fold pass**, only when a post-hoc step is enabled: the inner
   splits are replayed with the selected hyperparameters, one model per
   split, to collect out-of-fold (OOF) predictions for the outer training
   rows.
3. **Post-hoc fits** on those OOF predictions. Classification: calibration,
   then threshold selection and conformal quantiles, both on the calibrated
   probabilities. Regression: residual quantiles for prediction intervals.
4. **Refit and evaluation**: the selected configuration is refitted on the
   whole outer training fold, and the objects from step 3 are applied to its
   predictions on the outer test fold.

Consequences worth knowing:

- Step 2 fits one extra model per inner split, so enabling any post-hoc step
  roughly doubles the cost of a fold.
- The OOF rows of step 2 also took part in selecting the hyperparameters.
  Removing that overlap would take a third level of nesting; nestkit accepts
  it as an approximation.
- Classifiers take the splits of step 2 from ``calibration_cv`` (default:
  ``inner_cv``). An integer creates a new splitter, whose folds need not
  match those of the search. Regressors always use ``inner_cv``.
- ``scoring`` only drives the inner search. Outer folds are always scored on
  a fixed set of metrics: accuracy, balanced accuracy, precision, recall, F1
  and ROC AUC (macro-averaged when multiclass) for classifiers; MSE, RMSE,
  MAE, R\ :sup:`2` and MAPE for regressors; Harrell's C-index, Uno's C-index
  and the integrated Brier score for survival models.
- :class:`~nestkit.callbacks.FoldCallback` hooks fire at the boundaries of
  these steps: ``on_outer_fold_start``, ``on_inner_search_complete``,
  ``on_post_processing_complete``, ``on_outer_fold_complete``, and
  ``on_nested_cv_complete`` once at the end.


Classification
--------------

Calibration
^^^^^^^^^^^

``calibration_method`` (``"sigmoid"``, ``"isotonic"``, ``"beta"`` or
``"venn_abers"``) fits one calibrator per outer fold on the pooled OOF
probabilities. Multiclass problems are calibrated one-vs-rest and the
calibrated columns are renormalized to sum to one. Each calibrator sees only
its marginal binary problem, so the joint multiclass distribution is not
guaranteed to be calibrated.

Once calibration is enabled, every downstream quantity uses the calibrated
probabilities, including the *default* predictions (0.5 threshold, or argmax)
and ROC AUC in ``summary_default_``. The raw probabilities stay available in
the fold results. ``calibration_summary_`` reports ECE, MCE and Brier score
before and after calibration, measured on the outer test folds (binary
problems only).

Decision threshold
^^^^^^^^^^^^^^^^^^

``threshold_strategy`` selects the threshold on the calibrated OOF
probabilities:

- ``"pooled"``: a single search over all OOF predictions of the fold;
- ``"fold_specific"``: one search per inner split, averaged. The standard
  deviation of the per-split thresholds is kept as a stability diagnostic.

The candidates are 999 evenly spaced values in ``[0.001, 0.999]`` plus the
midpoints between consecutive distinct OOF probabilities, so every distinct
split of the OOF sample is evaluated even when the probabilities are packed
into a narrow range. ``threshold_criterion`` is one of ``"youden"``,
``"f_beta"``, ``"cost"``, ``"balanced_accuracy"``, ``"precision_at_recall"``,
or a callable ``(y_true, y_proba, threshold) -> float`` to maximize.

For multiclass problems, one threshold per class is selected one-vs-rest. A
sample gets the only class whose probability clears its threshold; when no
class or several classes do, it falls back to the argmax.

Scores at the default and at the optimized threshold are kept side by side
(``summary_default_``, ``summary_optimized_``, ``threshold_comparison()``).

Binary class labels
^^^^^^^^^^^^^^^^^^^

Binary targets may carry any labels: ``{0, 1}``, ``{1, 2}``, ``{"case",
"control"}``. ``NestedCVClassifier`` encodes them internally so that the
calibrators, threshold criteria and calibration diagnostics always see the
0/1 encoding they are defined on, and decodes predictions, conformal
prediction sets and ``results_`` back to the original labels.

Following the scikit-learn convention, ``classes_`` is sorted and
``classes_[1]`` is the positive class: the one whose probability is
thresholded, whose recall and F1 are reported, and that the calibration
diagnostics describe. With ``{"case", "control"}`` that makes ``"control"``
positive, so encode the target yourself if you need the other orientation.

Conformal prediction sets
^^^^^^^^^^^^^^^^^^^^^^^^^

With ``conformal_prediction=True``, each outer fold computes one quantile per
class (Mondrian, i.e. class-conditional) from the OOF nonconformity scores
``1 - p(y | x)``, taken as the exact order statistic
``ceil((n_c + 1)(1 - conformal_alpha))`` among the ``n_c`` OOF samples of that
class. Class ``c`` enters the prediction set when ``1 - p(c | x) <= q_c``, so
coverage is targeted for each class, not only on average. A class with too
few OOF samples for that order statistic gets ``q_c = 1`` and is always
included. Sets can be empty. ``conformal_report()`` gives, per outer fold,
the empirical coverage, the mean set size and the fractions of singleton and
empty sets.


Regression: prediction intervals
--------------------------------

With ``prediction_intervals=True``, intervals come from the *signed* OOF
residuals ``y - y_hat``, so they are asymmetric when the errors are skewed.
The two bounds are the exact order statistics at ``alpha/2`` and
``1 - alpha/2`` (``alpha = 1 - confidence_level``), with no interpolation,
added to the prediction of the refitted model.

``mondrian_bins`` makes the interval depend on the predicted value: OOF
predictions are cut into equal-frequency bins, each bin gets its own residual
quantiles, and a test point uses the bin its prediction falls in. Per-bin
coverage on the outer test folds is in ``mondrian_coverage_per_bin_``.

.. _conformal-bin-size:

Minimum calibration size
^^^^^^^^^^^^^^^^^^^^^^^^

The two order statistics above, ``floor((alpha/2)(n+1))`` and
``ceil((1-alpha/2)(n+1))``, both exist only when

.. math::

   n \ge \frac{2 - \alpha}{\alpha}

which is 39 residuals at the default ``alpha=0.05`` and 19 at
``alpha=0.1``. ``mondrian_min_bin_size`` is therefore raised to that floor
automatically, and bins are merged until each one clears it;
``mondrian_bins`` is reduced with a warning when the OOF set cannot support
the number requested. When even the whole OOF set is too small, the
corresponding bound is returned as infinite rather than clipped to the most
extreme residual, which would report a narrower interval than the coverage
guarantee allows.


Survival
--------

:class:`~nestkit.NestedCVSurvival` takes a ``[event, duration]`` target
(:func:`~nestkit.survival.make_survival_target`, or a DataFrame or structured
array with ``event`` and ``duration`` fields) and a scikit-learn compatible
survival estimator, such as :class:`~nestkit.survival.CoxPHWrapper` around
lifelines' ``CoxPHFitter``. Install with ``pip install nestkit[survival]``.

- **Splits.** An integer ``outer_cv`` or ``inner_cv`` becomes a shuffled
  K-fold stratified on the event indicator, so every fold keeps the
  censoring rate.
- **Inner scoring.** Uno's C-index by default. The scikit-learn scorer
  interface only exposes the validation rows, so during the inner search the
  censoring distribution is estimated on the validation split itself. On the
  outer fold it is estimated on the outer training rows.
- **Outer scoring.** Harrell's C-index, Uno's C-index truncated at the last
  event time observed in the outer training rows, and the integrated Brier
  score (IBS).
- **Coefficients.** ``coefficient_stability_`` gives mean and standard
  deviation of each Cox coefficient and hazard ratio across outer folds,
  under the original feature names.

IPCW and the Brier score
^^^^^^^^^^^^^^^^^^^^^^^^

Uno's C-index and the IBS weight each retained observation by ``1 / G(t)``,
where ``G`` is a Kaplan-Meier estimate of the censoring distribution fitted on
the **training** rows of the outer fold.

The Brier score at time *t* is normalized by the full test-fold size:

.. math::

   BS(t) = \frac{1}{n} \sum_i \left[
       \frac{S(t \mid x_i)^2 \, \mathbb{1}(T_i \le t, \delta_i = 1)}{G(T_i)}
     + \frac{(1 - S(t \mid x_i))^2 \, \mathbb{1}(T_i > t)}{G(t)}
   \right]

Samples censored before *t* contribute zero. Their weight is already carried
by the retained samples through the ``1 / G`` factors, so normalizing by the
number of retained samples instead of *n* would correct for censoring twice.

``G(t)`` reaches exactly zero past the largest censoring time, where the IPC
weights are undefined. Evaluation times beyond that horizon are dropped with a
``UserWarning`` rather than evaluated with an arbitrarily large weight, so the
reported IBS always stays on the ``[0, 1]`` probability scale.

.. _ibs-time-grid:

Where the IBS is evaluated
^^^^^^^^^^^^^^^^^^^^^^^^^^

By default the IBS is integrated over the 10th to 90th percentiles (9 points)
of the uncensored event times; ``ibs_time_grid`` chooses which rows they are
taken from:

- ``'per_fold'`` (default): the training rows of each outer fold. Nothing
  from the test fold influences the evaluation, but the domains differ
  slightly between folds, so the mean in ``summary_default_`` averages Brier
  scores over close but not identical domains.
- ``'global'``: the whole dataset. All folds are integrated over the same
  domain and are directly comparable, at the cost of letting the test folds'
  event times decide *where* the score is evaluated. The grid never enters
  model fitting.

Passing ``ibs_eval_times`` fixes the domain a priori and avoids the
trade-off. Without it, the IBS is skipped with a warning when the rows the
grid is taken from contain fewer than 5 uncensored events. The grid actually used on
each fold, after the tail truncation described above, is stored in
``results_.fold_results_[i].ibs_eval_times``.


.. _missing-values:

Missing values and preprocessing
--------------------------------

Preprocessing belongs in the estimator: a
:class:`~sklearn.pipeline.Pipeline` is cloned and refitted wherever the model
is, the OOF pass of step 2 included, and its parameters are tuned like any
other (``<step>__<parameter>``).

- ``X`` may contain ``NaN``, or ``pd.NA`` in pandas nullable dtypes
  (converted to ``NaN``). Infinite values in ``X`` and missing values in
  ``y`` raise a ``ValueError`` before any fold is run.
- If the estimator cannot handle ``NaN``, scikit-learn's ``ValueError``
  surfaces from the inner search: add an imputer to the Pipeline, or use an
  estimator with native support such as
  :class:`~sklearn.ensemble.HistGradientBoostingClassifier`.
- ``X`` is converted to a numeric NumPy array before the folds are created:
  encode string columns beforehand, and let a
  :class:`~sklearn.compose.ColumnTransformer` inside the Pipeline select
  columns by position, not by name. DataFrame column names and index are
  still used to label the results.

**Feature importances.** :class:`~nestkit.importance.FeatureImportanceAggregator`
reads importances (model-native or SHAP) from the final step of each
outer-fold Pipeline, with SHAP values computed on the preprocessed test fold,
and labels them with the input feature names. This requires the
preprocessing to keep one output column per input column, in the same
order. A different number of columns makes ``compute()`` raise a
``ValueError``; a reordering (e.g. by a ``ColumnTransformer``) is not
detected and mislabels the importances. For imputers this means:

- set ``keep_empty_features=True``: with the default (``False``), a column
  with no observed value in an outer training fold is dropped from that
  fold's model;
- do not set ``add_indicator=True``, which appends one column per feature
  with missing values.

The restriction concerns only the aggregated importances: scores,
predictions and calibration are unaffected.

**Survival.** ``CoxPHFitter`` does not accept ``NaN``, so survival models
need an imputer in the Pipeline. Cox coefficients are named after the output
features of the preprocessing steps (``get_feature_names_out``), so the
restriction above does not apply: with
``make_pipeline(SimpleImputer(add_indicator=True), CoxPHWrapper())`` the
indicators appear in the coefficients as ``missingindicator_<column name>``,
and a column dropped in one fold is absent from that fold only.


Inference across folds
----------------------

- **Confidence intervals.** ``ci_lower`` and ``ci_upper`` in every
  ``summary_*`` table use the Nadeau-Bengio correction
  ``sqrt(1/n + n_test/n_train)`` for folds sharing training rows, instead of
  the naive ``sqrt(1/n)``. A fold where a metric cannot be computed (e.g. ROC
  AUC on a single-class test fold) is excluded for that metric only.
- **Model comparison.** :class:`~nestkit.comparison.NestedCVComparator`
  rejects models whose outer test indices differ in any fold, so all models
  must be run with the same outer splits. It offers the Nadeau-Bengio
  corrected *t*-test (with ``n_train`` and ``n_test`` taken from the actual
  folds), a Bayesian correlated *t*-test giving the posterior probabilities
  of "A better", "equivalent within ``rope``" and "B better", and pairwise
  tests with Holm-Bonferroni correction. ``threshold="optimized"`` compares
  the threshold-optimized scores instead of the default ones.
- **Generalization gap.** ``generalization_gap_`` lists the best inner score
  next to all outer scores of each fold. Only the outer column matching the
  inner ``scoring`` is comparable (e.g. ``outer_roc_auc`` for
  ``scoring="roc_auc"``), up to the sign for ``neg_*`` scorers.
- **Stability.** :class:`~nestkit.diagnostics.HyperparameterStability`
  summarizes how consistently the inner search picks each hyperparameter
  across outer folds (modal value, entropy, agreement rate, pairwise Jaccard
  similarity). ``FeatureImportanceAggregator.stability_index(top_k)``
  applies the Nogueira stability index to the top-*k* features of each fold.
