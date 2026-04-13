Changelog
=========

v0.2.0 (2026-04-13)
--------------------

- Add CV+ Mondrian conformal prediction sets for classification (binary and multiclass)
- Add Mondrian-binned conditional prediction intervals for regression
- New ``conformal`` subpackage with ``MondrianClassifierConformal`` and ``MondrianRegressorConformal``
- New ``NestedCVClassifier`` parameters: ``conformal_prediction``, ``conformal_alpha``
- New ``NestedCVRegressor`` parameters: ``mondrian_bins``, ``mondrian_min_bin_size``
- New ``ClassifierResults`` attributes: ``conformal_coverage_``, ``conformal_set_size_stats_``, ``conformal_qhat_per_fold_``, ``conformal_report()``
- New ``RegressorResults`` attribute: ``mondrian_coverage_per_bin_``
- Fix RuntimeWarning in single-fold summary statistics

v0.1.1 (2026-03-09)
--------------------

- Standardize numerical epsilon constants across the codebase
- Fix prediction interval lower-quantile edge case
- Fix Nadeau-Bengio t-test for zero-variance differences
- Correct docstring parameter names and type references in plotting module

v0.1.0 (2026-03-06)
--------------------

Initial release.

- Nested cross-validation for classification and regression
- Post-hoc probability calibration (Platt, isotonic, beta, Venn-ABERS)
- Threshold optimization (Youden's J, F-beta, cost-sensitive, balanced accuracy, precision at recall)
- Statistical model comparison (Nadeau-Bengio corrected t-test, Bayesian correlated t-test)
- Hyperparameter stability diagnostics
- Feature importance aggregation with Nogueira stability index
- Callback system (progress, checkpointing, logging)
- 25+ plotting functions
- Full scikit-learn API compatibility
