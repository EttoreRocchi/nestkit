# Changelog

All notable changes to nestkit are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.3.0] - 2026-09-29

### Added

- Survival analysis: `NestedCVSurvival` with Cox PH models, Harrell's and Uno's concordance indices, integrated Brier score and coefficient stability. Install with `pip install nestkit[survival]`.
- `make_survival_target` to build a survival target from event and duration arrays.
- Missing values are accepted in `X`: an imputer placed in a `Pipeline` is fitted inside every fold.
- `explainability` extra for SHAP importances.
- Tutorial notebook on survival analysis.

### Changed

- matplotlib and seaborn are now core dependencies, and matplotlib 3.10 or newer is required.

### Fixed

- Binary targets whose labels are not `0` and `1` are now supported: predictions, conformal sets and `results_` come back in the original labels, and, as in scikit-learn, `classes_[1]` is the positive class. This also repairs isotonic calibration, threshold selection with Youden's J and the expected calibration error, which were all wrong on such targets.
- Mondrian prediction intervals fell short of their target coverage on small bins. Bins too small for the requested `alpha` are now merged, and a bound the sample cannot determine is returned as infinite.
- SHAP importances for `Pipeline` estimators were computed on the raw features instead of the preprocessed ones.
- `plot_comparison` raised `TypeError` on matplotlib 3.9 and newer.
- A fold whose `roc_auc` could not be computed no longer turns the whole `roc_auc` summary row into `NaN`.
- Comparing models from a single fold now reports that a variance needs at least two folds, instead of returning `NaN` with numpy warnings.
- The feature-selection stability index reported a perfect 1.0 when a fold's importances were all equal; it now returns `NaN` with a warning.
- Corrected the documented example for `holm_bonferroni_correction`.

### Removed

- The `plotting` extra, now that matplotlib and seaborn ship with nestkit.

## [0.2.0] - 2026-04-13

### Added

- Conformal prediction: CV+ Mondrian prediction sets for classification and Mondrian-binned prediction intervals for regression, with coverage diagnostics on both.

### Fixed

- Spurious warning when summarizing results from a single fold.

## [0.1.1] - 2026-03-09

### Fixed

- Numerical robustness of prediction intervals and of the Nadeau-Bengio t-test in edge cases.
- Incorrect parameter names and type references in the plotting docstrings.

## [0.1.0] - 2026-03-06

Initial release.
