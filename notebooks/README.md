# nestkit Tutorials

This directory contains Jupyter notebooks demonstrating nestkit's features.

## Notebooks

### [01  -  Basic Usage](01_basic_usage.ipynb)

A beginner-friendly tutorial covering core functionality:

- Binary classification with `NestedCVClassifier`
- Inspecting results (summary, predictions, generalization gap)
- Plotting (ROC curves, confusion matrices, outer scores)
- Adding probability calibration
- Adding threshold optimization
- Regression with `NestedCVRegressor` and prediction intervals

### [02  -  Advanced Workflows](02_advanced_workflows.ipynb)

A deeper-dive tutorial for experienced users:

- Multi-model comparison with `NestedCVComparator`
  - Nadeau-Bengio corrected t-test
  - Bayesian correlated t-test with ROPE
  - Critical difference diagrams
- Feature importance analysis with `FeatureImportanceAggregator`
  - Nogueira stability index
  - Consensus features
- Hyperparameter stability diagnostics
- Callbacks: progress, logging, checkpointing

## Requirements

```bash
pip install nestkit[full]
pip install jupyter
```
