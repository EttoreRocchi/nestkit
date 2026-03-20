:html_theme.sidebar_secondary.remove:

.. image:: _static/nestkit_logo.png
   :alt: nestkit
   :align: center
   :width: 300px

.. rst-class:: text-center

   *A nested cross-validation toolkit for scikit-learn*

.. rst-class:: text-center

   |pypi| |python| |license|

.. |pypi| image:: https://img.shields.io/pypi/v/nestkit
   :target: https://pypi.org/project/nestkit/
   :alt: PyPI

.. |python| image:: https://img.shields.io/pypi/pyversions/nestkit
   :target: https://pypi.org/project/nestkit/
   :alt: Python versions

.. |license| image:: https://img.shields.io/github/license/ettorerocchi/nestkit
   :target: https://github.com/ettorerocchi/nestkit/blob/main/LICENSE
   :alt: License

----

**nestkit** provides a nested cross-validation framework for
scikit-learn with integrated calibration, threshold optimization, statistical
comparison, and comprehensive diagnostics, all within a single, leakage-free
evaluation pipeline.

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: Nested CV
      :text-align: center

      Classification and regression with full scikit-learn API compatibility
      and leakage-free evaluation.

   .. grid-item-card:: Calibration
      :text-align: center

      Post-hoc probability calibration via Platt scaling, isotonic regression,
      beta calibration, and Venn-ABERS.

   .. grid-item-card:: Thresholding
      :text-align: center

      Decision-threshold optimization with Youden's J, F-beta, cost-sensitive,
      and precision-at-recall criteria.

   .. grid-item-card:: Model Comparison
      :text-align: center

      Nadeau-Bengio corrected t-test, Bayesian correlated t-test with ROPE,
      and Holm-Bonferroni correction.

   .. grid-item-card:: Diagnostics
      :text-align: center

      Hyperparameter stability and feature importance aggregation with
      Nogueira stability index.

   .. grid-item-card:: Plotting
      :text-align: center

      25+ visualizations: ROC curves, calibration diagrams,
      confusion matrices, critical difference diagrams, and more.

   .. grid-item-card:: Tutorials
      :text-align: center
      :link: tutorials
      :link-type: doc

      Interactive Jupyter notebooks covering basic usage
      and advanced workflows.

Getting started
---------------

.. code-block:: bash

   pip install nestkit

.. code-block:: python

   from sklearn.datasets import load_breast_cancer
   from sklearn.ensemble import RandomForestClassifier
   from nestkit import NestedCVClassifier

   X, y = load_breast_cancer(return_X_y=True)

   ncv = NestedCVClassifier(
       estimator=RandomForestClassifier(random_state=42),
       param_grid={"n_estimators": [50, 100], "max_depth": [3, 5, 10]},
       outer_cv=5,
       inner_cv=3,
       scoring="accuracy",
       random_state=42,
   )
   ncv.fit(X, y)
   print(ncv.results_.summary_default_)

----

.. toctree::
   :maxdepth: 1

   installation
   quickstart
   tutorials
   architecture
   api/index
   changelog
