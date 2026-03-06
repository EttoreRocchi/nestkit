Quick Start
===========

Classification
--------------

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

   results = ncv.results_
   print(results.summary_default_)
   print(results.best_params_per_fold_)

With calibration and threshold optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   ncv = NestedCVClassifier(
       estimator=RandomForestClassifier(random_state=42),
       param_grid={"n_estimators": [50, 100], "max_depth": [3, 5]},
       outer_cv=5,
       inner_cv=3,
       calibration_method="isotonic",
       threshold_strategy="pooled",
       threshold_criterion="youden",
       random_state=42,
   )
   ncv.fit(X, y)
   print(ncv.results_.threshold_comparison())

Regression
----------

.. code-block:: python

   from sklearn.datasets import load_diabetes
   from sklearn.linear_model import Ridge
   from nestkit import NestedCVRegressor

   X, y = load_diabetes(return_X_y=True)

   ncv = NestedCVRegressor(
       estimator=Ridge(),
       param_grid={"alpha": [0.01, 0.1, 1.0, 10.0]},
       outer_cv=5,
       inner_cv=3,
       prediction_intervals=True,
       random_state=42,
   )
   ncv.fit(X, y)

   results = ncv.results_
   print(results.summary_default_)
   print(f"PI coverage: {results.prediction_interval_coverage_['mean']:.3f}")
