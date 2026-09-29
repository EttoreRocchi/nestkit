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

With missing values
~~~~~~~~~~~~~~~~~~~

``X`` may contain ``NaN``. Put the imputer inside a
:class:`~sklearn.pipeline.Pipeline` so that it is fitted within each fold:

.. code-block:: python

   from sklearn.impute import SimpleImputer
   from sklearn.pipeline import make_pipeline

   ncv = NestedCVClassifier(
       estimator=make_pipeline(
           SimpleImputer(keep_empty_features=True),
           RandomForestClassifier(random_state=42),
       ),
       param_grid={"randomforestclassifier__max_depth": [3, 5, 10]},
       outer_cv=5,
       inner_cv=3,
       random_state=42,
   )
   ncv.fit(X, y)

See :ref:`missing-values` for details.

Survival Analysis
-----------------

.. code-block:: python

   from nestkit import NestedCVSurvival
   from nestkit.survival import CoxPHWrapper, make_survival_target
   from lifelines.datasets import load_rossi

   rossi = load_rossi()
   X = rossi.drop(columns=["week", "arrest"])
   y = make_survival_target(event=rossi["arrest"].values, duration=rossi["week"].values)

   ncv = NestedCVSurvival(
       estimator=CoxPHWrapper(),
       param_grid={"penalizer": [0.001, 0.01, 0.1, 1.0]},
       outer_cv=5,
       inner_cv=3,
       random_state=42,
   )
   ncv.fit(X, y)

   results = ncv.results_
   print(results.summary_default_)
   print(results.coefficient_stability_)

Install survival support with ``pip install nestkit[survival]``.

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
