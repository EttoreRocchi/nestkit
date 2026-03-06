"""Shared fixtures for nestkit tests."""

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


@pytest.fixture
def binary_data():
    """Binary classification dataset: 200 samples, 10 features."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=42,
    )
    return X, y


@pytest.fixture
def multiclass_data():
    """Multiclass classification dataset: 200 samples, 10 features, 3 classes."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    return X, y


@pytest.fixture
def regression_data():
    """Regression dataset: 200 samples, 10 features."""
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    return X, y


@pytest.fixture
def binary_data_df(binary_data):
    """Binary classification data as pandas DataFrame with column names."""
    X, y = binary_data
    columns = [f"feat_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=columns)
    y_series = pd.Series(y, name="target")
    return X_df, y_series


@pytest.fixture
def simple_param_grid():
    """A small parameter grid for fast tests."""
    return {"n_estimators": [10, 20], "max_depth": [3, 5]}


@pytest.fixture
def simple_classifier():
    """A small RandomForestClassifier for fast tests."""
    return RandomForestClassifier(random_state=42, n_estimators=10)


@pytest.fixture
def simple_regressor():
    """A small RandomForestRegressor for fast tests."""
    return RandomForestRegressor(random_state=42, n_estimators=10)
