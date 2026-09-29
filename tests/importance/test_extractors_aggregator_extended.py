"""Extended tests for feature importance extractors and aggregator covering SHAP, coef_, and consensus."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nestkit.importance.aggregator import FeatureImportanceAggregator
from nestkit.importance.extractors import (
    _unwrap_pipeline,
    extract_model_importance,
)


class _SimpleEstimator:
    """Simple estimator mock with controllable attributes."""

    pass


def _mock_estimator(feature_importances=None, coef=None):
    """Create a mock estimator with optional attributes."""
    est = _SimpleEstimator()
    if feature_importances is not None:
        est.feature_importances_ = np.asarray(feature_importances)
    if coef is not None:
        est.coef_ = np.asarray(coef)
    return est


@dataclass
class _FakeFoldResult:
    fitted_estimator: object
    test_indices: np.ndarray = field(default_factory=lambda: np.arange(20))


class _FakeResults:
    """Minimal results mock for FeatureImportanceAggregator."""

    def __init__(self, n_folds=3, n_features=5, feature_names=None, has_estimators=True):
        rng = np.random.RandomState(42)
        self.fold_results_ = []
        for _ in range(n_folds):
            est = _mock_estimator(feature_importances=rng.rand(n_features))
            if not has_estimators:
                est = None
            self.fold_results_.append(_FakeFoldResult(fitted_estimator=est))

        if feature_names is not None:
            self.feature_names_in_ = feature_names

    @property
    def has_fitted_estimators(self):
        if not self.fold_results_:
            return False
        return self.fold_results_[0].fitted_estimator is not None


class TestExtractModelImportanceCoef:
    def test_coef_1d(self):
        """Verify extract_model_importance returns abs(coef_) for 1D linear models."""
        est = _mock_estimator(coef=[-0.5, 0.3, -0.1, 0.8])
        result = extract_model_importance(est)
        expected = np.abs([-0.5, 0.3, -0.1, 0.8])
        np.testing.assert_array_almost_equal(result, expected)

    def test_coef_2d_multiclass(self):
        """Verify extract_model_importance averages abs(coef_) across classes for 2D."""
        coef = np.array(
            [
                [-0.5, 0.3, -0.1],
                [0.2, -0.4, 0.6],
            ]
        )
        est = _mock_estimator(coef=coef)
        result = extract_model_importance(est)
        expected = np.mean(np.abs(coef), axis=0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_no_attributes_raises(self):
        """Verify AttributeError when estimator lacks both feature_importances_ and coef_."""
        est = _mock_estimator()  # No feature_importances_, no coef_
        with pytest.raises(AttributeError, match="feature_importances_"):
            extract_model_importance(est)

    def test_feature_importances_preferred_over_coef(self):
        """Verify feature_importances_ is used when both attributes present."""
        fi = [0.4, 0.3, 0.2, 0.1]
        est = _mock_estimator(feature_importances=fi, coef=[-1, 1, -1, 1])
        result = extract_model_importance(est)
        np.testing.assert_array_almost_equal(result, fi)


class TestUnwrapPipeline:
    def test_pipeline_unwrap(self):
        """Verify pipeline unwrapping extracts the final step."""
        inner = _SimpleEstimator()

        class FakePipeline:
            steps: ClassVar[list] = [("scaler", None), ("clf", inner)]

            def __getitem__(self, idx):
                return inner

        result = _unwrap_pipeline(FakePipeline())
        assert result is inner

    def test_non_pipeline_passthrough(self):
        """Verify non-pipeline estimator is returned as-is."""
        est = _SimpleEstimator()
        result = _unwrap_pipeline(est)
        assert result is est


class TestComputeShapImportance:
    def test_tree_explainer(self):
        """Verify TreeExplainer is selected for tree-based models (mocked)."""
        mock_shap = MagicMock()
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.random.rand(10, 5)
        mock_shap.TreeExplainer.return_value = mock_explainer

        est = _mock_estimator(feature_importances=[0.2, 0.3, 0.1, 0.15, 0.25])
        X_test = np.random.rand(10, 5)

        with patch.dict("sys.modules", {"shap": mock_shap}):
            from nestkit.importance.extractors import compute_shap_importance

            mean_abs, raw = compute_shap_importance(est, X_test, shap_type="tree")

        mock_shap.TreeExplainer.assert_called_once()
        assert mean_abs.shape == (5,)
        assert raw.shape == (10, 5)

    def test_binary_list_handling(self):
        """Verify binary classifier SHAP returns (list of 2 arrays) uses index 1."""
        mock_shap = MagicMock()
        arr0 = np.ones((10, 5)) * -1
        arr1 = np.ones((10, 5)) * 2
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = [arr0, arr1]
        mock_shap.TreeExplainer.return_value = mock_explainer

        est = _mock_estimator(feature_importances=[0.2, 0.3, 0.1, 0.15, 0.25])
        X_test = np.random.rand(10, 5)

        with patch.dict("sys.modules", {"shap": mock_shap}):
            from nestkit.importance.extractors import compute_shap_importance

            mean_abs, raw = compute_shap_importance(est, X_test, shap_type="tree")

        np.testing.assert_array_almost_equal(raw, arr1)
        np.testing.assert_array_almost_equal(mean_abs, np.mean(np.abs(arr1), axis=0))

    def test_kernel_fallback(self):
        """Verify KernelExplainer is selected when estimator has no tree/linear attrs."""
        mock_shap = MagicMock()
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.random.rand(10, 5)
        mock_shap.KernelExplainer.return_value = mock_explainer
        mock_shap.kmeans.return_value = MagicMock()

        est = _mock_estimator()  # No feature_importances_ or coef_
        est.predict_proba = lambda x: np.random.rand(len(x), 2)
        X_test = np.random.rand(10, 5)

        with patch.dict("sys.modules", {"shap": mock_shap}):
            from nestkit.importance.extractors import compute_shap_importance

            _mean_abs, _raw = compute_shap_importance(est, X_test, shap_type="kernel")

        mock_shap.KernelExplainer.assert_called_once()

    def test_pipeline_transforms_before_explaining(self):
        """Verify a Pipeline's final step is explained on preprocessed (imputed) data."""
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        rng = np.random.RandomState(0)
        X = rng.rand(40, 5)
        y = (X[:, 0] > 0.5).astype(int)
        X[rng.rand(*X.shape) < 0.2] = np.nan
        pipe = make_pipeline(SimpleImputer(), LogisticRegression()).fit(X, y)

        mock_shap = MagicMock()
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.random.rand(10, 5)
        mock_shap.LinearExplainer.return_value = mock_explainer

        with patch.dict("sys.modules", {"shap": mock_shap}):
            from nestkit.importance.extractors import compute_shap_importance

            compute_shap_importance(pipe, X[:10], shap_type="linear")

        explained_est, background = mock_shap.LinearExplainer.call_args.args
        assert explained_est is pipe[-1]
        np.testing.assert_allclose(background, pipe[:-1].transform(X[:10]))
        assert not np.isnan(mock_explainer.shap_values.call_args.args[0]).any()


class TestResolveShapExplainer:
    def test_explicit_types(self):
        """Verify explicit shap_type maps to correct explainer class."""
        mock_shap = MagicMock()
        mock_shap.TreeExplainer = "tree"
        mock_shap.KernelExplainer = "kernel"
        mock_shap.LinearExplainer = "linear"

        with patch.dict("sys.modules", {"shap": mock_shap}):
            from nestkit.importance.extractors import _resolve_shap_explainer

            est = MagicMock(spec=[])
            assert _resolve_shap_explainer(est, "tree") == "tree"
            assert _resolve_shap_explainer(est, "kernel") == "kernel"
            assert _resolve_shap_explainer(est, "linear") == "linear"

    def test_auto_detects_tree(self):
        """Verify auto-detection selects TreeExplainer for tree models."""
        mock_shap = MagicMock()
        mock_shap.TreeExplainer = "tree"

        est = _mock_estimator(feature_importances=[0.1, 0.2])

        with patch.dict("sys.modules", {"shap": mock_shap}):
            from nestkit.importance.extractors import _resolve_shap_explainer

            assert _resolve_shap_explainer(est, "auto") == "tree"

    def test_auto_detects_linear(self):
        """Verify auto-detection selects LinearExplainer for linear models."""
        mock_shap = MagicMock()
        mock_shap.LinearExplainer = "linear"

        est = _mock_estimator(coef=[0.1, 0.2])

        with patch.dict("sys.modules", {"shap": mock_shap}):
            from nestkit.importance.extractors import _resolve_shap_explainer

            assert _resolve_shap_explainer(est, "auto") == "linear"

    def test_auto_fallback_kernel(self):
        """Verify auto-detection falls back to KernelExplainer."""
        mock_shap = MagicMock()
        mock_shap.KernelExplainer = "kernel"

        est = _mock_estimator()

        with patch.dict("sys.modules", {"shap": mock_shap}):
            from nestkit.importance.extractors import _resolve_shap_explainer

            assert _resolve_shap_explainer(est, "auto") == "kernel"


class TestAggregatorFeatureNames:
    def test_from_results(self):
        """Verify feature_names are resolved from results.feature_names_in_."""
        names = ["a", "b", "c", "d", "e"]
        results = _FakeResults(n_features=5, feature_names=names)
        agg = FeatureImportanceAggregator(results, feature_names=None)
        assert agg.feature_names == names

    def test_none_fallback(self):
        """Verify feature_names is None when neither explicit nor results provide them."""
        results = _FakeResults(n_features=5, feature_names=None)
        # Remove the attribute entirely
        if hasattr(results, "feature_names_in_"):
            delattr(results, "feature_names_in_")
        agg = FeatureImportanceAggregator(results, feature_names=None)
        assert agg.feature_names is None

    def test_explicit_overrides_results(self):
        """Verify explicit feature_names override results.feature_names_in_."""
        results = _FakeResults(n_features=5, feature_names=["x", "y", "z", "w", "v"])
        agg = FeatureImportanceAggregator(results, feature_names=["a", "b", "c", "d", "e"])
        assert agg.feature_names == ["a", "b", "c", "d", "e"]


class TestAggregatorCompute:
    def test_unknown_method_raises(self):
        """Verify ValueError for unknown importance method."""
        results = _FakeResults(n_features=5)
        agg = FeatureImportanceAggregator(results, method="unknown")
        with pytest.raises(ValueError, match="Unknown method"):
            agg.compute()

    def test_shap_without_x_raises(self):
        """Verify ValueError when method='shap' and X is None."""
        results = _FakeResults(n_features=5)
        agg = FeatureImportanceAggregator(results, method="shap")
        with pytest.raises(ValueError, match="X is required"):
            agg.compute(X=None)

    def test_no_estimators_raises(self):
        """Verify ValueError when results lack fitted estimators."""
        results = _FakeResults(n_features=5, has_estimators=False)
        with pytest.raises(ValueError, match="fitted estimators"):
            FeatureImportanceAggregator(results)


class TestAggregatorStabilityIndex:
    def test_stability_index_returns_float(self):
        """Verify stability_index returns a float value."""
        results = _FakeResults(n_folds=5, n_features=10)
        agg = FeatureImportanceAggregator(results)
        agg.compute()
        si = agg.stability_index(top_k=3)
        assert isinstance(si, float | np.floating)


class TestAggregatorConsensusFeatures:
    def test_top_k_criterion(self):
        """Verify consensus_features with top_k returns list of feature names."""
        results = _FakeResults(n_folds=3, n_features=10)
        agg = FeatureImportanceAggregator(results, feature_names=[f"f_{i}" for i in range(10)])
        agg.compute()
        features = agg.consensus_features(criterion="top_k", top_k=3)
        assert len(features) == 3
        assert all(isinstance(f, str) for f in features)

    def test_frequency_criterion(self):
        """Verify consensus_features with frequency returns features appearing often in top-k."""
        results = _FakeResults(n_folds=3, n_features=10)
        agg = FeatureImportanceAggregator(results, feature_names=[f"f_{i}" for i in range(10)])
        agg.compute()
        features = agg.consensus_features(criterion="frequency", top_k=5, min_frequency=0.5)
        assert isinstance(features, list)
        assert all(isinstance(f, str) for f in features)

    def test_unknown_criterion_raises(self):
        """Verify ValueError for unknown consensus criterion."""
        results = _FakeResults(n_folds=3, n_features=10)
        agg = FeatureImportanceAggregator(results, feature_names=[f"f_{i}" for i in range(10)])
        agg.compute()
        with pytest.raises(ValueError, match="Unknown criterion"):
            agg.consensus_features(criterion="bogus")
