"""End-to-end tests for NestedCVSurvival."""

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from nestkit.survival import CoxPHWrapper, NestedCVSurvival

pytest.importorskip("lifelines")


@pytest.fixture
def simple_survival_ncv():
    return NestedCVSurvival(
        estimator=CoxPHWrapper(penalizer=0.1),
        param_grid={"penalizer": [0.01, 0.1, 1.0]},
        outer_cv=3,
        inner_cv=2,
        random_state=42,
    )


@pytest.mark.slow
class TestNestedCVSurvivalBasic:
    def test_fit_returns_self(self, simple_survival_ncv, survival_data):
        X, y = survival_data
        result = simple_survival_ncv.fit(X, y)
        assert result is simple_survival_ncv

    def test_is_fitted(self, simple_survival_ncv, survival_data):
        X, y = survival_data
        simple_survival_ncv.fit(X, y)
        assert simple_survival_ncv.is_fitted_

    def test_results_exist(self, simple_survival_ncv, survival_data):
        X, y = survival_data
        simple_survival_ncv.fit(X, y)
        assert hasattr(simple_survival_ncv, "results_")

    def test_summary_has_concordance(self, simple_survival_ncv, survival_data):
        X, y = survival_data
        simple_survival_ncv.fit(X, y)
        summary = simple_survival_ncv.results_.summary_default_
        metrics = summary["metric"].tolist()
        assert "concordance_index" in metrics
        assert "uno_c_index" in metrics

    def test_ibs_in_scores(self, simple_survival_ncv, survival_data):
        X, y = survival_data
        simple_survival_ncv.fit(X, y)
        scores_df = simple_survival_ncv.results_.outer_scores_default_
        assert "integrated_brier_score" in scores_df.columns

    def test_predictions_dataframe(self, simple_survival_ncv, survival_data):
        X, y = survival_data
        simple_survival_ncv.fit(X, y)
        preds = simple_survival_ncv.results_.predictions_
        assert isinstance(preds, pd.DataFrame)
        assert "y_event" in preds.columns
        assert "y_duration" in preds.columns
        assert "risk_score" in preds.columns
        assert "fold_idx" in preds.columns
        assert len(preds) == len(y)

    def test_coefficient_stability(self, simple_survival_ncv, survival_data):
        X, y = survival_data
        simple_survival_ncv.fit(X, y)
        coef_stab = simple_survival_ncv.results_.coefficient_stability_
        assert isinstance(coef_stab, pd.DataFrame)
        assert "feature" in coef_stab.columns
        assert "coef_mean" in coef_stab.columns
        assert "hazard_ratio_mean" in coef_stab.columns
        assert len(coef_stab) == X.shape[1]

    def test_generalization_gap(self, simple_survival_ncv, survival_data):
        X, y = survival_data
        simple_survival_ncv.fit(X, y)
        gap = simple_survival_ncv.results_.generalization_gap_
        assert isinstance(gap, pd.DataFrame)
        assert "best_inner_score" in gap.columns

    def test_best_params_per_fold(self, simple_survival_ncv, survival_data):
        X, y = survival_data
        simple_survival_ncv.fit(X, y)
        params = simple_survival_ncv.results_.best_params_per_fold_
        assert len(params) == 3  # outer_cv=3
        assert all("penalizer" in p for p in params)


@pytest.mark.slow
class TestNestedCVSurvivalInputFormats:
    def test_dataframe_input(self, survival_data):
        X, y = survival_data
        X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])

        ncv = NestedCVSurvival(
            estimator=CoxPHWrapper(penalizer=0.1),
            param_grid={"penalizer": [0.1, 1.0]},
            outer_cv=3,
            inner_cv=2,
            random_state=42,
        )
        ncv.fit(X_df, y)
        assert ncv.results_.feature_names_in_ is not None
        assert ncv.results_.feature_names_in_[0] == "feat_0"

    def test_structured_array_target(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5)
        dt = np.dtype([("event", np.float64), ("duration", np.float64)])
        event = rng.binomial(1, 0.7, 200).astype(float)
        duration = rng.exponential(10, 200)
        y = np.array(list(zip(event, duration)), dtype=dt)

        ncv = NestedCVSurvival(
            estimator=CoxPHWrapper(penalizer=0.1),
            param_grid={"penalizer": [0.1, 1.0]},
            outer_cv=3,
            inner_cv=2,
            random_state=42,
        )
        ncv.fit(X, y)
        assert ncv.is_fitted_

    def test_dataframe_target(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5)
        y_df = pd.DataFrame(
            {
                "event": rng.binomial(1, 0.7, 200).astype(float),
                "duration": rng.exponential(10, 200),
            }
        )

        ncv = NestedCVSurvival(
            estimator=CoxPHWrapper(penalizer=0.1),
            param_grid={"penalizer": [0.1, 1.0]},
            outer_cv=3,
            inner_cv=2,
            random_state=42,
        )
        ncv.fit(X, y_df)
        assert ncv.is_fitted_


@pytest.mark.slow
class TestNestedCVSurvivalParamGrid:
    def test_penalizer_and_l1_ratio(self, survival_data):
        X, y = survival_data
        ncv = NestedCVSurvival(
            estimator=CoxPHWrapper(),
            param_grid={
                "penalizer": [0.01, 0.1],
                "l1_ratio": [0.0, 0.5],
            },
            outer_cv=3,
            inner_cv=2,
            random_state=42,
        )
        ncv.fit(X, y)
        params = ncv.results_.best_params_per_fold_
        assert all("penalizer" in p and "l1_ratio" in p for p in params)


@pytest.mark.slow
class TestNestedCVSurvivalStratification:
    def test_event_ratio_balanced(self, survival_data):
        """Folds should have roughly similar event ratios."""
        X, y = survival_data
        ncv = NestedCVSurvival(
            estimator=CoxPHWrapper(penalizer=0.1),
            param_grid={"penalizer": [0.1]},
            outer_cv=5,
            inner_cv=2,
            random_state=42,
        )
        ncv.fit(X, y)

        overall_rate = np.mean(y[:, 0])
        fold_rates = []
        for fr in ncv.results_.fold_results_:
            fold_rates.append(np.mean(fr.y_event))

        # Each fold's event rate should be within 15% of overall
        for rate in fold_rates:
            assert abs(rate - overall_rate) < 0.15


@pytest.mark.slow
class TestNestedCVSurvivalPipeline:
    @staticmethod
    def _fit(X, y, imputer):
        return NestedCVSurvival(
            estimator=make_pipeline(imputer, CoxPHWrapper()),
            param_grid={"coxphwrapper__penalizer": [0.01, 0.1]},
            outer_cv=3,
            inner_cv=2,
            random_state=42,
        ).fit(X, y)

    @pytest.fixture
    def survival_data_nan(self, survival_data):
        X, y = survival_data
        X = X.copy()
        X[np.random.default_rng(0).random(X.shape) < 0.1] = np.nan
        names = [f"feat_{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=names), y

    def test_nan_imputed_inside_folds(self, survival_data_nan):
        """Verify NaN in X works with an imputer pipeline and all metrics are computed."""
        X, y = survival_data_nan
        ncv = self._fit(X, y, SimpleImputer())
        for fr in ncv.results_.fold_results_:
            assert "integrated_brier_score" in fr.outer_scores
            assert list(fr.coefficients) == list(X.columns)

    def test_coefficient_names_follow_preprocessing_output(self, survival_data_nan):
        """Verify added columns (missing indicators) get their own names."""
        X, y = survival_data_nan
        ncv = self._fit(X, y, SimpleImputer(add_indicator=True))
        coef_names = list(ncv.results_.fold_results_[0].coefficients)
        assert coef_names[: X.shape[1]] == list(X.columns)
        assert all(n.startswith("missingindicator_") for n in coef_names[X.shape[1] :])


class TestNestedCVSurvivalSklearnCompat:
    def test_get_params_set_params(self):
        ncv = NestedCVSurvival(
            estimator=CoxPHWrapper(),
            param_grid={"penalizer": [0.1]},
            outer_cv=3,
            ibs_eval_times=[1.0, 5.0, 10.0],
        )
        params = ncv.get_params()
        assert params["outer_cv"] == 3
        assert params["ibs_eval_times"] == [1.0, 5.0, 10.0]

    def test_get_params_restores_after_fit(self, survival_data):
        """After fit, outer_cv should still be the original int, not the splitter."""
        X, y = survival_data
        ncv = NestedCVSurvival(
            estimator=CoxPHWrapper(penalizer=0.1),
            param_grid={"penalizer": [0.1]},
            outer_cv=3,
            inner_cv=2,
            random_state=42,
        )
        ncv.fit(X, y)
        assert ncv.outer_cv == 3
        assert ncv.inner_cv == 2
        assert ncv.scoring is None


@pytest.mark.slow
class TestIBSTimeGrid:
    """``ibs_time_grid`` selects where the integrated Brier score is evaluated."""

    @staticmethod
    def _fit(X, y, **kwargs):
        ncv = NestedCVSurvival(
            estimator=CoxPHWrapper(penalizer=0.1),
            param_grid={"penalizer": [0.01, 0.1]},
            outer_cv=3,
            inner_cv=2,
            random_state=42,
            **kwargs,
        )
        return ncv.fit(X, y)

    def test_rejects_unknown_grid(self, survival_data):
        X, y = survival_data
        with pytest.raises(ValueError, match="ibs_time_grid"):
            self._fit(X, y, ibs_time_grid="sometimes")

    def test_default_is_per_fold(self):
        ncv = NestedCVSurvival(estimator=CoxPHWrapper(), param_grid={"penalizer": [0.1]})
        assert ncv.get_params()["ibs_time_grid"] == "per_fold"

    def test_per_fold_grids_differ_between_folds(self, survival_data):
        """Each fold derives its grid from its own training rows."""
        X, y = survival_data
        ncv = self._fit(X, y, ibs_time_grid="per_fold")
        grids = [fr.ibs_eval_times for fr in ncv.results_.fold_results_]
        assert all(g is not None for g in grids)
        assert not all(np.array_equal(grids[0], g) for g in grids[1:])

    def test_global_grid_is_shared_by_every_fold(self, survival_data):
        X, y = survival_data
        ncv = self._fit(X, y, ibs_time_grid="global")
        grids = [fr.ibs_eval_times for fr in ncv.results_.fold_results_]
        assert all(np.array_equal(grids[0], g) for g in grids[1:])

    def test_explicit_times_override_the_grid_mode(self, survival_data):
        X, y = survival_data
        times = [2.0, 4.0, 6.0]
        ncv = self._fit(X, y, ibs_eval_times=times, ibs_time_grid="per_fold")
        for fr in ncv.results_.fold_results_:
            assert np.array_equal(fr.ibs_eval_times, np.asarray(times))

    @pytest.mark.parametrize("grid", ["per_fold", "global"])
    def test_ibs_is_a_probability_scale_score(self, survival_data, grid):
        """IBS must land in [0, 1]; the IPCW tail used to push it to ~1e8."""
        X, y = survival_data
        ncv = self._fit(X, y, ibs_time_grid=grid)
        ibs = ncv.results_.outer_scores_default_["integrated_brier_score"]
        assert ibs.notna().all()
        assert ((ibs >= 0.0) & (ibs <= 1.0)).all()

    def test_failure_warns_instead_of_being_swallowed(self, survival_data, monkeypatch):
        """A failing IBS must surface: the 'nestkit' logger has a NullHandler."""
        import nestkit.survival.survival as survival_mod

        def _boom(*args, **kwargs):
            raise RuntimeError("synthetic IBS failure")

        monkeypatch.setattr(survival_mod, "_compute_ibs", _boom)

        X, y = survival_data
        with pytest.warns(UserWarning, match="IBS computation failed"):
            ncv = self._fit(X, y)

        # The other metrics still summarize normally
        summary = ncv.results_.summary_default_.set_index("metric")
        assert "integrated_brier_score" not in summary.index
        assert not np.isnan(summary.loc["concordance_index", "mean"])
        assert all(fr.ibs_eval_times is None for fr in ncv.results_.fold_results_)

    @pytest.mark.parametrize(
        "bad, match",
        [
            ([], "at least one"),
            ([1.0, float("nan")], "finite"),
            ([1.0, float("inf")], "finite"),
            ([-1.0, 2.0], "positive"),
            ([0.0, 2.0], "positive"),
        ],
    )
    def test_rejects_invalid_explicit_times(self, survival_data, bad, match):
        X, y = survival_data
        with pytest.raises(ValueError, match=match):
            self._fit(X, y, ibs_eval_times=bad)

    def test_unsorted_explicit_times_are_normalized(self, survival_data):
        """The score is integrated with the trapezoidal rule, so order matters."""
        X, y = survival_data
        sorted_fit = self._fit(X, y, ibs_eval_times=[2.0, 4.0, 6.0])
        shuffled_fit = self._fit(X, y, ibs_eval_times=[6.0, 2.0, 4.0])

        for fr in shuffled_fit.results_.fold_results_:
            assert np.array_equal(fr.ibs_eval_times, np.array([2.0, 4.0, 6.0]))
        assert np.allclose(
            sorted_fit.results_.outer_scores_default_["integrated_brier_score"],
            shuffled_fit.results_.outer_scores_default_["integrated_brier_score"],
        )
