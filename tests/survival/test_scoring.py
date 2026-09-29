"""Tests for survival scoring functions."""

import numpy as np
import pytest

from nestkit.survival._scoring import (
    _compute_ibs,
    _compute_uno_c_index,
    _kaplan_meier_censoring,
    _truncate_ibs_eval_times,
    concordance_index_scorer,
)

pytest.importorskip("lifelines")


class TestKaplanMeierCensoring:
    def test_no_censoring(self):
        """With no censoring, G(t) should be 1 for all t < max(duration)."""
        event = np.ones(50)
        duration = np.arange(1, 51, dtype=float)
        G = _kaplan_meier_censoring(event, duration)
        # No censoring events, so G(t) = 1 everywhere
        assert G(0.0) == pytest.approx(1.0)
        assert G(25.0) == pytest.approx(1.0)

    def test_all_censored(self):
        """With all censored, G(t) should decrease."""
        event = np.zeros(50)
        duration = np.arange(1, 51, dtype=float)
        G = _kaplan_meier_censoring(event, duration)
        # G should decrease since every observation is a "censoring event"
        assert G(1.0) < 1.0
        assert G(25.0) < G(10.0)

    def test_returns_callable(self):
        event = np.array([1, 0, 1, 0, 1])
        duration = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        G = _kaplan_meier_censoring(event, duration)
        assert callable(G)

    def test_scalar_and_array_input(self):
        event = np.array([1, 0, 1, 0, 1])
        duration = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        G = _kaplan_meier_censoring(event, duration)
        scalar_result = G(2.0)
        array_result = G(np.array([2.0]))
        assert isinstance(scalar_result, float)
        assert scalar_result == pytest.approx(float(array_result[0]))


class TestUnoCIndex:
    def test_perfect_discrimination(self):
        """When risk scores perfectly rank patients, C-index should be high."""
        n = 100
        rng = np.random.RandomState(42)
        duration = np.sort(rng.exponential(10, n))[::-1]  # longer = lower risk
        event = np.ones(n)
        risk_scores = np.arange(n, dtype=float)  # higher index = higher risk = shorter duration

        c = _compute_uno_c_index(event, duration, event, duration, risk_scores)
        assert c > 0.9

    def test_random_scores(self):
        """Random scores should give C-index around 0.5."""
        n = 200
        rng = np.random.RandomState(42)
        duration = rng.exponential(10, n)
        event = np.ones(n)
        risk_scores = rng.randn(n)

        c = _compute_uno_c_index(event, duration, event, duration, risk_scores)
        assert 0.3 < c < 0.7

    def test_no_censoring_matches_harrell(self):
        """Without censoring, Uno's should be close to Harrell's."""
        import lifelines.utils

        n = 100
        rng = np.random.RandomState(42)
        duration = rng.exponential(10, n)
        event = np.ones(n)  # no censoring
        risk_scores = duration + rng.randn(n) * 0.5  # noisy perfect predictor

        uno_c = _compute_uno_c_index(event, duration, event, duration, risk_scores)
        harrell_c = lifelines.utils.concordance_index(duration, -risk_scores, event)
        assert abs(uno_c - harrell_c) < 0.1

    def test_returns_0_5_when_no_comparable_pairs(self):
        """Edge case: all censored, no comparable pairs."""
        event = np.zeros(10)
        duration = np.arange(1, 11, dtype=float)
        risk_scores = np.ones(10)
        c = _compute_uno_c_index(event, duration, event, duration, risk_scores)
        assert c == 0.5


class TestConcordanceIndexScorer:
    def test_scorer_protocol(self, survival_data):
        from nestkit.survival import CoxPHWrapper

        X, y = survival_data
        est = CoxPHWrapper(penalizer=0.1)
        est.fit(X, y)
        score = concordance_index_scorer(est, X, y)
        assert 0.0 <= score <= 1.0


class TestIBS:
    def test_perfect_model_low_ibs(self):
        """A model with perfect survival predictions should have low IBS."""
        n = 50
        event = np.ones(n)
        duration = np.arange(1, n + 1, dtype=float)
        eval_times = np.array([10.0, 20.0, 30.0, 40.0])

        # Perfect survival function: S(t|i) = 1 if duration_i > t, else 0
        sf = np.zeros((len(eval_times), n))
        for k, t in enumerate(eval_times):
            sf[k, :] = (duration > t).astype(float)

        ibs = _compute_ibs(event, duration, event, duration, sf, eval_times)
        assert ibs < 0.05

    def test_uninformative_model_higher_ibs(self):
        """A constant 0.5 survival model should have higher IBS."""
        n = 50
        event = np.ones(n)
        duration = np.arange(1, n + 1, dtype=float)
        eval_times = np.array([10.0, 20.0, 30.0, 40.0])

        sf = np.full((len(eval_times), n), 0.5)

        ibs = _compute_ibs(event, duration, event, duration, sf, eval_times)
        assert ibs > 0.1


def _censored_sample(rng, n, censoring_scale):
    """Draw an exponential survival sample with independent censoring."""
    duration = rng.exponential(10, n)
    censoring = rng.exponential(censoring_scale, n)
    event = (duration <= censoring).astype(float)
    return event, np.minimum(duration, censoring)


class TestKaplanMeierSupport:
    def test_matches_lifelines(self):
        """The censoring KM must equal lifelines fitted on the flipped indicator."""
        import lifelines

        rng = np.random.RandomState(7)
        event, duration = _censored_sample(rng, 300, 12.0)

        G = _kaplan_meier_censoring(event, duration)
        kmf = lifelines.KaplanMeierFitter().fit(duration, 1 - event)

        times = np.linspace(0.1, duration.max() * 0.99, 40)
        assert np.allclose(G(times), kmf.survival_function_at_times(times).values, atol=1e-10)

    def test_support_is_infinite_when_g_never_vanishes(self):
        """No censoring at all: G stays at 1, so weights are defined everywhere."""
        event = np.ones(20)
        duration = np.arange(1, 21, dtype=float)
        G = _kaplan_meier_censoring(event, duration)
        assert G.support_ == np.inf

    def test_support_marks_last_positive_time(self):
        """A censored largest observation drives G to exactly 0 past it."""
        event = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
        duration = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        G = _kaplan_meier_censoring(event, duration)
        # Not clipped to a small epsilon any more
        assert G(5.0) == 0.0
        assert G(G.support_) > 0.0
        assert G.support_ == pytest.approx(4.0)


class TestTruncateIBSEvalTimes:
    def test_keeps_times_within_support(self):
        event = np.ones(20)
        duration = np.arange(1, 21, dtype=float)
        times = np.array([2.0, 5.0, 10.0])
        kept = _truncate_ibs_eval_times(times, event, duration)
        assert np.array_equal(kept, times)

    def test_drops_times_past_support_with_warning(self):
        event = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
        duration = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.warns(UserWarning, match="IBS evaluation time"):
            kept = _truncate_ibs_eval_times(np.array([2.0, 3.0, 5.0, 7.0]), event, duration)
        assert np.array_equal(kept, np.array([2.0, 3.0]))

    def test_can_return_empty(self):
        event = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
        duration = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.warns(UserWarning):
            kept = _truncate_ibs_eval_times(np.array([6.0, 7.0]), event, duration)
        assert kept.shape == (0,)


class TestIBSNormalization:
    @pytest.mark.parametrize("censoring_scale", [1e6, 60.0, 25.0, 12.0, 6.0, 3.0])
    def test_constant_half_predictor_gives_quarter(self, censoring_scale):
        """A constant S(t) = 0.5 has squared error 0.25 at every t.

        The IPCW weights already compensate for the samples censored before
        t, so the Brier score must be normalized by the full test-set size.
        Dividing by the number of retained samples instead applies the
        censoring correction twice and inflates the score with the
        censoring rate (up to 0.62 at ~79% censoring).
        """
        rng = np.random.RandomState(0)
        event, duration = _censored_sample(rng, 400, censoring_scale)

        eval_times = np.quantile(duration[event.astype(bool)], np.linspace(0.1, 0.9, 9))
        eval_times = _truncate_ibs_eval_times(eval_times, event, duration, warn=False)
        sf = np.full((len(eval_times), 400), 0.5)

        ibs = _compute_ibs(event, duration, event, duration, sf, eval_times)
        assert ibs == pytest.approx(0.25, abs=1e-9)

    def test_stays_bounded_when_censoring_km_vanishes(self):
        """Evaluating past the censoring support must not blow the score up.

        With G(t) formerly clipped to 1e-10, a test event beyond the
        training horizon contributed a weight of 1e10 and produced an IBS
        of ~8.3e8 instead of a value in [0, 1].
        """
        train_event = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
        train_duration = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        test_event = np.array([1.0, 1.0, 1.0])
        test_duration = np.array([2.0, 3.0, 6.0])

        eval_times = _truncate_ibs_eval_times(
            np.array([2.0, 3.0, 5.0, 7.0]), train_event, train_duration, warn=False
        )
        sf = np.full((len(eval_times), 3), 0.5)

        ibs = _compute_ibs(train_event, train_duration, test_event, test_duration, sf, eval_times)
        assert 0.0 <= ibs <= 1.0

    def test_empty_grid_returns_nan(self):
        event = np.ones(10)
        duration = np.arange(1, 11, dtype=float)
        sf = np.zeros((0, 10))
        assert np.isnan(_compute_ibs(event, duration, event, duration, sf, np.array([])))

    def test_perfect_predictor_scores_zero(self):
        n = 60
        event = np.ones(n)
        duration = np.arange(1, n + 1, dtype=float)
        eval_times = np.array([10.0, 20.0, 30.0, 40.0])
        sf = np.array([(duration > t).astype(float) for t in eval_times])
        assert _compute_ibs(event, duration, event, duration, sf, eval_times) == pytest.approx(0.0)


def _uno_reference(train_event, train_duration, test_event, test_duration, risk_scores):
    """Naive O(n^2) double loop, kept as an oracle for the vectorized version."""
    G = _kaplan_meier_censoring(train_event, train_duration)
    test_event = np.asarray(test_event, dtype=bool)
    uncensored = np.asarray(train_duration)[np.asarray(train_event).astype(bool)]
    tau = float(np.max(uncensored)) if uncensored.size else np.inf

    numerator = denominator = 0.0
    for i in range(len(test_duration)):
        if not test_event[i] or test_duration[i] > tau:
            continue
        g_ti = float(G(test_duration[i]))
        if g_ti <= 0:
            continue
        w_i = 1.0 / (g_ti * g_ti)
        for j in range(len(test_duration)):
            if test_duration[j] <= test_duration[i]:
                continue
            denominator += w_i
            if risk_scores[i] > risk_scores[j]:
                numerator += w_i
            elif risk_scores[i] == risk_scores[j]:
                numerator += 0.5 * w_i
    return 0.5 if denominator == 0.0 else numerator / denominator


class TestUnoCIndexVectorization:
    @pytest.mark.parametrize("n", [50, 200])
    @pytest.mark.parametrize("censoring_scale", [1e6, 20.0, 6.0])
    def test_matches_reference_loop(self, n, censoring_scale):
        rng = np.random.RandomState(7)
        event, duration = _censored_sample(rng, n, censoring_scale)
        risk_scores = rng.randn(n)

        expected = _uno_reference(event, duration, event, duration, risk_scores)
        actual = _compute_uno_c_index(event, duration, event, duration, risk_scores)
        assert actual == pytest.approx(expected, abs=1e-12)

    def test_matches_reference_loop_with_ties(self):
        """Tied risk scores must still contribute exactly half a pair."""
        rng = np.random.RandomState(3)
        event, duration = _censored_sample(rng, 150, 20.0)
        risk_scores = np.round(rng.randn(150), 1)  # many ties

        expected = _uno_reference(event, duration, event, duration, risk_scores)
        actual = _compute_uno_c_index(event, duration, event, duration, risk_scores)
        assert actual == pytest.approx(expected, abs=1e-12)

    def test_chunking_does_not_change_result(self, monkeypatch):
        """The chunk size is a memory knob, never a numerical one."""
        import nestkit.survival._scoring as scoring

        rng = np.random.RandomState(11)
        event, duration = _censored_sample(rng, 200, 15.0)
        risk_scores = rng.randn(200)

        full = _compute_uno_c_index(event, duration, event, duration, risk_scores)
        monkeypatch.setattr(scoring, "_PAIR_CHUNK_BYTES", 8 * 3)  # 3 rows per chunk
        chunked = _compute_uno_c_index(event, duration, event, duration, risk_scores)
        assert chunked == pytest.approx(full, abs=1e-12)
