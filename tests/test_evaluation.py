import numpy as np
import optuna
import pandas as pd
import pytest

from wasserstein_score import benchmark_protocol
from wasserstein_score.evaluation import (
    DEFAULT_FALSE_ALARM_BUDGET_PER_1000,
    DEFAULT_FALSE_ALARM_CONFIDENCE,
    DEFAULT_N_TRIALS,
    DEFAULT_OPTUNA_SEED,
    DEFAULT_WASSERSTEIN_BOUNDS,
    _as_list_of_streams,
    _bootstrap_ci,
    _eval_single_stream,
    _make_sampler,
    _params_at_bound,
    _poisson_rate_upper_bound,
    _record_false_alarm_statistics,
    _run_wasserstein,
    _select_best,
    calibrate_and_test,
    evaluate_fixed_detector,
    evaluate_with_params,
)


def test_calibration_defaults_follow_canonical_benchmark_protocol():
    assert DEFAULT_WASSERSTEIN_BOUNDS == benchmark_protocol.WASSERSTEIN_BOUNDS
    assert DEFAULT_WASSERSTEIN_BOUNDS["threshold_drift_score"][0] >= 0.0
    assert DEFAULT_N_TRIALS == benchmark_protocol.N_OPTUNA_TRIALS
    assert DEFAULT_OPTUNA_SEED == benchmark_protocol.OPTUNA_SEED
    assert (
        DEFAULT_FALSE_ALARM_BUDGET_PER_1000
        == benchmark_protocol.FALSE_ALARM_BUDGET_PER_1000
    )
    assert (
        DEFAULT_FALSE_ALARM_CONFIDENCE
        == benchmark_protocol.FALSE_ALARM_CONFIDENCE
    )


def test_stream_normalisation_accepts_one_or_many_streams():
    single = _as_list_of_streams(np.array([1.0, 2.0, 3.0]))
    multiple = _as_list_of_streams(np.array([[1.0, 2.0], [3.0, 4.0]]))

    assert len(single) == 1
    assert len(multiple) == 2
    np.testing.assert_array_equal(multiple[1], [3.0, 4.0])


def test_bootstrap_interval_is_reproducible():
    values = [1.0, 2.0, 3.0, 4.0]

    first = _bootstrap_ci(values, 200, 0.95, np.random.default_rng(7))
    second = _bootstrap_ci(values, 200, 0.95, np.random.default_rng(7))

    assert first == second
    assert first[0] <= np.mean(values) <= first[1]


def test_optuna_sampler_is_reproducible():
    def sampled_params(seed: int) -> list[dict]:
        study = optuna.create_study(
            direction="minimize",
            sampler=_make_sampler(seed=seed, n_startup_trials=2),
        )
        study.optimize(
            lambda trial: trial.suggest_float("value", 0.0, 1.0),
            n_trials=5,
        )
        return [trial.params for trial in study.trials]

    assert sampled_params(DEFAULT_OPTUNA_SEED) == sampled_params(DEFAULT_OPTUNA_SEED)


def test_wasserstein_runner_and_stream_evaluation_smoke():
    rng = np.random.default_rng(12)
    stream = np.concatenate([rng.normal(0, 1, 100), rng.normal(5, 1, 100)])
    params = {"window_size": 10, "threshold_drift_score": 0.5}

    detections = _run_wasserstein(stream, **params, starting_idx=0)
    result = _eval_single_stream(
        "abrupt", "wasserstein", stream, _run_wasserstein, params, 100, 0
    )

    assert all(isinstance(index, int) for index in detections)
    assert result[:2] == ("abrupt", "wasserstein")
    assert result[3] >= 1


def test_stream_evaluation_counts_false_positives_on_common_window():
    def run_fixed(stream, starting_idx):
        del stream
        return [starting_idx + offset for offset in (1, 5, 7, 10, 12)]

    result = _eval_single_stream(
        "scenario",
        "detector",
        np.zeros(15),
        run_fixed,
        {},
        drift_at=10,
        starting_idx=0,
        common_exposure_start=6,
    )

    assert result == ("scenario", "detector", 3, 2, 0, 1, 0)


def test_evaluation_uses_maximum_warmup_for_common_exposure():
    stream = np.zeros(30)
    test_df, per_stream_df = evaluate_with_params(
        test_streams={"example": [stream]},
        best_params={
            "wasserstein": {"window_size": 5, "threshold_drift_score": 1.0},
            "adwin": {"delta": 0.01},
        },
        drift_at=20,
        algorithms=["wasserstein", "adwin"],
        n_bootstrap=10,
        verbose=False,
        detection_horizon=None,
    )

    assert (test_df["common_exposure_start"] == 15).all()
    assert (test_df["common_exposure_samples"] == 5).all()
    assert (per_stream_df["common_exposure_start"] == 15).all()
    assert (per_stream_df["common_exposure_samples"] == 5).all()
    assert dict(test_df.set_index("algorithm")["warmup_samples"]) == {
        "wasserstein": 15,
        "adwin": 10,
    }


def test_unknown_algorithm_is_rejected_before_calibration():
    with pytest.raises(ValueError, match="Unknown algorithm"):
        calibrate_and_test(
            calibration_streams=[np.zeros(20)],
            test_streams={"example": [np.zeros(20)]},
            drift_at=10,
            algorithms=["unknown"],
            verbose=False,
        )


def test_calibration_results_mark_exactly_one_selected_trial():
    rng = np.random.default_rng(3)
    calibration = [rng.normal(size=80)]
    test = np.concatenate([rng.normal(size=60), rng.normal(3, 1, size=20)])

    best, calibration_df, test_df, per_stream_df = calibrate_and_test(
        calibration_streams=calibration,
        test_streams={"example": [test]},
        drift_at=60,
        algorithms=["wasserstein"],
        wasserstein_bounds={
            "threshold_drift_score": (0.5, 1.0),
            "window_size": (5, 10),
        },
        n_trials=2,
        n_bootstrap=10,
        verbose=False,
        false_alarm_budget_per_1000=1.0,
        detection_horizon=None,
    )

    repeated_test_df, repeated_per_stream_df = evaluate_with_params(
        test_streams={"example": [test]},
        best_params=best,
        drift_at=60,
        algorithms=["wasserstein"],
        n_bootstrap=10,
        verbose=False,
        detection_horizon=None,
    )

    selected = calibration_df.loc[calibration_df["selected"]]
    assert len(selected) == 1
    assert "false_alarm_ucb_per_1000" in calibration_df
    assert "false_alarm_budget_met" in calibration_df
    assert (calibration_df["false_alarm_confidence"] == 0.95).all()
    assert (calibration_df["false_alarm_budget_per_1000"] == 1.0).all()
    assert (
        calibration_df["calibration_samples"]
        == 80 - 3 * calibration_df["window_size"]
    ).all()
    expected_objective = np.where(
        calibration_df["false_alarm_budget_met"],
        1.0 - calibration_df["mean_detections"],
        calibration_df["false_alarm_ucb_per_1000"],
    )
    np.testing.assert_allclose(calibration_df["selection_objective"], expected_objective)
    assert selected.iloc[0]["window_size"] == best["wasserstein"]["window_size"]
    assert selected.iloc[0]["threshold_drift_score"] == pytest.approx(
        best["wasserstein"]["threshold_drift_score"]
    )
    pd.testing.assert_frame_equal(repeated_test_df, test_df)
    pd.testing.assert_frame_equal(repeated_per_stream_df, per_stream_df)
    assert (test_df["common_exposure_start"] == 3 * best["wasserstein"]["window_size"]).all()
    assert (
        test_df["common_exposure_samples"]
        == 60 - 3 * best["wasserstein"]["window_size"]
    ).all()
    assert (
        per_stream_df["n_false_positives_common"]
        == per_stream_df["n_false_positives_active"]
    ).all()
    np.testing.assert_allclose(
        per_stream_df["false_positives_per_1000_common"],
        per_stream_df["false_positives_per_1000_active"],
    )


def test_poisson_upper_bound_handles_zero_observed_false_alarms():
    upper = _poisson_rate_upper_bound(0, exposure=1_000, confidence=0.95)

    assert upper == pytest.approx(-np.log(0.05), rel=1e-12)


def test_calibration_selection_saturates_budget_among_admissible_trials():
    trials = pd.DataFrame(
        {
            "threshold": [10.0, 1.0],
            "mean_detections": [0.2, 0.8],
            "std_detections": [0.1, 0.2],
            "false_alarm_ucb_per_1000": [0.5, 0.9],
        }
    )

    best, selected = _select_best(
        trials,
        param_cols=["threshold"],
        verbose=False,
        name="example",
        aggressiveness_sort=[("threshold", False)],
        false_alarm_budget_per_1000=1.0,
    )

    assert selected["mean_detections"] == 0.8
    assert bool(selected["false_alarm_budget_met"])
    assert best == {"threshold": 1.0}


def test_calibration_selection_enforces_ucb_instead_of_observed_mean():
    trials = pd.DataFrame(
        {
            "threshold": [10.0, 1.0],
            "mean_detections": [0.5, 0.9],
            "std_detections": [0.1, 0.1],
            "false_alarm_ucb_per_1000": [0.8, 1.1],
        }
    )

    best, selected = _select_best(
        trials,
        param_cols=["threshold"],
        verbose=False,
        name="example",
        aggressiveness_sort=[("threshold", False)],
        false_alarm_budget_per_1000=1.0,
    )

    assert selected["false_alarm_ucb_per_1000"] == 0.8
    assert best == {"threshold": 10.0}


def test_calibration_selection_prefers_stability_before_aggressiveness():
    trials = pd.DataFrame(
        {
            "threshold": [2.0, 1.0],
            "mean_detections": [0.8, 0.8],
            "std_detections": [0.1, 0.2],
            "false_alarm_ucb_per_1000": [0.9, 0.9],
        }
    )

    best, selected = _select_best(
        trials,
        param_cols=["threshold"],
        verbose=False,
        name="example",
        aggressiveness_sort=[("threshold", False)],
        false_alarm_budget_per_1000=1.0,
    )

    assert selected.name == 0
    assert best == {"threshold": 2.0}


def test_calibration_selection_uses_structural_aggressiveness_as_final_tie_breaker():
    trials = pd.DataFrame(
        {
            "threshold": [2.0, 1.0],
            "mean_detections": [0.8, 0.8],
            "std_detections": [0.1, 0.1],
            "false_alarm_ucb_per_1000": [0.9, 0.9],
        }
    )

    best, selected = _select_best(
        trials,
        param_cols=["threshold"],
        verbose=False,
        name="example",
        aggressiveness_sort=[("threshold", False)],
        false_alarm_budget_per_1000=1.0,
    )

    assert selected.name == 1
    assert best == {"threshold": 1.0}


def test_calibration_selection_flags_least_violating_fallback():
    trials = pd.DataFrame(
        {
            "threshold": [2.0, 1.0],
            "mean_detections": [0.5, 0.8],
            "std_detections": [0.1, 0.1],
            "false_alarm_ucb_per_1000": [1.1, 1.3],
        }
    )

    best, selected = _select_best(
        trials,
        param_cols=["threshold"],
        verbose=False,
        name="example",
        aggressiveness_sort=[("threshold", False)],
        false_alarm_budget_per_1000=1.0,
    )

    assert selected.name == 0
    assert not bool(selected["false_alarm_budget_met"])
    assert best == {"threshold": 2.0}


@pytest.mark.parametrize("budget", [-1.0, np.inf, np.nan])
def test_calibration_selection_rejects_invalid_false_alarm_budget(budget):
    trials = pd.DataFrame(
        {
            "threshold": [1.0],
            "mean_detections": [0.0],
            "std_detections": [0.0],
            "false_alarm_ucb_per_1000": [0.1],
        }
    )

    with pytest.raises(ValueError, match="finite and non-negative"):
        _select_best(
            trials,
            param_cols=["threshold"],
            verbose=False,
            name="example",
            false_alarm_budget_per_1000=budget,
        )


@pytest.mark.parametrize("confidence", [0.5, 1.0, np.nan])
def test_poisson_upper_bound_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError, match="between 0.5 and 1"):
        _poisson_rate_upper_bound(0, exposure=1_000, confidence=confidence)


class _RecordingTrial:
    def __init__(self):
        self.user_attrs = {}

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


def test_budget_and_horizon_bound_the_chance_detection_rate():
    assert benchmark_protocol.FALSE_ALARM_BUDGET_PER_1000 * (
        benchmark_protocol.DETECTION_HORIZON / 1000
    ) == pytest.approx(benchmark_protocol.MAX_CHANCE_DETECTIONS_PER_HORIZON)


def test_calibration_false_alarm_rate_excludes_warmup():
    trial = _RecordingTrial()
    loss = _record_false_alarm_statistics(
        trial,
        detections=[[500, 700], []],
        streams=[np.zeros(1_000), np.zeros(1_000)],
        confidence=0.95,
        budget_per_1000=10.0,
        warmup=500,
    )

    assert trial.user_attrs["calibration_samples"] == 1_000
    assert trial.user_attrs["mean"] == pytest.approx(2.0)
    assert trial.user_attrs["false_alarm_ucb_per_1000"] == pytest.approx(
        _poisson_rate_upper_bound(2, exposure=1_000, confidence=0.95)
    )
    assert loss == pytest.approx(10.0 - 2.0)


def test_calibration_rejects_streams_shorter_than_warmup():
    with pytest.raises(ValueError, match="longer than the detector warm-up"):
        _record_false_alarm_statistics(
            _RecordingTrial(), [[]], [np.zeros(100)], 0.95, 1.0, warmup=100
        )


def test_detection_requires_an_alarm_within_the_horizon():
    def run_fixed(stream, starting_idx):
        del stream
        return [starting_idx + offset for offset in (15, 30)]

    args = ("scenario", "detector", np.zeros(40), run_fixed, {}, 10, 0)

    assert _eval_single_stream(*args, detection_horizon=10)[4] == 5
    outside_short_horizon = _eval_single_stream(*args, detection_horizon=5)
    assert outside_short_horizon[4] is None
    assert outside_short_horizon[6] == 5
    assert _eval_single_stream(*args, detection_horizon=None)[4] == 5


def test_horizon_running_past_the_stream_is_rejected():
    with pytest.raises(ValueError, match="ends before"):
        evaluate_with_params(
            test_streams={"example": [np.zeros(30)]},
            best_params={"adwin": {"delta": 0.01}},
            drift_at=20,
            algorithms=["adwin"],
            verbose=False,
            detection_horizon=20,
        )


def test_evaluation_reports_placebo_and_excess_detection_rate():
    rng = np.random.default_rng(0)
    drifted = [
        np.concatenate([rng.normal(0, 1, 100), rng.normal(0, 30, 60)])
        for _ in range(3)
    ]
    null = [rng.normal(0, 1, 160) for _ in range(4)]
    kwargs = dict(
        test_streams={"example": drifted},
        best_params={"ph": {"threshold": 50.0, "delta": 0.005}},
        drift_at=100,
        algorithms=["ph"],
        n_bootstrap=10,
        verbose=False,
        detection_horizon=40,
    )

    with_null, _ = evaluate_with_params(**kwargs, null_streams=null)
    without_null, _ = evaluate_with_params(**kwargs)

    row = with_null.iloc[0]
    assert row["placebo_n_streams"] == 4
    assert 0.0 <= row["placebo_detection_rate"] <= 1.0
    assert row["excess_detection_rate"] == pytest.approx(
        row["detection_rate"] - row["placebo_detection_rate"]
    )
    assert np.isnan(without_null.iloc[0]["placebo_detection_rate"])
    assert without_null.iloc[0]["placebo_n_streams"] == 0


def test_selection_at_a_search_bound_is_flagged():
    bounds = {"delta": (1e-5, 1e-1), "threshold": (50.0, 500.0)}

    assert _params_at_bound(
        pd.Series({"delta": 1.05e-5, "threshold": 275.0}), bounds, ("delta",)
    ) == "delta=low"
    assert _params_at_bound(
        pd.Series({"delta": 1e-3, "threshold": 499.0}), bounds, ("delta",)
    ) == "threshold=high"
    assert _params_at_bound(
        pd.Series({"delta": 1e-3, "threshold": 275.0}), bounds, ("delta",)
    ) == ""


class _ScriptedAlarms:
    """Detector that alarms exactly on samples equal to 1."""

    window_size = 2  # Wasserstein warm-up: 6 samples

    def __init__(self):
        self.drift_detected = False

    def update(self, x):
        self.drift_detected = x == 1


def _alarms_at(positions, length=40):
    stream = np.zeros(length)
    stream[list(positions)] = 1
    return stream


def test_fixed_detector_evaluation_follows_benchmark_metric_definitions():
    (row,) = evaluate_fixed_detector(
        _ScriptedAlarms,
        calibration_streams=[_alarms_at([10]), _alarms_at([])],
        test_streams={"example": [_alarms_at([8, 21, 30, 31]), _alarms_at([27])]},
        drift_at=20,
        null_streams=[_alarms_at([22]), _alarms_at([26])],
        detection_horizon=5,
        false_alarm_budget_per_1000=0.1,
        n_jobs=1,
    )

    # One calibration alarm over 2 x (40 - 6) active samples.
    assert row["calibration_false_positive_rate"] == pytest.approx(1 / 68 * 1000)
    assert row["calibration_fp_ucb_per_1000"] == pytest.approx(
        _poisson_rate_upper_bound(1, 68)
    )
    assert row["calibration_budget_met"] is False
    # Pre-drift alarms over the 20 - 6 active pre-drift samples.
    assert row["test_false_positive_rate"] == pytest.approx((1 / 14 * 1000) / 2)
    # Only the alarm at 21 falls in [20, 25); the one at 27 is too late.
    assert row["detection_rate"] == 0.5
    # One detection, at the placebo rate: the delay is kept but not reported.
    assert row["mean_first_delay_raw"] == 1.0
    assert not row["conditional_delay_reportable"]
    assert np.isnan(row["mean_first_delay"]) and np.isnan(row["median_first_delay"])
    # Extra post-drift alarms, averaged over all streams, not only detected ones.
    assert row["flicker_rate"] == pytest.approx((2 / 20 * 1000 + 0) / 2)
    assert row["placebo_detection_rate"] == 0.5
    assert row["excess_detection_rate"] == 0.0
    assert (row["n_streams"], row["placebo_n_streams"]) == (2, 2)


def test_fixed_detector_evaluation_shares_calibration_across_scenarios():
    rows = evaluate_fixed_detector(
        _ScriptedAlarms,
        calibration_streams=[_alarms_at([])],
        test_streams={"a": [_alarms_at([21])], "b": [_alarms_at([])]},
        drift_at=20,
        detection_horizon=5,
        n_jobs=1,
    )

    assert [row["scenario"] for row in rows] == ["a", "b"]
    assert [row["detection_rate"] for row in rows] == [1.0, 0.0]
    assert rows[0]["calibration_fp_ucb_per_1000"] == rows[1]["calibration_fp_ucb_per_1000"]
    assert np.isnan(rows[0]["placebo_detection_rate"])
