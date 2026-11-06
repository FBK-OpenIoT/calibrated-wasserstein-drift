"""Matched-budget ablation, budget-matched sensitivity and seeded KSWIN."""

import numpy as np
import pandas as pd
import pytest

from experiments import ablation_study
from wasserstein_score import benchmark_protocol
from wasserstein_score.ablation_variants import VARIANT_CLASSES
from wasserstein_score.evaluation import (
    _calibrate_wasserstein_largest_window,
    calibrate_and_test,
    _poisson_rate_upper_bound,
    _run_wasserstein,
    budget_matched_threshold,
    calibrate_detectors,
    excursion_maxima,
    run_detector,
    wasserstein_score_trajectory,
)
from wasserstein_score.helper_functions import (
    river_friedman_drift_stream,
)

WINDOW = 20


def _streams(n_streams=3, length=400, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.exponential(size=length) for _ in range(n_streams)]


def test_excursion_maxima_returns_one_maximum_per_positive_run():
    scores = np.array([0, 1.0, 3.0, 0, 0, 2.0, 0, 0.5, 4.0, 1.0])
    np.testing.assert_array_equal(excursion_maxima(scores), [3.0, 2.0, 4.0])
    assert excursion_maxima(np.zeros(5)).size == 0


@pytest.mark.parametrize("threshold", [0.0, 0.3, 1.0, 2.5])
def test_excursion_count_reproduces_hysteresis_alarms(threshold):
    for stream in _streams():
        maxima = excursion_maxima(wasserstein_score_trajectory(stream, WINDOW))
        alarms = _run_wasserstein(stream, WINDOW, threshold, starting_idx=0)
        assert len(alarms) == int(np.sum(maxima > threshold))


def test_budget_matched_threshold_is_the_lowest_admissible_threshold():
    streams = _streams()
    budget = 15.0
    matched = budget_matched_threshold(
        streams, WINDOW, false_alarm_budget_per_1000=budget, n_jobs=1
    )
    exposure = sum(len(s) - 3 * WINDOW for s in streams)
    alarms = sum(
        len(_run_wasserstein(s, WINDOW, matched["threshold_drift_score"], 0)) for s in streams
    )

    assert matched["calibration_samples"] == exposure
    assert matched["n_false_alarms"] == alarms
    assert matched["false_alarm_budget_met"]
    assert _poisson_rate_upper_bound(alarms, exposure) <= budget
    # Any lower threshold raises more alarms than the budget admits.
    lower = np.nextafter(matched["threshold_drift_score"], -np.inf)
    more = sum(len(_run_wasserstein(s, WINDOW, lower, 0)) for s in streams)
    assert more > alarms
    assert _poisson_rate_upper_bound(more, exposure) > budget


def test_budget_matched_threshold_flags_an_unattainable_budget():
    streams = _streams()
    matched = budget_matched_threshold(
        streams, WINDOW, false_alarm_budget_per_1000=0.1, n_jobs=1
    )
    assert matched["n_false_alarms"] == 0
    assert not matched["false_alarm_budget_met"]


def test_kswin_runs_are_reproducible():
    stream = _streams(n_streams=1, length=600)[0]
    params = {"window_size": 100, "alpha": 0.05, "stat_size": 30}
    first = run_detector("kswin", stream, params)
    assert first == run_detector("kswin", stream, params)
    assert benchmark_protocol.KSWIN_SEED == 42


# --- Largest admissible window --------------------------------------------

def test_largest_admissible_window_takes_the_top_window_and_its_exact_threshold():
    streams = _streams()
    budget = 15.0
    bounds = {"threshold_drift_score": (0.0, 4.0), "window_size": (10, WINDOW)}
    params, row, table = _calibrate_wasserstein_largest_window(
        streams, bounds, 0, 1, False, false_alarm_budget_per_1000=budget, stream_backend="threading",
    )
    matched = budget_matched_threshold(streams, WINDOW, false_alarm_budget_per_1000=budget, n_jobs=1)
    assert params == {"window_size": WINDOW, "threshold_drift_score": matched["threshold_drift_score"]}
    assert len(table) == 1 and row["false_alarm_budget_met"]
    assert row["n_false_alarms"] == matched["n_false_alarms"]
    # The window is at its upper bound by design, which is not flagged.
    assert row["params_at_bound"] == ""


def test_largest_admissible_window_steps_down_until_the_budget_can_be_met():
    # With 2 x 400 samples, a warm-up of 3 x 120 leaves too little exposure for
    # even zero alarms to meet the budget; smaller windows leave enough.
    streams = _streams(n_streams=2)
    bounds = {"threshold_drift_score": (0.0, 4.0), "window_size": (20, 120)}
    params, row, table = _calibrate_wasserstein_largest_window(
        streams, bounds, 0, 1, False, false_alarm_budget_per_1000=6.0, stream_backend="threading",
    )
    assert list(table["window_size"])[:2] == [120, 70]
    assert not table["false_alarm_budget_met"].iloc[0]
    assert row["false_alarm_budget_met"]
    assert params["window_size"] == table["window_size"].iloc[-1] < 120


def test_largest_admissible_window_counts_alarms_without_hysteresis():
    streams = _streams()
    bounds = {"threshold_drift_score": (0.0, 4.0), "window_size": (10, WINDOW)}
    kwargs = dict(false_alarm_budget_per_1000=15.0, stream_backend="threading")
    # The calibrator raises if the predicted and actual alarm counts differ.
    _, full, _ = _calibrate_wasserstein_largest_window(
        streams, bounds, 0, 1, False, detector_cls=VARIANT_CLASSES["full"], **kwargs)
    _, no_hyst, _ = _calibrate_wasserstein_largest_window(
        streams, bounds, 0, 1, False, detector_cls=VARIANT_CLASSES["no_hysteresis"], **kwargs)
    assert full["false_alarm_budget_met"] and no_hyst["false_alarm_budget_met"]
    assert no_hyst["threshold_drift_score"] > full["threshold_drift_score"]


def test_largest_admissible_window_is_not_widened():
    streams = _streams(n_streams=2, length=300)
    bounds = {"threshold_drift_score": (0.0, 1e-3), "window_size": (10, 20)}
    params, table, final_bounds = calibrate_detectors_with_widening(
        streams, algorithms=["wasserstein"], max_rounds=2, verbose=False,
        # 2 x (300 - 60) samples admit one alarm at this budget, so it binds.
        wasserstein_bounds=bounds, false_alarm_budget_per_1000=10.0,
    )
    assert set(table["round"]) == {0}
    assert final_bounds["wasserstein"] == bounds
    assert params["wasserstein"]["window_size"] == 20
    # The threshold is solved over its domain, not the search range.
    assert params["wasserstein"]["threshold_drift_score"] > 1e-3
    assert table["selected"].sum() == 1


def test_unknown_window_selection_is_rejected():
    with pytest.raises(ValueError, match="window selection"):
        calibrate_detectors(_streams(), algorithms=["wasserstein"], verbose=False,
                            wasserstein_window_selection="smallest")


def test_ablation_calibrates_each_variant_with_the_benchmark_widening(monkeypatch):
    seen = {}

    def fake_widening(streams, algorithms, max_warmup_len, wasserstein_cls, **kwargs):
        seen.update(algorithms=algorithms, cls=wasserstein_cls, max_warmup_len=max_warmup_len)
        table = pd.DataFrame([{"selected": True, "round": 2, "params_at_bound": ""}])
        bounds = {"wasserstein": {"threshold_drift_score": (0.0, 16.0), "window_size": (500, 1000)}}
        return {"wasserstein": {"window_size": 600, "threshold_drift_score": 9.0}}, table, bounds

    monkeypatch.setattr(ablation_study, "calibrate_detectors_with_widening", fake_widening)
    params, row, bounds = ablation_study.calibrate_variant("no_hysteresis", [])

    assert seen["cls"] is VARIANT_CLASSES["no_hysteresis"]
    assert seen["algorithms"] == ["wasserstein"]
    assert seen["max_warmup_len"] == benchmark_protocol.DRIFT_AT - benchmark_protocol.STARTING_IDX
    assert params == {"window_size": 600, "threshold_drift_score": 9.0}
    assert row["round"] == 2 and bounds["threshold_drift_score"] == (0.0, 16.0)


def test_river_friedman_has_a_single_transition_scenario():
    _, _, _, _, X_test, _ = river_friedman_drift_stream(
        n=1_200, drift_at=600, n_calibrations_streams=1, n_test_streams=1,
        num_training_points=20, num_test_points=10,
    )
    assert set(X_test) == {"abrupt_local_lea", "abrupt_global_gsg", "gradual_fast"}


def test_calibrate_detectors_caps_windows_to_the_calibration_streams():
    streams = _streams(n_streams=2, length=300)
    params, calibration = calibrate_detectors(
        streams, algorithms=["wasserstein"],
        wasserstein_bounds={"threshold_drift_score": (0.0, 4.0), "window_size": (10, 500)},
        n_trials=3, verbose=False, false_alarm_budget_per_1000=1e6,
    )
    # 0.7 * 300 // 3: the warm-up fits inside every calibration stream.
    assert params["wasserstein"]["window_size"] <= 70
    assert calibration["selected"].sum() == 1


# --- Range widening -------------------------------------------------------

from wasserstein_score.evaluation import (  # noqa: E402
    PARAMETER_DOMAINS,
    calibrate_detectors_with_widening,
    widened_bounds,
)


def _row(budget_met, at_bound=""):
    return pd.Series({"false_alarm_budget_met": budget_met, "params_at_bound": at_bound})


def test_widening_extends_the_conservative_end_of_an_inadmissible_selection():
    new = widened_bounds("adwin", {"delta": (1e-5, 0.1)}, _row(False), max_warmup_len=10_000)
    assert new["delta"] == pytest.approx((1e-9, 0.1))
    new = widened_bounds("ph", {"threshold": (50.0, 500.0), "delta": (1e-5, 5.0)},
                         _row(False), max_warmup_len=10_000)
    assert new["threshold"] == (50.0, 950.0) and new["delta"] == (1e-5, 5.0)


def test_window_sizes_are_capped_at_the_detection_horizon():
    cap = benchmark_protocol.MAX_WINDOW_SIZE
    assert cap == benchmark_protocol.DETECTION_HORIZON
    for bounds in (benchmark_protocol.WASSERSTEIN_BOUNDS, benchmark_protocol.KSWIN_BOUNDS):
        assert bounds["window_size"][1] == cap
    for algo in ("wasserstein", "kswin"):
        assert PARAMETER_DOMAINS[algo]["window_size"][1] == cap


def test_widening_extends_flagged_bounds_within_domain_and_warmup():
    bounds = {"threshold_drift_score": (0.0, 4.0), "window_size": (500, 800)}
    new = widened_bounds("wasserstein", bounds, _row(True, "window_size=low"), 10_000)
    assert new["window_size"] == (200, 800)
    # The window domain stops at MAX_WINDOW_SIZE however long the streams.
    for algo, params in (("wasserstein", bounds),
                         ("kswin", {"window_size": (500, 800), "alpha": (1e-5, 1e-2),
                                    "stat_size": (20, 80)})):
        new = widened_bounds(algo, params, _row(True, "window_size=high"), 100_000)
        assert new["window_size"] == (500, benchmark_protocol.MAX_WINDOW_SIZE)
    # The warm-up cap (0.7 * 3000 // 3 = 700) limits the upper window first.
    new = widened_bounds("wasserstein", {**bounds, "window_size": (500, 600)},
                         _row(True, "window_size=high"), 3_000)
    assert new["window_size"] == (500, 700)
    assert isinstance(new["window_size"][0], int)
    # Nothing left to widen: theta already spans its domain's lower end.
    assert widened_bounds("wasserstein", bounds, _row(True, "threshold_drift_score=low"), 10_000) == bounds


def test_calibrate_and_test_widens_by_default_and_can_be_disabled():
    streams = _streams(n_streams=2, length=300)
    test = {"s": _streams(n_streams=1, length=300, seed=1)}
    kwargs = dict(
        drift_at=200, algorithms=["wasserstein"], n_trials=3, verbose=False,
        wasserstein_bounds={"threshold_drift_score": (0.0, 0.1), "window_size": (10, 20)},
        false_alarm_budget_per_1000=5.0, detection_horizon=50, n_bootstrap=10,
        wasserstein_window_selection="tpe",
    )
    _, widened, _, _ = calibrate_and_test(streams, test, **kwargs)
    _, fixed, _, _ = calibrate_and_test(streams, test, widening_rounds=0, **kwargs)
    assert widened["round"].max() >= 1
    assert set(fixed["round"]) == {0} and fixed["selected"].sum() == 1


def test_calibration_with_widening_selects_over_all_rounds():
    streams = _streams(n_streams=2, length=300)
    params, table, bounds = calibrate_detectors_with_widening(
        streams, algorithms=["wasserstein"], max_rounds=2, verbose=False,
        wasserstein_bounds={"threshold_drift_score": (0.0, 0.1), "window_size": (10, 20)},
        n_trials=3, false_alarm_budget_per_1000=5.0, wasserstein_window_selection="tpe",
    )
    assert table["round"].max() >= 1
    assert bounds["wasserstein"]["threshold_drift_score"][1] > 0.1
    assert table["selected"].sum() == 1
    chosen = table[table["selected"]].iloc[0]
    assert params["wasserstein"]["threshold_drift_score"] == chosen["threshold_drift_score"]


def test_widening_moves_towards_sensitivity_when_the_budget_does_not_bind():
    row = pd.Series({"false_alarm_budget_met": True, "params_at_bound": "",
                     "mean_detections": 0.0})
    new = widened_bounds("ph", {"threshold": (50.0, 500.0), "delta": (1e-5, 5.0)}, row, 10_000)
    assert new["threshold"] == (1e-3, 500.0)
    assert new["delta"] == pytest.approx((1e-5 / 5e5, 5.0))
    row["mean_detections"] = 0.05
    assert widened_bounds("ph", {"threshold": (50.0, 500.0), "delta": (1e-5, 5.0)}, row, 10_000) == {
        "threshold": (50.0, 500.0), "delta": (1e-5, 5.0)}


def test_widened_kswin_selection_keeps_integer_parameters():
    streams = _streams(n_streams=2, length=400)
    params, table, _ = calibrate_detectors_with_widening(
        streams, algorithms=["kswin", "adwin"], max_rounds=1, verbose=False,
        kswin_bounds={"window_size": (50, 60), "alpha": (1e-3, 1e-2), "stat_size": (10, 20)},
        n_trials=2, false_alarm_budget_per_1000=0.1,
    )
    assert isinstance(params["kswin"]["window_size"], int)
    assert isinstance(params["kswin"]["stat_size"], int)
    run_detector("kswin", streams[0], params["kswin"])


def test_widening_never_narrows_a_range():
    bounds = {"threshold_drift_score": (0.0, 4.0), "window_size": (5, 10)}
    new = widened_bounds("wasserstein", bounds, _row(True, "window_size=low"), 60)
    assert new == bounds
    # A cap below the current upper window keeps the current one.
    new = widened_bounds("wasserstein", {"threshold_drift_score": (0.0, 4.0), "window_size": (10, 30)},
                         _row(True, "window_size=high"), 60)
    assert new["window_size"] == (10, 30)
