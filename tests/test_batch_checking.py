"""Batch checking: samples arrive in batches and every detector may alarm only
on the last sample of a batch."""

import numpy as np
import pytest
from river import drift

from wasserstein_score.batch_detectors import BatchKSWIN, BatchPageHinkley
from wasserstein_score.evaluation import (
    _warmup_length,
    alarm_levels,
    calibrate_and_test,
    calibrate_detectors_with_widening,
    run_detector,
    wasserstein_score_trajectory,
)
from wasserstein_score.wasserstein_drift_detector import WassersteinDriftDetector

PARAMS = {
    "wasserstein": {"window_size": 40, "threshold_drift_score": 0.5},
    "adwin": {"delta": 0.01},
    "kswin": {"window_size": 60, "alpha": 0.01, "stat_size": 20},
    "ph": {"threshold": 20.0, "delta": 0.005},
}


def _drift_stream(seed=0, n_pre=600, n_post=600):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.exponential(size=n_pre), 6 * rng.exponential(size=n_post)])


def _alarms(detector, stream):
    alarms = []
    for i, x in enumerate(stream):
        detector.update(x)
        if detector.drift_detected:
            alarms.append(i)
    return alarms


@pytest.mark.parametrize("batch_cls, river_cls, kwargs", [
    (BatchKSWIN, drift.KSWIN, {"window_size": 60, "alpha": 0.01, "stat_size": 20, "seed": 42}),
    (BatchPageHinkley, drift.PageHinkley, {"threshold": 20.0, "delta": 0.005}),
])
def test_batch_size_one_reproduces_the_river_detector(batch_cls, river_cls, kwargs):
    stream = _drift_stream()
    expected = _alarms(river_cls(**kwargs), stream)
    assert expected
    assert _alarms(batch_cls(**kwargs, batch_size=1), stream) == expected


@pytest.mark.parametrize("algorithm", sorted(PARAMS))
@pytest.mark.parametrize("batch_size", [5, 20])
def test_alarms_fall_on_the_last_sample_of_a_batch(algorithm, batch_size):
    stream = _drift_stream()
    starting_idx = 1_000
    alarms = run_detector(
        algorithm, stream, {**PARAMS[algorithm], "batch_size": batch_size},
        starting_idx=starting_idx,
    )
    assert any(a >= starting_idx + 600 for a in alarms)
    assert all((a - starting_idx + 1) % batch_size == 0 for a in alarms)


def test_page_hinkley_alarms_at_the_end_of_the_batch_holding_the_crossing():
    # A persistent shift keeps the statistic above the threshold, so the
    # batched detector alarms on the last sample of the crossing's batch.
    rng = np.random.default_rng(1)
    stream = np.concatenate([rng.normal(size=203), rng.normal(5, 1, size=200)])
    first = run_detector("ph", stream, PARAMS["ph"])[0]
    batch_size = 10
    batched = run_detector("ph", stream, {**PARAMS["ph"], "batch_size": batch_size})[0]
    assert batched == (first // batch_size + 1) * batch_size - 1


def test_batched_wasserstein_tests_the_same_time_span():
    detector = WassersteinDriftDetector(window_size=100, batch_size=10)
    assert detector.score_history == 10
    assert detector.wasserstein_scores.maxlen == 20
    with pytest.raises(ValueError, match="fewer than two"):
        WassersteinDriftDetector(window_size=100, batch_size=100)
    with pytest.raises(ValueError, match="positive"):
        WassersteinDriftDetector(window_size=100, batch_size=0)
    with pytest.raises(ValueError, match="positive"):
        BatchKSWIN(batch_size=0)


@pytest.mark.parametrize("batch_size", [1, 10])
def test_batched_score_trajectory_predicts_the_alarm_count(batch_size):
    stream = _drift_stream(seed=2)
    scores = wasserstein_score_trajectory(stream, 40, batch_size=batch_size)
    assert len(scores) > 0
    for theta in (0.0, 0.5, 2.0):
        alarms = run_detector(
            "wasserstein", stream,
            {"window_size": 40, "threshold_drift_score": theta, "batch_size": batch_size},
        )
        assert len(alarms) == int(np.sum(alarm_levels(scores) > theta))


def test_warmup_rounds_up_to_whole_batches():
    assert _warmup_length("wasserstein", {"window_size": 500}) == 1_500
    assert _warmup_length("wasserstein", {"window_size": 500, "batch_size": 7}) == 1_505
    assert _warmup_length("ph", {"batch_size": 100}) == 100
    assert _warmup_length("kswin", {"window_size": 500, "batch_size": 100}) == 500


def test_calibration_records_the_batch_size():
    rng = np.random.default_rng(0)
    streams = [rng.exponential(size=1_000) for _ in range(2)]
    params, table, _ = calibrate_detectors_with_widening(
        streams, algorithms=["wasserstein", "adwin", "ph"], max_rounds=0, n_trials=3,
        verbose=False, false_alarm_budget_per_1000=5.0, batch_size=10,
        wasserstein_bounds={"threshold_drift_score": (0.0, 4.0), "window_size": (50, 100)},
    )
    assert {p["batch_size"] for p in params.values()} == {10}
    assert (table["batch_size"] == 10).all()
    for algorithm, selected in params.items():
        assert all((a + 1) % 10 == 0 for a in run_detector(algorithm, streams[0], selected))


@pytest.mark.parametrize("drift_at, horizon", [(605, 100), (600, 105)])
def test_batches_must_not_straddle_the_drift_or_the_horizon_end(drift_at, horizon):
    with pytest.raises(ValueError, match="multiple of batch_size"):
        calibrate_and_test(
            calibration_streams=[np.zeros(1_000)],
            test_streams={"example": [np.zeros(1_000)]},
            drift_at=drift_at, detection_horizon=horizon,
            algorithms=["adwin"], verbose=False, batch_size=10,
        )


def test_calibrate_and_test_runs_every_detector_in_batches():
    rng = np.random.default_rng(4)
    calibration = [rng.exponential(size=1_200) for _ in range(2)]
    test = [
        np.concatenate([rng.exponential(size=800), 6 * rng.exponential(size=400)])
        for _ in range(3)
    ]
    best, _, test_df, per_stream = calibrate_and_test(
        calibration_streams=calibration, test_streams={"example": test},
        drift_at=800, detection_horizon=400, batch_size=10,
        wasserstein_bounds={"threshold_drift_score": (0.0, 4.0), "window_size": (50, 100)},
        kswin_bounds={"window_size": (50, 100), "alpha": (1e-3, 1e-2), "stat_size": (10, 20)},
        n_trials=3, n_bootstrap=10, widening_rounds=0, verbose=False,
        false_alarm_budget_per_1000=5.0,
    )
    assert {p["batch_size"] for p in best.values()} == {10}
    assert set(test_df["algorithm"]) == {"wasserstein", "adwin", "kswin", "ph"}
    delays = per_stream["first_delay"].dropna()
    assert len(delays) > 0
    assert (delays % 10 == 9).all()
    assert (test_df["warmup_samples"] % 10 == 0).all()
