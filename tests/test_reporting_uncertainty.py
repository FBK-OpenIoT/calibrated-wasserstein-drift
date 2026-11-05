"""Regression tests for stream-level uncertainty and honest delay reporting."""

import json

import numpy as np
import pandas as pd
import pytest

from wasserstein_score import evaluation as ev


@pytest.mark.parametrize("successes, expected", [
    (0, (0.0, 0.0713475991)),
    (25, (0.3664451432, 0.6335548568)),
    (50, (0.9286524009, 1.0)),
])
def test_wilson_reference_values(successes, expected):
    assert ev._wilson_ci(successes, 50) == pytest.approx(expected)


def test_wilson_small_empty_and_invalid_samples():
    assert ev._wilson_ci(1, 1)[0] < 1
    assert all(np.isnan(ev._wilson_ci(0, 0)))
    with pytest.raises(ValueError):
        ev._wilson_ci(2, 1)
    with pytest.raises(ValueError):
        ev._wilson_ci(1, 2, 1.0)


class ScriptedDetector:
    window_size = 2

    def update(self, x):
        self.drift_detected = x == 1


def stream(*alarms, length=40):
    values = np.zeros(length)
    values[list(alarms)] = 1
    return values


def test_capped_delay_penalizes_misses_in_fixed_detector():
    (row,) = ev.evaluate_fixed_detector(
        ScriptedDetector, [stream()],
        {"example": [stream(20), stream(24), stream(25), stream()]},
        drift_at=20, detection_horizon=5, n_jobs=1, n_bootstrap=100,
        null_streams=[stream(), stream()],
    )
    assert row["detection_rate"] == 0.5
    assert row["mean_first_delay_raw"] == 2
    assert row["median_first_delay_raw"] == 2
    assert np.isnan(row["mean_first_delay"])  # two detections: below the minimum
    assert row["median_capped_delay"] == 4.5
    assert row["mean_capped_delay"] == 3.5  # (0 + 4 + 5 + 5) / 4
    assert row["mean_capped_delay_ci_low"] <= 3.5 <= row["mean_capped_delay_ci_high"]
    assert row["placebo_detection_rate_ci_high"] > 0
    assert np.isnan(row["null_rate_ci_high"])
    assert row["null_poisson_ucb_per_1000"] > 0
    assert np.isnan(row["calibration_ucb_below_null_ci"])


def run_scripted(values, starting_idx, **params):
    return [starting_idx + int(i) for i in np.flatnonzero(values)]


def test_evaluation_exports_capped_delays_and_audits_without_rerunning_null(monkeypatch, capsys, tmp_path):
    # Serial joblib makes the scripted runner visible and counts invocations.
    monkeypatch.setattr(ev, "Parallel", lambda **kw: lambda jobs: [f(*a, **k) for f, a, k in jobs])
    calls = []

    def runner(values, starting_idx, **params):
        calls.append(id(values))
        return run_scripted(values, starting_idx, **params)

    monkeypatch.setattr(ev, "_run_adwin", runner)
    null = [stream(12, 22), stream()]
    test, per_stream = ev.evaluate_with_params(
        {"example": [stream(20), stream(24), stream(25), stream()]},
        {"adwin": {"delta": 0.1}}, drift_at=20, detection_horizon=5,
        null_streams=null, n_bootstrap=100, verbose=True,
        calibration_results=pd.DataFrame([{
            "algorithm": "adwin", "selected": True,
            "false_alarm_ucb_per_1000": 1.0,
            "false_alarm_confidence": np.nan,  # Mixed legacy combination table.
            "false_alarm_budget_per_1000": 2.0,
        }]),
    )
    row = test.iloc[0]
    assert per_stream["capped_delay"].tolist() == [0, 4, 5, 5]
    assert row["mean_capped_delay"] == 3.5
    assert row["detection_rate_ci_low"] == pytest.approx(ev._wilson_ci(2, 4)[0])
    assert len(calls) == 6  # test + null, no second audit pass
    assert json.loads(row["null_alarm_counts"]) == [2, 0]
    assert row["null_n_false_alarms"] == 2
    assert row["null_active_samples"] == 60
    assert row["null_false_alarms_per_1000"] == pytest.approx(100 / 3)
    assert not row["calibration_ucb_covers_null_rate_proxy"]
    assert row["null_false_alarm_budget_per_1000"] == 2
    path = tmp_path / "summary.csv"
    test.to_csv(path, index=False)
    saved = pd.read_csv(path).iloc[0]
    assert json.loads(saved["null_alarm_counts"]) == [2, 0]
    assert json.loads(saved["null_active_exposures"]) == [30, 30]
    output = capsys.readouterr().out
    assert "(Wilson)" in output and "capped_delay=3.5" in output
    assert "proxy" in output and "assumption-dependent" in output


def test_calibration_pipeline_audits_selected_bound_without_tuning_on_null(monkeypatch):
    monkeypatch.setattr(ev, "Parallel", lambda **kw: lambda jobs: [f(*a, **k) for f, a, k in jobs])
    monkeypatch.setattr(ev, "_run_adwin", run_scripted)
    calibration = [stream(12)]

    def calibrate(streams, **kwargs):
        assert len(streams) == 1
        np.testing.assert_array_equal(streams[0], calibration[0])
        assert "null_streams" not in kwargs
        table = pd.DataFrame([
            {"algorithm": "adwin", "selected": False, "false_alarm_ucb_per_1000": 1000},
            {"algorithm": "adwin", "selected": True, "false_alarm_ucb_per_1000": 0.1},
        ])
        return {"adwin": {"delta": 0.1}}, table, {}

    monkeypatch.setattr(ev, "calibrate_detectors_with_widening", calibrate)
    _, _, summary, _ = ev.calibrate_and_test(
        calibration, {"example": [stream(20)]}, drift_at=20,
        algorithms=["adwin"], detection_horizon=5, n_bootstrap=50,
        null_streams=[stream(12), stream(13)],
        false_alarm_confidence=0.9, false_alarm_budget_per_1000=0.2, verbose=False,
    )
    row = summary.iloc[0]
    assert not row["calibration_ucb_covers_null_rate_proxy"]
    assert row["null_false_alarm_confidence"] == 0.9
    assert row["null_false_alarm_budget_per_1000"] == 0.2
    assert row["null_poisson_ucb_per_1000"] == pytest.approx(ev._poisson_rate_upper_bound(2, 60, 0.9))


@pytest.mark.parametrize("horizon, expected", [(5, 5.0), (None, np.nan)])
def test_all_missed_and_unbounded_delay(horizon, expected):
    (row,) = ev.evaluate_fixed_detector(
        ScriptedDetector, [stream()], {"example": [stream(), stream()]},
        drift_at=20, detection_horizon=horizon, n_jobs=1, n_bootstrap=20,
    )
    assert np.isnan(row["mean_first_delay"])
    if horizon is None:
        assert np.isnan(row["mean_capped_delay"])
    else:
        assert row["mean_capped_delay"] == expected
    assert row["detection_rate_ci_high"] > 0
    assert row["null_n_streams"] == 0
    assert np.isnan(row["null_poisson_budget_met"])


def audit(alarms, lengths, **kwargs):
    return ev._null_alarm_audit(
        alarms, [np.zeros(n) for n in lengths], warmup=10, starting_idx=100,
        n_bootstrap=2000, ci_level=0.95, bootstrap_seed=3,
        false_alarm_confidence=0.95, false_alarm_budget_per_1000=5,
        **kwargs,
    )


def test_null_audit_exposure_boundaries_and_unequal_lengths():
    row = audit([[100, 109, 110, 120, 121], [110, 111]], [20, 30], calibration_ucb=150)
    # Half-open active windows [110,120), [110,130).
    assert json.loads(row["null_alarm_counts"]) == [1, 2]
    assert json.loads(row["null_active_exposures"]) == [10, 20]
    assert row["null_false_alarms_per_1000"] == 100
    assert row["null_rate_ci_low"] == row["null_rate_ci_high"] == 100
    assert row["calibration_ucb_covers_null_rate_proxy"]
    assert row["null_count_dispersion"] == 0


def test_null_audit_detects_clustered_counts_and_undercoverage():
    row = audit([list(range(110, 120)), [], [], []], [30] * 4, calibration_ucb=0)
    assert row["null_count_dispersion"] == pytest.approx(10)
    assert row["null_rate_ci_high"] > row["null_false_alarms_per_1000"]
    assert not row["calibration_ucb_covers_null_rate_proxy"]
    stable = audit([[110, 111], [110, 111], [110, 111]], [30] * 3, calibration_ucb=1)
    assert stable["calibration_ucb_below_null_ci"]


def test_null_audit_zero_single_and_empty_are_not_false_certificates():
    for alarms, lengths in [([], []), ([[]], [20]), ([[], []], [20, 20])]:
        row = audit(alarms, lengths)
        assert np.isnan(row["null_rate_ci_high"])
        assert np.isnan(row["calibration_ucb_covers_null_rate_proxy"])
    row = audit([[]], [20])
    assert row["null_poisson_ucb_per_1000"] == pytest.approx(-np.log(0.05) / 10 * 1000)
    with pytest.raises(ValueError, match="warm-up"):
        audit([[]], [10])


@pytest.mark.parametrize("detected, n, placebo, expected", [
    (4, 50, 0.0, False),    # too few detections for a delay
    (5, 50, 0.0, True),
    (6, 50, 0.06, False),   # 0.12 detected, Wilson lower bound ~0.056 < placebo
    (46, 50, 0.02, True),
    (5, 50, np.nan, True),  # no placebo streams: only the minimum applies
])
def test_conditional_delay_requires_detections_above_chance(detected, n, placebo, expected):
    assert ev.delay_reportable(detected, n, placebo) is expected


def test_delay_summary_masks_unreportable_conditional_delays():
    rng = np.random.default_rng(0)
    delays, capped = [1.0, 2.0, 30.0], [1.0, 2.0, 30.0, 5.0]
    shown = ev._delay_summary(delays, capped, True, 50, 0.95, rng, rng)
    assert shown["median_first_delay"] == 2.0 and shown["mean_first_delay"] == 11.0
    assert shown["median_first_delay_ci_low"] <= 2.0 <= shown["median_first_delay_ci_high"]
    hidden = ev._delay_summary(delays, capped, False, 50, 0.95, rng, rng)
    assert all(np.isnan(hidden[k]) for k in (
        "mean_first_delay", "median_first_delay", "std_first_delay",
        "mean_first_delay_ci_low", "median_first_delay_ci_high"))
    assert hidden["median_first_delay_raw"] == 2.0
    assert hidden["mean_capped_delay"] == shown["mean_capped_delay"] == 9.5
    assert hidden["median_capped_delay"] == 3.5
