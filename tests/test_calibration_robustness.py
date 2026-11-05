import numpy as np
import pandas as pd
import pytest

from experiments.calibration_robustness import (
    SCENARIOS,
    _calibration_seed_base,
    _generate_scenario,
    _null_streams,
    build_markdown_report,
    detection_rate_ranges,
    parse_args,
    run_experiment,
    summarize_results,
)
from wasserstein_score import benchmark_protocol
from wasserstein_score.helper_functions import (
    abrupt_linear_drift_stream,
    bimodal_shift_streams,
    river_friedman_drift_stream,
)


@pytest.mark.parametrize(
    ("generator", "extra_kwargs"),
    [
        (abrupt_linear_drift_stream, {"noise": 0.1}),
        (bimodal_shift_streams, {"seed_base": 2_000, "separation": 0.9}),
    ],
)
def test_calibration_seed_changes_only_calibration_draws(generator, extra_kwargs):
    common = {
        "n": 30,
        "drift_at": 20,
        "n_calibrations_streams": 1,
        "n_test_streams": 1,
        **extra_kwargs,
    }
    first = generator(**common, calibration_seed_base=10_000)
    second = generator(**common, calibration_seed_base=20_000)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert not np.array_equal(first[2][0], second[2][0])
    np.testing.assert_array_equal(
        next(iter(first[4].values()))[0], next(iter(second[4].values()))[0]
    )
    np.testing.assert_array_equal(
        next(iter(first[5].values()))[0], next(iter(second[5].values()))[0]
    )


def test_river_calibration_seed_changes_only_calibration_draws():
    common = {
        "n": 30,
        "drift_at": 20,
        "n_calibrations_streams": 1,
        "n_test_streams": 1,
        "num_training_points": 5,
        "num_test_points": 5,
        "drift_scenarios": {"lea": {"river_type": "lea", "window": 0}},
    }
    first = river_friedman_drift_stream(
        **common, calibration_seed_base=10_000
    )
    second = river_friedman_drift_stream(
        **common, calibration_seed_base=20_000
    )

    np.testing.assert_array_equal(first[0], second[0])
    assert not np.array_equal(first[2][0], second[2][0])
    np.testing.assert_array_equal(
        next(iter(first[4].values()))[0], next(iter(second[4].values()))[0]
    )


def test_robustness_summary_uses_sample_dispersion():
    rows = []
    for value in (1.0, 3.0):
        rows.append({
            "scenario": "Linear Abrupt",
            "window_size": value,
            "threshold_drift_score": value,
            "calibration_fp_ucb_per_1000": value,
            "calibration_budget_met": True,
            "median_first_delay": value,
            "mean_first_delay": value,
            "mean_capped_delay": value,
            "detection_rate": value,
            "placebo_detection_rate": value,
            "excess_detection_rate": value,
            "mean_false_positives_active": value,
            "flicker_rate": value,
            "null_false_alarms_per_1000": value,
            "calibration_ucb_covers_null_rate_proxy": value == 1.0,
            "calibration_ucb_below_null_ci": value != 1.0,
        })

    summary = summarize_results(pd.DataFrame(rows))
    window = summary.loc[summary["metric"] == "window_size"].iloc[0]

    assert window["mean"] == 2.0
    assert window["std"] == pytest.approx(np.sqrt(2.0))
    assert (window["min"], window["max"], window["n"]) == (1.0, 3.0, 2)
    coverage = summary.loc[summary["metric"] == "calibration_ucb_covers_null_rate_proxy"].iloc[0]
    assert coverage["mean"] == 0.5
    assert coverage["n"] == 2


def test_report_leads_with_the_detection_rate_range_across_seeds():
    rows = [
        {"scenario": "Bimodal shift", "optuna_seed": seed, "window_size": window,
         "detection_rate": rate, "window_selection": "tpe"}
        for seed, window, rate in ((42, 1089, 1.0), (43, 907, 0.98), (44, 454, 0.72))
    ]
    results = pd.DataFrame(rows)
    ranges = detection_rate_ranges(results)
    (row,) = ranges.itertuples()
    assert (row.detection_rate_min, row.detection_rate_median, row.detection_rate_max) == (
        0.72, 0.98, 1.0)
    assert row.windows == "454, 907, 1089"

    report = build_markdown_report(
        results, pd.DataFrame(columns=["scenario", "metric"]),
        n_calibration_streams=20, n_test_streams=50, n_trials=40,
        false_alarm_budget_per_1000=0.1, false_alarm_confidence=0.95,
    )
    assert report.index("## Detection rate across calibration seeds") < report.index("## All metrics")
    assert "| Bimodal shift | 3 | 0.72-1.00 | 0.98 | 454, 907, 1089 |" in report


def test_cli_defaults_follow_canonical_benchmark_protocol(monkeypatch):
    monkeypatch.setattr("sys.argv", ["calibration_robustness.py"])
    args = parse_args()

    assert args.n_trials == benchmark_protocol.N_OPTUNA_TRIALS
    assert args.n_calibration_streams == benchmark_protocol.N_CALIBRATION_STREAMS
    assert args.n_test_streams == benchmark_protocol.N_TEST_STREAMS
    assert args.n_samples == benchmark_protocol.NUM_DATAPOINTS
    assert args.drift_at == benchmark_protocol.DRIFT_AT
    assert (args.threshold_min, args.threshold_max) == (
        benchmark_protocol.WASSERSTEIN_BOUNDS["threshold_drift_score"]
    )
    assert args.threshold_min >= 0.0
    assert args.false_alarm_budget_per_1000 == (
        benchmark_protocol.FALSE_ALARM_BUDGET_PER_1000
    )
    assert args.window_selection == benchmark_protocol.WASSERSTEIN_WINDOW_SELECTION
    assert args.batch_size == 1


def test_resume_rejects_results_at_another_batch_size(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["calibration_robustness.py", "--output-dir", str(tmp_path), "--resume",
         "--batch-size", "50"],
    )
    args = parse_args()
    metadata = {
        "n_calibration_streams": args.n_calibration_streams,
        "n_test_streams": args.n_test_streams,
        "n_trials": args.n_trials,
        "n_samples": args.n_samples,
        "drift_at": args.drift_at,
        "window_min": args.window_min,
        "window_max": args.window_max,
        "threshold_min": args.threshold_min,
        "threshold_max": args.threshold_max,
        "false_alarm_budget_per_1000": args.false_alarm_budget_per_1000,
        "false_alarm_confidence": args.false_alarm_confidence,
        "detection_horizon": benchmark_protocol.DETECTION_HORIZON,
        "window_selection": args.window_selection,
    }
    # No batch_size column: results that predate the option count as unbatched.
    pd.DataFrame([{"scenario": "Linear Abrupt", "optuna_seed": 42, **metadata}]).to_csv(
        tmp_path / "calibration_robustness_results.csv", index=False
    )

    with pytest.raises(ValueError, match="'batch_size' does not match 50"):
        run_experiment(args)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_robustness_null_streams_are_disjoint_from_every_replicate(scenario):
    config = dict(
        n=40, drift_at=30, n_calibration_streams=20, n_test_streams=50,
        n_training_points=20, n_validation_points=10,
    )
    first = _generate_scenario(
        scenario, calibration_seed_base=_calibration_seed_base(scenario, 0), **config
    )
    X_null, _ = _null_streams(
        scenario, 50, calibration_seed_base=_calibration_seed_base(scenario, 0), **config
    )
    used = {x.tobytes() for x in next(iter(first[4].values()))}
    for replicate in range(5):
        calib = _generate_scenario(
            scenario,
            calibration_seed_base=_calibration_seed_base(scenario, replicate),
            **dict(config, n_test_streams=0),
        )[2]
        used |= {x.tobytes() for x in calib}

    assert len(X_null) == 50
    assert not {x.tobytes() for x in X_null} & used


def test_resume_rejects_results_without_placebo_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["calibration_robustness.py", "--output-dir", str(tmp_path), "--resume"]
    )
    args = parse_args()
    metadata = {
        "n_calibration_streams": args.n_calibration_streams,
        "n_test_streams": args.n_test_streams,
        "n_trials": args.n_trials,
        "n_samples": args.n_samples,
        "drift_at": args.drift_at,
        "window_min": args.window_min,
        "window_max": args.window_max,
        "threshold_min": args.threshold_min,
        "threshold_max": args.threshold_max,
        "false_alarm_budget_per_1000": args.false_alarm_budget_per_1000,
        "false_alarm_confidence": args.false_alarm_confidence,
        "detection_horizon": benchmark_protocol.DETECTION_HORIZON,
        "window_selection": args.window_selection,
    }
    pd.DataFrame([{"scenario": "Linear Abrupt", "optuna_seed": 42, **metadata}]).to_csv(
        tmp_path / "calibration_robustness_results.csv", index=False
    )

    with pytest.raises(ValueError, match="placebo_detection_rate"):
        run_experiment(args)


def test_resume_rejects_old_raw_error_results(tmp_path, monkeypatch):
    from experiments import calibration_robustness as robustness

    monkeypatch.setattr(
        "sys.argv", ["calibration_robustness.py", "--output-dir", str(tmp_path), "--resume"]
    )
    args = parse_args()
    row = {column: 0.0 for column in robustness.REPORT_METRICS}
    for column in (
        "n_calibration_streams", "n_test_streams", "n_trials", "drift_at",
        "window_min", "window_max", "threshold_min", "threshold_max",
        "false_alarm_budget_per_1000", "false_alarm_confidence", "batch_size",
    ):
        row[column] = getattr(args, column)
    row.update(n_samples=args.n_samples, window_selection=args.window_selection,
               detection_horizon=benchmark_protocol.DETECTION_HORIZON,
               scenario="Linear Abrupt", optuna_seed=42)
    pd.DataFrame([row]).to_csv(tmp_path / "calibration_robustness_results.csv", index=False)

    with pytest.raises(ValueError, match="calibration-median normalization"):
        run_experiment(args)
