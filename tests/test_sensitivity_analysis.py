import numpy as np
import pytest

from experiments import sensitivity_analysis
from wasserstein_score.benchmark_protocol import (
    BIMODAL_SHIFT_SEPARATION,
    WASSERSTEIN_BOUNDS,
)
from wasserstein_score.helper_functions import benchmark_datasets


def test_sensitivity_covers_threshold_search_space_and_bimodal_scenario():
    low, high = WASSERSTEIN_BOUNDS["threshold_drift_score"]
    assert min(sensitivity_analysis.THRESHOLD_GRID) == pytest.approx(low)
    assert max(sensitivity_analysis.THRESHOLD_GRID) == pytest.approx(high)
    assert low in sensitivity_analysis.GRID_THRESHOLDS
    w_low, w_high = WASSERSTEIN_BOUNDS["window_size"]
    for grid in (sensitivity_analysis.WINDOW_SIZE_GRID, sensitivity_analysis.GRID_WINDOW_SIZES):
        assert (min(grid), max(grid)) == (w_low, w_high)

    bimodal = sensitivity_analysis.DATASETS["Bimodal Shift"]
    assert bimodal["kwargs"]["separation"] == pytest.approx(
        BIMODAL_SHIFT_SEPARATION
    )
    assert bimodal["kwargs"]["seed_base"] == 2000


def test_sensitivity_uses_the_benchmark_datasets():
    benchmark = {label: (g, kw) for label, g, kw in benchmark_datasets()}

    assert set(sensitivity_analysis.DATASETS) == set(benchmark)
    for label, config in sensitivity_analysis.DATASETS.items():
        assert config["generator"] is benchmark[label][0]
        assert config["kwargs"] == benchmark[label][1]
    assert sensitivity_analysis.NOISE_DATASET in benchmark


def test_bimodal_sensitivity_scenario_builds_residual_streams(monkeypatch):
    class ZeroModel:
        @staticmethod
        def predict(X):
            return np.zeros(len(X))

    monkeypatch.setattr(
        sensitivity_analysis,
        "get_model",
        lambda *args, **kwargs: ZeroModel(),
    )

    config = sensitivity_analysis.DATASETS["Bimodal Shift"]
    calibration, test, null = sensitivity_analysis.build_residual_streams(
        config["generator"],
        dict(config["kwargs"], n=30, drift_at=20, n_calibrations_streams=2, n_test_streams=3),
        rescale=True,
    )

    assert len(calibration) == 2
    assert list(test) == ["bimodal_shift"]
    assert len(test["bimodal_shift"]) == len(null) == 3
    assert all(stream.shape == (30,) for stream in calibration + test["bimodal_shift"] + null)
    assert np.median(np.concatenate(calibration)) == pytest.approx(1.0)
    assert not {s.tobytes() for s in null} & {
        s.tobytes() for s in calibration + test["bimodal_shift"]
    }


def test_sensitivity_sweep_retains_all_scenarios(monkeypatch):
    class ZeroModel:
        @staticmethod
        def predict(X):
            return np.zeros(len(X))

    def multi_scenario_generator(**kwargs):
        n = kwargs["n"]
        X = np.zeros((n, 1))
        return (
            X, np.zeros(n), [X], [np.ones(n)],
            {"abrupt": [X], "gradual": [X]},
            {"abrupt": [np.ones(n)], "gradual": [np.full(n, 2.0)]},
        )

    monkeypatch.setattr(sensitivity_analysis, "get_model", lambda *a, **kw: ZeroModel())
    monkeypatch.setattr(
        sensitivity_analysis, "held_out_null_streams",
        lambda generator, n_streams, **kwargs: (
            [np.zeros((kwargs["n"], 1))], [np.zeros(kwargs["n"])]
        ),
    )
    monkeypatch.setattr(sensitivity_analysis, "DATASETS", {
        "Multi": {"generator": multi_scenario_generator, "kwargs": {"n": 30, "n_test_streams": 1}}
    })
    monkeypatch.setattr(sensitivity_analysis, "WINDOW_SIZE_GRID", [500])
    monkeypatch.setattr(sensitivity_analysis, "THRESHOLD_GRID", [1.5])
    monkeypatch.setattr(
        sensitivity_analysis, "calibrate_operating_point", lambda calibration: (500, 1.5)
    )
    monkeypatch.setattr(
        sensitivity_analysis, "budget_matched_threshold",
        lambda calibration, window_size: {"threshold_drift_score": 2.0},
    )

    def evaluate_all(factory, calibration, test, **kwargs):
        assert set(test) == {"abrupt", "gradual"}
        np.testing.assert_array_equal(test["abrupt"][0], np.ones(30))
        np.testing.assert_array_equal(test["gradual"][0], np.full(30, 4.0))
        return [{"scenario": name, "detection_rate": 0.5,
                 "placebo_detection_rate": 0.0, "mean_first_delay": 1.0,
                 "median_first_delay": 1.0, "mean_capped_delay": 500.5,
                 "calibration_false_positive_rate": 0.0,
                 "calibration_budget_met": True} for name in test]

    monkeypatch.setattr(sensitivity_analysis, "evaluate_fixed_detector", evaluate_all)
    result = sensitivity_analysis.sweep_window_size_and_threshold()

    assert len(result) == 4
    assert set(result["scenario"]) == {"abrupt", "gradual"}
    assert result.groupby("parameter")["scenario"].nunique().to_dict() == {
        "window_size": 2, "threshold_drift_score": 2,
    }
    windows = result[result["parameter"] == "window_size"]
    thresholds = result[result["parameter"] == "threshold_drift_score"]
    # Window sweep at the budget-matched threshold; threshold sweep at W*.
    assert set(windows["threshold_drift_score"]) == {2.0}
    assert set(thresholds["window_size"]) == {500}
    assert result["calibrated_value"].all()
    assert sensitivity_analysis.OPERATING_POINTS["Multi"] == (500, 1.5)


def test_sensitivity_ranges_use_budget_meeting_rows_and_label_noise():
    import pandas as pd

    def row(parameter, value, detection, budget_met, dataset="D"):
        return {"dataset": dataset, "scenario": "s", "parameter": parameter,
                "value": value, "detection_rate": detection,
                "placebo_detection_rate": 0.0, "median_first_delay": 10.0,
                "calibration_false_positive_rate": 0.07,
                "calibration_budget_met": budget_met}

    oat = pd.DataFrame([
        row("window_size", 500, 0.8, True), row("window_size", 1000, 1.0, True),
        # Thresholds below the calibrated one violate the budget: excluded.
        row("threshold_drift_score", 0.0, 1.0, False),
        row("threshold_drift_score", 2.0, 0.9, True),
        row("threshold_drift_score", 4.0, 0.7, True),
        row("noise", 0.1, 1.0, True), row("noise", 0.4, 1.0, True),
    ])
    ranges = sensitivity_analysis.sensitivity_ranges(oat)
    assert ranges.loc["window_size", "detection_rate_range"] == pytest.approx(0.2)
    assert ranges.loc["threshold_drift_score", "detection_rate_range"] == pytest.approx(0.2)
    assert ranges.loc["threshold_drift_score", "n_values"] == 2
    assert ranges.loc["noise", "factor_type"] == "data condition"
    assert ranges.loc["window_size", "factor_type"] == "detector hyperparameter"
