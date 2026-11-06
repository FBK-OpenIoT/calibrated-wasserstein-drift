import json

import numpy as np
import pandas as pd
import pytest

from experiments import run_all_tests


def _dataset_result() -> run_all_tests.DatasetResult:
    test_results = pd.DataFrame([
        {
            "scenario": "scenario_a",
            "algorithm": "wasserstein",
            "mean_total_detections": 2.0,
            "mean_false_positives_common": 0.2,
            "mean_false_positives_active": 0.3,
            "mean_true_detections": 1.8,
            "detection_rate": 0.8,
            "detection_rate_ci_low": 0.6,
            "detection_rate_ci_high": 0.9,
            "placebo_detection_rate": 0.1,
            "excess_detection_rate": 0.7,
            "mean_first_delay": 10.0,
            "median_first_delay": 8.0,
            "mean_capped_delay": 208.0,
            "flicker_rate": 0.1,
        },
        {
            "scenario": "scenario_b",
            "algorithm": "wasserstein",
            "mean_total_detections": 4.0,
            "mean_false_positives_common": 0.4,
            "mean_false_positives_active": 0.5,
            "mean_true_detections": 3.6,
            "detection_rate": 1.0,
            "detection_rate_ci_low": 0.93,
            "detection_rate_ci_high": 1.0,
            "placebo_detection_rate": 0.1,
            "excess_detection_rate": 0.9,
            "mean_first_delay": 20.0,
            "median_first_delay": 18.0,
            "mean_capped_delay": 20.0,
            "flicker_rate": 0.3,
        },
    ])
    calibration_results = pd.DataFrame([
        {
            "algorithm": "wasserstein",
            "threshold_drift_score": 2.5,
            "window_size": 750,
            "selected": True,
        },
        {
            "algorithm": "wasserstein",
            "threshold_drift_score": 3.0,
            "window_size": 900,
            "selected": False,
        },
    ])
    per_stream_results = pd.DataFrame([
        {
            "scenario": "scenario_a",
            "algorithm": "wasserstein",
            "stream_id": 0,
        }
    ])
    return run_all_tests.DatasetResult(
        label="Synthetic dataset",
        best_params={
            "wasserstein": {
                "threshold_drift_score": 2.5,
                "window_size": 750,
            }
        },
        calibration_results=calibration_results,
        test_results=test_results,
        per_stream_results=per_stream_results,
        normalization_scale=2.0,
        rescale=True,
    )


def test_build_result_tables_persists_aggregates_parameters_and_revision():
    provenance = {"commit_sha": "abc123", "working_tree_dirty": False}

    test_summary, overall, parameters, per_stream = (
        run_all_tests.build_result_tables([_dataset_result()], provenance)
    )

    assert set(test_summary["dataset"]) == {"Synthetic dataset"}
    assert set(test_summary["commit_sha"]) == {"abc123"}
    assert overall.loc[0, "n_scenarios"] == 2
    assert overall.loc[0, "n_datasets"] == 1
    assert overall.loc[0, "detection_rate"] == pytest.approx(0.9)
    assert overall.loc[0, "median_first_delay"] == pytest.approx(13.0)
    assert overall.loc[0, "mean_first_delay"] == pytest.approx(15.0)
    assert overall.loc[0, "mean_capped_delay"] == pytest.approx(114.0)
    assert len(parameters) == 1
    assert parameters.loc[0, "threshold_drift_score"] == pytest.approx(2.5)
    assert parameters.loc[0, "window_size"] == 750
    assert set(per_stream["commit_sha"]) == {"abc123"}
    for frame in (test_summary, parameters, per_stream):
        assert set(frame["normalization_scale"]) == {2.0}


def test_overall_summary_weighs_datasets_equally():
    provenance = {"commit_sha": "abc123", "working_tree_dirty": False}
    two_scenarios = _dataset_result()  # detection rates 0.8 and 1.0
    one_scenario = _dataset_result()
    one_scenario.label = "Other dataset"
    one_scenario.test_results = one_scenario.test_results.iloc[:1].assign(
        scenario="scenario_c", detection_rate=0.0)

    _, overall, _, _ = run_all_tests.build_result_tables(
        [two_scenarios, one_scenario], provenance)

    # (mean(0.8, 1.0) + 0.0) / 2, not mean(0.8, 1.0, 0.0).
    assert overall.loc[0, "detection_rate"] == pytest.approx(0.45)
    assert (overall.loc[0, "n_datasets"], overall.loc[0, "n_scenarios"]) == (2, 3)


def test_save_benchmark_outputs_writes_metadata_and_all_tables(tmp_path, monkeypatch):
    empty_report = pd.DataFrame({"comparison": ["placeholder"]})
    monkeypatch.setattr(
        run_all_tests,
        "significance_report",
        lambda *args, **kwargs: {
            "delay_wilcoxon": empty_report,
            "fp_wilcoxon": empty_report,
            "detection_mcnemar": empty_report,
        },
    )
    provenance = {"commit_sha": "abc123", "working_tree_dirty": True}

    run_all_tests.save_benchmark_outputs([_dataset_result()], tmp_path, provenance)

    expected = {
        "test_summary_results.csv",
        "overall_summary_results.csv",
        "calibrated_parameters.csv",
        "per_stream_results.csv",
        "significance_delay_wilcoxon.csv",
        "significance_fp_wilcoxon.csv",
        "significance_detection_mcnemar.csv",
        "benchmark_run_metadata.json",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}

    metadata = json.loads(
        (tmp_path / "benchmark_run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["experiment"] == "main_benchmark"
    assert metadata["configuration"]["signal_normalization"] == "pooled_calibration_median"
    assert metadata["normalization_scales"] == {"Synthetic dataset": 2.0}
    assert metadata["configuration"]["wasserstein_window_selection"] == "largest_admissible"
    assert metadata["commit_sha"] == "abc123"
    assert metadata["working_tree_dirty"] is True
    assert metadata["configuration"]["optuna_seed"] == run_all_tests.OPTUNA_SEED
    assert metadata["configuration"]["wasserstein_bounds"][
        "threshold_drift_score"
    ] == list(run_all_tests.WASSERSTEIN_BOUNDS["threshold_drift_score"])
    assert metadata["configuration"]["detection_horizon"] == (
        run_all_tests.DETECTION_HORIZON
    )
    assert metadata["outputs"]["calibrated_parameters"] == (
        "calibrated_parameters.csv"
    )


@pytest.mark.parametrize("rescale", [False, True])
def test_run_dataset_reports_detection_and_delay_to_stdout(monkeypatch, capsys, rescale):
    class Model:
        def predict(self, features):
            return np.zeros(len(features))

    monkeypatch.setattr(run_all_tests, "get_model", lambda *args: Model())
    result = _dataset_result()

    def calibrate(**kwargs):
        assert kwargs["verbose"] is True
        assert set(kwargs["test_streams"]) == {"scenario_a"}
        divisor = 4.0 if rescale else 1.0
        assert kwargs["algorithms"] == ["adwin", "ph"]
        np.testing.assert_array_equal(kwargs["calibration_streams"][0], np.full(40, 4.0 / divisor))
        np.testing.assert_array_equal(kwargs["test_streams"]["scenario_a"][0], np.full(40, 16.0 / divisor))
        np.testing.assert_array_equal(kwargs["null_streams"][0], np.full(40, 36.0 / divisor))
        return (
            result.best_params, result.calibration_results,
            result.test_results, result.per_stream_results,
        )

    monkeypatch.setattr(run_all_tests, "calibrate_and_test", calibrate)
    n = run_all_tests.NUM_TRAINING_POINTS + run_all_tests.NUM_MODEL_TEST_POINTS
    streams = [np.zeros((40, 2))]
    actual = run_all_tests.run_dataset(
        "Stub dataset", np.zeros((n, 2)), np.zeros(n), streams, [np.full(40, 2.0)],
        {"scenario_a": streams}, {"scenario_a": [np.full(40, 4.0)]}, drift_at=30,
        null_data=(streams, [np.full(40, 6.0)]),
        rescale=rescale, algorithms=["adwin", "ph"],
    )
    assert actual.normalization_scale == (4.0 if rescale else 1.0)
    assert actual.rescale == rescale

    stdout = capsys.readouterr().out
    assert "# Stub dataset" in stdout
    assert "## Detection probability and delay" in stdout
    for column in ("median_first_delay", "mean_capped_delay", "detection_rate_ci_low"):
        assert column in stdout
    # Two scenarios share the dataset's calibration and pre-drift streams.
    assert "not independent replications" in stdout


def test_benchmark_cli_supports_detector_subset_and_optional_rescaling():
    defaults = run_all_tests.parse_args([])
    assert defaults.rescale is False
    assert defaults.algorithms == list(run_all_tests.RUNNING_ALGORITHMS)
    args = run_all_tests.parse_args(["--algorithms", "adwin", "ph", "--rescale",
                                     "--output-dir", "data/adwin_ph_median"])
    assert args.algorithms == ["adwin", "ph"]
    assert args.rescale is True
    assert args.output_dir.as_posix() == "data/adwin_ph_median"
    assert run_all_tests.parse_args(["--rescale", "--no-rescale"]).rescale is False


def test_adwin_ph_only_rescaled_calibration_and_export(tmp_path):
    from wasserstein_score.evaluation import calibrate_and_test
    from wasserstein_score.normalization import CalibrationMedianNormalizer

    rng = np.random.default_rng(21)
    calibration = [rng.exponential(10.0, 1500) for _ in range(2)]
    tests = [np.concatenate([rng.exponential(10.0, 1000), rng.exponential(100.0, 500)])
             for _ in range(4)]
    normalizer = CalibrationMedianNormalizer.fit(calibration)
    best, calibrated, test, per_stream = calibrate_and_test(
        calibration_streams=normalizer.transform(calibration),
        test_streams={"smoke": normalizer.transform(tests)},
        drift_at=1000, detection_horizon=500, algorithms=["adwin", "ph"],
        n_trials=2, widening_rounds=0, n_bootstrap=20,
        false_alarm_budget_per_1000=5.0, verbose=False,
    )
    result = run_all_tests.DatasetResult("Smoke", best, calibrated, test, per_stream,
                                         normalization_scale=normalizer.scale, rescale=True)
    run_all_tests.save_benchmark_outputs(
        [result], tmp_path, {"commit_sha": "smoke", "working_tree_dirty": True},
    )
    metadata = json.loads((tmp_path / "benchmark_run_metadata.json").read_text())
    assert set(metadata["configuration"]["algorithms"]) == {"adwin", "ph"}
    assert metadata["configuration"]["significance_reference"] == "adwin"
    assert metadata["configuration"]["signal_normalization"] == "pooled_calibration_median"
    assert metadata["normalization_scales"]["Smoke"] == normalizer.scale
    saved = pd.read_csv(tmp_path / "test_summary_results.csv")
    assert set(saved["algorithm"]) == {"adwin", "ph"}
    assert set(saved["signal_normalization"]) == {"pooled_calibration_median"}
