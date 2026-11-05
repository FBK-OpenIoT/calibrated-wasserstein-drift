from experiments import severity_sweep
from wasserstein_score import benchmark_protocol

import numpy as np
import pandas as pd


def test_severity_sweep_uses_main_benchmark_protocol():
    assert severity_sweep.NUM_DATAPOINTS == benchmark_protocol.NUM_DATAPOINTS
    assert severity_sweep.DRIFT_AT == benchmark_protocol.DRIFT_AT
    assert severity_sweep.N_CALIBRATION_STREAMS == benchmark_protocol.N_CALIBRATION_STREAMS
    assert severity_sweep.N_TEST_STREAMS == benchmark_protocol.N_TEST_STREAMS
    assert severity_sweep.NUM_TRAINING_POINTS == benchmark_protocol.NUM_TRAINING_POINTS
    assert severity_sweep.NUM_MODEL_TEST_POINTS == benchmark_protocol.NUM_MODEL_TEST_POINTS
    assert severity_sweep.N_OPTUNA_TRIALS == benchmark_protocol.N_OPTUNA_TRIALS
    assert severity_sweep.WASSERSTEIN_BOUNDS == benchmark_protocol.WASSERSTEIN_BOUNDS
    assert severity_sweep.DETECTION_HORIZON == benchmark_protocol.DETECTION_HORIZON


def test_severity_grids_include_the_main_benchmark_points():
    bimodal = severity_sweep.SEVERITY_GRIDS["bimodal_shift"]

    assert bimodal["reference_value"] == benchmark_protocol.BIMODAL_SHIFT_SEPARATION
    assert bimodal["reference_value"] in bimodal["grid"]


def test_severity_sweep_reuses_calibration_median_for_later_levels(monkeypatch):
    class Model:
        def predict(self, features):
            return np.zeros(len(features))

    def generator(separation, **kwargs):
        X = np.zeros((4, 1))
        # A changed calibration draw at the later level must not change the scale.
        calibration_y = np.full(4, 2.0 if separation == 1.0 else 100.0)
        test_y = np.full(4, 4.0 if separation == 1.0 else 8.0)
        return X, np.zeros(4), [X], [calibration_y], {"s": [X]}, {"s": [test_y]}

    monkeypatch.setattr(severity_sweep, "get_model", lambda *args: Model())
    monkeypatch.setattr(severity_sweep, "build_model_df", lambda *args: None)
    monkeypatch.setattr(severity_sweep, "held_out_null_streams", lambda *a, **kw: (
        [np.zeros((4, 1))], [np.full(4, 6.0)]
    ))
    calls = []

    def calibrate(**kwargs):
        calls.append("calibrate")
        np.testing.assert_array_equal(kwargs["calibration_streams"][0], np.ones(4))
        np.testing.assert_array_equal(kwargs["test_streams"]["s"][0], np.full(4, 4.0))
        np.testing.assert_array_equal(kwargs["null_streams"][0], np.full(4, 9.0))
        frame = pd.DataFrame([dict(algorithm="ph", scenario="s", detection_rate=0.5,
                                  placebo_detection_rate=0.1, null_n_false_alarms=0)])
        return {"ph": {"threshold": 2.0, "delta": 0.01}}, pd.DataFrame(), frame, pd.DataFrame()

    def evaluate(**kwargs):
        calls.append("evaluate")
        np.testing.assert_array_equal(kwargs["test_streams"]["s"][0], np.full(4, 16.0))
        return pd.DataFrame([dict(algorithm="ph", scenario="s", detection_rate=0.5)]), pd.DataFrame()

    monkeypatch.setattr(severity_sweep, "calibrate_and_test", calibrate)
    monkeypatch.setattr(severity_sweep, "evaluate_with_params", evaluate)
    result = severity_sweep.run_severity_sweep("s", dict(
        generator=generator, param="separation", grid=[1.0, 0.5],
        reference_value=1.0, fixed_kwargs={}, seed_base=0,
    ), rescale=True)
    assert calls == ["calibrate", "evaluate"]
    assert set(result["normalization_scale"]) == {4.0}
    assert set(result["placebo_detection_rate"]) == {0.1}
