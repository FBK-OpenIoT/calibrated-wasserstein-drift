import numpy as np
import pytest

from wasserstein_score.normalization import CalibrationMedianNormalizer


def test_uses_pooled_samples_and_keeps_scale_fixed_without_mutating_inputs():
    # The pooled median is 3; the median of the two stream medians is 51.5.
    calibration = [np.array([1.0, 2.0, 3.0, 4.0]), np.array([100.0])]
    originals = [stream.copy() for stream in calibration]
    normalizer = CalibrationMedianNormalizer.fit(calibration)
    assert normalizer.scale == 3.0
    assert np.median(np.concatenate(normalizer.transform(calibration))) == 1.0
    np.testing.assert_array_equal(normalizer.transform([[30.0, 300.0]])[0], [10.0, 100.0])
    np.testing.assert_array_equal(normalizer.transform([[6.0]])[0], [2.0])
    assert normalizer.scale == 3.0
    for stream, original in zip(calibration, originals):
        np.testing.assert_array_equal(stream, original)


@pytest.mark.parametrize("streams", [[], [[]], [[0.0, 0.0]], [[np.nan]], [[np.inf]], [[-1.0]], [[[1.0]]]])
def test_rejects_invalid_calibration_data(streams):
    with pytest.raises(ValueError):
        CalibrationMedianNormalizer.fit(streams)


@pytest.mark.parametrize("scale", [0.0, -1.0, np.inf, np.nan])
def test_rejects_invalid_restored_scale(scale):
    with pytest.raises(ValueError, match="finite and strictly positive"):
        CalibrationMedianNormalizer(scale)


def test_rescaling_can_be_disabled_even_with_zero_calibration_median():
    normalizer = CalibrationMedianNormalizer.fit([[0.0, 0.0, 0.0, 2.0]], enabled=False)
    assert normalizer.scale == 1.0
    assert normalizer.method == "none"
    np.testing.assert_array_equal(normalizer.transform([[0.0, 10.0]])[0], [0.0, 10.0])


def test_fixed_linear_scaling_preserves_wasserstein_alarms():
    from wasserstein_score.evaluation import _run_wasserstein

    rng = np.random.default_rng(12)
    calibration = [rng.exponential(10.0, 100)]
    stream = np.concatenate([rng.exponential(10.0, 100), rng.exponential(100.0, 100)])
    normalizer = CalibrationMedianNormalizer.fit(calibration)
    params = dict(window_size=10, threshold_drift_score=0.5, starting_idx=0)
    raw_alarms = _run_wasserstein(stream, **params)
    assert raw_alarms
    assert _run_wasserstein(normalizer.transform([stream])[0], **params) == raw_alarms
