import numpy as np
import pytest

from wasserstein_score import WassersteinDriftDetector


def test_public_api_and_initial_state():
    detector = WassersteinDriftDetector(window_size=10)

    assert detector.drift_detected is False
    assert detector.current_drift_score is None
    assert detector.update(0.0) is detector


def test_detector_respects_warmup():
    detector = WassersteinDriftDetector(window_size=4, starting_idx=0)

    for value in range(10):
        detector.update(float(value))

    assert detector.current_drift_score is None

    detector.update(10.0)
    assert detector.current_drift_score is not None


def test_iqr_rule_detects_a_large_outlier():
    detector = WassersteinDriftDetector(window_size=4, threshold_drift_score=1.2)
    values = np.array([0.9, 1.0, 1.0, 1.1, 10.0])

    score, detected = detector._check_drift_detected(values, threshold_drift_score=1.2)

    assert score > 1.2
    assert detected is True


def test_zero_spread_does_not_trigger_drift():
    detector = WassersteinDriftDetector(window_size=4)

    score, detected = detector._check_drift_detected(
        np.ones(5), threshold_drift_score=1.2
    )

    assert score == 0.0
    assert detected is False


def test_seeded_stationary_stream_has_no_early_alarm():
    rng = np.random.default_rng(42)
    detector = WassersteinDriftDetector(window_size=20, threshold_drift_score=2.0)

    alarms = []
    for index, value in enumerate(rng.normal(size=55)):
        detector.update(value)
        if detector.drift_detected:
            alarms.append(index)

    assert alarms == []


def _scripted_detector(scores, threshold):
    """Detector whose per-step drift scores follow ``scores`` exactly."""
    detector = WassersteinDriftDetector(window_size=1, threshold_drift_score=threshold)
    remaining = iter(scores)

    def scripted_check(values, threshold_drift_score):
        score = next(remaining)
        return score, score > threshold_drift_score

    detector._check_drift_detected = scripted_check
    return detector


def test_hysteresis_re_arms_only_when_score_returns_to_zero():
    # The 0.5 scores stay above theta = 0.3; re-arming only at a zero score
    # keeps them from raising repeated alarms.
    scores = [0.0, 2.0, 0.5, 0.5, 0.0, 0.5]
    detector = _scripted_detector(scores, threshold=0.3)

    detector.update(0.0)  # fills the two-sample buffer; no score yet
    alarms = []
    for step in range(len(scores)):
        detector.update(0.0)
        if detector.drift_detected:
            alarms.append(step)

    assert alarms == [1, 5]


def test_negative_threshold_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        WassersteinDriftDetector(window_size=10, threshold_drift_score=-0.1)
