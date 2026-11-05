"""Base drift detector interface following River library pattern."""

from collections import deque
from river.base.drift_detector import DriftDetector
import numpy as np
from scipy.stats import wasserstein_distance

class WassersteinDriftDetector(DriftDetector):
    """Drift detector based on Wasserstein distance, following River's drift detector interface."""

    # Hysteresis: one alarm per excursion of the drift score above zero.
    # Calibration relies on this to count alarms at every threshold from one
    # pass over the scores (``evaluation.budget_matched_threshold``).
    rearms_at_zero = True

    def __init__(self, window_size, threshold_drift_score=1.2, starting_idx=-1, batch_size=1):
        super().__init__()
        if threshold_drift_score < 0:
            raise ValueError("threshold_drift_score must be non-negative")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be a positive integer")
        self._drift_detected = False

        self.window_size = window_size

        self.current_index = starting_idx - 1
        self._is_drifting = False

        self.threshold_drift_score = threshold_drift_score
        self.current_drift_score = None

        # The distance is computed, and an alarm can be raised, only on the
        # last sample of every batch of ``batch_size`` samples. The fence then
        # compares the latest distance with those of the preceding
        # ``window_size`` to ``2 * window_size`` samples, as it does with
        # batch_size = 1, i.e. with ``score_history`` to twice as many scores.
        self.batch_size = int(batch_size)
        self.score_history = -(-int(window_size) // self.batch_size)
        if self.batch_size > 1 and self.score_history < 2:
            raise ValueError(
                f"batch_size={batch_size} leaves fewer than two distances per "
                f"window of {window_size} samples"
            )
        self._n_updates = 0

        self.datastream = deque(maxlen=int(2 * window_size))
        self.wasserstein_scores = deque(maxlen=2 * self.score_history)
        self.drift_scores = deque(maxlen=2 * self.score_history)
    
    def update(self, x) -> "WassersteinDriftDetector":
        """Update the drift detector with a new data point.

        Args:
            x: The new data point (e.g. model residual or feature value).
        """
        self.current_index += 1
        self._n_updates += 1
        self.datastream.append(x)
        if (
            len(self.datastream) < 2 * self.window_size
            or self._n_updates % self.batch_size
        ):
            # No test on this sample; the hysteresis state carries over.
            self._drift_detected = False
            return self

        data = np.asarray(self.datastream)
        split = int(3/2*self.window_size)
        result = self._compute_statistic(data[:split], data[split:])
        self.wasserstein_scores.append((self.current_index, result))

        self.update_drift_detected_property()
        return self

    def _compute_statistic(self, before: np.ndarray, after: np.ndarray) -> float:
        """The two-sample statistic comparing the reference and test windows.

        Overridable so alternative distributional statistics can be swapped
        in while the IQR fence, hysteresis, and window split stay fixed.
        """
        return wasserstein_distance(before, after)

    @property
    def drift_detected(self) -> bool:
        """Whether a drift has been detected after the last update."""
        return self._drift_detected

    def update_drift_detected_property(self) -> None:
        self._drift_detected = False

        if len(self.wasserstein_scores) < self.score_history:
            return

        values = np.asarray(self.wasserstein_scores)[:, 1]
        drift_score, dd = self._check_drift_detected(values, threshold_drift_score=self.threshold_drift_score)
        self.current_drift_score = drift_score
        self.drift_scores.append((self.current_index, drift_score))

        was_it_drifting = self._is_drifting
        if was_it_drifting:
            # Re-arm only once the distance is back inside Tukey's fence
            # (score zero), which lies below the raising threshold for every
            # theta >= 0.
            self._drift_detected = False
            self._is_drifting = drift_score > 0
        else:
            self._drift_detected = dd
            self._is_drifting = self._drift_detected
        return

    def _check_drift_detected(self, values: np.ndarray, threshold_drift_score: float) -> tuple[float, bool]:
        if len(values) < 2:
            return 0.0, False

        historical = values[:-1]
        latest = values[-1]

        Q1 = np.percentile(historical, 25)
        Q3 = np.percentile(historical, 75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR

        if latest > upper_bound and IQR > 0:
            drift_score = (latest - upper_bound) / IQR
        else:
            drift_score = 0.0

        return float(drift_score), bool(drift_score > threshold_drift_score)
