"""Ablation variants of :class:`WassersteinDriftDetector`.

Each variant changes one design choice and keeps the rest of the detector, so a
calibrated comparison isolates that choice:

  no_hysteresis    - every step is re-evaluated independently, without the
                      dead band that suppresses alarms until the score returns
                      to zero.
  zscore_bound     - a Gaussian (mean + 1.5*std) fence on the historical
                      distances replaces the IQR fence; the excess is divided
                      by the standard deviation.
  ks_statistic     - the Kolmogorov-Smirnov statistic replaces the Wasserstein
                      distance; fence, hysteresis and window split are kept.
  energy_statistic - energy distance replaces the Wasserstein distance.

The classes live in the package, rather than in the experiment script, so that
worker processes can unpickle them by reference.
"""

import numpy as np
from scipy.stats import energy_distance, ks_2samp

from wasserstein_score.wasserstein_drift_detector import WassersteinDriftDetector


class NoHysteresisDetector(WassersteinDriftDetector):
    """Same IQR bound, but no dead-band suppression after a drift fires."""

    # Every step whose score exceeds the threshold alarms.
    rearms_at_zero = False

    def update_drift_detected_property(self) -> None:
        self._drift_detected = False
        if len(self.wasserstein_scores) < self.score_history:
            return
        values = np.asarray(self.wasserstein_scores)[:, 1]
        drift_score, dd = self._check_drift_detected(
            values, threshold_drift_score=self.threshold_drift_score
        )
        self.current_drift_score = drift_score
        self.drift_scores.append((self.current_index, drift_score))
        self._drift_detected = dd


class ZScoreBoundDetector(WassersteinDriftDetector):
    """Same hysteresis logic, but a plain Gaussian (mean/std) bound instead of IQR."""

    def _check_drift_detected(self, values: np.ndarray, threshold_drift_score: float) -> tuple[float, bool]:
        if len(values) < 2:
            return 0.0, False

        historical = values[:-1]
        latest = values[-1]

        mu = np.mean(historical)
        sigma = np.std(historical)
        upper_bound = mu + 1.5 * sigma

        if latest > upper_bound and sigma > 0:
            drift_score = (latest - upper_bound) / sigma
        else:
            drift_score = 0.0

        return float(drift_score), bool(drift_score > threshold_drift_score)


class KSStatisticDetector(WassersteinDriftDetector):
    """Same IQR bound, hysteresis, and window split, but the Kolmogorov-Smirnov
    statistic in place of Wasserstein distance."""

    def _compute_statistic(self, before: np.ndarray, after: np.ndarray) -> float:
        return ks_2samp(before, after).statistic


class EnergyDistanceDetector(WassersteinDriftDetector):
    """Same IQR bound, hysteresis, and window split, but energy distance in
    place of Wasserstein distance."""

    def _compute_statistic(self, before: np.ndarray, after: np.ndarray) -> float:
        return energy_distance(before, after)


VARIANT_CLASSES = {
    "full": WassersteinDriftDetector,
    "no_hysteresis": NoHysteresisDetector,
    "zscore_bound": ZScoreBoundDetector,
    "ks_statistic": KSStatisticDetector,
    "energy_statistic": EnergyDistanceDetector,
}
