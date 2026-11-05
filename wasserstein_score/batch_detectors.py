"""River baselines that test for drift only at the end of each batch.

Samples arrive in batches of ``batch_size``. Every detector still reads every
sample, but it may raise an alarm only on the last sample of a batch, as the
Wasserstein detector does with its ``batch_size`` argument and ADWIN with
River's ``clock``. With ``batch_size=1`` both classes behave exactly as their
River parents.

The classes live in the package, rather than in the experiment script, so that
worker processes can unpickle them by reference.
"""

from river import drift


def _check_batch_size(batch_size) -> int:
    if int(batch_size) != batch_size or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size}")
    return int(batch_size)


class BatchKSWIN(drift.KSWIN):
    """KSWIN whose Kolmogorov-Smirnov test runs on the last sample of each batch.

    Between tests the sample only enters the sliding window. The batch count is
    kept apart from River's ``n``, which restarts after every alarm.
    """

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        seed: int | None = None,
        window=None,
        batch_size: int = 1,
    ):
        self.batch_size = _check_batch_size(batch_size)
        self._n_updates = 0
        super().__init__(
            alpha=alpha, window_size=window_size, stat_size=stat_size,
            seed=seed, window=window,
        )

    def update(self, x):
        self._n_updates += 1
        if self._n_updates % self.batch_size == 0:
            return super().update(x)
        if self.drift_detected:
            self._reset()
        self.n += 1
        self.window.append(x)
        self._drift_detected = False
        return self


class BatchPageHinkley(drift.PageHinkley):
    """Page-Hinkley whose threshold test applies on the last sample of each batch.

    The cumulative sums are updated on every sample. An exceedance inside a
    batch is not an alarm and resets nothing; the statistic carries it to the
    batch's last sample, where it alarms if it still exceeds the threshold.
    """

    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 1 - 0.0001,
        mode: str = "both",
        batch_size: int = 1,
    ):
        self.batch_size = _check_batch_size(batch_size)
        self._n_updates = 0
        super().__init__(
            min_instances=min_instances, delta=delta, threshold=threshold,
            alpha=alpha, mode=mode,
        )

    def update(self, x):
        self._n_updates += 1
        super().update(x)
        if self._n_updates % self.batch_size:
            self._drift_detected = False
        return self
