"""Fixed normalization of squared errors using drift-free calibration data."""

from dataclasses import dataclass

import numpy as np


def _squared_error_streams(streams) -> list[np.ndarray]:
    arrays = [np.asarray(stream, dtype=float) for stream in streams]
    if any(
        stream.ndim != 1
        or not np.isfinite(stream).all()
        or (stream < 0).any()
        for stream in arrays
    ):
        raise ValueError("Squared-error streams must be finite, non-negative 1-D arrays")
    return arrays


@dataclass(frozen=True)
class CalibrationMedianNormalizer:
    """Divide squared errors by one frozen, pooled calibration median.

    Fit once per dataset/model using only drift-free calibration streams.
    Reuse this instance for every test scenario, held-out null stream and
    severity level evaluated with the same calibrated detector parameters.
    This is linear scaling, with no centering, clipping or online adaptation;
    the transformed values are not restricted to [0, 1].
    """

    scale: float
    enabled: bool = True

    @property
    def method(self) -> str:
        return "pooled_calibration_median" if self.enabled else "none"

    def __post_init__(self):
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Calibration median must be finite and strictly positive")

    @classmethod
    def fit(cls, calibration_streams, *, enabled: bool = True) -> "CalibrationMedianNormalizer":
        """Use the median of all calibration samples, not the median of medians.

        Empty streams and a zero median raise an error: substituting an
        arbitrary epsilon would silently change the normalization protocol.
        With ``enabled=False``, use the identity scale without fitting a median;
        this permits raw-error runs even when the calibration median is zero.
        """
        if not enabled:
            return cls(scale=1.0, enabled=False)
        streams = _squared_error_streams(calibration_streams)
        if not streams or any(stream.size == 0 for stream in streams):
            raise ValueError("Calibration streams must be non-empty")
        return cls(scale=float(np.median(np.concatenate(streams))))

    def transform(self, streams) -> list[np.ndarray]:
        """Return scaled copies without modifying the supplied streams."""
        return [stream / self.scale for stream in _squared_error_streams(streams)]
