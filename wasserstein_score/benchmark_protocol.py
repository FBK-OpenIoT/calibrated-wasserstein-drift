"""Canonical protocol for the main synthetic benchmark and its severity sweep.

Keeping these values in one place prevents diagnostic experiments from silently
using shorter streams or smaller calibration/test samples than the benchmark
they are meant to explain.
"""

NUM_DATAPOINTS = 20_000
NUM_TRAINING_POINTS = 5_000
NUM_MODEL_TEST_POINTS = 2_500

DRIFT_AT = 15_000
STARTING_IDX = 0

N_CALIBRATION_STREAMS = 20
N_TEST_STREAMS = 50
N_OPTUNA_TRIALS = 40

# Canonical calibration search spaces. Main-benchmark diagnostics that are
# intended to explain or reproduce its operating points must import these
# bounds instead of maintaining local defaults.
#
# threshold_drift_score spans the detector's full domain: the drift score is
# zero inside Tukey's fence, so theta = 0 means "any distance beyond
# Q3 + 1.5 IQR", and the re-arm rule (score back to zero) keeps a valid
# hysteresis dead band for every theta >= 0.
#
# Every windowed detector (Wasserstein, KSWIN) is capped at MAX_WINDOW_SIZE,
# equal to DETECTION_HORIZON, both in its search range and as the limit of
# range widening (``evaluation.PARAMETER_DOMAINS``). Under the
# largest-admissible rule below the Wasserstein upper bound is the operating
# window, and at 1,000 its test segment (window_size / 2 = 500 samples) fills
# within the first half of the horizon. ADWIN's window is adaptive and
# Page-Hinkley has none, so neither takes this cap.
MAX_WINDOW_SIZE = 1_000
WASSERSTEIN_BOUNDS = {
    "threshold_drift_score": (0.0, 4.0),
    "window_size": (500, MAX_WINDOW_SIZE),
}
# How calibration chooses the Wasserstein window. The false-alarm budget pins
# the threshold at every window but not the window itself, so a TPE search
# settled on whichever admissible window it happened to sample (across
# calibration seeds the Bimodal detection rate followed it from 0.72 at
# W=454 to 1.00 at W=1089). "largest_admissible" takes the largest window in
# the search range at which some threshold meets the budget, with the lowest
# such threshold, computed exactly rather than searched. At a fixed alarm
# budget a longer window adds evidence against a persistent change; the cost
# is delay, which stays bounded because the test segment (window_size / 2,
# 500 samples at the upper bound) is shorter than DETECTION_HORIZON. "tpe"
# restores the joint Optuna search.
WASSERSTEIN_WINDOW_SELECTION = "largest_admissible"

ADWIN_BOUNDS = {
    "delta": (0.00001, 0.1),
}
KSWIN_BOUNDS = {
    "window_size": (500, MAX_WINDOW_SIZE),
    "alpha": (0.00001, 0.01),
    "stat_size": (20, 80),
}
PAGE_HINKLEY_BOUNDS = {
    "threshold": (50.0, 500.0),
    "delta": (0.00001, 5.0),
}

# A selected parameter closer than this fraction of its search range to either
# bound (measured in log space for log-scaled parameters) is flagged, so an
# operating point set by the search space rather than by the budget is visible.
AT_BOUND_TOLERANCE = 0.02

# A selection that misses the false-alarm budget, sits at a search bound, or
# meets the budget without a single calibration alarm (so the budget does not
# bind) is recalibrated over a widened range, for at most this many extra
# rounds (``evaluation.calibrate_detectors_with_widening``); 0 disables it.
CALIBRATION_WIDENING_ROUNDS = 4

RUNNING_ALGORITHMS = ("wasserstein", "ph", "kswin", "adwin")

# Preserve the raw squared-error benchmark unless median rescaling is requested.
RESCALE_BY_CALIBRATION_MEDIAN = False

# KSWIN draws its reference sample at random; seeding every instance makes the
# calibration, test and placebo results of a run reproducible.
KSWIN_SEED = 42

N_BOOTSTRAP = 2_000
CI_LEVEL = 0.95
BOOTSTRAP_SEED = 0
OPTUNA_SEED = 42

# A stream counts as detected only if an alarm falls in
# [drift_at, drift_at + DETECTION_HORIZON). The horizon covers the slowest
# delays of reliable detectors on the hard scenarios (a few hundred samples)
# while keeping the detection window short relative to the post-drift span.
DETECTION_HORIZON = 1_000

# Under a homogeneous Poisson working model, alarms at the budgeted rate give
# probability 1 - exp(-budget * horizon / 1000) = 1 - exp(-0.1) ~ 9.5%.
# This is not a guarantee for dependent sequential alarms or a configuration
# chosen by calibration; held-out null streams audit the actual behavior.
# 0.1 per 1,000 samples is equivalent to a mean of 10,000 samples between
# false alarms.
MAX_CHANCE_DETECTIONS_PER_HORIZON = 0.1

# A conditional delay (mean or median over detected streams) is reported only
# when at least this many streams were detected and the Wilson lower bound of
# the detection rate exceeds the placebo rate. Otherwise the "detections" are
# indistinguishable from chance alarms and their delay is not a property of
# the detector (e.g. one stream detected out of 50 with a placebo rate of
# 0.04). The horizon-capped delay remains defined for every detector.
MIN_DETECTED_STREAMS_FOR_DELAY = 5
FALSE_ALARM_BUDGET_PER_1000 = (
    MAX_CHANCE_DETECTIONS_PER_HORIZON * 1_000 / DETECTION_HORIZON
)
FALSE_ALARM_CONFIDENCE = 0.95

# Severity value used for the Bimodal point in the main benchmark.
BIMODAL_SHIFT_SEPARATION = 0.99

# Seed ranges of the held-out drift-free streams used for the placebo
# detection rate: N_TEST_STREAMS streams drawn from each generator's
# calibration branch, starting at these seeds. Each range is disjoint from the
# generator's calibration and test seeds (and their noise-seed offsets), and
# from the calibration-robustness ranges, which start at 10,000;
# helper_functions.held_out_null_streams verifies this for every call.
NULL_STREAM_SEED_BASES = {
    "abrupt_linear_drift_stream": 300,    # calib 100.., test 200..
    "friedman_f1_f2_switch": 500,         # calib 100.., test 120..
    "friedman_f2_f1_switch": 500,
    "river_friedman_drift_stream": 1_000,  # calib 10.., test 30..
}
# Generators whose seeds are all relative to a ``seed_base`` argument get a
# null range at the same relative position, so it moves with ``seed_base``.
NULL_STREAM_SEED_OFFSETS = {
    "bimodal_shift_streams": 300,  # calib seed_base+100.., test seed_base+200..
}
