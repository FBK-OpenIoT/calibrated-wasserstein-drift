"""
Automated hyperparameter calibration and evaluation for drift detectors.

Supported algorithms: Wasserstein, ADWIN, KSWIN, PageHinkley.

Calibration uses Bayesian optimisation (Optuna TPE) on no-drift streams. A
configuration is admissible when the one-sided Poisson working-model bound of its
false-alarm rate respects a common budget. Among admissible trials, selection
saturates that budget, then favours stability and structural aggressiveness.
The Wasserstein detector is calibrated without TPE by default: its window is
the largest one in the search range at which a threshold meets the budget, and
its threshold the lowest such one, computed exactly from the drift-score
trajectories (``benchmark_protocol.WASSERSTEIN_WINDOW_SELECTION``).
False-alarm rates are always measured over the samples on which a detector is
able to alarm, i.e. after its own warm-up.
The bound assumes Poisson counts for a fixed configuration; neither temporal
dependence nor selection is covered by its nominal confidence level.

With ``batch_size`` > 1 the samples arrive in batches and every detector may
alarm only on the last sample of a batch; calibration, false-alarm rates and
delays stay in samples.

Test evaluation reports detection rate, delay, and false positives for each
(scenario, algorithm) pair. A stream counts as detected only when an alarm falls
within a fixed horizon after the drift; the chance level of that criterion is
measured on held-out drift-free streams (the placebo detection rate).
"""

from __future__ import annotations

import functools
import json
import math
import os
from typing import Any, Callable, Optional

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from river import drift
from scipy.stats import chi2, norm

from wasserstein_score import helper_functions as md
from wasserstein_score.benchmark_protocol import (
    ADWIN_BOUNDS,
    AT_BOUND_TOLERANCE,
    BOOTSTRAP_SEED,
    CALIBRATION_WIDENING_ROUNDS,
    CI_LEVEL,
    DETECTION_HORIZON,
    FALSE_ALARM_BUDGET_PER_1000,
    FALSE_ALARM_CONFIDENCE,
    KSWIN_BOUNDS,
    KSWIN_SEED,
    MAX_WINDOW_SIZE,
    MIN_DETECTED_STREAMS_FOR_DELAY,
    N_BOOTSTRAP,
    N_OPTUNA_TRIALS,
    OPTUNA_SEED,
    PAGE_HINKLEY_BOUNDS,
    WASSERSTEIN_BOUNDS,
    WASSERSTEIN_WINDOW_SELECTION,
)
from wasserstein_score.batch_detectors import BatchKSWIN, BatchPageHinkley
from wasserstein_score.wasserstein_drift_detector import WassersteinDriftDetector


def _available_cpu_count() -> int:
    """CPUs actually usable by this process. Respects SLURM/cgroup affinity
    masks, unlike os.cpu_count() which reports the whole node and can cause
    oversubscription on shared partitions."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


os.environ["LOKY_MAX_CPU_COUNT"] = str(_available_cpu_count())

optuna.logging.set_verbosity(optuna.logging.WARNING)


_ALL_ALGORITHMS = (
    "wasserstein", "adwin", "kswin", "ph",
)

DEFAULT_WASSERSTEIN_BOUNDS: dict = dict(WASSERSTEIN_BOUNDS)
DEFAULT_ADWIN_BOUNDS: dict = dict(ADWIN_BOUNDS)
DEFAULT_KSWIN_BOUNDS: dict = dict(KSWIN_BOUNDS)
DEFAULT_PH_BOUNDS: dict = dict(PAGE_HINKLEY_BOUNDS)

DEFAULT_N_TRIALS: int = N_OPTUNA_TRIALS
DEFAULT_OPTUNA_SEED: int = OPTUNA_SEED
DEFAULT_FALSE_ALARM_BUDGET_PER_1000: float = FALSE_ALARM_BUDGET_PER_1000
DEFAULT_FALSE_ALARM_CONFIDENCE: float = FALSE_ALARM_CONFIDENCE

# Warm-up cost of each window-based detector, in multiples of `window_size`,
# before it can produce its first drift test
_WASSERSTEIN_WARMUP_MULTIPLIER = 3
_KSWIN_WARMUP_MULTIPLIER = 1

# River's own warm-ups for the detectors without a window: ADWIN only tests
# once more than `grace_period` samples have arrived, Page-Hinkley once
# `min_instances` have.
_ADWIN_WARMUP = drift.ADWIN().grace_period
_PH_WARMUP = drift.PageHinkley().min_instances

# Parameters sampled on a log scale, used when flagging selections at a bound.
_LOG_PARAMS = {
    "wasserstein": (),
    "adwin": ("delta",),
    "kswin": ("alpha",),
    "ph": ("delta",),
}


def _warmup_length(algo_name: str, params: dict) -> int:
    """Number of leading samples during which ``algo_name`` cannot raise its
    first alarm, given its calibrated ``params``. For ADWIN and Page-Hinkley
    this is River's fixed ``grace_period`` / ``min_instances``. With a
    ``batch_size`` in ``params`` it is rounded up to a whole number of batches,
    since alarms fall on the last sample of a batch.
    """
    if algo_name == "wasserstein":
        warmup = _WASSERSTEIN_WARMUP_MULTIPLIER * params["window_size"]
    elif algo_name == "kswin":
        warmup = _KSWIN_WARMUP_MULTIPLIER * params["window_size"]
    elif algo_name == "adwin":
        warmup = _ADWIN_WARMUP
    elif algo_name == "ph":
        warmup = _PH_WARMUP
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    batch_size = int(params.get("batch_size", 1))
    return -(-warmup // batch_size) * batch_size


def _max_window_for_warmup(
    pre_drift_len: int, warmup_multiplier: int, safety_factor: float = 0.7
) -> int:
    """Largest window_size whose warm-up (``warmup_multiplier * window_size``)
    fits within ``pre_drift_len`` with ``safety_factor`` to spare, so the
    detector has a genuine no-drift baseline before drift_at instead of
    finishing warm-up right as the drift arrives.
    """
    return int(pre_drift_len * safety_factor) // warmup_multiplier


def _run_wasserstein(
    series: np.ndarray,
    window_size: int,
    threshold_drift_score: float,
    starting_idx: int,
    detector_cls: type = WassersteinDriftDetector,
    batch_size: int = 1,
) -> list[int]:
    detector = detector_cls(
        window_size=window_size,
        starting_idx=starting_idx,
        threshold_drift_score=threshold_drift_score,
        batch_size=batch_size,
    )
    detections: list[int] = []
    for i, x in enumerate(series):
        detector.update(x)
        if detector.drift_detected:
            detections.append(starting_idx + i)
    return detections


def _run_adwin(
    series: np.ndarray,
    delta: float,
    starting_idx: int,
    batch_size: int = 1,
) -> list[int]:
    # River's clock: ADWIN tests for a cut every ``clock`` samples.
    detector = drift.ADWIN(delta=delta, clock=batch_size)
    detections: list[int] = []
    for i, x in enumerate(series):
        detector.update(x)
        if detector.drift_detected:
            detections.append(starting_idx + i)
    return detections


def _run_kswin(
    series: np.ndarray,
    window_size: int,
    alpha: float,
    stat_size: int,
    starting_idx: int,
    seed: int = KSWIN_SEED,
    batch_size: int = 1,
) -> list[int]:
    detector = BatchKSWIN(
        alpha=alpha, window_size=window_size, stat_size=stat_size, seed=seed,
        batch_size=batch_size,
    )
    detections: list[int] = []
    for i, x in enumerate(series):
        detector.update(x)
        if detector.drift_detected:
            detections.append(starting_idx + i)
    return detections


def _run_ph(
    series: np.ndarray,
    threshold: float,
    delta: float,
    starting_idx: int,
    batch_size: int = 1,
) -> list[int]:
    detector = BatchPageHinkley(threshold=threshold, delta=delta, batch_size=batch_size)
    detections: list[int] = []
    for i, x in enumerate(series):
        detector.update(x)
        if detector.drift_detected:
            detections.append(starting_idx + i)
    return detections


def _make_sampler(
    seed: int = DEFAULT_OPTUNA_SEED,
    n_startup_trials: int = 10,
) -> optuna.samplers.TPESampler:
    return optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials)


def _poisson_rate_upper_bound(
    n_events: int,
    exposure: int,
    confidence: float = DEFAULT_FALSE_ALARM_CONFIDENCE,
) -> float:
    """One-sided bound exact under a Poisson count model, per 1,000 samples.

    Overlapping windows, alarm suppression and hyperparameter selection can
    invalidate nominal coverage. This is a calibration heuristic, not a
    post-selection guarantee; audit the frozen detector on held-out streams.
    """
    if n_events < 0:
        raise ValueError("n_events must be non-negative")
    if exposure <= 0:
        raise ValueError("exposure must be positive")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")

    upper_event_count = 0.5 * chi2.ppf(confidence, 2 * (n_events + 1))
    return float(upper_event_count / exposure * 1000)


def _false_alarm_statistics(
    detections: list[list[int]],
    streams: list,
    confidence: float,
    budget_per_1000: float,
    warmup: int,
) -> dict:
    """No-drift false-alarm statistics of one configuration.

    Rates are normalised by each stream's active exposure (its length minus the
    detector's ``warmup``), because a detector cannot alarm while warming up.
    """
    counts = np.asarray([len(indices) for indices in detections], dtype=int)
    exposures = np.asarray([len(stream) for stream in streams], dtype=int) - warmup
    if np.any(exposures <= 0):
        raise ValueError(
            f"Calibration streams must be longer than the detector warm-up "
            f"({warmup} samples)."
        )
    rates = counts / exposures * 1000
    n_false_alarms = int(np.sum(counts))
    calibration_samples = int(np.sum(exposures))
    false_alarm_ucb = _poisson_rate_upper_bound(
        n_false_alarms,
        calibration_samples,
        confidence,
    )
    return {
        "mean": float(n_false_alarms / calibration_samples * 1000),
        "std_detections": float(np.std(rates)),
        "n_false_alarms": n_false_alarms,
        "calibration_samples": calibration_samples,
        "false_alarm_ucb_per_1000": false_alarm_ucb,
        "false_alarm_budget_met": bool(false_alarm_ucb <= budget_per_1000),
    }


def _record_false_alarm_statistics(
    trial: optuna.Trial,
    detections: list[list[int]],
    streams: list,
    confidence: float,
    budget_per_1000: float,
    warmup: int,
) -> float:
    """Record no-drift statistics and return a hard-budget optimisation loss."""
    stats = _false_alarm_statistics(
        detections, streams, confidence, budget_per_1000, warmup
    )
    mean = stats["mean"]
    false_alarm_ucb = stats["false_alarm_ucb_per_1000"]
    budget_met = stats["false_alarm_budget_met"]

    trial.set_user_attr("mean", mean)
    trial.set_user_attr("std_detections", stats["std_detections"])
    trial.set_user_attr("n_false_alarms", stats["n_false_alarms"])
    trial.set_user_attr("calibration_samples", stats["calibration_samples"])
    trial.set_user_attr("warmup_samples", int(warmup))

    # Every admissible loss is <= budget, while every violating loss is > budget.
    # Within the admissible region, minimising the gap maximises the observed rate;
    # outside it, minimising the UCB finds the least-violating configuration.
    selection_loss = budget_per_1000 - mean if budget_met else false_alarm_ucb

    trial.set_user_attr("false_alarm_ucb_per_1000", false_alarm_ucb)
    trial.set_user_attr("false_alarm_confidence", confidence)
    trial.set_user_attr("false_alarm_budget_met", budget_met)
    trial.set_user_attr("selection_objective", selection_loss)
    return selection_loss


def _trials_to_df(
    study: optuna.Study, param_cols: list[str], algorithm: str
) -> pd.DataFrame:
    """Convert completed Optuna trials to a calibration DataFrame."""
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {k: t.params[k] for k in param_cols}
        row["mean_detections"] = t.user_attrs.get("mean", t.values[0])
        row["std_detections"] = t.user_attrs.get("std_detections", np.nan)
        row["n_false_alarms"] = t.user_attrs.get("n_false_alarms", np.nan)
        row["calibration_samples"] = t.user_attrs.get("calibration_samples", np.nan)
        row["warmup_samples"] = t.user_attrs.get("warmup_samples", np.nan)
        row["false_alarm_ucb_per_1000"] = t.user_attrs.get(
            "false_alarm_ucb_per_1000", np.nan
        )
        row["false_alarm_confidence"] = t.user_attrs.get(
            "false_alarm_confidence", np.nan
        )
        row["selection_objective"] = t.user_attrs.get("selection_objective", t.values[0])
        rows.append(row)
    df = pd.DataFrame(rows)
    df["algorithm"] = algorithm
    return df


def _params_at_bound(
    row: pd.Series,
    bounds: dict,
    log_params: tuple[str, ...] = (),
    tolerance: float = AT_BOUND_TOLERANCE,
) -> str:
    """Comma-separated ``name=low``/``name=high`` flags for every parameter of
    ``row`` lying within ``tolerance`` of its search range (log space for
    ``log_params``); empty when the selection is interior."""
    flags = []
    for param, (low, high) in bounds.items():
        if param not in row.index:
            continue
        value = float(row[param])
        if param in log_params:
            value, low, high = np.log10(value), np.log10(low), np.log10(high)
        span = high - low
        if span <= 0:
            continue
        if value - low <= tolerance * span:
            flags.append(f"{param}=low")
        elif high - value <= tolerance * span:
            flags.append(f"{param}=high")
    return ",".join(flags)


def _select_best(
    df: pd.DataFrame,
    param_cols: list[str],
    verbose: bool,
    name: str,
    aggressiveness_sort: list[tuple[str, bool]] | None = None,
    false_alarm_budget_per_1000: float = DEFAULT_FALSE_ALARM_BUDGET_PER_1000,
    bounds: dict | None = None,
    log_params: tuple[str, ...] = (),
) -> tuple[dict, pd.Series]:
    """Select the most aggressive stable trial whose false-alarm UCB meets budget.

    Admissible configurations have a one-sided false-alarm-rate upper confidence
    bound no greater than ``false_alarm_budget_per_1000``. Among them, the mean
    rate is maximised to saturate the budget, then between-stream variability is
    minimised, and detector-specific structural aggressiveness breaks remaining
    ties. If no trial is admissible, the least-violating UCB is selected and the
    returned row has ``false_alarm_budget_met=False``. When ``bounds`` is given,
    ``params_at_bound`` flags trials whose parameters sit at the search-space
    edge.
    """
    if (
        not np.isfinite(false_alarm_budget_per_1000)
        or false_alarm_budget_per_1000 < 0
    ):
        raise ValueError("false_alarm_budget_per_1000 must be finite and non-negative")

    df["false_alarm_budget_per_1000"] = false_alarm_budget_per_1000
    df["false_alarm_budget_met"] = (
        df["false_alarm_ucb_per_1000"] <= false_alarm_budget_per_1000
    )
    df["params_at_bound"] = (
        df.apply(_params_at_bound, axis=1, bounds=bounds, log_params=log_params)
        if bounds and not df.empty
        else ""
    )

    admissible = df.loc[df["false_alarm_budget_met"]]
    budget_met = not admissible.empty
    candidates = admissible if budget_met else df

    if budget_met:
        sort_cols = ["mean_detections", "std_detections"]
        ascending = [False, True]
    else:
        sort_cols = ["false_alarm_ucb_per_1000", "std_detections"]
        ascending = [True, True]

    for col, higher_is_more_aggressive in aggressiveness_sort or []:
        sort_cols.append(col)
        ascending.append(not higher_is_more_aggressive)

    if not budget_met:
        sort_cols.append("mean_detections")
        ascending.append(True)

    ranked = candidates.sort_values(
        sort_cols,
        ascending=ascending,
        kind="mergesort",
        na_position="last",
    )
    best_row = ranked.iloc[0]

    # Convert numpy scalars to Python native types (River requires plain int/float)
    best_params = {
        k: best_row[k].item() if hasattr(best_row[k], "item") else best_row[k]
        for k in param_cols
    }
    if verbose:
        msg = (
            f"  [{name}] Best params: {best_params}"
            f"  (mean={best_row['mean_detections']:.3f}"
        )
        if "std_detections" in df.columns:
            msg += f", std={best_row['std_detections']:.3f}"
        msg += (
            f", UCB={best_row['false_alarm_ucb_per_1000']:.3f}"
            f", budget={false_alarm_budget_per_1000:g}"
            f", budget_met={budget_met})"
        )
        if best_row["params_at_bound"]:
            msg += f"  WARNING: at search bound ({best_row['params_at_bound']})"
        print(msg)
    return best_params, best_row


def _calibrate_wasserstein(
    calibration_streams: list,
    bounds: dict,
    starting_idx: int,
    n_trials: int,
    verbose: bool,
    optuna_seed: int = DEFAULT_OPTUNA_SEED,
    stream_backend: str = "loky",
    false_alarm_budget_per_1000: float = DEFAULT_FALSE_ALARM_BUDGET_PER_1000,
    false_alarm_confidence: float = DEFAULT_FALSE_ALARM_CONFIDENCE,
    detector_cls: type = WassersteinDriftDetector,
    batch_size: int = 1,
) -> tuple[dict, pd.Series, pd.DataFrame]:
    param_cols = ["threshold_drift_score", "window_size"]

    def objective(trial: optuna.Trial) -> float:
        threshold = trial.suggest_float(
            "threshold_drift_score", *bounds["threshold_drift_score"]
        )
        window = trial.suggest_int("window_size", *bounds["window_size"])
        raw = Parallel(n_jobs=-1, backend=stream_backend)(
            delayed(_run_wasserstein)(s, window_size=window,
                                 threshold_drift_score=threshold,
                                 starting_idx=starting_idx,
                                 detector_cls=detector_cls,
                                 batch_size=batch_size)
            for s in calibration_streams
        )
        return _record_false_alarm_statistics(
            trial,
            raw,
            calibration_streams,
            false_alarm_confidence,
            false_alarm_budget_per_1000,
            warmup=_warmup_length("wasserstein", {**trial.params, "batch_size": batch_size}),
        )

    if verbose:
        print(f"\nWasserstein: {n_trials} Optuna trials x {len(calibration_streams)} streams ...")
    study = optuna.create_study(
        direction="minimize", sampler=_make_sampler(seed=optuna_seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    df = _trials_to_df(study, param_cols, "wasserstein")
    df["optuna_seed"] = optuna_seed
    best, best_row = _select_best(
        df,
        param_cols,
        verbose,
        "wasserstein",
        bounds=bounds,
        log_params=_LOG_PARAMS["wasserstein"],
        aggressiveness_sort=[("threshold_drift_score", False), ("window_size", False)],
        false_alarm_budget_per_1000=false_alarm_budget_per_1000,
    )
    return best, best_row, df


WINDOW_SELECTION_MODES = ("largest_admissible", "tpe")

# Candidate windows are tried downward from the upper bound in steps of this
# many samples until one admits a threshold that meets the budget.
_WINDOW_SEARCH_STEP = 50


def _check_window_selection(window_selection: str) -> None:
    if window_selection not in WINDOW_SELECTION_MODES:
        raise ValueError(
            f"Unknown Wasserstein window selection {window_selection!r}; "
            f"choose one of {WINDOW_SELECTION_MODES}."
        )


def _calibrate_wasserstein_largest_window(
    calibration_streams: list,
    bounds: dict,
    starting_idx: int,
    n_trials: int,
    verbose: bool,
    optuna_seed: int = DEFAULT_OPTUNA_SEED,
    stream_backend: str = "loky",
    false_alarm_budget_per_1000: float = DEFAULT_FALSE_ALARM_BUDGET_PER_1000,
    false_alarm_confidence: float = DEFAULT_FALSE_ALARM_CONFIDENCE,
    detector_cls: type = WassersteinDriftDetector,
    batch_size: int = 1,
) -> tuple[dict, pd.Series, pd.DataFrame]:
    """Largest admissible window, with the lowest threshold meeting the budget.

    Windows are tried from ``bounds["window_size"][1]`` downward. At each, the
    threshold is solved for exactly (:func:`budget_matched_threshold`) within
    the detector's threshold domain, and the detector is then run at that
    operating point to record its calibration statistics; the first window
    meeting the budget is selected. The threshold bounds play no role, since
    nothing is searched, and neither do ``n_trials`` or ``optuna_seed``, which
    are accepted for signature compatibility with :func:`_calibrate_wasserstein`.
    If no window meets the budget, the least-violating one is returned with
    ``false_alarm_budget_met=False``. ``params_at_bound`` flags only a threshold
    clipped to the top of its domain: the window sits at its upper bound by
    design.
    """
    low, high = (int(v) for v in bounds["window_size"])
    theta_low, theta_high = PARAMETER_DOMAINS["wasserstein"]["threshold_drift_score"]
    windows = list(range(high, low - 1, -_WINDOW_SEARCH_STEP))
    if windows[-1] != low:
        windows.append(low)

    rows = []
    for window in windows:
        matched = budget_matched_threshold(
            calibration_streams, window, false_alarm_budget_per_1000,
            false_alarm_confidence, backend=stream_backend, detector_cls=detector_cls,
            batch_size=batch_size,
        )
        exact = matched["threshold_drift_score"]
        threshold = float(np.clip(exact, theta_low, theta_high))
        raw = Parallel(n_jobs=-1, backend=stream_backend)(
            delayed(_run_wasserstein)(s, window_size=window,
                                     threshold_drift_score=threshold,
                                     starting_idx=starting_idx,
                                     detector_cls=detector_cls,
                                     batch_size=batch_size)
            for s in calibration_streams
        )
        warmup = _warmup_length(
            "wasserstein", {"window_size": window, "batch_size": batch_size}
        )
        stats = _false_alarm_statistics(
            raw, calibration_streams, false_alarm_confidence,
            false_alarm_budget_per_1000, warmup,
        )
        if threshold == exact and stats["n_false_alarms"] != matched["n_false_alarms"]:
            raise RuntimeError(
                f"{detector_cls.__name__} raised {stats['n_false_alarms']} calibration "
                f"alarms at window_size={window}, threshold={threshold:.6g}, but its "
                f"score trajectories predict {matched['n_false_alarms']}; check its "
                "`rearms_at_zero` attribute."
            )
        met = stats["false_alarm_budget_met"]
        rows.append({
            "threshold_drift_score": threshold,
            "window_size": window,
            "mean_detections": stats["mean"],
            "std_detections": stats["std_detections"],
            "n_false_alarms": stats["n_false_alarms"],
            "calibration_samples": stats["calibration_samples"],
            "warmup_samples": warmup,
            "false_alarm_ucb_per_1000": stats["false_alarm_ucb_per_1000"],
            "false_alarm_confidence": false_alarm_confidence,
            "selection_objective": (
                false_alarm_budget_per_1000 - stats["mean"] if met
                else stats["false_alarm_ucb_per_1000"]
            ),
            "false_alarm_budget_per_1000": false_alarm_budget_per_1000,
            "false_alarm_budget_met": met,
            "params_at_bound": "threshold_drift_score=high" if exact > theta_high else "",
        })
        if met:
            break

    df = pd.DataFrame(rows)
    df["algorithm"] = "wasserstein"
    df["optuna_seed"] = optuna_seed
    df["window_selection"] = "largest_admissible"
    admissible = df.index[df["false_alarm_budget_met"]]
    best_index = (
        admissible[0] if len(admissible)
        else df.sort_values(["false_alarm_ucb_per_1000", "std_detections"],
                            kind="mergesort").index[0]
    )
    best_row = df.loc[best_index]
    best = {
        "threshold_drift_score": float(best_row["threshold_drift_score"]),
        "window_size": int(best_row["window_size"]),
    }
    if verbose:
        print(
            f"  [wasserstein] Largest admissible window: {best}"
            f"  (mean={best_row['mean_detections']:.3f}"
            f", UCB={best_row['false_alarm_ucb_per_1000']:.3f}"
            f", budget={false_alarm_budget_per_1000:g}"
            f", budget_met={bool(best_row['false_alarm_budget_met'])})"
        )
    return best, best_row, df


def _calibrate_adwin(
    calibration_streams: list,
    bounds: dict,
    starting_idx: int,
    n_trials: int,
    verbose: bool,
    optuna_seed: int = DEFAULT_OPTUNA_SEED,
    stream_backend: str = "loky",
    false_alarm_budget_per_1000: float = DEFAULT_FALSE_ALARM_BUDGET_PER_1000,
    false_alarm_confidence: float = DEFAULT_FALSE_ALARM_CONFIDENCE,
    batch_size: int = 1,
) -> tuple[dict, pd.Series, pd.DataFrame]:
    param_cols = ["delta"]

    def objective(trial: optuna.Trial) -> float:
        delta = trial.suggest_float("delta", *bounds["delta"], log=True)
        raw = Parallel(n_jobs=-1, backend=stream_backend)(
            delayed(_run_adwin)(s, delta=delta, starting_idx=starting_idx,
                                batch_size=batch_size)
            for s in calibration_streams
        )
        return _record_false_alarm_statistics(
            trial,
            raw,
            calibration_streams,
            false_alarm_confidence,
            false_alarm_budget_per_1000,
            warmup=_warmup_length("adwin", {**trial.params, "batch_size": batch_size}),
        )

    if verbose:
        print(f"\nADWIN: {n_trials} Optuna trials x {len(calibration_streams)} streams ...")
    study = optuna.create_study(
        direction="minimize", sampler=_make_sampler(seed=optuna_seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    df = _trials_to_df(study, param_cols, "adwin")
    df["optuna_seed"] = optuna_seed
    best, best_row = _select_best(
        df,
        param_cols,
        verbose,
        "adwin",
        bounds=bounds,
        log_params=_LOG_PARAMS["adwin"],
        aggressiveness_sort=[("delta", True)],
        false_alarm_budget_per_1000=false_alarm_budget_per_1000,
    )
    return best, best_row, df


def _calibrate_kswin(
    calibration_streams: list,
    bounds: dict,
    starting_idx: int,
    n_trials: int,
    verbose: bool,
    optuna_seed: int = DEFAULT_OPTUNA_SEED,
    stream_backend: str = "loky",
    false_alarm_budget_per_1000: float = DEFAULT_FALSE_ALARM_BUDGET_PER_1000,
    false_alarm_confidence: float = DEFAULT_FALSE_ALARM_CONFIDENCE,
    batch_size: int = 1,
    require_attainable_alpha: bool = False,
) -> tuple[dict, pd.Series, pd.DataFrame]:
    param_cols = ["window_size", "alpha", "stat_size"]

    def objective(trial: optuna.Trial) -> float:
        window = trial.suggest_int("window_size", *bounds["window_size"])
        alpha = trial.suggest_float("alpha", *bounds["alpha"], log=True)
        # KSWIN draws its stat_size reference points from the window's first
        # window_size - stat_size points, so stat_size can be at most half the window
        stat_size = trial.suggest_int(
            "stat_size", bounds["stat_size"][0], min(bounds["stat_size"][1], window // 2)
        )
        # Equal-size exact two-sided KS tests cannot attain a smaller p-value,
        # even for disjoint samples. Such a configuration can never alarm.
        if require_attainable_alpha and alpha < 2 / math.comb(2 * stat_size, stat_size):
            raise optuna.TrialPruned("alpha is below the exact KS p-value floor")
        raw = Parallel(n_jobs=-1, backend=stream_backend)(
            delayed(_run_kswin)(s, window_size=window, alpha=alpha,
                                stat_size=stat_size, starting_idx=starting_idx,
                                batch_size=batch_size)
            for s in calibration_streams
        )
        return _record_false_alarm_statistics(
            trial,
            raw,
            calibration_streams,
            false_alarm_confidence,
            false_alarm_budget_per_1000,
            warmup=_warmup_length("kswin", {**trial.params, "batch_size": batch_size}),
        )

    if verbose:
        print(f"\nKSWIN: {n_trials} Optuna trials x {len(calibration_streams)} streams ...")
    study = optuna.create_study(
        direction="minimize", sampler=_make_sampler(seed=optuna_seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    df = _trials_to_df(study, param_cols, "kswin")
    if df.empty:
        raise ValueError("No attainable KSWIN trials; increase n_trials or the alpha range")
    df["optuna_seed"] = optuna_seed
    best, best_row = _select_best(
        df,
        param_cols,
        verbose,
        "kswin",
        bounds=bounds,
        log_params=_LOG_PARAMS["kswin"],
        aggressiveness_sort=[("alpha", True)],
        false_alarm_budget_per_1000=false_alarm_budget_per_1000,
    )
    return best, best_row, df


def _calibrate_ph(
    calibration_streams: list,
    bounds: dict,
    starting_idx: int,
    n_trials: int,
    verbose: bool,
    optuna_seed: int = DEFAULT_OPTUNA_SEED,
    stream_backend: str = "loky",
    false_alarm_budget_per_1000: float = DEFAULT_FALSE_ALARM_BUDGET_PER_1000,
    false_alarm_confidence: float = DEFAULT_FALSE_ALARM_CONFIDENCE,
    batch_size: int = 1,
) -> tuple[dict, pd.Series, pd.DataFrame]:
    param_cols = ["threshold", "delta"]

    def objective(trial: optuna.Trial) -> float:
        threshold = trial.suggest_float("threshold", *bounds["threshold"])
        delta = trial.suggest_float("delta", *bounds["delta"], log=True)
        raw = Parallel(n_jobs=-1, backend=stream_backend)(
            delayed(_run_ph)(s, threshold=threshold, delta=delta,
                             starting_idx=starting_idx, batch_size=batch_size)
            for s in calibration_streams
        )
        return _record_false_alarm_statistics(
            trial,
            raw,
            calibration_streams,
            false_alarm_confidence,
            false_alarm_budget_per_1000,
            warmup=_warmup_length("ph", {**trial.params, "batch_size": batch_size}),
        )

    if verbose:
        print(f"\nPageHinkley: {n_trials} Optuna trials x {len(calibration_streams)} streams ...")
    study = optuna.create_study(
        direction="minimize", sampler=_make_sampler(seed=optuna_seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    df = _trials_to_df(study, param_cols, "ph")
    df["optuna_seed"] = optuna_seed
    best, best_row = _select_best(
        df,
        param_cols,
        verbose,
        "ph",
        bounds=bounds,
        log_params=_LOG_PARAMS["ph"],
        aggressiveness_sort=[("threshold", False), ("delta", False)],
        false_alarm_budget_per_1000=false_alarm_budget_per_1000,
    )
    return best, best_row, df


def _eval_single_stream(
    scenario: str,
    algorithm: str,
    stream: np.ndarray,
    run_fn: Callable,
    params: dict,
    drift_at: int,
    starting_idx: int,
    common_exposure_start: Optional[int] = None,
    detection_horizon: Optional[int] = None,
) -> tuple:
    """Run one detector on one stream; return raw counts for aggregation.

    ``common_exposure_start`` defines the first sample on which false positives
    are comparable across detectors. The unqualified false-positive count is
    retained for active-exposure reporting; the common-window count follows the
    first five fields.

    The fifth field is the delay of the first alarm in
    ``[drift_at, drift_at + detection_horizon)`` (``None`` when there is none),
    i.e. the detection criterion; ``detection_horizon=None`` accepts any
    post-drift alarm. The last field is the delay of the first post-drift alarm
    regardless of the horizon.
    """
    if common_exposure_start is None:
        common_exposure_start = starting_idx
    if not starting_idx <= common_exposure_start < drift_at:
        raise ValueError(
            "common_exposure_start must be in the pre-drift interval "
            f"[{starting_idx}, {drift_at})"
        )

    dets = run_fn(stream, **params, starting_idx=starting_idx)
    before = [d for d in dets if d < drift_at]
    before_common = [d for d in before if d >= common_exposure_start]
    after = [d for d in dets if d >= drift_at]
    in_horizon = (
        after
        if detection_horizon is None
        else [d for d in after if d < drift_at + detection_horizon]
    )
    return (
        scenario,
        algorithm,
        len(before),
        len(after),
        in_horizon[0] - drift_at if in_horizon else None,
        len(before_common),
        after[0] - drift_at if after else None,
    )


def _check_detection_horizon(
    streams: dict[str, list],
    drift_at: int,
    starting_idx: int,
    detection_horizon: Optional[int],
) -> None:
    """Reject a horizon that is not positive or that runs past a stream's end."""
    if detection_horizon is None:
        return
    if detection_horizon <= 0:
        raise ValueError("detection_horizon must be positive or None")
    for name, group in streams.items():
        for stream in group:
            if starting_idx + len(stream) < drift_at + detection_horizon:
                raise ValueError(
                    f"{name}: a stream of length {len(stream)} ends before "
                    f"drift_at + detection_horizon = {drift_at + detection_horizon}."
                )


def _check_batch_alignment(
    batch_size: int,
    drift_at: int,
    starting_idx: int,
    detection_horizon: Optional[int],
) -> None:
    """Reject a batch size whose batches straddle the drift or the horizon end.

    Batches start at ``starting_idx``. A drift in the middle of a batch would
    hand that batch's alarm a delay shorter than the batch, and a horizon
    ending mid-batch would cut the last batch short.
    """
    if int(batch_size) != batch_size or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size}")
    if (drift_at - starting_idx) % batch_size:
        raise ValueError(
            f"drift_at - starting_idx = {drift_at - starting_idx} is not a "
            f"multiple of batch_size={batch_size}"
        )
    if detection_horizon is not None and detection_horizon % batch_size:
        raise ValueError(
            f"detection_horizon={detection_horizon} is not a multiple of "
            f"batch_size={batch_size}"
        )


def _fmt_delay(mean: float, std: float) -> str:
    if np.isnan(mean):
        return "NaN"
    std_str = f"{std:.1f}" if not np.isnan(std) else "N/A"
    return f"{mean:.1f} +/- {std_str}"


def _fmt_ci(lo: float, hi: float, decimals: int = 1) -> str:
    if np.isnan(lo) or np.isnan(hi):
        return "[CI: N/A]"
    return f"[CI: {lo:.{decimals}f}, {hi:.{decimals}f}]"


def _wilson_ci(successes: int, n: int, ci_level: float = CI_LEVEL) -> tuple[float, float]:
    """Two-sided Wilson score interval for independent stream-level outcomes."""
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must be between 0 and 1")
    if n < 0 or not 0 <= successes <= n:
        raise ValueError("require 0 <= successes <= n")
    if n == 0:
        return np.nan, np.nan
    z2 = float(norm.ppf((1 + ci_level) / 2)) ** 2
    p = successes / n
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    radius = np.sqrt(p * (1 - p) / n * z2 + z2**2 / (4 * n**2)) / denominator
    return (0.0 if successes == 0 else float(center - radius),
            1.0 if successes == n else float(center + radius))


def _capped_delay(delay: Optional[float], horizon: Optional[int]) -> float:
    # There is no common cap when the caller requests an unbounded horizon.
    if horizon is None:
        return np.nan
    return float(horizon if delay is None else min(delay, horizon))


def _null_alarm_audit(
    alarms: list[list[int]],
    streams: list,
    warmup: int,
    starting_idx: int,
    *,
    n_bootstrap: int,
    ci_level: float,
    bootstrap_seed: int,
    false_alarm_confidence: float,
    false_alarm_budget_per_1000: float,
    calibration_ucb: float = np.nan,
) -> dict:
    """Audit frozen parameters without selecting or adjusting them.

    Resample independent *whole streams*, preserving within-stream alarm
    dependence. The ratio of pooled counts to pooled exposure handles unequal
    lengths. Bootstrap intervals are unavailable with fewer than two streams
    or no alarms (resampling zeros cannot establish an upper rate bound).
    Coverage of an independently estimated rate is only a coverage proxy,
    never coverage of a known population parameter. Shared null streams across
    calibration replicates also make the resulting proxy indicators dependent.
    Counts/exposures are JSON arrays in input stream order for CSV persistence.
    """
    if len(alarms) != len(streams):
        raise ValueError("alarms and streams must have the same length")
    counts = np.asarray([
        sum(starting_idx + warmup <= a < starting_idx + len(s) for a in ds)
        for ds, s in zip(alarms, streams)
    ], dtype=int)
    exposures = np.asarray([len(s) - warmup for s in streams], dtype=int)
    if np.any(exposures <= 0):
        raise ValueError("Null streams must be longer than the detector warm-up")
    total = int(counts.sum())
    exposure = int(exposures.sum())
    rate = total / exposure * 1000 if exposure else np.nan
    lo = hi = dispersion = np.nan
    if len(counts) >= 2 and total > 0:
        rng = np.random.default_rng(bootstrap_seed)
        idx = rng.integers(0, len(counts), size=(n_bootstrap, len(counts)))
        rates = counts[idx].sum(axis=1) / exposures[idx].sum(axis=1) * 1000
        alpha = (1 - ci_level) / 2
        lo, hi = map(float, np.quantile(rates, [alpha, 1 - alpha]))
        expected = total * exposures / exposure
        # Pearson dispersion accommodates unequal exposures; >1 suggests
        # overdispersion relative to the Poisson working model.
        dispersion = float(np.sum((counts - expected)**2 / expected) / (len(counts) - 1))
    ucb = (_poisson_rate_upper_bound(total, exposure, false_alarm_confidence)
           if exposure else np.nan)
    available = np.isfinite(rate) and np.isfinite(calibration_ucb)
    return {
        "null_n_streams": len(counts),
        "null_n_false_alarms": total,
        "null_active_samples": exposure,
        "null_alarm_counts": json.dumps(counts.tolist()),
        "null_active_exposures": json.dumps(exposures.tolist()),
        "null_false_alarms_per_1000": rate,
        "null_rate_ci_low": lo,
        "null_rate_ci_high": hi,
        "null_rate_ci_method": "whole_stream_percentile_bootstrap",
        "null_count_dispersion": dispersion,
        "null_poisson_ucb_per_1000": ucb,
        "null_false_alarm_confidence": false_alarm_confidence if exposure else np.nan,
        "null_false_alarm_budget_per_1000": false_alarm_budget_per_1000 if exposure else np.nan,
        "null_observed_budget_met": bool(rate <= false_alarm_budget_per_1000) if exposure else np.nan,
        "null_poisson_budget_met": bool(ucb <= false_alarm_budget_per_1000) if exposure else np.nan,
        "calibration_ucb_covers_null_rate_proxy": bool(rate <= calibration_ucb) if available else np.nan,
        "calibration_ucb_below_null_ci": bool(calibration_ucb < lo) if available and np.isfinite(lo) else np.nan,
    }


def _bootstrap_ci(
    values: list,
    n_boot: int,
    ci_level: float,
    rng: np.random.Generator,
    statistic: Callable = np.mean,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``statistic`` (default the mean) of `values`.

    Needs >= 2 observations to say anything about uncertainty; with fewer,
    returns (nan, nan) rather than a degenerate zero-width interval.
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot_stats = statistic(arr[idx], axis=1)
    alpha = (1 - ci_level) / 2
    lo, hi = np.percentile(boot_stats, [100 * alpha, 100 * (1 - alpha)])
    return float(lo), float(hi)


def delay_reportable(
    n_detected: int, n_streams: int, placebo_rate: float, ci_level: float = CI_LEVEL
) -> bool:
    """Whether a conditional delay describes the detector rather than chance.

    Requires at least :data:`MIN_DETECTED_STREAMS_FOR_DELAY` detected streams
    and, when a placebo rate is available, a Wilson lower bound of the detection
    rate above it. Otherwise the detections are indistinguishable from chance
    alarms, and their delay measures where those alarms happened to fall.
    """
    if n_detected < MIN_DETECTED_STREAMS_FOR_DELAY:
        return False
    if placebo_rate is None or np.isnan(placebo_rate):
        return True
    return _wilson_ci(n_detected, n_streams, ci_level)[0] > placebo_rate


def _delay_summary(
    delays: list,
    capped_delays: list,
    reportable: bool,
    n_bootstrap: int,
    ci_level: float,
    rng: np.random.Generator,
    median_rng: np.random.Generator,
) -> dict:
    """Conditional and horizon-capped delay statistics of one (scenario, detector).

    Conditional statistics (over detected streams) are NaN unless
    ``reportable``; their unmasked mean and median are kept with a ``_raw``
    suffix. Medians accompany the means because a few late detections can
    dominate a mean. ``median_rng`` keeps the mean intervals' bootstrap draws
    unchanged by the median intervals.
    """
    mean_lo, mean_hi = _bootstrap_ci(delays, n_bootstrap, ci_level, rng)
    capped_lo, capped_hi = _bootstrap_ci(capped_delays, n_bootstrap, ci_level, rng)
    median_lo, median_hi = _bootstrap_ci(delays, n_bootstrap, ci_level, median_rng, np.median)
    mean = float(np.mean(delays)) if delays else np.nan
    median = float(np.median(delays)) if delays else np.nan
    conditional = {
        "mean_first_delay": mean,
        "std_first_delay": float(np.std(delays)) if len(delays) > 1 else np.nan,
        "mean_first_delay_ci_low": mean_lo,
        "mean_first_delay_ci_high": mean_hi,
        "median_first_delay": median,
        "median_first_delay_ci_low": median_lo,
        "median_first_delay_ci_high": median_hi,
    }
    if not reportable:
        conditional = {key: np.nan for key in conditional}
    return {
        **conditional,
        "mean_first_delay_raw": mean,
        "median_first_delay_raw": median,
        "conditional_delay_reportable": bool(reportable),
        "mean_capped_delay": float(np.mean(capped_delays)) if capped_delays else np.nan,
        "mean_capped_delay_ci_low": capped_lo,
        "mean_capped_delay_ci_high": capped_hi,
        "median_capped_delay": float(np.median(capped_delays)) if capped_delays else np.nan,
    }


def _as_list_of_streams(streams) -> list:
    """Normalise a single stream or collection of streams to a list of 1-D arrays."""
    try:
        arr = np.asarray(streams)
    except ValueError:
        return [np.asarray(s) for s in streams]
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        return [arr]
    if arr.ndim == 2:
        return [arr[i] for i in range(arr.shape[0])]
    return [np.asarray(s) for s in streams]


def evaluate_with_params(
    test_streams: dict,
    best_params: dict,
    drift_at: int,
    starting_idx: int = 0,
    algorithms: Optional[list] = None,
    n_bootstrap: int = N_BOOTSTRAP,
    ci_level: float = CI_LEVEL,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    verbose: bool = True,
    detection_horizon: Optional[int] = DETECTION_HORIZON,
    null_streams: Optional[list] = None,
    calibration_results: Optional[pd.DataFrame] = None,
    false_alarm_budget_per_1000: float = DEFAULT_FALSE_ALARM_BUDGET_PER_1000,
    false_alarm_confidence: float = DEFAULT_FALSE_ALARM_CONFIDENCE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate already-calibrated detectors with the benchmark test protocol.

    This is the test phase used by :func:`calibrate_and_test`, exposed
    separately for experiments whose test distribution changes while their
    training and no-drift calibration data remain identical. False positives
    are reported primarily on a common window that begins after the largest
    warm-up among the evaluated detectors; detector-specific active-exposure
    rates are retained as supplementary outputs.

    A stream is detected when an alarm falls in
    ``[drift_at, drift_at + detection_horizon)`` (any post-drift alarm when the
    horizon is ``None``); delays are measured on those alarms. ``null_streams``
    are held-out drift-free streams from the same process as the pre-drift
    test data, disjoint from the calibration streams. Applying the same
    criterion to them at ``drift_at`` gives the placebo detection rate, i.e. the
    detection rate a detector reaches by chance. Detection/placebo intervals
    use Wilson scores. ``mean_capped_delay`` includes every stream, assigning
    misses ``detection_horizon``; it is NaN when the horizon is None.
    Conditional delays (``mean_first_delay``, ``median_first_delay`` and
    their intervals) are NaN unless :func:`delay_reportable`; the unmasked
    values are kept as ``mean_first_delay_raw``/``median_first_delay_raw``.

    Full post-warm-up null counts and whole-stream bootstrap intervals audit
    the frozen detector. Optional ``calibration_results`` supplies selected
    calibration bounds (one row per algorithm, or a trial table with
    ``selected``). Their coverage of the held-out estimated rate is a proxy,
    not a post-selection coverage guarantee. Null data must be independent of
    tuning and must not subsequently be used to choose configurations.
    """
    _wilson_ci(0, 0, ci_level)  # Validate even when no streams are supplied.
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")
    _poisson_rate_upper_bound(0, 1, false_alarm_confidence)
    if not np.isfinite(false_alarm_budget_per_1000) or false_alarm_budget_per_1000 < 0:
        raise ValueError("false_alarm_budget_per_1000 must be finite and non-negative")
    selected_calibration = {}
    if calibration_results is not None and not calibration_results.empty:
        selected = calibration_results
        if "selected" in selected:
            selected = selected.loc[selected["selected"]]
        if selected["algorithm"].duplicated().any():
            raise ValueError("Expected one selected calibration row per algorithm")
        selected_calibration = selected.set_index("algorithm").to_dict("index")
    active_algos = set(algorithms) if algorithms is not None else set(best_params)
    unknown = active_algos - set(_ALL_ALGORITHMS)
    if unknown:
        raise ValueError(f"Unknown algorithm(s): {unknown}. Valid choices: {_ALL_ALGORITHMS}")
    missing_params = active_algos - set(best_params)
    if missing_params:
        raise ValueError(f"Missing calibrated parameters for algorithm(s): {missing_params}")

    test_streams = {k: _as_list_of_streams(v) for k, v in test_streams.items()}
    null_streams = _as_list_of_streams(null_streams) if null_streams is not None else []
    _check_detection_horizon(test_streams, drift_at, starting_idx, detection_horizon)
    _check_detection_horizon(
        {"null_streams": null_streams}, drift_at, starting_idx, detection_horizon
    )

    if verbose:
        md.h2(
            f"Test ({len(test_streams)} scenarios, drift_at={drift_at}, "
            f"detection_horizon={detection_horizon})"
        )

    algo_runners: list[tuple[str, Callable, dict]] = [
        (name, run_fn, best_params[name])
        for name, run_fn in [
            ("wasserstein", _run_wasserstein),
            ("adwin", _run_adwin),
            ("kswin", _run_kswin),
            ("ph", _run_ph),
        ]
        if name in active_algos
    ]

    pre_drift_len = drift_at - starting_idx
    warmup_samples = {
        algo_name: _warmup_length(algo_name, params)
        for algo_name, _, params in algo_runners
    }
    common_warmup_samples = max(warmup_samples.values())
    common_exposure_start = starting_idx + common_warmup_samples
    common_exposure_samples = drift_at - common_exposure_start
    if common_exposure_samples <= 0:
        raise ValueError(
            "The maximum detector warm-up leaves no common pre-drift exposure: "
            f"max_warmup={common_warmup_samples}, pre_drift_len={pre_drift_len}."
        )

    active_pre_drift_len = {
        algo_name: pre_drift_len - warmup
        for algo_name, warmup in warmup_samples.items()
    }
    for algo_name, active_len in active_pre_drift_len.items():
        if active_len <= 0:
            raise ValueError(
                f"{algo_name}: warm-up ({warmup_samples[algo_name]} samples) "
                f"does not fit within pre_drift_len ({pre_drift_len}); cannot "
                "compute an active-sample false-positive rate."
            )

    stream_tasks = [
        (scenario, algo_name, stream_id, stream, run_fn, params)
        for scenario, streams in test_streams.items()
        for algo_name, run_fn, params in algo_runners
        for stream_id, stream in enumerate(streams)
    ]

    raw_stream_results = Parallel(
        n_jobs=-1, backend="loky", verbose=10 if verbose else 0
    )(
        delayed(_eval_single_stream)(
            scenario,
            algo_name,
            stream,
            run_fn,
            params,
            drift_at,
            starting_idx,
            common_exposure_start,
            detection_horizon,
        )
        for scenario, algo_name, stream_id, stream, run_fn, params in stream_tasks
    )

    placebo_tasks = [
        (algo_name, stream, run_fn, params)
        for algo_name, run_fn, params in algo_runners
        for stream in null_streams
    ]
    null_alarms = Parallel(n_jobs=-1, backend="loky")(
        delayed(run_fn)(stream, **params, starting_idx=starting_idx)
        for _, stream, run_fn, params in placebo_tasks
    )
    placebo_detected: dict[str, list[bool]] = {name: [] for name, _, _ in algo_runners}
    alarms_by_algorithm = {name: [] for name, _, _ in algo_runners}
    end = np.inf if detection_horizon is None else drift_at + detection_horizon
    for (algo_name, _, _, _), alarms in zip(placebo_tasks, null_alarms):
        placebo_detected[algo_name].append(any(drift_at <= d < end for d in alarms))
        alarms_by_algorithm[algo_name].append(alarms)
    null_audits = {}
    for algo_name, _, _ in algo_runners:
        calibration = selected_calibration.get(algo_name, {})
        calibration_confidence = calibration.get("false_alarm_confidence", np.nan)
        if pd.isna(calibration_confidence):
            calibration_confidence = false_alarm_confidence
        null_audits[algo_name] = _null_alarm_audit(
            alarms_by_algorithm[algo_name], null_streams,
            warmup_samples[algo_name], starting_idx,
            n_bootstrap=n_bootstrap, ci_level=ci_level, bootstrap_seed=bootstrap_seed,
            false_alarm_confidence=calibration_confidence,
            false_alarm_budget_per_1000=calibration.get(
                "false_alarm_budget_per_1000", false_alarm_budget_per_1000),
            calibration_ucb=calibration.get("false_alarm_ucb_per_1000", np.nan),
        )

    groups: dict[tuple, dict] = {}
    for sc, al, nb_active, na, fd, nb_common, _ in raw_stream_results:
        key = (sc, al)
        if key not in groups:
            groups[key] = {
                "nb_active": [],
                "nb_common": [],
                "na": [],
                "fd": [],
                "detected": [],
                "capped_delays": [],
            }
        groups[key]["nb_active"].append(nb_active)
        groups[key]["nb_common"].append(nb_common)
        groups[key]["na"].append(na)
        groups[key]["detected"].append(1 if fd is not None else 0)
        groups[key]["capped_delays"].append(_capped_delay(fd, detection_horizon))
        if fd is not None:
            groups[key]["fd"].append(fd)

    # Per-stream rows (Parallel preserves input order, so zip with the tasks).
    # Paired across algorithms by (scenario, stream_id) for significance tests.
    per_stream_rows: list[dict] = []
    for (
        (scenario, algo_name, stream_id, stream, _, _),
        (_, _, nb_active, na, fd, nb_common, fd_any),
    ) in zip(stream_tasks, raw_stream_results):
        stream_len = len(stream)
        post_drift_len = stream_len - pre_drift_len
        per_stream_rows.append(
            {
                "scenario": scenario,
                "algorithm": algo_name,
                "stream_id": stream_id,
                "n_false_positives_common": nb_common,
                "false_positives_per_1000_common": (
                    nb_common / common_exposure_samples * 1000
                ),
                "common_exposure_start": common_exposure_start,
                "common_exposure_samples": common_exposure_samples,
                "n_false_positives_active": nb_active,
                "false_positives_per_1000_active": (
                    nb_active / active_pre_drift_len[algo_name] * 1000
                ),
                "warmup_samples": warmup_samples[algo_name],
                # Backward-compatible columns. These cover all pre-drift
                # alarms and are not used for detector comparisons.
                "n_false_positives": nb_active,
                "false_positives_per_1000": nb_active / pre_drift_len * 1000,
                "detected": fd is not None,
                "first_delay": float(fd) if fd is not None else np.nan,
                "capped_delay": _capped_delay(fd, detection_horizon),
                "detection_horizon": detection_horizon,
                "detected_any": na > 0,
                "first_delay_any": float(fd_any) if fd_any is not None else np.nan,
                "n_true_detections": na,
                "flicker_per_1000": max(na - 1, 0) / post_drift_len * 1000,
                "stream_len": stream_len,
            }
        )
    per_stream_results = pd.DataFrame(per_stream_rows)
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    median_rng = np.random.default_rng([bootstrap_seed, 1])
    test_rows: list[dict] = []
    for scenario in test_streams:
        stream_lens = [len(s) for s in test_streams[scenario]]
        post_drift_lens = [length - pre_drift_len for length in stream_lens]
        for algo_name, _, _ in algo_runners:
            key = (scenario, algo_name)
            if key not in groups:
                continue
            g = groups[key]
            nb_active = g["nb_active"]
            nb_common = g["nb_common"]
            na, fd = g["na"], g["fd"]
            detected = g["detected"]
            detection_rate = float(np.mean(detected))
            placebo = placebo_detected[algo_name]
            placebo_rate = float(np.mean(placebo)) if placebo else np.nan
            det_rate_ci_low, det_rate_ci_high = _wilson_ci(sum(detected), len(detected), ci_level)
            placebo_lo, placebo_hi = _wilson_ci(sum(placebo), len(placebo), ci_level)
            test_rows.append({
                "scenario": scenario,
                "algorithm": algo_name,
                **_delay_summary(
                    fd, g["capped_delays"],
                    delay_reportable(len(fd), len(detected), placebo_rate, ci_level),
                    n_bootstrap, ci_level, bootstrap_rng, median_rng,
                ),
                "mean_false_positives_common": (
                    float(np.mean(nb_common)) / common_exposure_samples * 1000
                ),
                "common_exposure_start": common_exposure_start,
                "common_exposure_samples": common_exposure_samples,
                "mean_false_positives_active": (
                    float(np.mean(nb_active)) / active_pre_drift_len[algo_name] * 1000
                ),
                "warmup_samples": warmup_samples[algo_name],
                # Backward-compatible full-pre-drift normalisation.
                "mean_false_positives": (
                    float(np.mean(nb_active)) / pre_drift_len * 1000
                ),
                "mean_true_detections": float(np.mean(na)),
                "mean_total_detections": float(
                    np.mean(
                        [
                            (before + after) / length * 1000
                            for before, after, length in zip(
                                nb_active, na, stream_lens
                            )
                        ]
                    )
                ),
                "detection_rate": detection_rate,
                "detection_rate_ci_low": det_rate_ci_low,
                "detection_rate_ci_high": det_rate_ci_high,
                "detection_rate_ci_method": "wilson",
                "ci_level": ci_level,
                "detection_horizon": detection_horizon,
                "detection_rate_any": float(np.mean([n > 0 for n in na])),
                "placebo_detection_rate": placebo_rate,
                "placebo_detection_rate_ci_low": placebo_lo,
                "placebo_detection_rate_ci_high": placebo_hi,
                "placebo_n_streams": len(placebo),
                "excess_detection_rate": detection_rate - placebo_rate,
                "flicker_rate": float(
                    np.mean(
                        [
                            max(count - 1, 0) / length * 1000
                            for count, length in zip(na, post_drift_lens)
                        ]
                    )
                ),
                "n_streams": len(nb_active),
                "n_streams_detected": len(fd),
                **null_audits[algo_name],
            })

    if verbose:
        current_scenario = None
        for row in test_rows:
            if row["scenario"] != current_scenario:
                current_scenario = row["scenario"]
                md.h3(
                    f"Scenario: {current_scenario} "
                    f"({len(test_streams[current_scenario])} streams)"
                )
            det_rate_ci_str = _fmt_ci(
                row["detection_rate_ci_low"], row["detection_rate_ci_high"], decimals=3
            )
            if row["conditional_delay_reportable"]:
                delay_str = (
                    f"median {row['median_first_delay']:.1f} "
                    + _fmt_ci(row["median_first_delay_ci_low"], row["median_first_delay_ci_high"])
                    + f", mean {_fmt_delay(row['mean_first_delay'], row['std_first_delay'])} "
                    + _fmt_ci(row["mean_first_delay_ci_low"], row["mean_first_delay_ci_high"])
                )
            else:
                delay_str = (
                    f"n/a ({row['n_streams_detected']}/{row['n_streams']} detected, "
                    "not distinguishable from chance)"
                )
            md.bullet(
                f"**{row['algorithm']}** "
                f"- det_rate={row['detection_rate']:.3f} {det_rate_ci_str} (Wilson)"
                f"; conditional_delay={delay_str}"
                f"; capped_delay={row['mean_capped_delay']:.1f}"
                f"; mean_false_positives_common="
                f"{row['mean_false_positives_common']:.5f}"
                f"; placebo={row['placebo_detection_rate']:.2f}"
                f"; flicker={row['flicker_rate']:.2f}"
            )
        if null_streams:
            md.h3("Held-out null audit (Poisson bounds are assumption-dependent)")
            for algo_name, audit in null_audits.items():
                md.bullet(
                    f"**{algo_name}**: {audit['null_n_false_alarms']} alarms / "
                    f"{audit['null_active_samples']} active samples; "
                    f"rate={audit['null_false_alarms_per_1000']:.4f}/1,000 "
                    + _fmt_ci(audit['null_rate_ci_low'], audit['null_rate_ci_high'], 4)
                    + " (whole-stream bootstrap); "
                    f"Poisson UCB={audit['null_poisson_ucb_per_1000']:.4f}; "
                    f"calibration UCB covers held-out rate estimate (proxy)="
                    f"{audit['calibration_ucb_covers_null_rate_proxy']}"
                )
        md.para("Done.")

    return pd.DataFrame(test_rows), per_stream_results


def _run_fresh_detector(
    detector_factory: Callable[[], Any], series: np.ndarray, starting_idx: int
) -> list[int]:
    detector = detector_factory()
    detections: list[int] = []
    for i, x in enumerate(series):
        detector.update(x)
        if detector.drift_detected:
            detections.append(starting_idx + i)
    return detections


def evaluate_fixed_detector(
    detector_factory: Callable[[], Any],
    calibration_streams: list,
    test_streams: dict,
    drift_at: int,
    null_streams: Optional[list] = None,
    starting_idx: int = 0,
    detection_horizon: Optional[int] = DETECTION_HORIZON,
    false_alarm_budget_per_1000: float = DEFAULT_FALSE_ALARM_BUDGET_PER_1000,
    false_alarm_confidence: float = DEFAULT_FALSE_ALARM_CONFIDENCE,
    n_jobs: int = -1,
    n_bootstrap: int = N_BOOTSTRAP,
    ci_level: float = CI_LEVEL,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> list[dict]:
    """Evaluate one fixed Wasserstein-type configuration with the benchmark's
    metric definitions, for diagnostics (ablation, sensitivity) that hold
    hyperparameters fixed instead of calibrating them.

    ``detector_factory()`` returns a fresh detector exposing ``window_size``;
    its warm-up is the Wasserstein detector's. Calibration false alarms are
    normalised by active exposure, with the same one-sided Poisson bound and
    budget flag as calibration, so a fixed operating point that violates the
    benchmark budget is visible. On test streams a drift is detected only by an
    alarm in ``[drift_at, drift_at + detection_horizon)``, flicker counts extra
    post-drift alarms averaged over all streams, and ``null_streams`` give the
    placebo detection rate. Returns one row per scenario of ``test_streams``;
    calibration and placebo statistics are shared by all scenarios.
    """
    _wilson_ci(0, 0, ci_level)
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")
    calibration_streams = _as_list_of_streams(calibration_streams)
    test_streams = {k: _as_list_of_streams(v) for k, v in test_streams.items()}
    null_streams = _as_list_of_streams(null_streams) if null_streams is not None else []
    _check_detection_horizon(test_streams, drift_at, starting_idx, detection_horizon)
    _check_detection_horizon(
        {"null_streams": null_streams}, drift_at, starting_idx, detection_horizon
    )

    probe = detector_factory()
    warmup = _warmup_length(
        "wasserstein",
        {"window_size": probe.window_size, "batch_size": getattr(probe, "batch_size", 1)},
    )
    pre_drift_len = drift_at - starting_idx
    active_pre_drift_len = pre_drift_len - warmup
    if active_pre_drift_len <= 0:
        raise ValueError(
            f"warm-up ({warmup} samples) does not fit within pre_drift_len "
            f"({pre_drift_len})."
        )
    end = np.inf if detection_horizon is None else drift_at + detection_horizon

    tasks = (
        [("calibration", None, s) for s in calibration_streams]
        + [("null", None, s) for s in null_streams]
        + [
            ("test", scenario, s)
            for scenario, streams in test_streams.items()
            for s in streams
        ]
    )
    detections = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_fresh_detector)(detector_factory, stream, starting_idx)
        for _, _, stream in tasks
    )

    calibration = _false_alarm_statistics(
        [d for (kind, _, _), d in zip(tasks, detections) if kind == "calibration"],
        calibration_streams,
        false_alarm_confidence,
        false_alarm_budget_per_1000,
        warmup,
    )
    placebo = [
        any(drift_at <= a < end for a in d)
        for (kind, _, _), d in zip(tasks, detections)
        if kind == "null"
    ]
    placebo_rate = float(np.mean(placebo)) if placebo else np.nan
    placebo_lo, placebo_hi = _wilson_ci(sum(placebo), len(placebo), ci_level)
    audit = _null_alarm_audit(
        [d for (kind, _, _), d in zip(tasks, detections) if kind == "null"],
        null_streams, warmup, starting_idx,
        n_bootstrap=n_bootstrap, ci_level=ci_level, bootstrap_seed=bootstrap_seed,
        false_alarm_confidence=false_alarm_confidence,
        false_alarm_budget_per_1000=false_alarm_budget_per_1000,
        calibration_ucb=calibration["false_alarm_ucb_per_1000"],
    )
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    median_rng = np.random.default_rng([bootstrap_seed, 1])

    rows = []
    for scenario in test_streams:
        fps, detected, delays, flickers, capped = [], [], [], [], []
        for (kind, name, stream), d in zip(tasks, detections):
            if kind != "test" or name != scenario:
                continue
            before = [a for a in d if a < drift_at]
            after = [a for a in d if a >= drift_at]
            in_horizon = [a for a in after if a < end]
            fps.append(len(before) / active_pre_drift_len * 1000)
            detected.append(bool(in_horizon))
            capped.append(_capped_delay(
                in_horizon[0] - drift_at if in_horizon else None, detection_horizon))
            if in_horizon:
                delays.append(in_horizon[0] - drift_at)
            flickers.append(
                max(len(after) - 1, 0) / (len(stream) - pre_drift_len) * 1000
            )
        detection_rate = float(np.mean(detected))
        detection_lo, detection_hi = _wilson_ci(sum(detected), len(detected), ci_level)
        delay_stats = _delay_summary(
            delays, capped,
            delay_reportable(len(delays), len(detected), placebo_rate, ci_level),
            n_bootstrap, ci_level, bootstrap_rng, median_rng,
        )
        rows.append({
            "scenario": scenario,
            "calibration_false_positive_rate": calibration["mean"],
            "calibration_fp_ucb_per_1000": calibration["false_alarm_ucb_per_1000"],
            "calibration_budget_met": calibration["false_alarm_budget_met"],
            "test_false_positive_rate": float(np.mean(fps)),
            "detection_rate": detection_rate,
            "detection_rate_ci_low": detection_lo,
            "detection_rate_ci_high": detection_hi,
            "detection_rate_ci_method": "wilson",
            "ci_level": ci_level,
            **delay_stats,
            "n_streams_detected": len(delays),
            "flicker_rate": float(np.mean(flickers)),
            "placebo_detection_rate": placebo_rate,
            "placebo_detection_rate_ci_low": placebo_lo,
            "placebo_detection_rate_ci_high": placebo_hi,
            "excess_detection_rate": detection_rate - placebo_rate,
            "detection_horizon": detection_horizon,
            "n_streams": len(detected),
            "placebo_n_streams": len(placebo),
            **audit,
        })
    return rows


_RUNNERS: dict[str, Callable] = {
    "wasserstein": _run_wasserstein,
    "adwin": _run_adwin,
    "kswin": _run_kswin,
    "ph": _run_ph,
}


def run_detector(
    algorithm: str, series, params: dict, starting_idx: int = 0
) -> list[int]:
    """Alarm indices of a fresh ``algorithm`` detector with ``params`` on ``series``."""
    if algorithm not in _RUNNERS:
        raise ValueError(f"Unknown algorithm: {algorithm}. Valid choices: {_ALL_ALGORITHMS}")
    return _RUNNERS[algorithm](np.asarray(series), **params, starting_idx=starting_idx)


def wasserstein_score_trajectory(
    series,
    window_size: int,
    detector_cls: type = WassersteinDriftDetector,
    batch_size: int = 1,
) -> np.ndarray:
    """Drift score at every step on which the detector can alarm.

    The score depends on the window size and batch size but not on the
    threshold or the alarm state, so one pass serves every threshold.
    """
    detector = detector_cls(
        window_size=window_size, threshold_drift_score=np.inf, batch_size=batch_size
    )
    scores: list[float] = []
    for x in series:
        detector.update(x)
        if detector.drift_scores and detector.drift_scores[-1][0] == detector.current_index:
            scores.append(detector.drift_scores[-1][1])
    return np.asarray(scores, dtype=float)


def excursion_maxima(scores: np.ndarray) -> np.ndarray:
    """Maximum of every maximal run of positive scores.

    With hysteresis, an armed detector alarms once per run whose maximum exceeds
    the threshold and re-arms when the score returns to zero, so a threshold
    ``theta`` yields ``sum(excursion_maxima(scores) > theta)`` alarms.
    """
    scores = np.asarray(scores, dtype=float)
    positive = np.concatenate(([False], scores > 0, [False]))
    edges = np.flatnonzero(np.diff(positive.astype(int)))
    starts, ends = edges[::2], edges[1::2]
    return np.asarray([scores[a:b].max() for a, b in zip(starts, ends)], dtype=float)


def alarm_levels(scores: np.ndarray, detector_cls: type = WassersteinDriftDetector) -> np.ndarray:
    """Score levels whose exceedance raises an alarm: a threshold ``theta``
    yields ``sum(alarm_levels(scores) > theta)`` alarms.

    A detector that re-arms at zero (``rearms_at_zero``) alarms once per
    positive excursion (:func:`excursion_maxima`); one without hysteresis
    alarms at every positive step above the threshold.
    """
    if getattr(detector_cls, "rearms_at_zero", True):
        return excursion_maxima(scores)
    scores = np.asarray(scores, dtype=float)
    return scores[scores > 0]


def budget_matched_threshold(
    calibration_streams: list,
    window_size: int,
    false_alarm_budget_per_1000: float = DEFAULT_FALSE_ALARM_BUDGET_PER_1000,
    false_alarm_confidence: float = DEFAULT_FALSE_ALARM_CONFIDENCE,
    n_jobs: int = -1,
    backend: str = "loky",
    detector_cls: type = WassersteinDriftDetector,
    batch_size: int = 1,
) -> dict:
    """Lowest Wasserstein threshold meeting the false-alarm budget at a fixed window.

    Applies the calibration selection rule exactly in one dimension: the
    admissible configuration with the most calibration alarms, and among those
    the lowest threshold. Exposure excludes the ``3 * window_size`` warm-up as in
    calibration. When even zero alarms violate the budget, the threshold that
    removes every calibration alarm is returned with ``budget_met=False``.
    ``detector_cls`` may be any Wasserstein-type variant whose drift score does
    not depend on the threshold (see :func:`alarm_levels`).
    """
    calibration_streams = _as_list_of_streams(calibration_streams)
    warmup = _warmup_length(
        "wasserstein", {"window_size": window_size, "batch_size": batch_size}
    )
    exposure = int(sum(len(s) - warmup for s in calibration_streams))
    if exposure <= 0:
        raise ValueError(
            f"Calibration streams must be longer than the warm-up ({warmup} samples)."
        )
    trajectories = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(wasserstein_score_trajectory)(s, window_size, detector_cls, batch_size)
        for s in calibration_streams
    )
    maxima = np.sort(np.concatenate(
        [alarm_levels(t, detector_cls) for t in trajectories]
    ))[::-1]

    def ucb(k: int) -> float:
        return _poisson_rate_upper_bound(k, exposure, false_alarm_confidence)

    budget_met = ucb(0) <= false_alarm_budget_per_1000
    max_alarms = 0
    while budget_met and ucb(max_alarms + 1) <= false_alarm_budget_per_1000:
        max_alarms += 1
    threshold = float(maxima[max_alarms]) if len(maxima) > max_alarms else 0.0
    n_alarms = int(np.sum(maxima > threshold))
    return {
        "window_size": int(window_size),
        "threshold_drift_score": threshold,
        "n_false_alarms": n_alarms,
        "calibration_samples": exposure,
        "mean_detections": n_alarms / exposure * 1000,
        "false_alarm_ucb_per_1000": ucb(n_alarms),
        "false_alarm_budget_met": bool(ucb(n_alarms) <= false_alarm_budget_per_1000),
    }


def calibrate_detectors(
    calibration_streams: list,
    algorithms: Optional[list] = None,
    max_warmup_len: Optional[int] = None,
    starting_idx: int = 0,
    wasserstein_bounds: Optional[dict] = None,
    adwin_bounds: Optional[dict] = None,
    kswin_bounds: Optional[dict] = None,
    ph_bounds: Optional[dict] = None,
    n_trials: int = DEFAULT_N_TRIALS,
    verbose: bool = True,
    optuna_seed: int = DEFAULT_OPTUNA_SEED,
    false_alarm_budget_per_1000: float = DEFAULT_FALSE_ALARM_BUDGET_PER_1000,
    false_alarm_confidence: float = DEFAULT_FALSE_ALARM_CONFIDENCE,
    wasserstein_cls: type = WassersteinDriftDetector,
    wasserstein_window_selection: str = WASSERSTEIN_WINDOW_SELECTION,
    batch_size: int = 1,
    kswin_require_attainable_alpha: bool = False,
) -> tuple[dict, pd.DataFrame]:
    """One calibration round for every algorithm, without range widening.

    Every algorithm is tuned with ``n_trials`` Optuna trials on the drift-free
    ``calibration_streams`` under the common false-alarm budget, except the
    Wasserstein detector under ``wasserstein_window_selection="largest_admissible"``,
    which takes the largest admissible window and solves for its threshold
    (:func:`_calibrate_wasserstein_largest_window`). Window search
    ranges are capped so the detector's warm-up fits within ``max_warmup_len``
    samples (by default the shortest calibration stream), which is the
    pre-drift length when the selection is later evaluated on drift streams.

    ``wasserstein_cls`` replaces the Wasserstein detector by a variant class
    with the same constructor and warm-up (see ``ablation_variants``).
    ``batch_size`` lets every detector alarm only on the last sample of each
    batch of that many samples (see :mod:`wasserstein_score.batch_detectors`).
    ``kswin_require_attainable_alpha`` excludes trials whose alpha is below
    the smallest attainable equal-sample exact KS p-value. It defaults to
    False to preserve the synthetic benchmark's search.

    Returns the selected parameters keyed by algorithm and the trial table
    described under ``calibration_results`` in :func:`calibrate_and_test`.
    """
    active_algos = set(algorithms) if algorithms is not None else set(_ALL_ALGORITHMS)
    unknown = active_algos - set(_ALL_ALGORITHMS)
    if unknown:
        raise ValueError(f"Unknown algorithm(s): {unknown}. Valid choices: {_ALL_ALGORITHMS}")
    _check_window_selection(wasserstein_window_selection)
    calibration_streams = _as_list_of_streams(calibration_streams)
    if max_warmup_len is None:
        max_warmup_len = min(len(stream) for stream in calibration_streams)

    if verbose:
        md.h2("Calibration")
        md.bullet(f"Active algorithms: {sorted(active_algos)}")

    wass_b = dict(wasserstein_bounds or DEFAULT_WASSERSTEIN_BOUNDS)
    adwin_b = dict(adwin_bounds or DEFAULT_ADWIN_BOUNDS)
    kswin_b = dict(kswin_bounds or DEFAULT_KSWIN_BOUNDS)
    ph_b = dict(ph_bounds or DEFAULT_PH_BOUNDS)

    def _cap_window_bound(bounds: dict, warmup_multiplier: int, name: str, key: str = "window_size") -> None:
        """Cap bounds[key] upper end so the detector's warm-up
        (``warmup_multiplier * bounds[key]``) fits within ``max_warmup_len``, and warn
        when that throttles the requested search space."""
        requested_hi = bounds[key][1]
        capped_hi = min(requested_hi, _max_window_for_warmup(max_warmup_len, warmup_multiplier))
        if capped_hi < bounds[key][0]:
            raise ValueError(
                f"max_warmup_len={max_warmup_len} is too short for the {name} window search space: "
                f"warm-up needs ~{warmup_multiplier}x {key} samples, giving an "
                f"upper bound of {capped_hi}, which is below the lower bound {bounds[key][0]}. "
                f"Provide a smaller {key} lower bound or longer streams."
            )
        bounds[key] = (bounds[key][0], capped_hi)
        if verbose and capped_hi < requested_hi:
            md.bullet(
                f"INFO: {name} {key} upper bound throttled from {requested_hi} to {capped_hi}. Search space is {bounds[key]}."
            )

    if "wasserstein" in active_algos:
        _cap_window_bound(wass_b, _WASSERSTEIN_WARMUP_MULTIPLIER, "Wasserstein")

    if "kswin" in active_algos:
        _cap_window_bound(kswin_b, _KSWIN_WARMUP_MULTIPLIER, "KSWIN")

    if verbose:
        md.bullet(
            f"{len(calibration_streams)} calibration streams, "
            f"{n_trials} trials/algorithm, Optuna seed={optuna_seed}, "
            f"false-alarm budget={false_alarm_budget_per_1000:g}/1,000 samples, "
            f"nominal one-sided Poisson confidence={false_alarm_confidence:.1%} "
            "(working model; no post-selection guarantee)"
        )

    best_params: dict = {}
    calib_dfs: list[pd.DataFrame] = []

    _calib_registry: list[tuple[Callable, dict, str]] = [
        (fn, bounds, name)
        for fn, bounds, name in [
            (functools.partial(
                _calibrate_wasserstein_largest_window
                if wasserstein_window_selection == "largest_admissible"
                else _calibrate_wasserstein,
                detector_cls=wasserstein_cls,
            ), wass_b, "wasserstein"),
            (_calibrate_adwin,       adwin_b, "adwin"),
            (functools.partial(_calibrate_kswin,
                               require_attainable_alpha=kswin_require_attainable_alpha),
             kswin_b, "kswin"),
            (_calibrate_ph,          ph_b,    "ph"),
        ]
        if name in active_algos
    ]
    n_calib_workers = min(len(_calib_registry), _available_cpu_count())
    # When only 1 algorithm runs (main process), use loky processes for stream-level
    # parallelism inside each Optuna trial. When multiple algorithms each occupy a loky
    # worker, use threading to avoid nested process spawning (unsafe on Windows).
    calib_stream_backend = "loky" if n_calib_workers == 1 else "threading"
    n_stream_workers = _available_cpu_count()

    if verbose:
        md.bullet(
            f"Running calibration for {len(_calib_registry)} algorithms "
            f"in parallel ({n_calib_workers} workers), "
            f"{len(calibration_streams)} streams/trial "
            f"({calib_stream_backend}, up to {n_stream_workers} workers) ..."
        )

    calib_results_parallel = Parallel(n_jobs=n_calib_workers, backend="loky")(
        delayed(fn)(
            calibration_streams, bounds, starting_idx, n_trials,
            verbose=False,  # verbose=False in workers; parent prints summary below
            optuna_seed=optuna_seed,
            stream_backend=calib_stream_backend,
            false_alarm_budget_per_1000=false_alarm_budget_per_1000,
            false_alarm_confidence=false_alarm_confidence,
            batch_size=batch_size,
        )
        for fn, bounds, _ in _calib_registry
    )

    for (_, _, name), (bp, best_row, df) in zip(_calib_registry, calib_results_parallel):
        best_params[name] = bp
        df["selected"] = False
        df.loc[best_row.name, "selected"] = True
        calib_dfs.append(df)
        if verbose:
            md.bullet(
                f"**{name}** - best params: {bp}"
                f"; mean={best_row['mean_detections']:.3f}"
                f", std={best_row['std_detections']:.3f}"
                f", UCB={best_row['false_alarm_ucb_per_1000']:.3f}"
                f", budget met={bool(best_row['false_alarm_budget_met'])}"
            )
            if best_row["params_at_bound"]:
                md.bullet(
                    f"WARNING: {name} selection is at its search bound "
                    f"({best_row['params_at_bound']}); widen the range."
                )

    return best_params, pd.concat(calib_dfs, ignore_index=True)


# Per-algorithm selection settings, as used by the ``_calibrate_*`` functions:
# parameter columns and the structural-aggressiveness tie-break.
_SELECTION_SETTINGS: dict[str, tuple[list[str], list[tuple[str, bool]]]] = {
    "wasserstein": (
        ["threshold_drift_score", "window_size"],
        [("threshold_drift_score", False), ("window_size", False)],
    ),
    "adwin": (["delta"], [("delta", True)]),
    "kswin": (["window_size", "alpha", "stat_size"], [("alpha", True)]),
    "ph": (["threshold", "delta"], [("threshold", False), ("delta", False)]),
}

# Valid domain of each searched parameter; widening never leaves it.
PARAMETER_DOMAINS: dict[str, dict[str, tuple[float, float]]] = {
    "wasserstein": {"threshold_drift_score": (0.0, 64.0), "window_size": (50, MAX_WINDOW_SIZE)},
    "adwin": {"delta": (1e-15, 0.999)},
    "kswin": {"window_size": (50, MAX_WINDOW_SIZE), "alpha": (1e-15, 0.5), "stat_size": (5, np.inf)},
    "ph": {"threshold": (1e-3, 1e6), "delta": (1e-12, 1e3)},
}

# The end of each range that lowers the false-alarm rate, widened when a
# selection cannot meet the budget.
CONSERVATIVE_ENDS: dict[str, dict[str, str]] = {
    "wasserstein": {"threshold_drift_score": "high"},
    "adwin": {"delta": "low"},
    "kswin": {"alpha": "low"},
    "ph": {"threshold": "high"},
}

# The opposite ends, which raise the false-alarm rate: widened when a selection
# meets the budget without a single calibration alarm, i.e. when the budget
# does not constrain the search at all.
AGGRESSIVE_ENDS: dict[str, dict[str, str]] = {
    "wasserstein": {"threshold_drift_score": "low"},
    "adwin": {"delta": "high"},
    "kswin": {"alpha": "high"},
    "ph": {"threshold": "low", "delta": "low"},
}

_INTEGER_PARAMS = ("window_size", "stat_size")

_WARMUP_MULTIPLIERS = {
    "wasserstein": _WASSERSTEIN_WARMUP_MULTIPLIER,
    "kswin": _KSWIN_WARMUP_MULTIPLIER,
}


def widened_bounds(
    algorithm: str, bounds: dict, selected: pd.Series, max_warmup_len: int
) -> dict:
    """Search bounds for the next calibration round of ``algorithm``.

    Every parameter flagged at a search bound has that end extended by the
    current span (in log space for log-sampled parameters). When the selection
    misses the false-alarm budget, the detector's conservative ends are
    extended too; when it meets the budget with no calibration alarm, so the
    budget does not bind, its aggressive ends are. Extensions stop at :data:`PARAMETER_DOMAINS` and, for window
    sizes, at the largest window whose warm-up fits within ``max_warmup_len``.
    An unchanged result means the range cannot usefully be widened.
    """
    ends = set()
    for flag in filter(None, str(selected.get("params_at_bound", "")).split(",")):
        param, side = flag.split("=")
        ends.add((param, side))
    if not bool(selected["false_alarm_budget_met"]):
        ends |= set(CONSERVATIVE_ENDS[algorithm].items())
    elif selected.get("mean_detections", np.nan) == 0:
        ends |= set(AGGRESSIVE_ENDS[algorithm].items())

    new = dict(bounds)
    for param, side in sorted(ends):
        low, high = new[param]
        dom_low, dom_high = PARAMETER_DOMAINS[algorithm][param]
        if param == "window_size":
            dom_high = min(dom_high, _max_window_for_warmup(
                max_warmup_len, _WARMUP_MULTIPLIERS[algorithm]
            ))
        if param in _LOG_PARAMS[algorithm]:
            factor = high / low
            target = low / factor if side == "low" else high * factor
        else:
            target = low - (high - low) if side == "low" else high + (high - low)
        # Each end only moves outward, and not past the parameter's domain.
        if side == "low":
            low = min(low, max(target, dom_low))
        else:
            high = max(high, min(target, dom_high))
        if isinstance(bounds[param][0], (int, np.integer)):
            low, high = int(np.ceil(low)), int(np.floor(high))
        new[param] = (low, high)
    return new


def calibrate_detectors_with_widening(
    calibration_streams: list,
    algorithms: Optional[list] = None,
    max_warmup_len: Optional[int] = None,
    max_rounds: int = CALIBRATION_WIDENING_ROUNDS,
    verbose: bool = True,
    window_size_limits: tuple[int, int] | None = None,
    **kwargs,
) -> tuple[dict, pd.DataFrame, dict]:
    """:func:`calibrate_detectors`, widening search ranges that bind.

    After each round, an algorithm whose selection misses the budget or sits at
    a search bound is recalibrated with :func:`widened_bounds`, for at most
    ``max_rounds`` further rounds. The final selection applies the calibration
    rule to the trials of all rounds, so widening never discards an admissible
    configuration found earlier. A Wasserstein detector calibrated by the
    largest-admissible-window rule is never widened: its window sits at the
    upper bound by design and its threshold is solved over its whole domain.
    Returns the selected parameters, the trial
    table (with a ``round`` column) and the final bounds per algorithm. The
    selected parameters include the ``batch_size`` they were calibrated at.
    ``window_size_limits`` optionally keeps the window search and all widening
    rounds within a case-specific inclusive range (for example 21--42 days).
    """
    window_selection = kwargs.get("wasserstein_window_selection", WASSERSTEIN_WINDOW_SELECTION)
    _check_window_selection(window_selection)
    calibration_streams = _as_list_of_streams(calibration_streams)
    batch_size = int(kwargs.get("batch_size", 1))
    active = list(algorithms) if algorithms is not None else list(_ALL_ALGORITHMS)
    if max_warmup_len is None:
        max_warmup_len = min(len(stream) for stream in calibration_streams)
    budget = kwargs.get("false_alarm_budget_per_1000", DEFAULT_FALSE_ALARM_BUDGET_PER_1000)
    defaults = {
        "wasserstein": DEFAULT_WASSERSTEIN_BOUNDS, "adwin": DEFAULT_ADWIN_BOUNDS,
        "kswin": DEFAULT_KSWIN_BOUNDS, "ph": DEFAULT_PH_BOUNDS,
    }
    bounds = {algo: dict(kwargs.pop(f"{algo}_bounds", None) or defaults[algo]) for algo in active}
    if window_size_limits is not None:
        lo, hi = window_size_limits
        if not (1 <= lo <= hi) or int(lo) != lo or int(hi) != hi:
            raise ValueError("window_size_limits must be positive ordered integers")
        for algo in ("wasserstein", "kswin"):
            if algo in bounds:
                low, high = bounds[algo]["window_size"]
                if low < lo or high > hi:
                    raise ValueError(f"{algo} initial window bounds exceed window_size_limits")
    for algo, key in (("wasserstein", "window_size"), ("kswin", "window_size")):
        if algo in bounds:
            cap = _max_window_for_warmup(max_warmup_len, _WARMUP_MULTIPLIERS[algo])
            bounds[algo][key] = (bounds[algo][key][0], min(bounds[algo][key][1], cap))

    trials: dict[str, list[pd.DataFrame]] = {algo: [] for algo in active}
    pending = list(active)
    best_params: dict = {}
    selected: dict[str, pd.Series] = {}
    for round_index in range(max_rounds + 1):
        if verbose and round_index:
            md.bullet(
                f"Widening round {round_index}: "
                + "; ".join(f"{algo} {bounds[algo]}" for algo in pending)
            )
        _, table = calibrate_detectors(
            calibration_streams, algorithms=pending, max_warmup_len=max_warmup_len,
            verbose=verbose and not round_index,
            **{f"{algo}_bounds": bounds[algo] for algo in pending}, **kwargs,
        )
        for algo in pending:
            df = table[table["algorithm"] == algo]
            selected_mask = df["selected"].to_numpy(dtype=bool)
            df = df.drop(columns=["selected"])
            # The joint table pads other algorithms' parameters with NaN.
            df = df.dropna(axis=1, how="all").copy()
            for col in _INTEGER_PARAMS:
                if col in df.columns:
                    df[col] = df[col].astype(int)
            df["round"] = round_index
            if algo == "wasserstein" and window_selection == "largest_admissible":
                # A rule, not a search: keep its choice and do not widen.
                df = df.reset_index(drop=True)
                selected[algo] = df.iloc[int(np.flatnonzero(selected_mask)[0])]
                best_params[algo] = {
                    "threshold_drift_score": float(selected[algo]["threshold_drift_score"]),
                    "window_size": int(selected[algo]["window_size"]),
                }
                trials[algo] = [df]
                continue
            trials[algo].append(df)
            union = pd.concat(trials[algo], ignore_index=True)
            param_cols, aggressiveness = _SELECTION_SETTINGS[algo]
            best_params[algo], selected[algo] = _select_best(
                union, param_cols, False, algo,
                aggressiveness_sort=aggressiveness,
                false_alarm_budget_per_1000=budget,
                bounds=bounds[algo], log_params=_LOG_PARAMS[algo],
            )
            trials[algo] = [union]
        # Do not report a wider, unevaluated search range after the last round.
        if round_index == max_rounds:
            break
        next_pending = []
        for algo in pending:
            if algo == "wasserstein" and window_selection == "largest_admissible":
                continue
            new = widened_bounds(algo, bounds[algo], selected[algo], max_warmup_len)
            if window_size_limits is not None and "window_size" in new:
                low, high = new["window_size"]
                new["window_size"] = (max(low, lo), min(high, hi))
            if new != bounds[algo]:
                bounds[algo] = new
                next_pending.append(algo)
        if not next_pending:
            break
        pending = next_pending

    tables = []
    for algo in active:
        best_params[algo] = {**best_params[algo], "batch_size": batch_size}
        df = trials[algo][0].copy()
        df["selected"] = False
        df.loc[selected[algo].name, "selected"] = True
        df["batch_size"] = batch_size
        tables.append(df)
        if verbose:
            row = selected[algo]
            md.bullet(
                f"**{algo}** - selected after widening: {best_params[algo]}"
                f"; mean={row['mean_detections']:.3f}, UCB={row['false_alarm_ucb_per_1000']:.3f}"
                f", budget met={bool(row['false_alarm_budget_met'])}"
                f", final bounds={bounds[algo]}"
                + (f", at bound: {row['params_at_bound']}" if row["params_at_bound"] else "")
            )
    return best_params, pd.concat(tables, ignore_index=True), bounds


def calibrate_and_test(
    calibration_streams: list,
    test_streams: dict,
    drift_at: int,
    starting_idx: int = 0,
    wasserstein_bounds: Optional[dict] = None,
    adwin_bounds: Optional[dict] = None,
    kswin_bounds: Optional[dict] = None,
    ph_bounds: Optional[dict] = None,
    n_trials: int = DEFAULT_N_TRIALS,
    verbose: bool = True,
    algorithms: Optional[list] = None,
    n_bootstrap: int = N_BOOTSTRAP,
    ci_level: float = CI_LEVEL,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    optuna_seed: int = DEFAULT_OPTUNA_SEED,
    false_alarm_budget_per_1000: float = DEFAULT_FALSE_ALARM_BUDGET_PER_1000,
    false_alarm_confidence: float = DEFAULT_FALSE_ALARM_CONFIDENCE,
    detection_horizon: Optional[int] = DETECTION_HORIZON,
    null_streams: Optional[list] = None,
    widening_rounds: int = CALIBRATION_WIDENING_ROUNDS,
    wasserstein_window_selection: str = WASSERSTEIN_WINDOW_SELECTION,
    batch_size: int = 1,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Bayesian hyperparameter optimisation on calibration data, then evaluate on test data.

    Parameters
    ----------
    calibration_streams : list of array-like
        Streams without drift. Each element is a 1-D numeric sequence.
    test_streams : dict[str, list of array-like]
        Mapping from scenario name to test streams, each containing a single
        drift event at ``drift_at``.
    drift_at : int
        Absolute index of the first post-drift sample. Detections strictly
        before this index are false positives; at or after are true positives.
    starting_idx : int
        Absolute index of the first element of each stream. Default 0.
    wasserstein_bounds, adwin_bounds, kswin_bounds, ph_bounds : dict, optional
        Search bounds for each algorithm. See DEFAULT_*_BOUNDS for keys.
    n_trials : int
        Number of Optuna trials per algorithm.
    verbose : bool
        Print progress and results. Default True.
    algorithms : list of str, optional
        Subset of algorithms to run. Valid values are the entries of
        ``_ALL_ALGORITHMS``: ``"wasserstein"``, ``"adwin"``, ``"kswin"``,
        ``"ph"``. Defaults to all.
    n_bootstrap : int
        Number of percentile-bootstrap resamples used for the CIs on
        conditional/capped delay means and the held-out null rate. Default 2000.
    ci_level : float
        Confidence level for bootstrap and Wilson intervals, e.g. 0.95.
    bootstrap_seed : int
        Seed for the bootstrap RNG, for reproducible interval widths.
    optuna_seed : int
        Seed for each algorithm's Optuna TPE sampler. Default 42.
    false_alarm_budget_per_1000 : float
        Common upper budget for false alarms per 1,000 no-drift samples on
        which the detector is able to alarm (its warm-up is excluded).
        Admissibility is based on the one-sided confidence bound, not only on
        the observed mean. Defaults to the benchmark protocol value.
    false_alarm_confidence : float
        Nominal confidence level of the one-sided Poisson working-model bound.
        Not a post-selection guarantee. Must be between 0.5 and 1. Default 0.95.
    detection_horizon : int or None
        A test stream is detected only if an alarm falls within this many
        samples of ``drift_at``; ``None`` accepts any post-drift alarm.
        Defaults to the benchmark protocol value.
    null_streams : list of array-like, optional
        Held-out drift-free streams, disjoint from ``calibration_streams``, on
        which the placebo detection rate is measured (see
        :func:`evaluate_with_params`).
    widening_rounds : int
        Maximum extra calibration rounds over widened search ranges (see
        :func:`calibrate_detectors_with_widening`); 0 keeps the given bounds.
    wasserstein_window_selection : str
        ``"largest_admissible"`` (the largest window in the search range with a
        budget-meeting threshold, and its lowest such threshold, solved
        exactly; ``n_trials`` and ``optuna_seed`` do not affect it) or
        ``"tpe"`` (joint Optuna search). Defaults to the benchmark protocol value.
    batch_size : int
        Samples arrive in batches of this size and every detector may alarm
        only on the last sample of a batch, both in calibration and in test.
        The drift must start, and the detection horizon end, on a batch
        boundary. Default 1 (sample-by-sample checking).

    Returns
    -------
    best_params : dict
        Selected hyperparameters keyed by algorithm name
        (``"wasserstein"``, ``"adwin"``, ``"kswin"``, ``"ph"``).
    calibration_results : pd.DataFrame
        All Optuna trials for all algorithms with ``mean_detections``,
        ``std_detections``, ``n_false_alarms``, ``calibration_samples`` (active
        exposure), ``warmup_samples``, ``params_at_bound``,
        ``false_alarm_ucb_per_1000``, ``false_alarm_confidence``,
        ``false_alarm_budget_per_1000``, ``false_alarm_budget_met``,
        ``selection_objective``, ``algorithm``, ``optuna_seed`` and ``selected``
        columns. Exactly one trial per algorithm is selected.
    test_results : pd.DataFrame
        Per (scenario × algorithm): ``mean_total_detections``,
        ``mean_false_positives_common`` (the primary false-positive rate,
        measured on the same post-maximum-warm-up pre-drift window for every
        detector), ``mean_false_positives_active`` (supplementary rate over
        each detector's own post-warm-up pre-drift samples),
        ``mean_false_positives`` (legacy full-pre-drift normalisation),
        ``common_exposure_start``/``common_exposure_samples``,
        ``warmup_samples`` (the subtracted warm-up length for that algorithm),
        ``mean_true_detections``, ``detection_rate``,
        ``mean_first_delay``, ``std_first_delay`` (both over alarms within the
        detection horizon), ``detection_horizon``, ``detection_rate_any``
        (any post-drift alarm), ``placebo_detection_rate``,
        ``placebo_n_streams``, ``excess_detection_rate``,
        ``flicker_rate`` (mean extra signals after first post-drift detection),
        ``n_streams``, ``n_streams_detected``,
        ``mean_first_delay_ci_low``/``_ci_high``,
        ``detection_rate_ci_low``/``_ci_high`` and placebo counterparts (Wilson
        intervals; NaN only for zero contributing streams), ``mean_capped_delay``
        and its bootstrap CI (misses receive the finite horizon), plus ``null_*``
        audit fields and ``calibration_ucb_*`` coverage proxies. Delay bootstrap
        CIs require at least two contributing streams. Null bootstrap CIs also
        require at least one alarm. See :func:`evaluate_with_params`.
    per_stream_results : pd.DataFrame
        One row per (scenario x algorithm x stream): ``stream_id``,
        ``n_false_positives_common``/``false_positives_per_1000_common``
        (primary common-window metrics), ``n_false_positives_active``/
        ``false_positives_per_1000_active`` (supplementary detector-specific
        exposure metrics), ``n_false_positives`` and
        ``false_positives_per_1000`` (legacy full-pre-drift metrics),
        ``common_exposure_start``/``common_exposure_samples``,
        ``warmup_samples``, ``detected`` (alarm within the horizon),
        ``first_delay`` (NaN when undetected), ``capped_delay`` (finite horizon
        assigned to misses; NaN for unbounded horizon), ``detection_horizon``, ``detected_any``,
        ``first_delay_any``, ``n_true_detections``,
        ``flicker_per_1000``, ``stream_len``. All algorithms see the same
        streams, so rows are paired by (scenario, stream_id) — this is the
        input for ``wasserstein_score.significance``.
    """
    active_algos = set(algorithms) if algorithms is not None else set(_ALL_ALGORITHMS)
    unknown = active_algos - set(_ALL_ALGORITHMS)
    if unknown:
        raise ValueError(f"Unknown algorithm(s): {unknown}. Valid choices: {_ALL_ALGORITHMS}")
    if (
        not np.isfinite(false_alarm_budget_per_1000)
        or false_alarm_budget_per_1000 < 0
    ):
        raise ValueError("false_alarm_budget_per_1000 must be finite and non-negative")
    if not 0.5 < false_alarm_confidence < 1.0:
        raise ValueError("false_alarm_confidence must be between 0.5 and 1")

    _check_batch_alignment(batch_size, drift_at, starting_idx, detection_horizon)

    calibration_streams = _as_list_of_streams(calibration_streams)
    test_streams = {k: _as_list_of_streams(v) for k, v in test_streams.items()}
    # Fail before calibration rather than after it.
    _check_detection_horizon(test_streams, drift_at, starting_idx, detection_horizon)
    if null_streams is not None:
        null_streams = _as_list_of_streams(null_streams)
        _check_detection_horizon(
            {"null_streams": null_streams}, drift_at, starting_idx, detection_horizon
        )

    best_params, calibration_results, _ = calibrate_detectors_with_widening(
        calibration_streams,
        algorithms=sorted(active_algos),
        max_warmup_len=drift_at - starting_idx,
        max_rounds=widening_rounds,
        starting_idx=starting_idx,
        wasserstein_bounds=wasserstein_bounds,
        adwin_bounds=adwin_bounds,
        kswin_bounds=kswin_bounds,
        ph_bounds=ph_bounds,
        n_trials=n_trials,
        verbose=verbose,
        optuna_seed=optuna_seed,
        false_alarm_budget_per_1000=false_alarm_budget_per_1000,
        false_alarm_confidence=false_alarm_confidence,
        wasserstein_window_selection=wasserstein_window_selection,
        batch_size=batch_size,
    )

    test_results, per_stream_results = evaluate_with_params(
        test_streams=test_streams,
        best_params=best_params,
        drift_at=drift_at,
        starting_idx=starting_idx,
        algorithms=algorithms,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        bootstrap_seed=bootstrap_seed,
        verbose=verbose,
        detection_horizon=detection_horizon,
        null_streams=null_streams,
        calibration_results=calibration_results,
        false_alarm_budget_per_1000=false_alarm_budget_per_1000,
        false_alarm_confidence=false_alarm_confidence,
    )
    return best_params, calibration_results, test_results, per_stream_results


def pivot_test_results(test_results: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot test_results into a scenario × algorithm table for a single metric."""
    return test_results.pivot(index="scenario", columns="algorithm", values=metric)


