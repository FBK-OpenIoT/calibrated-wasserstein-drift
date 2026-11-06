"""
Sensitivity analysis for WassersteinDriftDetector around its calibrated
operating point.

Each dataset's operating point (W*, theta*) is the main benchmark's
Wasserstein calibration on the same streams. Factors are then perturbed one
at a time (OAT), plus a small 2-D grid to check for interaction between the
two hyperparameters:

  window_size            - swept under the false-alarm budget: at every window
                            the threshold is re-chosen as the lowest one that
                            meets the budget on the calibration streams
                            (``evaluation.budget_matched_threshold``), so the
                            sweep shows what the window changes at a fixed
                            alarm constraint.
  threshold_drift_score  - swept at W* across the calibration search space,
                            including theta*, tracing the delay / false-alarm
                            trade-off through the operating point.
  noise                  - label noise in the underlying data-generating
                            process (a data-side factor, not a detector
                            hyperparameter); at every noise level the threshold
                            at W* is re-matched to the budget on calibration
                            streams with that noise, as a deployment would.

For each swept value we report calibration false-positive rate (with the
calibration budget's upper bound and whether the value meets it), test
detection rate, the placebo detection rate on held-out drift-free streams,
median detection delay and flicker rate, then rank the factors by how much they
move detection and delay across the budget-meeting part of their sweep
(``sensitivity_ranges``), with the data-side noise factor labelled as such.
Under the benchmark's largest-admissible window rule W* is the top of the
window grid. Datasets, stream geometry and metric
definitions are those of the main benchmark
(``evaluation.evaluate_fixed_detector``), so a permissive threshold's
detection rate can be read against its chance level.
"""

import argparse
import functools
import os
from typing import Any

import pandas as pd
import plotly.express as px

from wasserstein_score.normalization import CalibrationMedianNormalizer
from wasserstein_score.wasserstein_drift_detector import WassersteinDriftDetector
from wasserstein_score.benchmark_protocol import (
    DRIFT_AT,
    N_OPTUNA_TRIALS,
    NUM_MODEL_TEST_POINTS,
    NUM_TRAINING_POINTS,
    OPTUNA_SEED,
    RESCALE_BY_CALIBRATION_MEDIAN,
    STARTING_IDX,
)
from wasserstein_score.evaluation import (
    budget_matched_threshold,
    calibrate_detectors_with_widening,
    evaluate_fixed_detector,
)
from wasserstein_score import helper_functions as md
from wasserstein_score.helper_functions import (
    get_model, build_model_df, benchmark_datasets, held_out_null_streams,
)

pd.options.plotting.backend = "plotly"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Both grids span the calibration search space (WASSERSTEIN_BOUNDS); each
# dataset's calibrated value is added to its sweep.
WINDOW_SIZE_GRID = [500, 600, 750, 900, 1000]
THRESHOLD_GRID = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
NOISE_GRID = [0.05, 0.1, 0.2, 0.4]

GRID_WINDOW_SIZES = [500, 750, 1000]
GRID_THRESHOLDS = [0.0, 0.5, 1.5, 3.0]

OUTPUT_DIR = "data"


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_wasserstein(window_size, threshold_drift_score, calib_streams, test_streams,
                          null_streams) -> list[dict]:
    factory = functools.partial(
        WassersteinDriftDetector,
        window_size=window_size,
        threshold_drift_score=threshold_drift_score,
    )
    return evaluate_fixed_detector(
        factory, calib_streams, test_streams,
        drift_at=DRIFT_AT, null_streams=null_streams, starting_idx=STARTING_IDX,
    )


def calibrate_operating_point(calib_streams) -> tuple[int, float]:
    """The main benchmark's Wasserstein selection (W*, theta*) on these streams."""
    params, _, _ = calibrate_detectors_with_widening(
        calib_streams, algorithms=["wasserstein"], max_warmup_len=DRIFT_AT - STARTING_IDX,
        starting_idx=STARTING_IDX, n_trials=N_OPTUNA_TRIALS, optuna_seed=OPTUNA_SEED,
        verbose=False,
    )
    return int(params["wasserstein"]["window_size"]), float(params["wasserstein"]["threshold_drift_score"])


def _log_row(scenario: str, label: str, metrics: dict) -> None:
    md.bullet(f"[{scenario}] {label} — det_rate={metrics['detection_rate']:.2f},"
              f" placebo={metrics['placebo_detection_rate']:.2f},"
              f" median_delay={metrics['median_first_delay']:.1f},"
              f" capped_delay={metrics['mean_capped_delay']:.1f},"
              f" calib_fp={metrics['calibration_false_positive_rate']:.3f}"
              f" (budget met={metrics['calibration_budget_met']})")


def _residuals(model, X_streams, y_streams) -> list:
    return [
        (model.predict(X_streams[i]) - y_streams[i]) ** 2
        for i in range(len(X_streams))
    ]


def build_residual_streams(generator, generator_kwargs,
                           rescale: bool = RESCALE_BY_CALIBRATION_MEDIAN):
    """Calibration, all-scenario test and held-out null residual streams.

    ``generator_kwargs`` are complete generator arguments (a benchmark
    dataset's, possibly with one factor changed); as in the benchmark, as many
    null streams as test streams are drawn.
    """
    X_training, y_training, X_calib, y_calib, X_test, y_test = generator(**generator_kwargs)
    X_null, y_null = held_out_null_streams(
        generator, generator_kwargs["n_test_streams"], **generator_kwargs
    )
    n_features = X_training.shape[1]

    model = get_model(
        build_model_df(X_training, y_training),
        NUM_TRAINING_POINTS, NUM_MODEL_TEST_POINTS, range(n_features), 'target',
    )

    calibration = _residuals(model, X_calib, y_calib)
    normalizer = CalibrationMedianNormalizer.fit(calibration, enabled=rescale)
    return (
        normalizer.transform(calibration),
        {
            scenario_name: normalizer.transform(_residuals(model, X_streams, y_test[scenario_name]))
            for scenario_name, X_streams in X_test.items()
        },
        normalizer.transform(_residuals(model, X_null, y_null)),
    )


# ---------------------------------------------------------------------------
# One-at-a-time sweeps
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict[str, Any]] = {
    label: {"generator": generator, "kwargs": kwargs}
    for label, generator, kwargs in benchmark_datasets()
}
NOISE_DATASET = "Linear Abrupt"


OPERATING_POINTS: dict[str, tuple[int, float]] = {}


def sweep_window_size_and_threshold(rescale: bool = RESCALE_BY_CALIBRATION_MEDIAN) -> pd.DataFrame:
    rows = []
    for dataset_name, config in DATASETS.items():
        md.h2(f"{dataset_name}: window_size / threshold sweeps")
        streams = build_residual_streams(config["generator"], config["kwargs"], rescale=rescale)
        calibration = streams[0]

        w_star, theta_star = calibrate_operating_point(calibration)
        OPERATING_POINTS[dataset_name] = (w_star, theta_star)
        md.bullet(f"Calibrated operating point: window_size={w_star}, threshold={theta_star:.4f}")

        for window_size in sorted(set(WINDOW_SIZE_GRID) | {w_star}):
            matched = budget_matched_threshold(calibration, window_size)
            threshold = matched["threshold_drift_score"]
            for metrics in evaluate_wasserstein(window_size, threshold, *streams):
                _log_row(metrics["scenario"],
                         f"window_size={window_size} (budget-matched threshold={threshold:.4f})",
                         metrics)
                rows.append({"dataset": dataset_name, "parameter": "window_size", "value": window_size,
                             "window_size": window_size, "threshold_drift_score": threshold,
                             "calibrated_value": window_size == w_star, **metrics})

        for threshold in sorted(set(THRESHOLD_GRID) | {theta_star}):
            for metrics in evaluate_wasserstein(w_star, threshold, *streams):
                _log_row(metrics["scenario"], f"threshold={threshold:.4f} (window_size={w_star})",
                         metrics)
                rows.append({"dataset": dataset_name, "parameter": "threshold_drift_score",
                             "value": threshold, "window_size": w_star,
                             "threshold_drift_score": threshold,
                             "calibrated_value": threshold == theta_star, **metrics})

    return pd.DataFrame(rows)


def sweep_noise(rescale: bool = RESCALE_BY_CALIBRATION_MEDIAN) -> pd.DataFrame:
    md.h2(f"{NOISE_DATASET}: noise sweep")
    config = DATASETS[NOISE_DATASET]
    if NOISE_DATASET not in OPERATING_POINTS:
        streams = build_residual_streams(config["generator"], config["kwargs"], rescale=rescale)
        OPERATING_POINTS[NOISE_DATASET] = calibrate_operating_point(streams[0])
    w_star, _ = OPERATING_POINTS[NOISE_DATASET]
    rows = []
    for noise in NOISE_GRID:
        streams = build_residual_streams(
            config["generator"], dict(config["kwargs"], noise=noise), rescale=rescale,
        )
        threshold = budget_matched_threshold(streams[0], w_star)["threshold_drift_score"]
        for metrics in evaluate_wasserstein(w_star, threshold, *streams):
            _log_row(metrics["scenario"],
                     f"noise={noise} (window_size={w_star}, budget-matched threshold={threshold:.4f})",
                     metrics)
            rows.append({"dataset": NOISE_DATASET, "parameter": "noise", "value": noise,
                         "window_size": w_star, "threshold_drift_score": threshold,
                         "calibrated_value": noise == config["kwargs"]["noise"], **metrics})

    return pd.DataFrame(rows)


def sweep_grid(rescale: bool = RESCALE_BY_CALIBRATION_MEDIAN) -> pd.DataFrame:
    md.h2(f"{NOISE_DATASET}: window_size x threshold grid")
    config = DATASETS[NOISE_DATASET]
    streams = build_residual_streams(config["generator"], config["kwargs"], rescale=rescale)

    rows = []
    for window_size in GRID_WINDOW_SIZES:
        for threshold in GRID_THRESHOLDS:
            for metrics in evaluate_wasserstein(
                window_size, threshold, *streams
            ):
                rows.append({"dataset": NOISE_DATASET, "window_size": window_size,
                             "threshold_drift_score": threshold, **metrics})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

RANGE_METRICS = {
    "detection_rate_range": "detection_rate",
    "placebo_detection_rate_range": "placebo_detection_rate",
    "median_first_delay_range": "median_first_delay",
    "calibration_fp_range": "calibration_false_positive_rate",
}
FACTOR_TYPES = {
    "window_size": "detector hyperparameter",
    "threshold_drift_score": "detector hyperparameter",
    "noise": "data condition",
}


def sensitivity_ranges(oat_df: pd.DataFrame) -> pd.DataFrame:
    """Range (max - min) of each metric across each factor's sweep, computed
    per scenario, averaged within each dataset and then across datasets.

    Only budget-meeting rows enter: a range spanning configurations that
    violate the false-alarm budget mixes a trade-off along the budget with
    sensitivity at it. Window and noise sweeps re-match the threshold to the
    budget at every value, so all their rows enter; the threshold sweep keeps
    only thresholds at or above the calibrated one.
    """
    matched = oat_df.loc[oat_df["calibration_budget_met"].astype(bool)]
    records = []
    for (dataset, scenario, parameter), group in matched.groupby(
        ["dataset", "scenario", "parameter"]
    ):
        records.append({
            "dataset": dataset, "parameter": parameter,
            "n_values": group["value"].nunique(),
            **{name: group[column].max() - group[column].min()
               for name, column in RANGE_METRICS.items()},
        })
    per_dataset = (
        pd.DataFrame(records)
        .groupby(["parameter", "dataset"])[["n_values", *RANGE_METRICS]].mean()
    )
    summary = per_dataset.groupby("parameter").mean()
    summary.insert(0, "factor_type", summary.index.map(FACTOR_TYPES))
    summary.insert(1, "n_datasets", per_dataset.groupby("parameter").size())
    return summary.sort_values("detection_rate_range", ascending=False)


def print_sensitivity_ranking(oat_df: pd.DataFrame) -> None:
    md.h1("Sensitivity at the false-alarm budget")
    md.para(
        "Range (max - min) of each metric across each factor's sweep, computed per "
        "scenario, averaged within each dataset and then across datasets, over "
        "budget-meeting configurations only. The window and noise sweeps re-match "
        "the threshold to the budget at every value. The budget pins the threshold "
        "at a given window, so the threshold row covers only thresholds at or above "
        "the calibrated one (more conservative operating points); thresholds below "
        "it violate the budget and appear only in the sweep rows above, as a "
        "trade-off curve. Noise is a property of the data, not a detector setting. "
        "Delay ranges use the median conditional delay, n/a where detections are "
        "indistinguishable from chance."
    )
    md.table(sensitivity_ranges(oat_df).round(4))


def save_plots(oat_df: pd.DataFrame, grid_df: pd.DataFrame) -> None:
    for parameter in ["window_size", "threshold_drift_score"]:
        sub = oat_df[oat_df["parameter"] == parameter].copy()
        sub["dataset_scenario"] = sub["dataset"] + " / " + sub["scenario"]
        fig = px.line(
            sub, x="value", y=["detection_rate", "calibration_false_positive_rate"],
            color="dataset_scenario", facet_col="variable",
            title=f"Sensitivity to {parameter}",
        )
        fig.write_html(os.path.join(OUTPUT_DIR, f"sensitivity_{parameter}.html"))

    heatmap = grid_df.pivot(index="window_size", columns="threshold_drift_score", values="detection_rate")
    fig = px.imshow(heatmap, text_auto=True, title="Detection rate: window_size x threshold_drift_score")
    fig.write_html(os.path.join(OUTPUT_DIR, "sensitivity_grid_detection_rate.html"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescale", action=argparse.BooleanOptionalAction,
                        default=RESCALE_BY_CALIBRATION_MEDIAN,
                        help="divide squared errors by the pooled calibration median (default: off)")
    args = parser.parse_args()
    if args.rescale:
        OUTPUT_DIR = os.path.join(OUTPUT_DIR, "median_rescaled")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    oat_results = sweep_window_size_and_threshold(rescale=args.rescale)
    noise_results = sweep_noise(rescale=args.rescale)
    oat_results = pd.concat([oat_results, noise_results], ignore_index=True)

    grid_results = sweep_grid(rescale=args.rescale)

    print_sensitivity_ranking(oat_results)

    md.h2("window_size x threshold grid (detection_rate)")
    md.table(grid_results.pivot(index="window_size", columns="threshold_drift_score", values="detection_rate"))
    md.h2("window_size x threshold grid (placebo_detection_rate)")
    md.table(grid_results.pivot(index="window_size", columns="threshold_drift_score", values="placebo_detection_rate"))

    oat_results.to_csv(os.path.join(OUTPUT_DIR, "sensitivity_analysis_oat_results.csv"), index=False)
    grid_results.to_csv(os.path.join(OUTPUT_DIR, "sensitivity_analysis_grid_results.csv"), index=False)

    try:
        save_plots(oat_results, grid_results)
        md.para(f"Saved interactive plots to {OUTPUT_DIR}/sensitivity_*.html")
    except Exception as exc:
        md.para(f"Skipped plot generation ({exc})")

    md.para(f"Saved results to {OUTPUT_DIR}/sensitivity_analysis_oat_results.csv and "
            f"{OUTPUT_DIR}/sensitivity_analysis_grid_results.csv")
