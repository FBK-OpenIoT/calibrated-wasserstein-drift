"""Repeat Wasserstein calibration on independent draws and Optuna seeds.

The experiment isolates calibration uncertainty: for each scenario the trained
model and final test streams are fixed, while every replicate receives a
disjoint set of no-drift calibration streams and a different Optuna TPE seed.
Under the benchmark's largest-admissible window rule the TPE seed plays no
role, so the replicates differ by their calibration draws alone. With
``--batch-size`` the detector may alarm only on the last sample of each batch,
as in ``batch_checking.py``, and is calibrated at that batch size. It writes
replicate-level results, dispersion summaries, and a compact Markdown report
under ``data/`` by default. The report leads with each scenario's detection
rate as a range across replicates: that range, not one calibration's rate, is
the figure to report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wasserstein_score.normalization import CalibrationMedianNormalizer
from wasserstein_score.benchmark_protocol import (
    BIMODAL_SHIFT_SEPARATION,
    DETECTION_HORIZON,
    DRIFT_AT,
    FALSE_ALARM_BUDGET_PER_1000,
    FALSE_ALARM_CONFIDENCE,
    N_BOOTSTRAP,
    N_CALIBRATION_STREAMS,
    N_OPTUNA_TRIALS,
    N_TEST_STREAMS,
    NUM_DATAPOINTS,
    NUM_MODEL_TEST_POINTS,
    NUM_TRAINING_POINTS,
    RESCALE_BY_CALIBRATION_MEDIAN,
    WASSERSTEIN_BOUNDS,
    WASSERSTEIN_WINDOW_SELECTION,
)
from wasserstein_score.evaluation import (
    WINDOW_SELECTION_MODES,
    calibrate_and_test,
)
from wasserstein_score.helper_functions import (
    abrupt_linear_drift_stream,
    bimodal_shift_streams,
    build_model_df,
    get_model,
    held_out_null_streams,
    river_friedman_drift_stream,
)


DEFAULT_SEEDS = (42, 43, 44, 45, 46)
SCENARIOS = (
    "Linear Abrupt",
    "River Friedman (LEA)",
    "Bimodal shift",
)
SCENARIO_SEED_OFFSETS = {
    scenario: scenario_index * 100_000
    for scenario_index, scenario in enumerate(SCENARIOS)
}

REPORT_METRICS = (
    "window_size",
    "threshold_drift_score",
    "calibration_fp_ucb_per_1000",
    "calibration_budget_met",
    "median_first_delay",
    "mean_first_delay",
    "mean_capped_delay",
    "detection_rate",
    "placebo_detection_rate",
    "excess_detection_rate",
    "mean_false_positives_active",
    "flicker_rate",
    "null_false_alarms_per_1000",
    "calibration_ucb_covers_null_rate_proxy",
    "calibration_ucb_below_null_ci",
)

def _calibration_seed_base(scenario: str, replicate_index: int) -> int:
    """Return non-overlapping seed ranges for every scenario and replicate."""
    return 10_000 + SCENARIO_SEED_OFFSETS[scenario] + replicate_index * 1_000


def _scenario_config(
    scenario: str,
    *,
    calibration_seed_base: int,
    n: int,
    drift_at: int,
    n_calibration_streams: int,
    n_test_streams: int,
    n_training_points: int,
    n_validation_points: int,
) -> tuple:
    """Generator and keyword arguments of ``scenario``."""
    common = {
        "n": n,
        "drift_at": drift_at,
        "n_calibrations_streams": n_calibration_streams,
        "n_test_streams": n_test_streams,
        "calibration_seed_base": calibration_seed_base,
    }

    if scenario == "Linear Abrupt":
        return abrupt_linear_drift_stream, dict(
            common,
            noise=0.1,
            scenario_name="linear_abrupt",
        )
    if scenario == "River Friedman (LEA)":
        return river_friedman_drift_stream, dict(
            common,
            num_training_points=n_training_points,
            num_test_points=n_validation_points,
            drift_scenarios={
                "lea": {"river_type": "lea", "window": 0},
            },
        )
    if scenario == "Bimodal shift":
        return bimodal_shift_streams, dict(
            common,
            base_sigma=1.0,
            separation=BIMODAL_SHIFT_SEPARATION,
            seed_base=2_000,
        )
    raise ValueError(f"Unknown scenario: {scenario}")


def _generate_scenario(scenario: str, **config) -> tuple:
    generator, kwargs = _scenario_config(scenario, **config)
    return generator(**kwargs)


def _null_streams(scenario: str, n_streams: int, **config) -> tuple:
    """Held-out drift-free streams for the placebo detection rate, disjoint
    from the calibration and test streams of ``scenario``."""
    generator, kwargs = _scenario_config(scenario, **config)
    return held_out_null_streams(generator, n_streams, **kwargs)


def _fit_model(X_training: np.ndarray, y_training: np.ndarray, n_train: int):
    return get_model(
        build_model_df(X_training, y_training),
        n_train,
        0,
        range(X_training.shape[1]),
        "target",
    )


def _residual_streams(model, X_streams: list, y_streams: list) -> list[np.ndarray]:
    return [
        np.square(model.predict(X_stream) - y_stream)
        for X_stream, y_stream in zip(X_streams, y_streams)
    ]


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Return mean, sample SD, and range for parameters and final metrics."""
    missing = set(REPORT_METRICS) - set(results.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")

    long = results.melt(
        id_vars="scenario",
        value_vars=REPORT_METRICS,
        var_name="metric",
        value_name="value",
    )
    long["value"] = pd.to_numeric(long["value"])
    return (
        long.groupby(["scenario", "metric"], sort=False)["value"]
        .agg(mean="mean", std="std", min="min", max="max", n="count")
        .reset_index()
    )


def _format_stat(row: pd.Series, decimals: int) -> str:
    if pd.isna(row["mean"]):
        return "N/A"
    sd = row["std"]
    sd_text = "N/A" if pd.isna(sd) else f"{sd:.{decimals}f}"
    return (
        f"{row['mean']:.{decimals}f} +/- {sd_text} "
        f"[{row['min']:.{decimals}f}, {row['max']:.{decimals}f}]"
    )


def detection_rate_ranges(results: pd.DataFrame) -> pd.DataFrame:
    """Per scenario, the detection rate's range across calibration replicates,
    with the windows those replicates selected."""
    rows = []
    for scenario in SCENARIOS:
        replicates = results.loc[results["scenario"] == scenario]
        if replicates.empty:
            continue
        rates = replicates["detection_rate"].astype(float)
        windows = sorted(set(replicates["window_size"].astype(int)))
        rows.append({
            "scenario": scenario,
            "n_replicates": len(replicates),
            "detection_rate_min": rates.min(),
            "detection_rate_median": rates.median(),
            "detection_rate_max": rates.max(),
            "windows": ", ".join(map(str, windows)),
        })
    return pd.DataFrame(rows)


def build_markdown_report(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    n_calibration_streams: int,
    n_test_streams: int,
    n_trials: int,
    false_alarm_budget_per_1000: float,
    false_alarm_confidence: float,
) -> str:
    """Build a paper-ready compact report from completed replicates."""
    completed_seeds = ", ".join(str(seed) for seed in sorted(results["optuna_seed"].unique()))
    selection = (
        ", ".join(sorted(results["window_selection"].astype(str).unique()))
        if "window_selection" in results else "unknown"
    )
    batch_sizes = (
        ", ".join(str(int(b)) for b in sorted(results["batch_size"].unique()))
        if "batch_size" in results else "1"
    )
    lines = [
        "# Calibration robustness across independent seeds",
        "",
        "## Detection rate across calibration seeds",
        "",
        (
            "Report each scenario's detection rate as this range: a single "
            "calibration's rate is one draw from it. Wasserstein window "
            f"selection: {selection}. Batch size: {batch_sizes}."
        ),
        "",
        "| Scenario | Replicates | Detection rate (min-max) | Median | Selected windows |",
        "|---|---:|---:|---:|---|",
    ]
    for row in detection_rate_ranges(results).itertuples():
        lines.append(
            f"| {row.scenario} | {row.n_replicates} | "
            f"{row.detection_rate_min:.2f}-{row.detection_rate_max:.2f} | "
            f"{row.detection_rate_median:.2f} | {row.windows} |"
        )
    lines += [
        "",
        "## All metrics",
        "",
        (
            f"Each replicate uses {n_calibration_streams} independent no-drift calibration "
            f"streams, {n_trials} Optuna trials, and one of the TPE seeds "
            f"{completed_seeds}. The trained model and the {n_test_streams} final test "
            "streams are held fixed within each scenario, so performance dispersion "
            "measures calibration uncertainty. Calibration requires the one-sided "
            f"{false_alarm_confidence:.1%} working-model Poisson upper bound to be at most "
            f"{false_alarm_budget_per_1000:g} false alarms per 1,000 post-warm-up "
            f"samples. A stream is detected only by an alarm within "
            f"{DETECTION_HORIZON:,} samples of the drift; the placebo rate applies "
            "the same criterion to held-out drift-free streams, and the excess "
            "rate is the detection rate minus the placebo rate. "
            "Capped delay assigns misses the full horizon. Poisson coverage is "
            "assumption-dependent and is not guaranteed after selection. "
            "The coverage proxy checks whether each selected calibration UCB "
            "covers its held-out null rate estimate; the below-CI flag checks "
            "whether it lies below that estimate's whole-stream bootstrap CI. "
            "These are diagnostics, not coverage against a known true rate. "
            "Null streams are shared across replicates, so proxy outcomes are "
            "dependent; their means are descriptive fractions. "
            "Values are mean +/- sample SD [min, max]."
        ),
        "",
        "| Scenario | $W$ | $\\theta$ | Calibration FP UCB | Budget met | Median conditional delay | Mean conditional delay | Capped delay | Detection rate | Placebo rate | Excess rate | FP / 1,000 active | Flicker / 1,000 | Null FP / 1,000 | UCB coverage proxy | UCB below null CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    decimals = {
        "window_size": 1,
        "threshold_drift_score": 2,
        "calibration_fp_ucb_per_1000": 3,
        "calibration_budget_met": 3,
        "median_first_delay": 1,
        "mean_first_delay": 1,
        "mean_capped_delay": 1,
        "detection_rate": 3,
        "placebo_detection_rate": 3,
        "excess_detection_rate": 3,
        "mean_false_positives_active": 3,
        "flicker_rate": 3,
        "null_false_alarms_per_1000": 3,
        "calibration_ucb_covers_null_rate_proxy": 3,
        "calibration_ucb_below_null_ci": 3,
    }
    for scenario in SCENARIOS:
        scenario_rows = summary[summary["scenario"] == scenario].set_index("metric")
        if scenario_rows.empty:
            continue
        cells = [
            _format_stat(scenario_rows.loc[metric], decimals[metric])
            for metric in REPORT_METRICS
        ]
        lines.append(f"| {scenario} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "The replicate-level CSV contains the calibration false-positive objective, its "
        "within-draw standard deviation, its one-sided upper bound, and whether the "
        "selected configuration met the common budget.",
        "It also retains null alarm counts and active exposures in stream order, "
        "null rate intervals and dispersion, Wilson detection intervals, and "
        "the calibration-bound coverage diagnostics. Missing bootstrap intervals "
        "(e.g. zero null alarms) produce unavailable below-CI diagnostics, not passes.",
        "",
    ])
    return "\n".join(lines)


def run_experiment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds must not contain duplicates")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "calibration_robustness_results.csv"
    summary_path = output_dir / "calibration_robustness_summary.csv"
    report_path = output_dir / "calibration_robustness_report.md"

    if args.resume and raw_path.exists():
        existing = pd.read_csv(raw_path)
        # Results written before --batch-size existed were all unbatched.
        if "batch_size" not in existing:
            existing["batch_size"] = 1
        expected_metadata = {
            "n_calibration_streams": args.n_calibration_streams,
            "n_test_streams": args.n_test_streams,
            "n_trials": args.n_trials,
            "n_samples": args.n_samples,
            "drift_at": args.drift_at,
            "window_min": args.window_min,
            "window_max": args.window_max,
            "threshold_min": args.threshold_min,
            "threshold_max": args.threshold_max,
            "false_alarm_budget_per_1000": args.false_alarm_budget_per_1000,
            "false_alarm_confidence": args.false_alarm_confidence,
            "detection_horizon": DETECTION_HORIZON,
            "window_selection": args.window_selection,
            "batch_size": args.batch_size,
        }
        for column, expected in expected_metadata.items():
            if column not in existing or not (existing[column] == expected).all():
                raise ValueError(
                    f"Cannot resume: existing {column!r} does not match {expected!r}"
                )
        missing = set(REPORT_METRICS) - set(existing.columns)
        if missing:
            raise ValueError(
                f"Cannot resume: existing results lack {sorted(missing)}; "
                "they predate the current protocol."
            )
        if (
            "signal_normalization" not in existing
            or not (existing["signal_normalization"] == (
                "pooled_calibration_median" if args.rescale else "none"
            )).all()
            or "normalization_scale" not in existing
            or not np.isfinite(existing["normalization_scale"]).all()
            or not (existing["normalization_scale"] > 0).all()
        ):
            raise ValueError(
                "Cannot resume: existing results lack matching calibration-median normalization metadata; "
                "start a new run to avoid mixing raw and normalized results."
            )
        records = existing.to_dict("records")
    else:
        records = []
    completed = {
        (row["scenario"], int(row["optuna_seed"]))
        for row in records
    }

    for scenario in args.scenarios:
        pending = [seed for seed in args.seeds if (scenario, seed) not in completed]
        if not pending:
            print(f"{scenario}: all requested seeds already completed; skipping.")
            continue

        first_seed = pending[0]
        first_replicate_index = args.seeds.index(first_seed)
        first_calibration_seed_base = _calibration_seed_base(
            scenario, first_replicate_index
        )
        scenario_config = dict(
            calibration_seed_base=first_calibration_seed_base,
            n=args.n_samples,
            drift_at=args.drift_at,
            n_calibration_streams=args.n_calibration_streams,
            n_test_streams=args.n_test_streams,
            n_training_points=args.n_training_points,
            n_validation_points=args.n_validation_points,
        )
        base_data = _generate_scenario(scenario, **scenario_config)
        X_training, y_training, X_calib, y_calib, X_test, y_test = base_data
        model = _fit_model(X_training, y_training, args.n_training_points)
        fixed_test_residuals = _residual_streams(
            model,
            next(iter(X_test.values())),
            next(iter(y_test.values())),
        )
        # Like the test streams, the null streams are fixed across replicates.
        X_null, y_null = _null_streams(
            scenario, args.n_test_streams, **scenario_config
        )
        null_residuals = _residual_streams(model, X_null, y_null)

        for seed in pending:
            replicate_index = args.seeds.index(seed)
            calibration_seed_base = _calibration_seed_base(
                scenario, replicate_index
            )
            if seed != first_seed:
                calibration_data = _generate_scenario(
                    scenario,
                    calibration_seed_base=calibration_seed_base,
                    n=args.n_samples,
                    drift_at=args.drift_at,
                    n_calibration_streams=args.n_calibration_streams,
                    n_test_streams=0,
                    n_training_points=args.n_training_points,
                    n_validation_points=args.n_validation_points,
                )
                X_calib, y_calib = calibration_data[2], calibration_data[3]

            calibration_residuals = _residual_streams(model, X_calib, y_calib)
            normalizer = CalibrationMedianNormalizer.fit(calibration_residuals, enabled=args.rescale)
            print(
                f"\n{scenario}: Optuna seed={seed}, "
                f"calibration seed base={calibration_seed_base}"
            )
            best, calibration_df, test_df, _ = calibrate_and_test(
                calibration_streams=normalizer.transform(calibration_residuals),
                test_streams={scenario: normalizer.transform(fixed_test_residuals)},
                drift_at=args.drift_at,
                algorithms=["wasserstein"],
                wasserstein_bounds={
                    "threshold_drift_score": (
                        args.threshold_min,
                        args.threshold_max,
                    ),
                    "window_size": (args.window_min, args.window_max),
                },
                n_trials=args.n_trials,
                n_bootstrap=args.n_bootstrap,
                optuna_seed=seed,
                bootstrap_seed=seed,
                false_alarm_budget_per_1000=args.false_alarm_budget_per_1000,
                false_alarm_confidence=args.false_alarm_confidence,
                detection_horizon=DETECTION_HORIZON,
                null_streams=normalizer.transform(null_residuals),
                verbose=not args.quiet,
                wasserstein_window_selection=args.window_selection,
                batch_size=args.batch_size,
            )
            selected = calibration_df.loc[calibration_df["selected"]]
            if len(selected) != 1:
                raise RuntimeError(
                    f"Expected one selected Wasserstein trial, found {len(selected)}"
                )
            selected_row = selected.iloc[0]
            test_row = test_df.iloc[0]
            records.append({
                "scenario": scenario,
                "signal_normalization": normalizer.method,
                "normalization_scale": normalizer.scale,
                "replicate": replicate_index,
                "optuna_seed": seed,
                "calibration_seed_base": calibration_seed_base,
                "n_calibration_streams": args.n_calibration_streams,
                "n_test_streams": args.n_test_streams,
                "n_trials": args.n_trials,
                "n_samples": args.n_samples,
                "drift_at": args.drift_at,
                "window_min": args.window_min,
                "window_max": args.window_max,
                "threshold_min": args.threshold_min,
                "threshold_max": args.threshold_max,
                "false_alarm_budget_per_1000": args.false_alarm_budget_per_1000,
                "false_alarm_confidence": args.false_alarm_confidence,
                "detection_horizon": DETECTION_HORIZON,
                "window_selection": args.window_selection,
                "batch_size": args.batch_size,
                "window_size": int(best["wasserstein"]["window_size"]),
                "threshold_drift_score": float(
                    best["wasserstein"]["threshold_drift_score"]
                ),
                "calibration_fp_per_1000": float(selected_row["mean_detections"]),
                "calibration_fp_std_per_1000": float(
                    selected_row["std_detections"]
                ),
                "calibration_fp_ucb_per_1000": float(
                    selected_row["false_alarm_ucb_per_1000"]
                ),
                "calibration_budget_met": bool(
                    selected_row["false_alarm_budget_met"]
                ),
                "calibration_false_alarms": int(selected_row["n_false_alarms"]),
                "calibration_samples": int(selected_row["calibration_samples"]),
                "params_at_bound": selected_row["params_at_bound"],
                "mean_first_delay": float(test_row["mean_first_delay"]),
                "median_first_delay": float(test_row["median_first_delay"]),
                "conditional_delay_reportable": bool(test_row["conditional_delay_reportable"]),
                "mean_capped_delay": float(test_row["mean_capped_delay"]),
                **{column: test_row[column] for column in test_df.columns
                   if column.startswith(("null_", "calibration_ucb_", "detection_rate_ci_",
                                         "placebo_detection_rate_ci_", "mean_capped_delay_ci_"))},
                "detection_rate": float(test_row["detection_rate"]),
                "placebo_detection_rate": float(
                    test_row["placebo_detection_rate"]
                ),
                "excess_detection_rate": float(test_row["excess_detection_rate"]),
                "mean_false_positives_active": float(
                    test_row["mean_false_positives_active"]
                ),
                "flicker_rate": float(test_row["flicker_rate"]),
            })
            pd.DataFrame(records).to_csv(raw_path, index=False)

    results = pd.DataFrame(records)
    if results.empty:
        raise RuntimeError("No calibration robustness results were produced")
    results = results.sort_values(["scenario", "replicate"]).reset_index(drop=True)
    results.to_csv(raw_path, index=False)
    summary = summarize_results(results)
    summary.to_csv(summary_path, index=False)
    report_path.write_text(
        build_markdown_report(
            results,
            summary,
            n_calibration_streams=args.n_calibration_streams,
            n_test_streams=args.n_test_streams,
            n_trials=args.n_trials,
            false_alarm_budget_per_1000=args.false_alarm_budget_per_1000,
            false_alarm_confidence=args.false_alarm_confidence,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {raw_path}, {summary_path}, and {report_path}")
    return results, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--n-trials", type=int, default=N_OPTUNA_TRIALS)
    parser.add_argument(
        "--n-calibration-streams", type=int, default=N_CALIBRATION_STREAMS
    )
    parser.add_argument("--n-test-streams", type=int, default=N_TEST_STREAMS)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--n-samples", type=int, default=NUM_DATAPOINTS)
    parser.add_argument("--drift-at", type=int, default=DRIFT_AT)
    parser.add_argument(
        "--n-training-points", type=int, default=NUM_TRAINING_POINTS
    )
    parser.add_argument(
        "--n-validation-points", type=int, default=NUM_MODEL_TEST_POINTS
    )
    parser.add_argument(
        "--window-min", type=int, default=WASSERSTEIN_BOUNDS["window_size"][0]
    )
    parser.add_argument(
        "--window-max", type=int, default=WASSERSTEIN_BOUNDS["window_size"][1]
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=WASSERSTEIN_BOUNDS["threshold_drift_score"][0],
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=WASSERSTEIN_BOUNDS["threshold_drift_score"][1],
    )
    parser.add_argument(
        "--false-alarm-budget-per-1000",
        type=float,
        default=FALSE_ALARM_BUDGET_PER_1000,
    )
    parser.add_argument(
        "--false-alarm-confidence",
        type=float,
        default=FALSE_ALARM_CONFIDENCE,
    )
    parser.add_argument(
        "--window-selection", choices=WINDOW_SELECTION_MODES,
        default=WASSERSTEIN_WINDOW_SELECTION,
        help="Wasserstein window selection rule (default: the benchmark's)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="alarm only on the last sample of each batch of this size (default: 1)",
    )
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--rescale", action=argparse.BooleanOptionalAction,
                        default=RESCALE_BY_CALIBRATION_MEDIAN,
                        help="divide squared errors by the pooled calibration median (default: off)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.rescale and args.output_dir == "data":
        args.output_dir = "data/median_rescaled"
    return args


if __name__ == "__main__":
    run_experiment(parse_args())
