import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

import pandas as pd

from wasserstein_score.normalization import CalibrationMedianNormalizer
from wasserstein_score.benchmark_protocol import (
    ADWIN_BOUNDS,
    BIMODAL_SHIFT_SEPARATION,
    BOOTSTRAP_SEED,
    CALIBRATION_WIDENING_ROUNDS,
    CI_LEVEL,
    DETECTION_HORIZON,
    DRIFT_AT,
    FALSE_ALARM_BUDGET_PER_1000,
    FALSE_ALARM_CONFIDENCE,
    KSWIN_BOUNDS,
    KSWIN_SEED,
    MIN_DETECTED_STREAMS_FOR_DELAY,
    N_BOOTSTRAP,
    N_CALIBRATION_STREAMS,
    N_OPTUNA_TRIALS,
    N_TEST_STREAMS,
    NUM_DATAPOINTS,
    NUM_MODEL_TEST_POINTS,
    NUM_TRAINING_POINTS,
    NULL_STREAM_SEED_BASES,
    NULL_STREAM_SEED_OFFSETS,
    OPTUNA_SEED,
    PAGE_HINKLEY_BOUNDS,
    RUNNING_ALGORITHMS,
    RESCALE_BY_CALIBRATION_MEDIAN,
    STARTING_IDX,
    WASSERSTEIN_BOUNDS,
    WASSERSTEIN_WINDOW_SELECTION,
)
from wasserstein_score.evaluation import (
    PARAMETER_DOMAINS,
    calibrate_and_test,
)
from wasserstein_score.significance import significance_report
from wasserstein_score import helper_functions as md
from wasserstein_score.helper_functions import (
    get_model, build_model_df, benchmark_datasets, held_out_null_streams,
)

# Reference algorithm for the paired significance tests, and FWER level
significance_reference = "wasserstein"
significance_alpha = 0.05
OUTPUT_DIR = Path("data")

SUMMARY_METRICS = [
    "mean_total_detections",
    "mean_false_positives_common",
    "mean_false_positives_active",
    "mean_true_detections",
    "detection_rate",
    "placebo_detection_rate",
    "excess_detection_rate",
    "mean_first_delay",
    "median_first_delay",
    "mean_capped_delay",
    "flicker_rate",
]

# Columns of the per-dataset Markdown table: detection with its Wilson
# interval, the robust (median) and mean conditional delays, which are n/a
# where detections are indistinguishable from chance, and the capped delay.
REPORT_COLUMNS = [
    "detection_rate", "detection_rate_ci_low", "detection_rate_ci_high",
    "placebo_detection_rate", "median_first_delay", "mean_first_delay",
    "mean_capped_delay",
]


@dataclass
class DatasetResult:
    label: str
    best_params: dict[str, dict[str, Any]]
    calibration_results: pd.DataFrame
    test_results: pd.DataFrame
    per_stream_results: pd.DataFrame
    normalization_scale: float = 1.0
    rescale: bool = RESCALE_BY_CALIBRATION_MEDIAN


def collect_git_provenance(repo_root: Path) -> dict[str, Any]:
    """Return the revision that produced a run without making Git mandatory."""
    git_prefix = ["git", "-c", f"safe.directory={repo_root}"]
    try:
        commit_sha = subprocess.run(
            [*git_prefix, "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [*git_prefix, "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return {
            "commit_sha": commit_sha,
            "working_tree_dirty": bool(status),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "commit_sha": "unknown",
            "working_tree_dirty": None,
            "git_error": str(exc),
        }


def build_result_tables(
    results: list[DatasetResult],
    provenance: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Combine one run's result frames into the persisted benchmark tables."""
    if not results:
        raise ValueError("At least one dataset result is required")
    if len({result.rescale for result in results}) != 1:
        raise ValueError("Cannot combine raw and median-rescaled results in one run")

    def _tag(frame: pd.DataFrame, result: DatasetResult) -> pd.DataFrame:
        tagged = frame.copy()
        tagged.insert(0, "dataset", result.label)
        tagged["normalization_scale"] = result.normalization_scale
        tagged["signal_normalization"] = (
            "pooled_calibration_median" if result.rescale else "none"
        )
        tagged["commit_sha"] = provenance["commit_sha"]
        tagged["working_tree_dirty"] = provenance["working_tree_dirty"]
        return tagged

    test_summary = pd.concat(
        [_tag(result.test_results, result) for result in results],
        ignore_index=True,
    )
    per_stream = pd.concat(
        [_tag(result.per_stream_results, result) for result in results],
        ignore_index=True,
    )

    selected_frames = []
    for result in results:
        selected = result.calibration_results.loc[
            result.calibration_results["selected"]
        ]
        if len(selected) != len(result.best_params):
            raise ValueError(
                f"{result.label}: expected {len(result.best_params)} selected "
                f"calibration rows, found {len(selected)}"
            )
        selected_frames.append(_tag(selected, result))
    calibrated_parameters = pd.concat(selected_frames, ignore_index=True)

    missing_metrics = set(SUMMARY_METRICS) - set(test_summary.columns)
    if missing_metrics:
        raise ValueError(
            f"Cannot build overall summary; missing metrics: {sorted(missing_metrics)}"
        )
    # Average within each dataset first: scenarios of one dataset share its
    # calibration and pre-drift streams, so a dataset with three scenarios
    # must not weigh three times as much as one with a single scenario.
    overall_summary = (
        test_summary.groupby(["algorithm", "dataset"], sort=False)[SUMMARY_METRICS]
        .mean()
        .groupby("algorithm", sort=False)
        .mean()
        .reset_index()
    )
    counts = (
        test_summary.groupby("algorithm", sort=False)
        .agg(n_datasets=("dataset", "nunique"), n_scenarios=("scenario", "nunique"))
        .reset_index()
    )
    overall_summary = counts.merge(
        overall_summary, on="algorithm", how="left", validate="one_to_one"
    )
    overall_summary["commit_sha"] = provenance["commit_sha"]
    overall_summary["working_tree_dirty"] = provenance["working_tree_dirty"]

    return test_summary, overall_summary, calibrated_parameters, per_stream


def build_run_metadata(
    provenance: dict[str, Any], output_files: dict[str, str],
    *, algorithms=None, rescale: bool = RESCALE_BY_CALIBRATION_MEDIAN,
) -> dict[str, Any]:
    return {
        "schema_version": 4,
        "interval_methods": {
            "detection_rate": "wilson",
            "placebo_detection_rate": "wilson",
            "delay_means": "stream_percentile_bootstrap",
            "null_rate": "whole_stream_percentile_bootstrap",
            "calibration_bound": "poisson_working_model_not_post_selection_guarantee",
        },
        "experiment": "main_benchmark",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **provenance,
        "configuration": {
            "signal_normalization": "pooled_calibration_median" if rescale else "none",
            "num_datapoints": NUM_DATAPOINTS,
            "num_training_points": NUM_TRAINING_POINTS,
            "num_model_test_points": NUM_MODEL_TEST_POINTS,
            "drift_at": DRIFT_AT,
            "starting_idx": STARTING_IDX,
            "n_calibration_streams": N_CALIBRATION_STREAMS,
            "n_test_streams": N_TEST_STREAMS,
            "n_optuna_trials": N_OPTUNA_TRIALS,
            "algorithms": list(RUNNING_ALGORITHMS if algorithms is None else algorithms),
            "n_bootstrap": N_BOOTSTRAP,
            "ci_level": CI_LEVEL,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "optuna_seed": OPTUNA_SEED,
            "false_alarm_budget_per_1000": FALSE_ALARM_BUDGET_PER_1000,
            "false_alarm_confidence": FALSE_ALARM_CONFIDENCE,
            "detection_horizon": DETECTION_HORIZON,
            "null_stream_seed_bases": NULL_STREAM_SEED_BASES,
            "null_stream_seed_offsets": NULL_STREAM_SEED_OFFSETS,
            "wasserstein_bounds": WASSERSTEIN_BOUNDS,
            "adwin_bounds": ADWIN_BOUNDS,
            "kswin_bounds": KSWIN_BOUNDS,
            "page_hinkley_bounds": PAGE_HINKLEY_BOUNDS,
            "calibration_widening_rounds": CALIBRATION_WIDENING_ROUNDS,
            "wasserstein_window_selection": WASSERSTEIN_WINDOW_SELECTION,
            "min_detected_streams_for_delay": MIN_DETECTED_STREAMS_FOR_DELAY,
            "parameter_domains": PARAMETER_DOMAINS,
            "kswin_seed": KSWIN_SEED,
            "bimodal_shift_separation": BIMODAL_SHIFT_SEPARATION,
        },
        "outputs": output_files,
    }


def run_dataset(label, X_training, y_training, X_calib_data, y_calib_data,
                X_test_data, y_test_data, drift_at, n_train=NUM_TRAINING_POINTS,
                n_test=NUM_MODEL_TEST_POINTS, null_data=None, batch_size=1,
                algorithms=None, rescale: bool = RESCALE_BY_CALIBRATION_MEDIAN):
    assert (n_train, n_test) == (NUM_TRAINING_POINTS, NUM_MODEL_TEST_POINTS), (
        f"{label}: n_train/n_test ({n_train}/{n_test}) differ from the shared "
        f"NUM_TRAINING_POINTS/NUM_MODEL_TEST_POINTS "
        f"({NUM_TRAINING_POINTS}/{NUM_MODEL_TEST_POINTS}) "
        "used for model training/validation across all datasets."
    )
    assert len(X_training) >= n_train + n_test, (
        f"{label}: X_training has only {len(X_training)} rows, but n_train+n_test="
        f"{n_train + n_test} were requested for the model train/validation split."
    )

    md.h1(label)

    n_features = X_training.shape[1]
    model = get_model(
        build_model_df(X_training, y_training),
        n_train, n_test, range(n_features), 'target',
    )

    y_calib_residuals = [
        (model.predict(X_calib_data[i]) - y_calib_data[i]) ** 2
        for i in range(len(X_calib_data))
    ]
    y_test_residuals = {
        name: [
            (model.predict(X_test_data[name][i]) - y_test_data[name][i]) ** 2
            for i in range(len(X_test_data[name]))
        ]
        for name in X_test_data
    }
    y_null_residuals = None
    if null_data is not None:
        X_null, y_null = null_data
        y_null_residuals = [
            (model.predict(X_null[i]) - y_null[i]) ** 2 for i in range(len(X_null))
        ]

    normalizer = CalibrationMedianNormalizer.fit(y_calib_residuals, enabled=rescale)
    y_calib_residuals = normalizer.transform(y_calib_residuals)
    y_test_residuals = {
        name: normalizer.transform(streams) for name, streams in y_test_residuals.items()
    }
    if y_null_residuals is not None:
        y_null_residuals = normalizer.transform(y_null_residuals)
    md.bullet(f"Signal normalization: {normalizer.method}; divisor: {normalizer.scale:g}")

    best_params, calib_df, test_df, per_stream_df = calibrate_and_test(
        calibration_streams=y_calib_residuals,
        test_streams=y_test_residuals,
        drift_at=drift_at,
        starting_idx=STARTING_IDX,
        algorithms=RUNNING_ALGORITHMS if algorithms is None else algorithms,
        wasserstein_bounds=WASSERSTEIN_BOUNDS,
        adwin_bounds=ADWIN_BOUNDS,
        kswin_bounds=KSWIN_BOUNDS,
        ph_bounds=PAGE_HINKLEY_BOUNDS,
        n_trials=N_OPTUNA_TRIALS,
        n_bootstrap=N_BOOTSTRAP,
        ci_level=CI_LEVEL,
        bootstrap_seed=BOOTSTRAP_SEED,
        optuna_seed=OPTUNA_SEED,
        false_alarm_budget_per_1000=FALSE_ALARM_BUDGET_PER_1000,
        false_alarm_confidence=FALSE_ALARM_CONFIDENCE,
        detection_horizon=DETECTION_HORIZON,
        null_streams=y_null_residuals,
        verbose=True,
        batch_size=batch_size,
    )
    md.h2("Detection probability and delay (misses receive the horizon cap)")
    md.table(test_df.set_index(["scenario", "algorithm"])[REPORT_COLUMNS].round(3))
    if test_df["scenario"].nunique() > 1:
        md.para(
            "These scenarios share one calibration and their pre-drift test "
            "streams: they are distinct drifts, not independent replications, "
            "and their false-positive data are tested once."
        )
    return DatasetResult(
        label=label,
        best_params=best_params,
        calibration_results=calib_df,
        test_results=test_df,
        per_stream_results=per_stream_df,
        normalization_scale=normalizer.scale,
        rescale=rescale,
    )


def run_benchmark(*, algorithms=None, datasets=None,
                  rescale: bool = RESCALE_BY_CALIBRATION_MEDIAN) -> list[DatasetResult]:
    return [
        run_dataset(
            label,
            *generator(**kwargs),
            drift_at=DRIFT_AT,
            null_data=held_out_null_streams(generator, N_TEST_STREAMS, **kwargs),
            algorithms=algorithms,
            rescale=rescale,
        )
        for label, generator, kwargs in benchmark_datasets()
        if datasets is None or label in datasets
    ]


def save_benchmark_outputs(
    results: list[DatasetResult],
    output_dir: Path,
    provenance: dict[str, Any],
) -> None:
    """Persist aggregate, calibration, per-stream, significance and provenance data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    test_summary, overall_summary, calibrated_parameters, per_stream = (
        build_result_tables(results, provenance)
    )

    output_files = {
        "test_summary": "test_summary_results.csv",
        "overall_summary": "overall_summary_results.csv",
        "calibrated_parameters": "calibrated_parameters.csv",
        "per_stream": "per_stream_results.csv",
        "delay_wilcoxon": "significance_delay_wilcoxon.csv",
        "fp_wilcoxon": "significance_fp_wilcoxon.csv",
        "detection_mcnemar": "significance_detection_mcnemar.csv",
    }

    test_summary.to_csv(output_dir / output_files["test_summary"], index=False)
    overall_summary.to_csv(output_dir / output_files["overall_summary"], index=False)
    calibrated_parameters.to_csv(
        output_dir / output_files["calibrated_parameters"], index=False
    )
    per_stream.to_csv(output_dir / output_files["per_stream"], index=False)

    active_algorithms = list(dict.fromkeys(test_summary["algorithm"]))
    reference = (
        significance_reference if significance_reference in active_algorithms
        else active_algorithms[0]
    )
    significance = (
        significance_report(per_stream, reference=reference, alpha=significance_alpha)
        if len(active_algorithms) > 1 else {
            name: pd.DataFrame(columns=["scenario", "algorithm", "reference", "p_value"])
            for name in ("delay_wilcoxon", "fp_wilcoxon", "detection_mcnemar")
        }
    )
    for name in ("delay_wilcoxon", "fp_wilcoxon", "detection_mcnemar"):
        significance[name].to_csv(output_dir / output_files[name], index=False)

    metadata_name = "benchmark_run_metadata.json"
    output_files["metadata"] = metadata_name
    metadata = build_run_metadata(
        provenance, output_files, algorithms=active_algorithms, rescale=results[0].rescale,
    )
    metadata["configuration"]["datasets"] = [result.label for result in results]
    metadata["configuration"]["significance_reference"] = reference
    metadata["normalization_scales"] = {
        result.label: result.normalization_scale for result in results
    }
    (output_dir / metadata_name).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    md.h1("Saved benchmark artifacts")
    for path in output_files.values():
        md.bullet(str(output_dir / path))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate and evaluate the synthetic benchmark.")
    parser.add_argument("--algorithms", nargs="+", choices=RUNNING_ALGORITHMS,
                        default=list(RUNNING_ALGORITHMS), help="detectors to calibrate and evaluate")
    parser.add_argument("--datasets", nargs="+", choices=[d[0] for d in benchmark_datasets()],
                        metavar="DATASET",
                        help="dataset families to run (default: all)")
    parser.add_argument("--rescale", action=argparse.BooleanOptionalAction,
                        default=RESCALE_BY_CALIBRATION_MEDIAN,
                        help="divide squared errors by the pooled calibration median (default: off)")
    parser.add_argument("--output-dir", type=Path,
                        help="result directory (default: data, or data/median_rescaled with --rescale)")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    provenance = collect_git_provenance(repo_root)
    if provenance["working_tree_dirty"]:
        md.para(
            "WARNING: the benchmark started from a dirty working tree; "
            "the commit SHA alone does not identify every code change."
        )
    results = run_benchmark(algorithms=args.algorithms, datasets=args.datasets, rescale=args.rescale)
    output_dir = args.output_dir or (OUTPUT_DIR / "median_rescaled" if args.rescale else OUTPUT_DIR)
    save_benchmark_outputs(results, output_dir, provenance)


if __name__ == "__main__":
    main()
