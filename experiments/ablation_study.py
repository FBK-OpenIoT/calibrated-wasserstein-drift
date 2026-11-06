"""
Matched-budget ablation study for WassersteinDriftDetector.

Isolates the contribution of each design choice in
wasserstein_score/wasserstein_drift_detector.py by replacing it with an
alternative (wasserstein_score/ablation_variants.py):

  full             - the detector as shipped: split-window comparison,
                      IQR (Q1/Q3, 1.5*IQR) adaptive bound, hysteresis
                      that re-arms once the score returns to zero.
  no_hysteresis    - removes the hysteresis dead band.
  zscore_bound     - Gaussian (mean + 1.5*std) bound instead of the IQR bound.
  ks_statistic     - Kolmogorov-Smirnov statistic instead of Wasserstein.
  energy_statistic - energy distance instead of Wasserstein.

Every variant is calibrated separately with the main benchmark's protocol
(same calibration streams, window-selection rule, false-alarm budget and
Poisson upper bound), so variants are compared at the same false-alarm
constraint rather than at shared numerical settings at which they raise very
different numbers of false alarms. Under the benchmark's largest-admissible
window rule every variant gets the largest window at which it can meet the
budget and its threshold is solved over the whole threshold domain, so the
'full' variant reproduces the benchmark's Wasserstein selection, and variants
that need thresholds far above the benchmark range, such as the one without
hysteresis, get them (``evaluation.calibrate_detectors_with_widening``).

The datasets, stream lengths, drift point and calibration/test sizes are those
of the main benchmark, and every metric follows the benchmark's definitions
(``evaluation.evaluate_fixed_detector``): detection within the detection
horizon, the placebo detection rate on held-out drift-free streams, and the
calibration false-alarm rate with its budget flag.
"""

import argparse
import functools
import os

import pandas as pd

from wasserstein_score.normalization import CalibrationMedianNormalizer
from wasserstein_score.ablation_variants import VARIANT_CLASSES
from wasserstein_score.benchmark_protocol import (
    DRIFT_AT,
    N_OPTUNA_TRIALS,
    N_TEST_STREAMS,
    NUM_MODEL_TEST_POINTS,
    NUM_TRAINING_POINTS,
    OPTUNA_SEED,
    RESCALE_BY_CALIBRATION_MEDIAN,
    STARTING_IDX,
)
from wasserstein_score.evaluation import (
    calibrate_detectors_with_widening,
    evaluate_fixed_detector,
)
from wasserstein_score import helper_functions as md
from wasserstein_score.helper_functions import (
    get_model, build_model_df, benchmark_datasets, held_out_null_streams,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_CSV = "data/ablation_study_results.csv"


def make_factory(variant: str, window_size: int, threshold_drift_score: float):
    return functools.partial(
        VARIANT_CLASSES[variant],
        window_size=window_size,
        threshold_drift_score=threshold_drift_score,
    )


def residual_streams(model, X_streams, y_streams) -> list:
    return [
        (model.predict(X_streams[i]) - y_streams[i]) ** 2
        for i in range(len(X_streams))
    ]


def calibrate_variant(variant: str, calibration_streams: list) -> tuple[dict, pd.Series, dict]:
    """Benchmark calibration of ``variant``, with the benchmark's range widening."""
    params, table, bounds = calibrate_detectors_with_widening(
        calibration_streams, algorithms=["wasserstein"], max_warmup_len=DRIFT_AT - STARTING_IDX,
        starting_idx=STARTING_IDX, n_trials=N_OPTUNA_TRIALS, optuna_seed=OPTUNA_SEED,
        wasserstein_cls=VARIANT_CLASSES[variant], verbose=False,
    )
    selected = table[table["selected"]].iloc[0]
    return params["wasserstein"], selected, bounds["wasserstein"]


def run_ablation(rescale: bool = RESCALE_BY_CALIBRATION_MEDIAN) -> pd.DataFrame:
    md.h1("Ablation Study (matched false-alarm budget)")
    rows = []
    for dataset_name, generator, kwargs in benchmark_datasets():
        md.h2(dataset_name)
        X_training, y_training, X_calib, y_calib, X_test, y_test = generator(**kwargs)
        X_null, y_null = held_out_null_streams(generator, N_TEST_STREAMS, **kwargs)
        n_features = X_training.shape[1]

        model = get_model(
            build_model_df(X_training, y_training),
            NUM_TRAINING_POINTS, NUM_MODEL_TEST_POINTS, range(n_features), 'target',
        )

        calib_residuals = residual_streams(model, X_calib, y_calib)
        null_residuals = residual_streams(model, X_null, y_null)
        test_residuals = {
            scenario_name: residual_streams(model, X_list, y_test[scenario_name])
            for scenario_name, X_list in X_test.items()
        }

        normalizer = CalibrationMedianNormalizer.fit(calib_residuals, enabled=rescale)
        calib_residuals = normalizer.transform(calib_residuals)
        null_residuals = normalizer.transform(null_residuals)
        test_residuals = {
            name: normalizer.transform(streams) for name, streams in test_residuals.items()
        }

        for variant in VARIANT_CLASSES:
            params, selected, bounds = calibrate_variant(variant, calib_residuals)
            window_size = int(params["window_size"])
            threshold = float(params["threshold_drift_score"])
            md.bullet(
                f"**{variant}** calibrated: window_size={window_size},"
                f" threshold={threshold:.4f} (final ranges {bounds}),"
                f" UCB={selected['false_alarm_ucb_per_1000']:.3f},"
                f" budget met={bool(selected['false_alarm_budget_met'])}"
                + (f", at bound: {selected['params_at_bound']}" if selected["params_at_bound"] else "")
            )
            factory = make_factory(variant, window_size, threshold)
            for metrics in evaluate_fixed_detector(
                factory, calib_residuals, test_residuals,
                drift_at=DRIFT_AT, null_streams=null_residuals,
                starting_idx=STARTING_IDX,
            ):
                md.bullet(
                    f"[{metrics['scenario']}] **{variant}** —"
                    f" det_rate={metrics['detection_rate']:.2f},"
                    f" placebo={metrics['placebo_detection_rate']:.2f},"
                    f" median_delay={metrics['median_first_delay']:.1f},"
                    f" capped_delay={metrics['mean_capped_delay']:.1f},"
                    f" calib_fp={metrics['calibration_false_positive_rate']:.3f}"
                    f" (UCB {metrics['calibration_fp_ucb_per_1000']:.3f},"
                    f" budget met={metrics['calibration_budget_met']}),"
                    f" flicker={metrics['flicker_rate']:.2f}"
                )
                rows.append({
                    "dataset": dataset_name,
                    "normalization_scale": normalizer.scale,
                    "signal_normalization": normalizer.method,
                    "variant": variant,
                    "window_size": window_size,
                    "threshold_drift_score": threshold,
                    "threshold_upper_bound": bounds["threshold_drift_score"][1],
                    "window_lower_bound": bounds["window_size"][0],
                    "widening_rounds": int(selected["round"]),
                    "params_at_bound": selected["params_at_bound"],
                    **metrics,
                })

    return pd.DataFrame(rows)


def _flatten_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = [" / ".join(map(str, idx)) for idx in df.index]
    return df


SUMMARY_METRICS = [
    "window_size",
    "threshold_drift_score",
    "detection_rate",
    "placebo_detection_rate",
    "excess_detection_rate",
    "median_first_delay",
    "mean_first_delay",
    "mean_capped_delay",
    "calibration_false_positive_rate",
    "calibration_budget_met",
    "flicker_rate",
]


def print_summary(df: pd.DataFrame) -> None:
    md.h1("Ablation Summary")
    for metric in SUMMARY_METRICS:
        md.h3(metric)
        pivot = df.pivot_table(index=["dataset", "scenario"], columns="variant", values=metric)
        pivot = pivot[[c for c in VARIANT_CLASSES if c in pivot.columns]]
        md.table(_flatten_index(pivot), index_label="dataset / scenario")

    md.h2("Delta vs 'full' (variant − full)")
    full = df[df["variant"] == "full"].set_index(["dataset", "scenario"])
    for metric in SUMMARY_METRICS:
        deltas = []
        for variant in VARIANT_CLASSES:
            if variant == "full":
                continue
            other = df[df["variant"] == variant].set_index(["dataset", "scenario"])
            delta = (
                other[metric].astype(float) - full[metric].astype(float)
            ).rename(variant)
            deltas.append(delta)
        table = pd.concat(deltas, axis=1)
        md.h3(metric)
        md.table(_flatten_index(table.map(lambda v: f"{v:+.3f}")), index_label="dataset / scenario")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescale", action=argparse.BooleanOptionalAction,
                        default=RESCALE_BY_CALIBRATION_MEDIAN,
                        help="divide squared errors by the pooled calibration median (default: off)")
    args = parser.parse_args()
    if args.rescale:
        OUTPUT_CSV = "data/median_rescaled/ablation_study_results.csv"
    results = run_ablation(rescale=args.rescale)
    print_summary(results)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results.to_csv(OUTPUT_CSV, index=False)
    md.para(f"Saved full results to {OUTPUT_CSV}")
