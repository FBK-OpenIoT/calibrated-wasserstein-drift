"""Severity sweep for the mean/variance-preserving bimodal scenario.

Turns the single fixed-severity comparison in run_all_tests.py
(separation=0.99) into a sweep over effect size, so we can see
detection_rate/mean_first_delay as a function of
how subtle the drift is, instead of asserting an advantage at one point.

Calibration is done once per scenario family, not once per severity level:
The generator only uses its severity parameter (separation) in the post-drift
branch of the *test* stream generator — the training data, the trained model, and the
calibration streams are built from code paths that never reference it, so
they are identical across the whole severity grid for a given seed_base.
Calibrating fresh at every grid point would just rerun Optuna on identical
inputs. We calibrate and evaluate at the main benchmark's reference severity
via the normal calibrate_and_test call, then reuse those best_params for the
remaining grid points through the exact same shared test-evaluation path.
"""

import argparse
import os

import pandas as pd
import plotly.express as px

from wasserstein_score.normalization import CalibrationMedianNormalizer
from wasserstein_score.benchmark_protocol import (
    ADWIN_BOUNDS,
    BIMODAL_SHIFT_SEPARATION,
    BOOTSTRAP_SEED,
    CI_LEVEL,
    DETECTION_HORIZON,
    DRIFT_AT,
    FALSE_ALARM_BUDGET_PER_1000,
    FALSE_ALARM_CONFIDENCE,
    KSWIN_BOUNDS,
    N_BOOTSTRAP,
    N_CALIBRATION_STREAMS,
    N_OPTUNA_TRIALS,
    N_TEST_STREAMS,
    NUM_DATAPOINTS,
    NUM_MODEL_TEST_POINTS,
    NUM_TRAINING_POINTS,
    OPTUNA_SEED,
    PAGE_HINKLEY_BOUNDS,
    RUNNING_ALGORITHMS,
    RESCALE_BY_CALIBRATION_MEDIAN,
    STARTING_IDX,
    WASSERSTEIN_BOUNDS,
)
from wasserstein_score.evaluation import (
    calibrate_and_test,
    evaluate_with_params,
)
from wasserstein_score import helper_functions as md
from wasserstein_score.helper_functions import (
    get_model, build_model_df,
    bimodal_shift_streams,
    held_out_null_streams,
)

pd.options.plotting.backend = "plotly"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = "data"

# Grids run from the clearly-detectable end down toward each scenario's floor.
SEVERITY_GRIDS = {
    "bimodal_shift": {
        "generator": bimodal_shift_streams,
        "param": "separation",
        "grid": [BIMODAL_SHIFT_SEPARATION, 0.95, 0.85, 0.7, 0.55, 0.4, 0.25],
        "reference_value": BIMODAL_SHIFT_SEPARATION,
        "fixed_kwargs": dict(base_sigma=1.0),
        "seed_base": 2000,
    },
}

# ---------------------------------------------------------------------------
# Sweep one scenario family
# ---------------------------------------------------------------------------

def build_residuals(model, X_data, y_data):
    return [(model.predict(X_data[i]) - y_data[i]) ** 2 for i in range(len(X_data))]


def run_severity_sweep(scenario_name: str, config: dict,
                       rescale: bool = RESCALE_BY_CALIBRATION_MEDIAN) -> pd.DataFrame:
    generator = config["generator"]
    param = config["param"]
    grid = config["grid"]
    base_kwargs = dict(
        n=NUM_DATAPOINTS, drift_at=DRIFT_AT,
        n_test_streams=N_TEST_STREAMS, n_calibrations_streams=N_CALIBRATION_STREAMS,
        seed_base=config["seed_base"], **config["fixed_kwargs"],
    )

    reference_value = config["reference_value"]
    if reference_value not in grid:
        raise ValueError(
            f"{scenario_name}: benchmark reference {param}={reference_value} is missing "
            "from the severity grid"
        )
    md.h2(f"{scenario_name}: calibrating at {param}={reference_value}")
    kwargs = dict(base_kwargs, **{param: reference_value})
    X_training, y_training, X_calib, y_calib, X_test, y_test = generator(**kwargs)
    n_features = X_training.shape[1]
    model = get_model(
        build_model_df(X_training, y_training),
        NUM_TRAINING_POINTS, NUM_MODEL_TEST_POINTS, range(n_features), 'target',
    )

    calib_residuals = build_residuals(model, X_calib, y_calib)
    test_residuals = {scenario_name: build_residuals(model, X_test[scenario_name], y_test[scenario_name])}
    # The severity parameter only enters the post-drift branch, so one set of
    # drift-free streams gives the placebo detection rate for the whole grid.
    X_null, y_null = held_out_null_streams(generator, N_TEST_STREAMS, **kwargs)
    null_residuals = build_residuals(model, X_null, y_null)

    normalizer = CalibrationMedianNormalizer.fit(calib_residuals, enabled=rescale)
    calib_residuals = normalizer.transform(calib_residuals)
    test_residuals = {
        name: normalizer.transform(streams) for name, streams in test_residuals.items()
    }
    null_residuals = normalizer.transform(null_residuals)

    best_params, _, test_df, _ = calibrate_and_test(
        calibration_streams=calib_residuals,
        test_streams=test_residuals,
        drift_at=DRIFT_AT,
        starting_idx=STARTING_IDX,
        algorithms=RUNNING_ALGORITHMS,
        wasserstein_bounds=WASSERSTEIN_BOUNDS,
        adwin_bounds=ADWIN_BOUNDS,
        kswin_bounds=KSWIN_BOUNDS,
        ph_bounds=PAGE_HINKLEY_BOUNDS,
        n_trials=N_OPTUNA_TRIALS,
        optuna_seed=OPTUNA_SEED,
        n_bootstrap=N_BOOTSTRAP,
        ci_level=CI_LEVEL,
        bootstrap_seed=BOOTSTRAP_SEED,
        false_alarm_budget_per_1000=FALSE_ALARM_BUDGET_PER_1000,
        false_alarm_confidence=FALSE_ALARM_CONFIDENCE,
        detection_horizon=DETECTION_HORIZON,
        null_streams=null_residuals,
        verbose=False,
    )
    # Frozen parameters and identical null distribution: reuse the entire
    # audit, including intervals and counts, rather than rerunning detectors.
    placebo = test_df.set_index("algorithm")[[
        c for c in test_df if c.startswith(("placebo_", "null_", "calibration_ucb_"))
    ]]
    test_df["severity_param"] = param
    test_df["severity_value"] = reference_value
    test_df["normalization_scale"] = normalizer.scale
    test_df["signal_normalization"] = normalizer.method
    rows = [test_df]

    for value in grid:
        if value == reference_value:
            continue
        md.bullet(f"{scenario_name}: evaluating {param}={value} (reusing calibration)")
        kwargs = dict(base_kwargs, **{param: value})
        _, _, _, _, X_test, y_test = generator(**kwargs)
        test_residuals = {scenario_name: normalizer.transform(
            build_residuals(model, X_test[scenario_name], y_test[scenario_name])
        )}

        point_df, _ = evaluate_with_params(
            test_streams=test_residuals,
            best_params=best_params,
            drift_at=DRIFT_AT,
            starting_idx=STARTING_IDX,
            algorithms=RUNNING_ALGORITHMS,
            n_bootstrap=N_BOOTSTRAP,
            ci_level=CI_LEVEL,
            bootstrap_seed=BOOTSTRAP_SEED,
            detection_horizon=DETECTION_HORIZON,
            verbose=False,
        )
        for column in placebo.columns:
            point_df[column] = point_df["algorithm"].map(placebo[column])
        point_df["excess_detection_rate"] = (
            point_df["detection_rate"] - point_df["placebo_detection_rate"]
        )
        point_df["severity_param"] = param
        point_df["severity_value"] = value
        point_df["normalization_scale"] = normalizer.scale
        point_df["signal_normalization"] = normalizer.method
        rows.append(point_df)

    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plots(df: pd.DataFrame) -> None:
    for metric in ["detection_rate", "mean_first_delay", "mean_capped_delay"]:
        lo_col, hi_col = f"{metric}_ci_low", f"{metric}_ci_high"
        sub = df.dropna(subset=[metric]).sort_values("severity_value")
        fig = px.line(
            sub, x="severity_value", y=metric, color="algorithm", facet_col="scenario",
            error_y=(sub[hi_col] - sub[metric]).clip(lower=0),
            error_y_minus=(sub[metric] - sub[lo_col]).clip(lower=0),
            markers=True,
            title=(f"{metric} vs. severity ({CI_LEVEL:.0%} "
                   f"{'Wilson' if metric == 'detection_rate' else 'bootstrap'} CI)"),
            hover_data=["detection_rate", "mean_first_delay", "mean_capped_delay"],
        )
        fig.update_xaxes(matches=None, showticklabels=True)
        fig.update_yaxes(matches=None)
        fig.write_html(os.path.join(OUTPUT_DIR, f"severity_sweep_{metric}.html"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescale", action=argparse.BooleanOptionalAction,
                        default=RESCALE_BY_CALIBRATION_MEDIAN,
                        help="divide squared errors by the pooled calibration median (default: off)")
    args = parser.parse_args()
    if args.rescale:
        OUTPUT_DIR = os.path.join(OUTPUT_DIR, "median_rescaled")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_rows = [run_severity_sweep(name, cfg, rescale=args.rescale)
                for name, cfg in SEVERITY_GRIDS.items()]
    sweep_df = pd.concat(all_rows, ignore_index=True)

    md.h1("Severity sweep summary")
    for scenario_name in SEVERITY_GRIDS:
        sub = sweep_df[sweep_df["scenario"] == scenario_name]
        for metric in ("detection_rate", "excess_detection_rate", "mean_capped_delay"):
            md.h3(f"{scenario_name}: {metric}")
            md.table(sub.pivot(index="severity_value", columns="algorithm", values=metric))

    out_csv = os.path.join(OUTPUT_DIR, "severity_sweep_results.csv")
    sweep_df.to_csv(out_csv, index=False)
    md.para(f"Saved results to {out_csv}")

    try:
        save_plots(sweep_df)
        md.para(f"Saved plots to {OUTPUT_DIR}/severity_sweep_detection_rate.html and "
                f"{OUTPUT_DIR}/severity_sweep_mean_first_delay.html")
    except Exception as exc:
        md.para(f"Skipped plot generation ({exc})")
