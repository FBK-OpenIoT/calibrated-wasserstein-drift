# Residual Drift Detectors under a Common False-Alarm Budget

Code accompanying *Residual Drift Detectors under a Common False-Alarm Budget: A Calibrated Comparison with a Wasserstein Detector*, by P. Grazieschi, R. Silvestri, M. Vecchio, M. Antonini and F. Antonelli.

The code includes the detector, synthetic calibration and benchmark workflows, and the soil-moisture upstream-change case.

**Reproducing the paper** (details in [REPRODUCIBILITY.md](REPRODUCIBILITY.md)):

- **Synthetic experiments** (Sections 5–6, Appendices B–F): the workflows in `experiments/` produced the reported results, with the seeds and settings in `wasserstein_score/benchmark_protocol.py`.
- **Soil-moisture case** (Section 7, Appendix G, Figure 4): the paper's results come from Open-Meteo series retrieved on 5 November 2026 and released as `soil_moisture_inputs_2026-11-05.zip`. With those inputs, `experiments/soil_moisture.py` regenerates every value checked by `experiments/check_soil_moisture_paper.py`, and `experiments/plot_soil_moisture.py` draws Figure 4.

This repository is distributed under the [BSD 3-Clause licence](LICENSE).

## About Concept Drift

Concept drift refers to the phenomenon where the statistical properties of a data-generating process change over time, degrading the performance of models trained on historical data. The **Wasserstein distance** (Earth Mover's Distance) provides a statistically principled way to quantify distribution shifts between a reference window and a current window of a stream.

## Key Features

- **Wasserstein-based detector**: `WassersteinDriftDetector` uses a split-window comparison with an IQR-based adaptive threshold and hysteresis to suppress flicker after a detection: once an alarm is raised, no new alarm is raised until the drift score returns to zero (the distance is back inside Tukey's fence).
- **River-compatible interface**: the detector subclasses `river.base.DriftDetector`, so it drops into River pipelines alongside ADWIN/KSWIN/PageHinkley.
- **Automated calibration + evaluation** (`wasserstein_score.evaluation`): Bayesian hyperparameter search (Optuna TPE) against no-drift calibration streams, then evaluation with Wilson detection intervals, conditional and horizon-capped delays, and held-out null alarm audits.
- **Statistical significance testing** (`wasserstein_score.significance`): paired Wilcoxon signed-rank, exact McNemar, and Friedman + Nemenyi tests (with Holm correction) across detectors.
- **Synthetic scenario generators** (`wasserstein_score.helper_functions`): abrupt linear drift, Friedman F1↔F2 function switches, River's `FriedmanDrift`, and a mean/variance-preserving bimodal-shift residual scenario.

## Project Structure

The repository contains a reusable library, the research workflows behind the
paper, and their documentation.

```text
calibrated-wasserstein-drift/
├── wasserstein_score/                        # Public Python library
│   ├── __init__.py                           # Exports WassersteinDriftDetector
│   ├── wasserstein_drift_detector.py          # Split-window + IQR-threshold detector
│   ├── batch_detectors.py                     # KSWIN / Page-Hinkley that test only at batch ends
│   ├── ablation_variants.py                   # Detector variants used by the ablation study
│   ├── benchmark_protocol.py                  # Shared protocol constants (streams, budget, search spaces)
│   ├── evaluation.py                          # Calibration (Optuna) + test evaluation pipeline
│   ├── helper_functions.py                    # Synthetic stream generators, model training, print helpers
│   └── significance.py                        # Paired significance tests (Wilcoxon, McNemar, Friedman/Nemenyi)
│
├── experiments/                               # Reproducible research workflows
│   ├── run_all_tests.py                       # Full synthetic-scenario sweep + significance report
│   ├── severity_sweep.py                      # Detection rate/delay vs. bimodal drift severity
│   ├── calibration_robustness.py              # Detection-rate range across independent calibration draws
│   ├── ablation_study.py                      # Matched-budget ablation of hysteresis, bound and statistic
│   ├── sensitivity_analysis.py                # Sweeps around the calibrated operating point, with placebo rates
│   ├── batch_checking.py                      # Main benchmark with residuals arriving in batches
│   ├── soil_moisture.py                       # Real-data upstream-change case
│   ├── plot_soil_moisture.py                  # Draws the soil-case figure (paper Figure 4)
│   └── check_soil_moisture_paper.py           # Compares a soil run with the paper's printed values
│
├── notebooks/
│   └── synthetic_drift_scenarios.ipynb        # Interactive walkthrough of synthetic drift scenarios
│
├── tests/                                     # pytest suite
├── data/                                      # Generated outputs (created on demand, Gitignored)
│
├── REPRODUCIBILITY.md                         # How to reproduce each result of the paper
├── DATASETS.md                                # Data sources, retrieval and preprocessing
├── pyproject.toml                             # Project dependencies and metadata
└── README.md
```

Synthetic experiments generate inputs from fixed seeds. The soil-moisture case retrieves data from Open-Meteo and caches it locally; see [DATASETS.md](DATASETS.md). All experiment outputs are written under `data/`.

## Installation

This project uses `uv` for dependency management, but can also be installed with pip.

### Using uv (Recommended)
```bash
uv sync
```

For the evaluation pipeline and experiments:

```bash
uv sync --locked --extra research
```

For the soil-case figure, add the `figures` extra (matplotlib):

```bash
uv sync --locked --extra research --extra figures
```

For the curated notebook as well:

```bash
uv sync --extra research --extra notebook
```

### Using pip
```bash
pip install -e .

# Evaluation pipeline and experiment scripts
pip install -e ".[research]"
```

### Python Version
- **Supported**: Python 3.11–3.13
- **Development version**: Python 3.13 (see `.python-version`)

## Dependencies

The base package is intentionally small:

| Package | Purpose |
|---------|---------|
| **numpy / scipy** | Numerical operations and Wasserstein distance |
| **river** | Online ML detector interface |
| **typing-extensions** | Runtime compatibility required by River |

The `research` extra adds the direct dependencies of the evaluation pipeline
and experiment scripts: pandas, Optuna, joblib, scikit-learn, XGBoost, and
Plotly. The
`notebook` extra adds the interactive notebook and rendering stack.

## Quick Start

### 1. Basic Usage — Single Detector

The detector implements River's `DriftDetector` interface: feed it one value per `update()` call (typically a model residual or squared error) and check `.drift_detected` afterward.

```python
from wasserstein_score import WassersteinDriftDetector
import numpy as np

detector = WassersteinDriftDetector(window_size=500, threshold_drift_score=1.2)

stream = np.concatenate([
    np.random.normal(0, 1, 2000),    # baseline
    np.random.normal(2, 1.5, 2000),  # drifted
])

for x in stream:
    detector.update(x)
    if detector.drift_detected:
        print("Drift detected!")
```

### 2. Calibration + Evaluation Pipeline

```python
from wasserstein_score.evaluation import calibrate_and_test
from wasserstein_score.significance import significance_report

# calibration_streams: no-drift streams; test_streams: {scenario_name: [streams with a drift at `drift_at`]}
# null_streams: further no-drift streams, disjoint from calibration_streams
best_params, calibration_results, test_results, per_stream_results = calibrate_and_test(
    calibration_streams=calibration_streams,
    test_streams=test_streams,
    drift_at=10_000,
    algorithms=["wasserstein", "adwin", "kswin", "ph"],
    false_alarm_budget_per_1000=0.1,
    false_alarm_confidence=0.95,
    detection_horizon=1_000,
    null_streams=null_streams,  # optional held-out no-drift streams
)

print(test_results[["scenario", "algorithm", "detection_rate",
                    "detection_rate_ci_low", "detection_rate_ci_high",
                    "mean_first_delay", "mean_capped_delay"]])

significance = significance_report(per_stream_results, reference="wasserstein")
```

The synthetic experiment scripts use raw squared residuals by default. Pass
`--rescale` to divide them by the **median of all samples pooled across the
corresponding drift-free calibration streams** (`--no-rescale` disables it).
One scale is fitted per dataset/model and held fixed for calibration, every
test scenario and held-out null stream, and subsequent severity levels. Every
detector receives the same normalized signal. The helper
`wasserstein_score.normalization.CalibrationMedianNormalizer` rejects empty
calibration streams, non-finite or negative squared errors, and a non-positive
median. It does not center, clip, or adapt the scale online. Median division
does not bound the signal to `[0, 1]`, so it does not establish ADWIN's nominal
bounded-input guarantees. The main and batch benchmark outputs retain
`normalization_scale` in their calibration and result tables and record the
normalization method and dataset scales in the run metadata. Previous raw-loss
results require fresh calibration before comparison with normalized runs.
Raw and rescaled runs use separate default output directories (`data/` and
`data/median_rescaled/`). In the notebook, set
`rescale_by_calibration_median = True` to enable the same preprocessing.

The general-purpose `calibrate_and_test` and `evaluate_with_params` functions
consume the signal supplied by the caller without additional scaling. To use
the synthetic benchmark's preprocessing outside the experiment scripts:

```python
from wasserstein_score.normalization import CalibrationMedianNormalizer

normalizer = CalibrationMedianNormalizer.fit(calibration_streams)
calibration_streams = normalizer.transform(calibration_streams)
test_streams = {name: normalizer.transform(streams)
                for name, streams in test_streams.items()}
null_streams = normalizer.transform(null_streams)  # when provided
# Reuse this normalizer for future streams evaluated with the fitted parameters.
```

Calibration targets a common false-alarm budget using a one-sided Poisson
upper confidence bound computed from the no-drift exposure. This bound is exact
only under a Poisson count model for a fixed configuration. Overlapping windows,
hysteresis and selecting among many configurations can invalidate its nominal
coverage. `false_alarm_budget_met` records a calibration selection criterion;
it is not a post-selection false-alarm guarantee. Exposure counts only
the samples on which a detector is able to alarm: each calibration stream's
length minus that detector's warm-up (`3 * window_size` for Wasserstein,
`window_size` for KSWIN, River's `grace_period` for ADWIN and `min_instances`
for Page-Hinkley). Among trials whose
bound respects the budget, it selects the highest observed rate (the most
aggressive configuration that saturates the constraint), then prefers lower
between-stream variability and detector-specific structural aggressiveness. If
no trial is admissible, the least-violating upper bound is selected and
`false_alarm_budget_met` is `False` in `calibration_results`. Any selected
parameter within 2% of its search range is listed in `params_at_bound` and
reported as a warning, since the search space rather than the budget then sets
the operating point.

Such selections trigger range widening
(`calibrate_detectors_with_widening`, used by `calibrate_and_test`). A detector
is recalibrated over a wider range when its selection misses the budget, sits
at a search bound, or meets the budget without a single calibration alarm (the
budget then does not bind). The flagged end of the range is extended by the
current span (in log space for log-sampled parameters). A missed budget also
extends the end that lowers the alarm rate, and a zero-alarm selection the end
that raises it. Extensions stay within each parameter's domain
(`PARAMETER_DOMAINS`) and the warm-up limit, for at most
`CALIBRATION_WIDENING_ROUNDS` (4) extra rounds of `n_trials` trials. The final
selection applies the rule above to the trials of all rounds. The trial table's
`round` column records the round of each trial. Pass `widening_rounds=0` to
keep the given bounds.

**Wasserstein window rule.** The false-alarm budget pins the Wasserstein
threshold at every window but not the window itself. As stated in the paper
(Section 4.2), the window-selection rule and the search range were fixed before
any test stream was evaluated. The range is 500 to 1,000 samples; its upper
end was set by stream length, since the warm-up of about `3W` samples and the
exposure lost to it grow with `W`, rather than by detection results.
By default (`WASSERSTEIN_WINDOW_SELECTION = "largest_admissible"`),
the Wasserstein window is the largest one in the search range at which some
threshold meets the budget, tried
downward from the upper bound in steps of 50. Its threshold is the lowest one
meeting the budget, solved exactly from the drift-score trajectories
(`budget_matched_threshold`) over the threshold's whole domain, then confirmed
by running the detector. At a fixed alarm budget a longer window accumulates
more evidence against a persistent change, at the cost of delay.
The rule is deterministic, so `n_trials` and `optuna_seed` do not affect the
Wasserstein selection, and it is never widened:
the window sits at its upper bound by design. Pass
`wasserstein_window_selection="tpe"` to enable a joint Optuna search as an
alternative mode. Ablation variants use the same rule. The variant without
hysteresis counts one alarm per
step above the threshold rather than one per excursion (its `rearms_at_zero`
class attribute is `False`).

A test stream counts as detected only when an alarm falls within
`detection_horizon` samples of `drift_at` (1,000 in the benchmark); delays are
measured on those alarms (`mean_first_delay`, conditional on detection).
`mean_capped_delay` averages over **all** streams: an alarm in the horizon keeps
its delay, while a miss (including an alarm at or beyond the horizon boundary)
receives the full horizon. Lower is better. For example, a delay of 10 on one
of ten streams with a 1,000-sample horizon yields a conditional delay of 10 but
a capped delay of 901. `capped_delay` and `detection_horizon` are also retained
per stream for paired analysis. When `detection_horizon=None`, capped metrics
are unavailable (`NaN`), rather than using incomparable stream-specific caps.
Both delay means have percentile-bootstrap intervals over independent streams.

A few late detections can dominate a mean, so `median_first_delay` (with a
bootstrap interval) and `median_capped_delay` accompany the means, and the
Markdown reports lead with the median. A conditional delay is reported only when
at least `MIN_DETECTED_STREAMS_FOR_DELAY` (5) streams were detected and the
detection rate's Wilson lower bound exceeds the placebo rate
(`delay_reportable`). Otherwise `mean_first_delay`, `median_first_delay` and
their intervals are `NaN` (shown as n/a), because the "detections" are
indistinguishable from chance alarms. One detected stream out of 50 against a
placebo rate of 0.04 is an example. `conditional_delay_reportable` records the
decision, `mean_first_delay_raw` and `median_first_delay_raw` keep the unmasked
values, and the capped delay remains defined for every detector.

Detection and placebo probabilities use two-sided **Wilson score intervals**,
including when every stream has the same outcome: 50/50 gives approximately
`[0.92865, 1]`, and 0/50 gives `[0, 0.07135]` at 95% confidence.
`ci_level` controls these intervals; `n_bootstrap` controls only bootstrap
intervals. Both methods assume independent streams, not independent time steps.

The budget is tied to the horizon: under a homogeneous Poisson model, 0.1 false
alarms per 1,000 samples gives a 1,000-sample chance-detection probability of
`1 - exp(-0.1)`, about 9.5%. This is a model-based reference, not a guarantee
for these sequential detectors. When `null_streams` are
supplied, the same criterion applied to them at `drift_at` gives each detector's
`placebo_detection_rate`, and `excess_detection_rate` is the detection rate
above that chance level. `detection_rate_any` keeps the looser "any post-drift
alarm" criterion as a secondary column.

The same held-out null detector runs also audit **all post-warm-up alarms**.
Test-summary CSVs retain `null_alarm_counts` and `null_active_exposures` as JSON
arrays in input stream order, their pooled count/exposure and rate, a Poisson
working-model UCB, observed/model-based budget flags, and Pearson count
dispersion (values above one suggest overdispersion). `null_rate_ci_low/high`
resample whole independent streams and recompute the pooled count/exposure
ratio, preserving within-stream alarm dependence and unequal stream lengths.
With fewer than two streams or zero total alarms, these bootstrap bounds are
`NaN`: resampling zeros cannot establish an upper rate bound. The Poisson UCB
remains positive for zero alarms, but is still assumption-dependent.

`calibrate_and_test` additionally compares the selected calibration UCB with
the independent null rate estimate (`calibration_ucb_covers_null_rate_proxy`)
and flags bounds below the null bootstrap interval (`calibration_ucb_below_null_ci`).
For `evaluate_with_params`, pass `calibration_results` to obtain these checks.
The calibration-robustness experiment summarizes their fractions across
recalibrations. These are **coverage diagnostics against an estimated rate**,
not measured coverage against a known population rate or certification at 95%.
Null streams are shared across robustness replicates, so those fractions are
descriptive; unavailable intervals are excluded, not counted as passes.
Keep audit streams independent of calibration and never select parameters
using the audit. Increase the null sample and independent recalibrations when
investigating coverage; retaining the nominal Poisson bound alone is insufficient.

The canonical Wasserstein search space, the starting point of widening, is
`threshold_drift_score` in `[0, 4]` and `window_size` in `[500, 1000]`. KSWIN searches `window_size` in `[500, 1000]` too: every windowed detector is capped at `MAX_WINDOW_SIZE` (1,000, the detection horizon), and range widening never exceeds it. ADWIN's window is adaptive and Page-Hinkley has none, so the cap does not apply to them. The main benchmark, severity
sweep, and calibration-robustness experiment all import these bounds and the
shared stream/sample settings from `wasserstein_score.benchmark_protocol`.

Structural aggressiveness is only a final tie-breaker: lower threshold then
smaller window for Wasserstein under `"tpe"`, higher `delta` for ADWIN, higher `alpha` for
KSWIN, and lower threshold then lower `delta` for Page-Hinkley. These are
declared parameter-level proxies; detection rate and delay remain test-only
metrics and are never used during calibration.

Test-set false positives are compared on one common exposure window. Its first
sample is `starting_idx + max(detector warm-up)` and its last sample is the one
immediately before `drift_at`. The primary outputs are
`mean_false_positives_common`, `n_false_positives_common`, and
`false_positives_per_1000_common`; the paired Wilcoxon and Friedman tests use
the common raw counts. Detector-specific active-exposure rates remain available
through the `*_active` columns as supplementary diagnostics.

Scenarios of one dataset share its calibration and pre-drift test streams (the
three River Friedman scenarios, for instance). They are distinct drifts but not
independent replications. Their false-positive data are identical, so
`significance_report` tests them once under a label joining their names
(`shared_false_positive_groups`) instead of repeating one test per scenario.
Cross-scenario summaries average within each dataset first, so a dataset with
three scenarios weighs as much as one with a single scenario.

Report detection rates that depend on calibration as a range across
calibration seeds. The calibration-robustness experiment's report opens with
each scenario's detection-rate range (min-max, median, selected windows); a
single calibration's rate is one draw from it.

The sensitivity analysis ranks factors over budget-meeting configurations only.
Window and noise sweeps re-match the threshold to the budget at every value,
and the threshold row covers thresholds at or above the calibrated one.
Thresholds below it violate the budget and are shown as a trade-off curve,
not a sensitivity range. Noise is labelled as a data condition, not a detector
setting.

See `wasserstein_score/helper_functions.py` for the synthetic stream generators (`abrupt_linear_drift_stream`, `friedman_f1_f2_switch`, `river_friedman_drift_stream`, `bimodal_shift_streams`) that build `calibration_streams`/`test_streams` from a trained model's residuals, and `experiments/run_all_tests.py` for a complete end-to-end example.

## How Wasserstein Distance Works for Drift Detection

1. **Reference window**: the older portion of the sliding window.
2. **Current window**: the more recent portion of the sliding window.
3. **Compute distance**: Wasserstein(reference, current).
4. **Adaptive threshold**: compare the latest distance against the historical distribution of distances using an IQR-based bound rather than a fixed cutoff.
5. **Detect drift**: a distance that is an outlier relative to its own recent history signals drift.

## Running the synthetic experiments

Install the `research` extra, then run each script from the repository root:

```bash
uv sync --locked --extra research
```

| Paper section | Command |
|---------------|---------|
| Main results (coverage, speed, alarm behaviour), statistical comparisons, selected parameters | `uv run python experiments/run_all_tests.py` |
| Mixture-separation sweep | `uv run python experiments/severity_sweep.py` |
| Repeated calibration | `uv run python experiments/calibration_robustness.py` |
| Ablations at a matched false-alarm budget | `uv run python experiments/ablation_study.py` |
| Hyperparameter and noise sensitivity | `uv run python experiments/sensitivity_analysis.py` |
| Residuals arriving in batches | `uv run python experiments/batch_checking.py --all` |
| Soil-moisture upstream-change case | `uv run --extra research python -m experiments.soil_moisture` |

To reproduce the paper's synthetic numbers, add `--rescale` to the main
benchmark, mixture-separation sweep and batch commands. The paper's results
come from median-rescaled runs of all four detectors; Wasserstein and KSWIN
are unaffected by the rescaling. The raw default changes ADWIN and
Page-Hinkley. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md#synthetic-experiments).

The soil runner fits one fixed 2020 XGBoost forecaster per site and target,
calibrates all four detectors jointly across the four 2021–2022 streams, and
monitors 2023–30 December 2025 without resetting at the November 2024 upgrade.
It writes calibration trials, selected parameters, daily residuals, dated
alarms, pooled rates, episode scores and scoring-sensitivity tables under
`data/soil_moisture/forecast_run/`, with input hashes and provenance in `metadata.json`.
It works from a clean clone with no data files: missing IFS soil moisture,
eight observed weather variables, five forecast variables and the ERA5-Land
control are downloaded automatically for all four sites over 2020–2025.
Downloads are stored as CSVs, so the default run needs only the research extra.
The five forecast variables are temperature, relative humidity,
precipitation, wind speed and reference evapotranspiration (ET₀). It reuses
local values and automatically downloads missing values from Open-Meteo's
ECMWF historical-forecast archive. These are stitched short-lead forecasts,
averaged over the previous day as in the paper; they do not have a fixed
24-hour lead. Every forecaster uses 15 regressors. `--forecast-dir` supplies
local forecasts and sets the cache root; `--offline` prevents downloads, and
`--prepare-only` downloads/validates inputs without running calibration.
See the input contract in [`DATASETS.md`](DATASETS.md). `--no-control` explicitly skips the control and
episode scoring. All downloads are cached, interrupted downloads resume, and
temporary API rate limits are retried.

For a short execution check, add `--n-trials 8 --widening-rounds 0
--output-dir data/soil_moisture/forecast_smoke`. Paper defaults are 40 trials per round,
seed 42, and up to four widening rounds. Use a distinct output directory for
each configuration; results in the selected directory are overwritten.

To reproduce the paper's soil numbers exactly, extract the archived inputs
`soil_moisture_inputs_2026-11-05.zip` (a release asset) at the repository root
and run offline, so nothing is downloaded. Then compare with the values printed
in the paper and redraw Figure 4:

```bash
uv run --extra research python -m experiments.soil_moisture --offline
uv run python experiments/check_soil_moisture_paper.py --run-dir data/soil_moisture/forecast_run
uv run --extra research --extra figures python -m experiments.plot_soil_moisture --run-dir data/soil_moisture/forecast_run
```

The check exits with 0 when all 197 compared values agree and 1 otherwise.
With the archived inputs it exits with 0. A fresh download may differ if
Open-Meteo has revised its archives since 5 November 2026; see
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). `--budget` recalibrates the case at
another false-alarm budget per 1,000 days.

To recalibrate and evaluate only ADWIN and Page-Hinkley with median rescaling:

```bash
uv run python experiments/run_all_tests.py --algorithms adwin ph --rescale --output-dir data/adwin_ph_median
```

This runs all five synthetic dataset families at the canonical stream counts,
trial budget and seeds. `--datasets "Linear Abrupt" "Bimodal Shift"`, for
example, restricts the dataset families. The saved metadata records the actual
detector subset, normalization mode, dataset scales and significance reference
(ADWIN for an ADWIN/PH-only run).

Each script prints a Markdown report to standard output and writes its results
(CSVs, and HTML plots where applicable) to `data/`, which is created on demand
and is gitignored. The main and batch benchmarks additionally write
scenario-level and overall summaries, the selected calibration rows, and a JSON
provenance manifest containing the Git commit SHA, dirty-tree flag, protocol
configuration, and output inventory.

`batch_checking.py --batch-size B` runs a single batch size (1, 10, 50 or 100)
and writes to `data/batch_checking/batch_<B>/`, so the four runs can proceed
in parallel; `--summarize` then combines the completed runs. `--all` runs them
in sequence and summarizes.

`calibration_robustness.py` keeps the trained model and final test streams fixed
within each scenario while varying both the calibration-data draw and the
Optuna seed. It checkpoints after every replicate; pass `--resume` to continue
an interrupted full run.

The full experiments perform repeated calibration and evaluation across many
streams and are CPU intensive: the paper's results come from cluster jobs with
20 CPUs and a 20-hour limit per script. They use all CPUs available to the process (respecting
SLURM and cgroup affinity). Start with fewer streams and Optuna trials when
validating a new environment.

## Development checks

```bash
uv sync --extra research --group dev
uv run ruff check .
uv run pytest --cov=wasserstein_score
uv build
uvx twine check dist/*
```

Continuous integration runs the tests on Linux and Windows with Python
3.11–3.13 and verifies that the built wheel can be installed in isolation.

## Citation

If you use this software, please cite the accompanying paper. Citation metadata
for the software itself is in [`CITATION.cff`](CITATION.cff); GitHub shows it
under "Cite this repository".

## Project documentation

- [Reproducing the paper](REPRODUCIBILITY.md)
- [Dataset provenance](DATASETS.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [Citation metadata](CITATION.cff)
- [Security policy](SECURITY.md)

## Known Limitations

- Requires a minimum window of samples (window-size-dependent) before the detector can produce its first drift test — see the warm-up handling in `wasserstein_score/evaluation.py`.
- `WassersteinDriftDetector`'s IQR-based bound (and the ablation `ZScoreBoundDetector` variant) never flags drift while the historical window has zero spread, since the bound requires `IQR > 0` / `sigma > 0`.
- Performance in residual-based scenarios depends on the quality of the model producing those residuals.

## References

- **Wasserstein Distance**: [Optimal Transport Theory](https://en.wikipedia.org/wiki/Wasserstein_distance)
- **Concept Drift**: [Gama et al., 2014](https://link.springer.com/article/10.1145/2523813)
- **River ML**: [Online Machine Learning](https://riverml.xyz/)
- **ADWIN**: [Bifet & Gavaldà, 2007](https://dl.acm.org/doi/10.1145/1296243.1296277)
- **Friedman & Nemenyi comparison methodology**: Demšar (2006), JMLR

## License

This project is licensed under the [BSD 3-Clause License](LICENSE).
