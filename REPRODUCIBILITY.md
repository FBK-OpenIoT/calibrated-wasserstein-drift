# Reproducibility

This document describes how to reproduce the results of the accompanying
paper, *Residual Drift Detectors under a Common False-Alarm Budget: A
Calibrated Comparison with a Wasserstein Detector*. Table, figure and section
numbers refer to the paper.

## Summary

| Paper results | Workflow |
|---------------|----------|
| Sections 5 and 6, Tables 1–6, Figures 2–3, Appendices B–F | `experiments/run_all_tests.py`, `severity_sweep.py`, `calibration_robustness.py`, `ablation_study.py`, `sensitivity_analysis.py`, `batch_checking.py` |
| Section 7, Tables 7–9, Table G.1, Figure 4 | `experiments/soil_moisture.py`, `plot_soil_moisture.py`, checked by `check_soil_moisture_paper.py` |
| Figure 1 | Pipeline diagram; no code involved |

## Synthetic experiments

The synthetic results in Sections 5 and 6 and Appendices B–F were produced by
the workflows in `experiments/`, with the stream counts, search spaces, trial
budgets and seeds fixed in `wasserstein_score/benchmark_protocol.py`. Every
input is generated from those seeds, so no data files are needed. The paper's
runs used cluster jobs with 20 CPUs and up to 20 hours per script.

**Use `--rescale`.** The paper's numbers come from runs in which all four
detectors read squared errors divided by the calibration median. The
Wasserstein score is invariant to this rescaling and KSWIN is rank-based, so
their results equal those on raw squared errors (Section 4.1); ADWIN and
Page-Hinkley need the rescaled signal:

```bash
uv run python experiments/run_all_tests.py --rescale
uv run python experiments/severity_sweep.py --rescale
uv run python experiments/batch_checking.py --all --rescale
```

Without `--rescale`, ADWIN and Page-Hinkley read raw squared errors and their
results differ from the paper's. Use runs that include Wasserstein for the
paper's tables: the common false-positive window starts after the largest
warm-up among the detectors in a run, and only with Wasserstein is it the
paper's [3000, 15000).

`severity_sweep.py` writes interactive HTML plots of the quantities in
Figure 3; Figure 2 is drawn from the CSV outputs of `run_all_tests.py`.

## Soil-moisture case

### Inputs

The soil results come from Open-Meteo series retrieved in UTC on 5 November
2026 for the four sites: IFS soil moisture and eight observed weather
variables, five ECMWF IFS forecast variables, and the ERA5-Land control (see
[DATASETS.md](DATASETS.md)). Because Open-Meteo can revise its archives, those
files are released with the code:

| Asset | Contents | SHA-256 |
|-------|----------|---------|
| `soil_moisture_inputs_2026-11-05.zip` (10.2 MB) | The 96 input files (CSVs, provenance sidecars, raw yearly API responses) at their paths under `data/`, with a `SHA256SUMS` manifest | `9ef41110c90acb7342a80eeec62bec9702d212340ac00314aa2cc9a51a6f6d8c` |
| `soil_moisture_outputs_2026-11-05.zip` (1.2 MB) | The outputs of the paper's run under `data/soil_moisture/paper_run/`, including `metadata.json` with input hashes, package versions and settings | `119e4b1e97e3feb587952847388a6be5570b7aba015450817887ee7620646dea` |

### Reproducing the paper's numbers

From the repository root, after extracting the inputs asset there:

```bash
uv run --extra research python -m experiments.soil_moisture --offline
uv run python experiments/check_soil_moisture_paper.py --run-dir data/soil_moisture/forecast_run \
    --output data/soil_moisture/forecast_run/paper_comparison.csv
uv run --extra research --extra figures python -m experiments.plot_soil_moisture \
    --run-dir data/soil_moisture/forecast_run
```

The run takes a few minutes on a workstation and is deterministic given its
inputs. `check_soil_moisture_paper.py` compares 197 values from Tables 7–9,
Section 7.3, Table G.1 and Appendix G with the run's outputs and exits with 0
when all agree. With the archived inputs and the locked dependencies, all 197
agree and the outputs are identical to those in the outputs asset.
`plot_soil_moisture.py` writes Figure 4 to
`data/soil_moisture/soil_upstream_change.pdf`.

Running without the inputs asset downloads fresh series, which can differ from
the archived ones if Open-Meteo has revised its products.

Section 7.2 states that, without the exclusion of unattainable significance
levels, the control KSWIN search would select a configuration that can never
alarm. This follows from calibrating the archived control residuals with
`kswin_require_attainable_alpha=False` in
`wasserstein_score.evaluation.calibrate_detectors_with_widening`: the selection
is m = 5 with α ≈ 0.0061, below the smallest attainable p-value
2/C(10, 5) ≈ 0.0079.
