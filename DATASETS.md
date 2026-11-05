# Dataset provenance

## Synthetic scenarios

All synthetic experiments generate their inputs locally from fixed random seeds, using
NumPy generators and River's synthetic Friedman datasets. No dataset is
downloaded and no generated stream is committed to the repository. The
generators and the benchmark datasets built from them (`benchmark_datasets`)
are in `wasserstein_score/helper_functions.py`; the shared stream geometry,
search spaces and seeds are in `wasserstein_score/benchmark_protocol.py`.

## Local data

Generated results belong under `data/`, which is excluded from Git. Any new
real-world dataset must have its canonical source, retrieval procedure,
redistribution terms, preprocessing, and evaluation convention documented here
before its workflow is published.

## Real soil-moisture upstream-change case

`experiments/soil_moisture.py` implements the paper's real upstream-change
case (Section 7), with no injected perturbations. The source is the
[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
and its [Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api).
Soil moisture is a gridded model estimate at 0–7 cm, in m³/m³, rather than a
field measurement. The intervention date is 12 November 2024, the
[ECMWF IFS Cycle 49r1 upgrade](https://confluence.ecmwf.int/display/FCST/Implementation+of+IFS+Cycle+49r1).
Open-Meteo data are provided under [CC BY 4.0](https://open-meteo.com/en/licence),
with attribution to Open-Meteo and the underlying ECMWF/Copernicus products.

| Site | Latitude | Longitude |
|------|----------|-----------|
| A | 42.9415 | 11.985 |
| B | 44.7153 | 7.7105 |
| C | 45.2835 | 8.287 |
| D | 46.245002 | 11.164803 |

### Observations and control

For each site, the runner downloads IFS soil moisture and eight observed
weather variables from the historical archive with `models=ecmwf_ifs`, the
coordinates above and UTC hourly data for 2020–2025. It saves `A.csv` through
`D.csv` under `--data-dir` (default `data/tests`), caches the raw yearly JSON
responses, and writes `.provenance.json` sidecars with the request URLs,
retrieval dates and raw-file hashes. Existing files are reused. Each
observation file contains `datetime`, `soil_moisture_0_to_7cm`, and these eight
weather columns:

```text
temperature_2m, relative_humidity_2m, precipitation,
et0_fao_evapotranspiration, wind_speed_10m, soil_temperature_0_to_7cm,
direct_radiation, shortwave_radiation
```

Inputs must cover every UTC hour from 1 January 2020 through 31 December 2025,
without duplicates, gaps or missing required values. Numeric columns use the
archive's default units (°C, %, mm, km/h and W/m² as applicable).

The ERA5-Land control is retrieved in the same way with `models=era5_land` and
`hourly=soil_moisture_0_to_7cm`, and saved under `--control-dir` (default
`data/soil_moisture/control`). Control CSVs can also be supplied manually and
need only `datetime` and the target column. `--no-control` disables the control
and all episode scoring; otherwise the control must cover all four sites.

### Forecast variables

Every forecaster uses five forecasted-weather variables:

```text
forecasted_temperature_2m
forecasted_relative_humidity_2m
forecasted_precipitation
forecasted_wind_speed_10m
forecasted_et0_fao_evapotranspiration
```

They are downloaded from `https://historical-forecast-api.open-meteo.com/v1/forecast`
with `models=ecmwf_ifs`, the site's coordinates, `timezone=UTC`, Celsius
temperatures, km/h wind and mm precipitation, one request per year. Validated
CSVs and their `.provenance.json` sidecars are saved under
`data/soil_moisture/forecasts/ecmwf_ifs/` (or `<forecast-dir>/ecmwf_ifs/`).
Local forecast files with these columns at hourly UTC forecast valid time can
be supplied as `A.csv` through `D.csv` under `--forecast-dir`; downloads fill
any gaps.

The Historical Forecast API joins short-lead ECMWF IFS forecasts into a
continuous series by valid time. Its values are averaged over UTC day d−1, like
the other weather variables, to predict soil moisture on day d (Section 7.2).
Missing or non-finite forecast values cause an error before model training.

### Preprocessing and evaluation

Hourly values are averaged over complete UTC calendar days. The outcome on
day d uses soil moisture on d−1 and d−2 and all weather means on d−1, giving 15
regressors. Separate XGBoost models for each site and target are fitted on all
366 days of 2020, with 100 trees, depth 5, learning rate 0.05, squared-error
objective and seed 42. Missing pre-2020 lags in the first two training rows are
left to XGBoost's missing-value handling. Models remain fixed. The ERA5-Land
control changes the target and target lags while keeping the same weather
regressors; its models and detector calibration are fitted separately.

The four 730-day 2021–2022 squared-error streams are pooled to fit one median
scale per target, fixed for every detector and all monitoring days. Calibration
uses the shared 95% Poisson working-model bound with a budget of 2.38 alarms per
1,000 days. Wasserstein takes the largest admissible window and solves its
threshold; the baselines use Optuna TPE (40 trials, 10 startup draws, seed 42)
with up to four widening rounds. Window bounds stay at 21–42 days in every
round. KSWIN initially searches statistic sizes 5–21 (capped at half its sampled
window) and excludes any draw with alpha below `2 / comb(2m, m)`, the smallest
attainable exact KS p-value, since such configurations can never alarm.
Excluded draws use up a trial. Other initial bounds and the widening and
selection rules come from the shared benchmark protocol.

Monitoring spans outcome dates 1 January 2023–30 December 2025 (1,095 days):
681 days before the upgrade and 414 afterwards. Each site has its own detector
from the start of monitoring, which runs through the upgrade without a reset.
Rate denominators exclude the protocol's initial warm-up (3W for Wasserstein,
W for KSWIN, 10 days for ADWIN and 30 for Page-Hinkley), while counts keep all
emitted alarms, as in calibration.

Episodes use the trailing 30-day mean of daily IFS minus ERA5-Land, below its
site-specific 1st percentile over 1 January 2021–11 November 2024 (linear
interpolation). After the upgrade, a new onset requires more than 30 days since
the preceding below-threshold day. Primary alarm matching covers onset through
onset + 42 days and is clipped at the end of monitoring. Counts inside matching
windows use their union. The runner also scores control alarms against the
same episodes as a placebo, writes the 15/30/60-day × 0.5/1/2-percentile
definition grid and the 0/15/30-day lead × 14/42/90-day lag matching grid, and
reports the one-sided binomial and two-sided Fisher count comparisons.

### Running and outputs

```bash
uv run --extra research python -m experiments.soil_moisture
```

This works from a clean clone with no `data/` directory and downloads all
three input products for the four sites. Add `--offline` to prohibit network
access; a missing input then causes an error. `--prepare-only` downloads and
validates the inputs without training or calibration. `--budget` changes the
false-alarm budget per 1,000 days (default 2.38). For a quick run use
`--n-trials 8 --widening-rounds 0 --output-dir data/soil_moisture/forecast_smoke`.
Downloads are cached per site and year, so interrupted runs resume; temporary
HTTP 429 and server errors are retried.

The paper's soil results use the inputs retrieved on 5 November 2026, archived
as the release asset `soil_moisture_inputs_2026-11-05.zip`. Extract it at the
repository root, which places every file at its default path under `data/`, and
add `--offline` so the run uses exactly those files. The asset's `SHA256SUMS`
lists every file's hash. Fresh downloads can differ if Open-Meteo revises its
archives. Figure 4 is drawn from a completed run with

```bash
uv run --extra research --extra figures python -m experiments.plot_soil_moisture \
    --run-dir data/soil_moisture/forecast_run
```

Outputs include per-target calibration trials and selections, parameter JSON,
model files, per-site daily residuals, `alarms.csv`, `site_summary.csv`, pooled
`summary.csv`, target-gap summaries and time series, episode definitions and
matching results, and both sensitivity tables. `metadata.json` records input
SHA-256 hashes, package versions, model and calibration settings, feature names
and target-specific scales. Results in the selected output directory are
overwritten; use a different `--output-dir` for each configuration.
`experiments/check_soil_moisture_paper.py` compares a completed run with the
values printed in the paper (see [REPRODUCIBILITY.md](REPRODUCIBILITY.md#soil-moisture-case)).
