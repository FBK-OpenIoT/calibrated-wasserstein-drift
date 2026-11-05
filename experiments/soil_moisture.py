"""Run the real upstream-change soil case (paper Section 'Real-World Soil-Moisture Case').

From the repository root: uv run --extra research python -m experiments.soil_moisture
Missing inputs are downloaded from Open-Meteo, including observations, all five
forecast variables and the ERA5-Land control. No local dataset is required.
See DATASETS.md for the input contract.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact
from xgboost import XGBRegressor

from wasserstein_score.benchmark_protocol import (
    ADWIN_BOUNDS, CALIBRATION_WIDENING_ROUNDS, FALSE_ALARM_CONFIDENCE,
    N_OPTUNA_TRIALS, OPTUNA_SEED, PAGE_HINKLEY_BOUNDS,
)
from wasserstein_score.evaluation import (
    _warmup_length,
    calibrate_detectors_with_widening, run_detector,
)
from wasserstein_score.normalization import CalibrationMedianNormalizer


SITES = {
    "A": ("42dot9415_11dot985", 42.9415, 11.985),
    "B": ("44dot7153_7dot7105", 44.7153, 7.7105),
    "C": ("45dot2835_8dot287", 45.2835, 8.287),
    "D": ("46dot245002_11dot164803", 46.245002, 11.164803),
}
TARGET = "soil_moisture_0_to_7cm"
WEATHER = (
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "et0_fao_evapotranspiration", "wind_speed_10m", "soil_temperature_0_to_7cm",
    "direct_radiation", "shortwave_radiation",
)
FORECAST_VARIABLES = (
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "wind_speed_10m", "et0_fao_evapotranspiration",
)
FORECAST_COLUMNS = tuple(f"forecasted_{name}" for name in FORECAST_VARIABLES)
FORECAST_ARCHIVE = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FORECAST_MODEL = "ecmwf_ifs"
FORECAST_DIR = Path("data/soil_moisture/forecasts")
ALGORITHMS = ("wasserstein", "kswin", "adwin", "ph")
START = pd.Timestamp("2020-01-01")
END = pd.Timestamp("2025-12-31")
MONITOR_START = pd.Timestamp("2023-01-01")
MONITOR_END = pd.Timestamp("2025-12-30")
UPGRADE = pd.Timestamp("2024-11-12")
BUDGET = 2.38
HORIZON = 42
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def request_json(url: str) -> dict:
    """Retry transient API limits; yearly caches make interrupted runs resumable."""
    for attempt in range(6):
        try:
            with urlopen(url, timeout=60) as response:
                return json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            try:
                reason = json.loads(detail).get("reason", detail)
            except ValueError:
                reason = detail
            try:
                delay = float(exc.headers.get("Retry-After", 60 if exc.code == 429 else 2 ** attempt))
            except (TypeError, ValueError):
                delay = 60 if exc.code == 429 else 2 ** attempt
            long_limit = exc.code == 429 and any(word in reason.lower() for word in ("daily", "hourly"))
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 5 or long_limit or delay > 120:
                raise ValueError(f"Could not fetch {url}: HTTP {exc.code}: {reason}. "
                                 "Rerun to resume cached years.") from exc
            delay = max(delay, 1.)
            print(f"Open-Meteo HTTP {exc.code}; retrying in {delay:g}s "
                  f"(attempt {attempt + 2}/6)", flush=True)
            while delay > 0:
                pause = min(delay, 30.)
                time.sleep(pause)
                delay -= pause
        except (URLError, TimeoutError) as exc:
            raise ValueError(f"Could not fetch {url}: {exc}. Rerun to resume cached years.") from exc
    raise AssertionError("Request retry loop exhausted")


def read_hourly(path: Path, *, allow_gaps: bool = False) -> pd.DataFrame:
    """Read an hourly UTC CSV or Parquet snapshot; reject gaps and duplicates."""
    if path.suffix == ".parquet":
        try:
            frame = pd.read_parquet(path)
        except ImportError as exc:
            raise ValueError("Parquet inputs need pyarrow; install it or provide CSV inputs") from exc
    else:
        frame = pd.read_csv(path)
    if "datetime" not in frame:
        raise ValueError(f"{path}: missing datetime column")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("datetime"), utc=True)).tz_localize(None)
    frame = frame.sort_index().loc[START:END + pd.Timedelta(hours=23)]
    expected = pd.date_range(START, END + pd.Timedelta(hours=23), freq="h")
    if allow_gaps:
        if not frame.index.is_unique or not frame.index.equals(frame.index.floor("h")):
            raise ValueError(f"{path}: forecast timestamps must be unique UTC hours")
        return frame.reindex(expected)
    if not frame.index.equals(expected):
        raise ValueError(f"{path}: expected unique, contiguous hourly UTC rows for 2020-2025")
    return frame


def find_input(root: Path, site: str, *, required: bool = True) -> Path | None:
    name = SITES[site][0]
    candidates = (root / name / "all_data.parquet", root / name / "all_data.csv",
                  root / f"{site}.csv", root / f"{site}.parquet")
    for path in candidates:
        if path.is_file():
            return path
    if required:
        raise ValueError(f"Missing {site} input under {root}; expected {site}.csv or {name}/all_data.*")
    return None


def download_archive(site: str, root: Path, *, model: str, variables: tuple[str, ...],
                     offline: bool = False) -> Path:
    """Retrieve complete hourly observations or a control from a clean checkout."""
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{site}.csv"
    if destination.exists():
        frame = read_hourly(destination)
        daily_means(frame, list(variables))
        return destination
    _, latitude, longitude = SITES[site]
    frames, requests = [], []
    for year in range(START.year, END.year + 1):
        cache = root / f"{site}_{model}_{year}.json"
        query = dict(latitude=latitude, longitude=longitude,
                     start_date=str(max(START, pd.Timestamp(year, 1, 1)).date()),
                     end_date=str(min(END, pd.Timestamp(year, 12, 31)).date()),
                     hourly=",".join(variables), models=model, timezone="UTC")
        url = ARCHIVE + "?" + urlencode(query)
        if cache.exists():
            record = json.loads(cache.read_text(encoding="utf-8"))
            if record.get("url") != url:
                raise ValueError(f"{cache}: cached archive request differs from this run")
        else:
            if offline:
                raise ValueError(f"Missing {model} inputs for site {site}, {year}; "
                                 "rerun without --offline to download from Open-Meteo")
            print(f"Downloading {model} archive: site {site}, {year}", flush=True)
            payload = request_json(url)
            record = dict(url=url, retrieved_at=utc_now(), response=payload)
            cache.write_text(json.dumps(record), encoding="utf-8")
        payload = record["response"]
        if not {"time", *variables}.issubset(payload.get("hourly", {})):
            raise ValueError(f"{cache}: Open-Meteo response lacks required archive variables")
        frame = pd.DataFrame(payload["hourly"])[["time", *variables]]
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("time"), utc=True)).tz_localize(None)
        expected = pd.date_range(query["start_date"],
                                 pd.Timestamp(query["end_date"]) + pd.Timedelta(hours=23), freq="h")
        if not frame.index.equals(expected):
            raise ValueError(f"{cache}: incomplete or unordered archive timestamps")
        daily_means(frame, list(variables))
        frames.append(frame)
        requests.append({**input_record(cache), "url": url, "retrieved_at": record["retrieved_at"]})
    provenance = dict(provider="Open-Meteo", endpoint=ARCHIVE, model=model,
                      product="historical_weather_archive", time_axis="valid_time_utc",
                      variables=list(variables), requests=requests)
    destination.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    pd.concat(frames).to_csv(destination, index_label="datetime")
    return destination


def download_observations(site: str, root: Path, *, offline: bool = False) -> Path:
    return download_archive(site, root, model="ecmwf_ifs",
                            variables=(TARGET, *WEATHER), offline=offline)


def download_control(site: str, root: Path, *, offline: bool = False) -> Path:
    return download_archive(site, root, model="era5_land", variables=(TARGET,), offline=offline)


def record_source(path: Path, inputs: list) -> dict:
    """Retain request provenance when supplied with a downloaded snapshot."""
    inputs.append(input_record(path))
    sidecar = path.with_suffix(".provenance.json")
    if sidecar.is_file():
        inputs.append(input_record(sidecar))
        return json.loads(sidecar.read_text(encoding="utf-8"))
    return dict(path=str(path), product="local_snapshot", original_request_metadata="unavailable")


def forecast_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the five forecast variables, accepting local next-period column aliases."""
    selected = pd.DataFrame(index=frame.index)
    for variable, canonical in zip(FORECAST_VARIABLES, FORECAST_COLUMNS):
        aliases = [canonical, f"{canonical}_next_1period"]
        if variable == "temperature_2m":
            aliases.append(f"{canonical}_mean_next_1period")
        series = pd.Series(np.nan, index=frame.index, dtype=float)
        for name in aliases:
            if name in frame:
                series = series.fillna(pd.to_numeric(frame[name], errors="raise").astype(float))
        selected[canonical] = series
    return selected


def validate_forecasts(frame: pd.DataFrame, context: str) -> None:
    """Never train with omitted forecast variables or incomplete archive responses."""
    values = frame[list(FORECAST_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad_row, bad_column = np.argwhere(~np.isfinite(values))[0]
        raise ValueError(f"{context}: missing/non-finite {FORECAST_COLUMNS[bad_column]} at "
                         f"{frame.index[bad_row]}; all five forecast variables are required")


def download_forecasts(site: str, root: Path, *, offline: bool = False) -> Path:
    """Download short-lead ECMWF archive values at valid time, not 24-hour leads.

    The paper's previous-day averaging happens later in day_ahead_problem.
    Cache raw yearly responses with their full requests and retrieval times.
    """
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{site}.csv"
    if destination.exists():
        validate_forecasts(forecast_columns(read_hourly(destination)), str(destination))
        if not destination.with_suffix(".provenance.json").is_file():
            raise ValueError(f"{destination}: downloaded forecast cache lacks its provenance sidecar; "
                             "place manually supplied forecasts directly under --forecast-dir")
        return destination
    _, latitude, longitude = SITES[site]
    frames, requests = [], []
    for year in range(START.year, END.year + 1):
        query = dict(latitude=latitude, longitude=longitude,
                     start_date=str(max(START, pd.Timestamp(year, 1, 1)).date()),
                     end_date=str(min(END, pd.Timestamp(year, 12, 31)).date()),
                     hourly=",".join(FORECAST_VARIABLES), models=FORECAST_MODEL,
                     timezone="UTC", temperature_unit="celsius", wind_speed_unit="kmh",
                     precipitation_unit="mm")
        url = FORECAST_ARCHIVE + "?" + urlencode(query)
        cache = root / f"{site}_{FORECAST_MODEL}_{year}.json"
        if cache.exists():
            record = json.loads(cache.read_text(encoding="utf-8"))
            if record.get("url") != url:
                raise ValueError(f"{cache}: cached forecast request differs from this run")
        else:
            if offline:
                raise ValueError(f"Missing cached forecasts for site {site}, {year}; "
                                 "rerun without --offline to download from Open-Meteo")
            print(f"Downloading ECMWF archived forecasts: site {site}, {year}", flush=True)
            payload = request_json(url)
            record = dict(url=url, retrieved_at=utc_now(), response=payload)
            cache.write_text(json.dumps(record), encoding="utf-8")
        payload = record["response"]
        hourly = payload.get("hourly", {})
        if not {"time", *FORECAST_VARIABLES}.issubset(hourly):
            raise ValueError(f"{cache}: Open-Meteo response lacks required forecast fields")
        frame = pd.DataFrame(hourly)[["time", *FORECAST_VARIABLES]].rename(
            columns={"time": "datetime", **{v: f"forecasted_{v}" for v in FORECAST_VARIABLES}})
        index = pd.DatetimeIndex(pd.to_datetime(frame.pop("datetime"), utc=True)).tz_localize(None)
        frame.index = index
        expected = pd.date_range(query["start_date"],
                                 pd.Timestamp(query["end_date"]) + pd.Timedelta(hours=23), freq="h")
        if not index.equals(expected):
            raise ValueError(f"{cache}: incomplete or unordered forecast timestamps")
        validate_forecasts(frame, str(cache))
        frames.append(frame)
        requests.append({**input_record(cache), "url": url, "retrieved_at": record["retrieved_at"]})
    combined = pd.concat(frames)
    provenance = dict(provider="Open-Meteo", endpoint=FORECAST_ARCHIVE, model=FORECAST_MODEL,
                      product="stitched_short_lead_historical_forecasts",
                      time_axis="forecast_valid_time_utc", fixed_lead_hours=None,
                      feature_alignment="UTC daily mean on d-1 to predict soil moisture on d",
                      variables=list(FORECAST_VARIABLES), requests=requests)
    destination.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    combined.to_csv(destination, index_label="datetime")
    return destination


def attach_forecasts(frame: pd.DataFrame, site: str, args):
    """Preserve local forecasts, fill gaps from the selected public archive."""
    selected = forecast_columns(frame)
    records, sources = [], []
    if selected.notna().any().any():
        sources.append(dict(kind="local_input_columns", provenance="supplied_with_observations"))
    root = args.forecast_dir or FORECAST_DIR
    supplied = find_input(root, site, required=False)
    if supplied is not None:
        selected = selected.fillna(forecast_columns(read_hourly(supplied, allow_gaps=True)))
        records.append(input_record(supplied))
        sources.append(dict(kind="local_forecast_file", path=str(supplied)))
    if selected.isna().any().any():
        downloaded = download_forecasts(site, root / FORECAST_MODEL, offline=args.offline)
        selected = selected.fillna(forecast_columns(read_hourly(downloaded)))
        records.append(input_record(downloaded))
        sidecar = downloaded.with_suffix(".provenance.json")
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        records.append(input_record(sidecar))
        sources.append(dict(kind="open_meteo_download", **provenance))
    validate_forecasts(selected, f"Site {site}")
    result = frame.copy()
    result[list(FORECAST_COLUMNS)] = selected[list(FORECAST_COLUMNS)]
    return result, records, sources


def daily_means(hourly: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    values = hourly[columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Required hourly inputs contain missing or non-finite values")
    daily = values.resample("D").mean().where(values.resample("D").count() == 24)
    if daily.isna().any().any():
        raise ValueError("Daily means require 24 finite hourly values per variable")
    return daily


def day_ahead_problem(daily: pd.DataFrame, weather: list[str]):
    """Outcome-dated rows: target lags 1/2 and weather lag 1; no future rows."""
    features = pd.DataFrame({"soil_previous_day": daily[TARGET].shift(1),
                             "soil_two_days_before": daily[TARGET].shift(2)}, index=daily.index)
    for column in weather:
        features[f"{column}_previous_day"] = daily[column].shift(1)
    # XGBoost handles the first two training rows' missing pre-2020 lags.
    return features, daily[TARGET]


def fit_residuals(daily: pd.DataFrame, weather: list[str]):
    features, outcomes = day_ahead_problem(daily, weather)
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05,
                         objective="reg:squarederror", random_state=42, n_jobs=1)
    training = features.index.year == 2020
    model.fit(features.loc[training], outcomes.loc[training])
    result = pd.DataFrame({"observed": outcomes, "prediction": model.predict(features)})
    result["squared_error"] = (result["prediction"] - result["observed"]) ** 2
    result["split"] = np.where(result.index.year == 2020, "training",
                               np.where(result.index.year <= 2022, "calibration", "monitoring"))
    return model, result.loc[:MONITOR_END], list(features.columns)


def define_episodes(gaps: dict[str, pd.Series], smoothing: int = 30, percentile: float = 1):
    """Episode separation is always >30 days, independently of smoothing."""
    rows, diagnostics = [], []
    for site, gap in gaps.items():
        smoothed = gap.rolling(smoothing, min_periods=smoothing).mean()
        reference = smoothed.loc["2021-01-01":UPGRADE - pd.Timedelta(days=1)]
        threshold = float(reference.quantile(percentile / 100, interpolation="linear"))
        below = smoothed.loc[UPGRADE:MONITOR_END] < threshold
        previous = None
        for day in below.index[below]:
            if previous is None or (day - previous).days > 30:
                rows.append(dict(site=site, onset=day, smoothing_days=smoothing,
                                 percentile=percentile, threshold=threshold))
            previous = day
        diagnostics.extend(dict(site=site, date=day, gap=value, smoothed_gap=smoothed.loc[day],
                                threshold=threshold, below_threshold=bool(smoothed.loc[day] < threshold))
                           for day, value in gap.loc[MONITOR_START:MONITOR_END].items())
    return (pd.DataFrame(rows, columns=["site", "onset", "smoothing_days", "percentile", "threshold"]),
            pd.DataFrame(diagnostics))


def score_episodes(episodes: pd.DataFrame, alarms: pd.DataFrame,
                   target: str, algorithm: str, lead: int = 0, lag: int = HORIZON):
    subset = alarms[(alarms.target == target) & (alarms.algorithm == algorithm)]
    subset = subset[subset.date >= UPGRADE]
    matched = pd.Series(False, index=subset.index)
    detail = []
    for episode in episodes.itertuples():
        start = max(UPGRADE, episode.onset - pd.Timedelta(days=lead))
        end = min(MONITOR_END, episode.onset + pd.Timedelta(days=lag))
        mask = (subset.site == episode.site) & subset.date.between(start, end)
        matched |= mask
        first = subset.loc[mask, "date"].min()
        detail.append(dict(target=target, algorithm=algorithm, site=episode.site,
                           onset=episode.onset, window_start=start, window_end=end,
                           detected=bool(mask.any()), first_alarm=first,
                           delay_days=(first - episode.onset).days if pd.notna(first) else np.nan))
    return dict(episodes=len(episodes), detected_episodes=sum(row["detected"] for row in detail),
                post_alarms_in_windows=int(matched.sum()), post_alarms=len(subset)), detail


def calibrate(streams, trials: int, seed: int, rounds: int, budget: float = BUDGET):
    return calibrate_detectors_with_widening(
        streams, algorithms=list(ALGORITHMS), n_trials=trials, optuna_seed=seed,
        max_rounds=rounds, false_alarm_budget_per_1000=budget,
        false_alarm_confidence=FALSE_ALARM_CONFIDENCE,
        wasserstein_bounds={"window_size": (21, 42), "threshold_drift_score": (0., 4.)},
        kswin_bounds={"window_size": (21, 42), "alpha": (1e-5, 0.01), "stat_size": (5, 21)},
        adwin_bounds=ADWIN_BOUNDS, ph_bounds=PAGE_HINKLEY_BOUNDS,
        window_size_limits=(21, 42), kswin_require_attainable_alpha=True,
    )


def monitor(target: str, residuals: dict, params: dict):
    rows, alarms = [], []
    for site, residual in residuals.items():
        stream = residual.loc[MONITOR_START:MONITOR_END, "normalized_squared_error"]
        for algorithm in ALGORITHMS:
            configuration = params[algorithm]
            indices = run_detector(algorithm, stream.to_numpy(), configuration)
            # Use the protocol's conservative initial warm-up for rate exposure.
            # Keep every emitted alarm, as in calibration; never reset at the upgrade.
            active_dates = stream.index[_warmup_length(algorithm, configuration):]
            dates = stream.index[np.asarray(indices, dtype=int)]
            before, after = int((dates < UPGRADE).sum()), int((dates >= UPGRADE).sum())
            rows.append(dict(target=target, site=site, algorithm=algorithm,
                             pre_alarms=before, post_alarms=after,
                             pre_exposure=int((active_dates < UPGRADE).sum()),
                             post_exposure=int((active_dates >= UPGRADE).sum())))
            alarms.extend(dict(target=target, site=site, algorithm=algorithm,
                               index=int(index), date=day) for index, day in zip(indices, dates))
    return rows, alarms


def summarize(site_summary: pd.DataFrame, alarms: pd.DataFrame, episodes: pd.DataFrame | None):
    pooled = site_summary.groupby(["target", "algorithm"], sort=False)[
        ["pre_alarms", "post_alarms", "pre_exposure", "post_exposure"]].sum().reset_index()
    details = []
    for index, row in pooled.iterrows():
        for period in ("pre", "post"):
            pooled.loc[index, f"{period}_rate_per_1000"] = (
                1000 * row[f"{period}_alarms"] / row[f"{period}_exposure"])
        count = int(row.pre_alarms + row.post_alarms)
        pooled.loc[index, "post_rate_increase_pvalue"] = binomtest(
            int(row.post_alarms), count,
            row.post_exposure / (row.pre_exposure + row.post_exposure),
            alternative="greater").pvalue if count else 1.0
        if episodes is not None:
            scores, detail = score_episodes(episodes, alarms, row.target, row.algorithm)
            for key, value in scores.items():
                pooled.loc[index, key] = value
            details.extend(detail)
    if set(pooled.target) == {"ifs", "era5_land"}:
        for algorithm in ALGORITHMS:
            pair = pooled[pooled.algorithm == algorithm].set_index("target")
            pvalue = fisher_exact(pair.loc[["ifs", "era5_land"],
                                           ["pre_alarms", "post_alarms"]].to_numpy(dtype=int)).pvalue
            pooled.loc[pooled.algorithm == algorithm, "target_period_fisher_pvalue"] = pvalue
    return pooled, pd.DataFrame(details)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def input_record(path: Path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def run(args):
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    hourly, weather, controls, inputs, deviations, forecast_sources = {}, {}, {}, [], [], {}
    observation_sources, control_sources = {}, {}
    for site in SITES:
        path = find_input(args.data_dir, site, required=False)
        if path is None:
            path = download_observations(site, args.data_dir, offline=args.offline)
        frame = read_hourly(path)
        observation_sources[site] = record_source(path, inputs)
        frame, forecast_inputs, forecast_sources[site] = attach_forecasts(frame, site, args)
        inputs.extend(forecast_inputs)
        weather[site] = list(WEATHER) + list(FORECAST_COLUMNS)
        missing = set([TARGET, *weather[site]]) - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        hourly[site] = frame
        if args.download_control:
            control_path = find_input(args.control_dir, site, required=False)
            if control_path is None:
                control_path = download_control(site, args.control_dir, offline=args.offline)
            controls[site] = read_hourly(control_path)
            control_sources[site] = record_source(control_path, inputs)
    if controls and len(controls) != len(SITES):
        raise ValueError("ERA5-Land control must cover all four sites; use --download-control")
    if not controls:
        deviations.append("ERA5-Land control disabled with --no-control: control monitoring, "
                          "target-gap episodes and scoring sensitivity are skipped.")
    if len({tuple(columns) for columns in weather.values()}) != 1:
        raise ValueError("All four sites must use the same weather columns")
    for message in deviations:
        warnings.warn(message, stacklevel=2)

    metadata = dict(status="running", started_at=utc_now(), argv=[Path(sys.argv[0]).name, *sys.argv[1:]],
                    inputs=inputs, deviations=deviations,
                    sites=SITES, dates=dict(training="2020", calibration="2021-2022",
                    monitoring=[str(MONITOR_START.date()), str(MONITOR_END.date())],
                    upgrade=str(UPGRADE.date())), budget_per_1000=args.budget,
                    confidence=FALSE_ALARM_CONFIDENCE, window_range=[21, 42],
                    horizon_days=HORIZON, n_trials=args.n_trials, optuna_seed=args.seed,
                    widening_rounds=args.widening_rounds, kswin_require_attainable_alpha=True,
                    initial_search_bounds=dict(
                        wasserstein={"window_size": [21, 42], "threshold_drift_score": [0., 4.]},
                        kswin={"window_size": [21, 42], "alpha": [1e-5, 0.01], "stat_size": [5, 21]},
                        adwin=ADWIN_BOUNDS, ph=PAGE_HINKLEY_BOUNDS),
                    calibration_search_note="KSWIN draws with an unattainable alpha are "
                    "excluded and use up a trial.",
                    signal_normalization="pooled_calibration_median_per_target",
                    observation_sources=observation_sources, control_sources=control_sources,
                    forecast_sources=forecast_sources, forecast_variables=list(FORECAST_VARIABLES),
                    regressors_per_model=15,
                    forecast_note="Short-lead ECMWF IFS forecasts joined by valid time, "
                    "averaged over the previous UTC day.",
                    observation_note="Observations are retrieved with models=ecmwf_ifs; inputs "
                    "without a provenance sidecar are recorded as local snapshots.",
                    versions={name: version(name) for name in
                              ("numpy", "scipy", "pandas", "river", "xgboost", "optuna")},
                    model_params=dict(n_estimators=100, max_depth=5, learning_rate=0.05,
                                      objective="reg:squarederror", random_state=42, n_jobs=1),
                    training_initial_lags="First two days retain missing pre-2020 lags for XGBoost",
                    targets={})
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.prepare_only:
        for site, frame in hourly.items():
            daily_means(frame, [TARGET, *weather[site]])
            if controls:
                daily_means(controls[site], [TARGET])
        metadata["status"] = "inputs_ready"
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"All four sites have 15 regressors; input manifest saved to {output}")
        return None
    all_sites, all_alarms, daily_targets = [], [], {}
    for target in (["ifs", "era5_land"] if controls else ["ifs"]):
        print(f"\nFitting four fixed 2020 forecasters: {target}", flush=True)
        residuals, daily_targets[target], features = {}, {}, {}
        for site, frame in hourly.items():
            source = frame.copy()
            if target == "era5_land":
                source[TARGET] = controls[site][TARGET]
            daily = daily_means(source, [TARGET, *weather[site]])
            daily_targets[target][site] = daily[TARGET]
            model, residuals[site], features[site] = fit_residuals(daily, weather[site])
            model.save_model(output / f"{target}_{site}_model.json")
        calibration = [frame.loc["2021":"2022", "squared_error"].to_numpy()
                       for frame in residuals.values()]
        if any(len(stream) != 730 for stream in calibration):
            raise ValueError("Expected exactly 730 calibration days per site")
        normalizer = CalibrationMedianNormalizer.fit(calibration)
        for site, frame in residuals.items():
            frame["normalized_squared_error"] = frame.squared_error / normalizer.scale
            frame.to_csv(output / f"{target}_{site}_residuals.csv", index_label="date")
        print(f"Calibrating {target}; pooled median squared error = {normalizer.scale:.8g}", flush=True)
        params, trials, bounds = calibrate(normalizer.transform(calibration), args.n_trials,
                                          args.seed, args.widening_rounds, args.budget)
        trials["normalization_scale"] = normalizer.scale
        trials.to_csv(output / f"{target}_calibration.csv", index=False)
        selected = trials[trials.selected]
        selected.to_csv(output / f"{target}_selection.csv", index=False)
        for row in selected.itertuples():
            if not row.false_alarm_budget_met:
                warnings.warn(f"{target}/{row.algorithm}: calibration budget not met", stacklevel=2)
        (output / f"{target}_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
        rows, alarms = monitor(target, residuals, params)
        all_sites.extend(rows)
        all_alarms.extend(alarms)
        metadata["targets"][target] = dict(normalization_scale=normalizer.scale,
                                           feature_names=features, final_search_bounds=bounds)
    alarms = pd.DataFrame(all_alarms, columns=["target", "site", "algorithm", "index", "date"])
    alarms["date"] = pd.to_datetime(alarms.date)
    alarms.to_csv(output / "alarms.csv", index=False)
    sites = pd.DataFrame(all_sites)
    sites.to_csv(output / "site_summary.csv", index=False)
    episodes = None
    if controls:
        gaps = {site: daily_targets["ifs"][site] - daily_targets["era5_land"][site] for site in SITES}
        episodes, diagnostics = define_episodes(gaps)
        episodes.to_csv(output / "episodes.csv", index=False)
        diagnostics.to_csv(output / "target_gaps.csv", index=False)
        gap_rows = []
        for site, gap in gaps.items():
            for period, values in (("before", gap.loc[MONITOR_START:UPGRADE - pd.Timedelta(days=1)]),
                                   ("after", gap.loc[UPGRADE:MONITOR_END])):
                gap_rows.append(dict(site=site, period=period, days=len(values),
                                     mean=values.mean(), median=values.median()))
        pd.DataFrame(gap_rows).to_csv(output / "target_gap_summary.csv", index=False)
        sensitivity = []
        for smoothing in (15, 30, 60):
            for percentile in (0.5, 1., 2.):
                variant, _ = define_episodes(gaps, smoothing, percentile)
                for algorithm in ALGORITHMS:
                    scores, _ = score_episodes(variant, alarms, "ifs", algorithm)
                    sensitivity.append(dict(smoothing_days=smoothing, percentile=percentile,
                                            algorithm=algorithm, **scores))
        pd.DataFrame(sensitivity).to_csv(output / "episode_definition_sensitivity.csv", index=False)
        matching = []
        for lead in (0, 15, 30):
            for lag in (14, 42, 90):
                for algorithm in ALGORITHMS:
                    scores, _ = score_episodes(episodes, alarms, "ifs", algorithm, lead, lag)
                    matching.append(dict(lead_days=lead, lag_days=lag, algorithm=algorithm, **scores))
        pd.DataFrame(matching).to_csv(output / "alarm_matching_sensitivity.csv", index=False)
    pooled, scores = summarize(sites, alarms, episodes)
    pooled.to_csv(output / "summary.csv", index=False)
    if episodes is not None:
        scores.to_csv(output / "episode_scores.csv", index=False)
    metadata["finished_at"] = utc_now()
    metadata["status"] = "complete"
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pooled.to_string(index=False))
    print(f"\nSaved results to {output}")
    return pooled


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/tests"),
                        help="Observation inputs/cache; missing sites are downloaded automatically")
    parser.add_argument("--forecast-dir", type=Path,
                        help="Local forecast inputs and cache root (default: data/soil_moisture/forecasts)")
    parser.add_argument("--offline", action="store_true", help="Use local/cached inputs; never download")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare and validate inputs, then stop")
    parser.add_argument("--control-dir", type=Path, default=Path("data/soil_moisture/control"))
    control = parser.add_mutually_exclusive_group()
    control.add_argument("--download-control", action="store_true", dest="download_control",
                         help="Download missing ERA5-Land inputs (default)")
    control.add_argument("--no-control", action="store_false", dest="download_control",
                         help="Explicitly skip the ERA5-Land control and episode scoring")
    parser.set_defaults(download_control=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/soil_moisture/forecast_run"))
    parser.add_argument("--n-trials", type=int, default=N_OPTUNA_TRIALS)
    parser.add_argument("--seed", type=int, default=OPTUNA_SEED)
    parser.add_argument("--widening-rounds", type=int, default=CALIBRATION_WIDENING_ROUNDS)
    parser.add_argument("--budget", type=float, default=BUDGET,
                        help="False-alarm budget per 1,000 days (default: the paper's 2.38)")
    args = parser.parse_args(argv)
    if args.n_trials < 1 or args.widening_rounds < 0:
        parser.error("--n-trials must be positive and --widening-rounds non-negative")
    if not args.budget > 0:
        parser.error("--budget must be positive")
    return args


def main():
    try:
        run(parse_args())
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise SystemExit(f"Soil-moisture case: {exc}") from exc


if __name__ == "__main__":
    main()
