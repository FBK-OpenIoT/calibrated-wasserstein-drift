"""Date alignment, episode scoring and short-window soil calibration."""

import math
from io import BytesIO
import json
from urllib.parse import parse_qs, urlparse
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import pytest

from experiments import soil_moisture as soil
from wasserstein_score import evaluation as ev


@pytest.fixture
def short_forecast_archive(monkeypatch):
    monkeypatch.setattr(soil, "START", pd.Timestamp("2020-01-01"))
    monkeypatch.setattr(soil, "END", pd.Timestamp("2020-01-02"))
    dates = pd.date_range(soil.START, soil.END + pd.Timedelta(hours=23), freq="h")
    payload = {"hourly": {"time": dates.strftime("%Y-%m-%dT%H:%M").tolist(),
                          **{name: (np.arange(len(dates)) + i).tolist()
                             for i, name in enumerate(soil.FORECAST_VARIABLES)}}}
    return dates, payload


def test_downloads_all_five_forecasts_from_correct_product_and_reuses_cache(
        monkeypatch, tmp_path, short_forecast_archive):
    dates, payload = short_forecast_archive
    urls = []
    def fetch(url, **kwargs):
        urls.append(url)
        return BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr(soil, "urlopen", fetch)
    path = soil.download_forecasts("A", tmp_path)
    result = soil.read_hourly(path)
    assert result.index.equals(dates)
    assert list(result.columns) == list(soil.FORECAST_COLUMNS)
    query = parse_qs(urlparse(urls[0]).query)
    assert urls[0].startswith(soil.FORECAST_ARCHIVE + "?")
    assert query["hourly"] == [",".join(soil.FORECAST_VARIABLES)]
    assert query["models"] == ["ecmwf_ifs"]
    assert query["timezone"] == ["UTC"]
    assert query["wind_speed_unit"] == ["kmh"]
    provenance = json.loads(path.with_suffix(".provenance.json").read_text())
    assert provenance["fixed_lead_hours"] is None
    assert provenance["product"] == "stitched_short_lead_historical_forecasts"
    assert provenance["requests"][0]["sha256"]
    assert soil.download_forecasts("A", tmp_path, offline=True) == path
    assert len(urls) == 1


def test_null_forecast_response_cannot_create_usable_snapshot(
        monkeypatch, tmp_path, short_forecast_archive):
    _, payload = short_forecast_archive
    payload["hourly"]["et0_fao_evapotranspiration"][0] = None
    monkeypatch.setattr(soil, "urlopen", lambda *a, **kw: BytesIO(json.dumps(payload).encode()))
    with pytest.raises(ValueError, match="missing/non-finite forecasted_et0"):
        soil.download_forecasts("A", tmp_path)
    assert not (tmp_path / "A.csv").exists()


def test_missing_forecasts_fail_offline_instead_of_reducing_feature_count(
        monkeypatch, tmp_path, short_forecast_archive):
    dates, _ = short_forecast_archive
    def no_network(*args, **kwargs):
        pytest.fail("Offline mode attempted a network call")
    monkeypatch.setattr(soil, "urlopen", no_network)
    args = soil.parse_args(["--offline", "--forecast-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="Missing cached forecasts"):
        soil.attach_forecasts(pd.DataFrame(index=dates), "A", args)


def test_complete_local_forecasts_do_not_trigger_downloads(
        monkeypatch, tmp_path, short_forecast_archive):
    dates, payload = short_forecast_archive
    def no_network(*args, **kwargs):
        pytest.fail("Complete local forecasts should not need downloading")
    monkeypatch.setattr(soil, "urlopen", no_network)
    existing = pd.DataFrame({f"forecasted_{v}": payload["hourly"][v]
                             for v in soil.FORECAST_VARIABLES}, index=dates)
    args = soil.parse_args(["--offline", "--forecast-dir", str(tmp_path)])
    result, records, sources = soil.attach_forecasts(existing, "A", args)
    pd.testing.assert_frame_equal(result, existing, check_dtype=False)
    assert records == []
    assert [source["kind"] for source in sources] == ["local_input_columns"]


def test_existing_forecasts_are_preserved_and_gaps_are_downloaded(
        monkeypatch, tmp_path, short_forecast_archive):
    dates, payload = short_forecast_archive
    monkeypatch.setattr(soil, "urlopen", lambda *a, **kw: BytesIO(json.dumps(payload).encode()))
    # Reuse both embedded columns and the existing next-period naming convention.
    observed = pd.DataFrame(index=dates)
    observed["forecasted_temperature_2m_mean_next_1period"] = np.nan
    observed.loc[dates[:24], "forecasted_temperature_2m_mean_next_1period"] = 99.
    supplied = pd.DataFrame({"datetime": dates[:12],
                             "forecasted_precipitation_next_1period": np.full(12, 77.)})
    supplied.to_csv(tmp_path / "A.csv", index=False)
    args = soil.parse_args(["--forecast-dir", str(tmp_path)])
    result, records, sources = soil.attach_forecasts(observed, "A", args)
    assert result.loc[dates[0], "forecasted_temperature_2m"] == 99
    assert result.loc[dates[25], "forecasted_temperature_2m"] == 25
    assert result.loc[dates[0], "forecasted_precipitation"] == 77
    assert result.loc[dates[13], "forecasted_precipitation"] == 15
    assert len(records) == 3
    assert {source["kind"] for source in sources} == {
        "local_input_columns", "local_forecast_file", "open_meteo_download"}
    soil.validate_forecasts(result, "test")
    assert "forecasted_precipitation" not in observed  # The caller's data stay untouched.


def test_five_forecasts_enter_the_15_regressors_on_the_previous_day(short_forecast_archive):
    dates, payload = short_forecast_archive
    hourly = pd.DataFrame({soil.TARGET: np.full(len(dates), 0.3),
                           **{name: np.ones(len(dates)) for name in soil.WEATHER},
                           **{f"forecasted_{name}": values for name, values in payload["hourly"].items()
                              if name != "time"}}, index=dates)
    weather = [*soil.WEATHER, *soil.FORECAST_COLUMNS]
    daily = soil.daily_means(hourly, [soil.TARGET, *weather])
    features, _ = soil.day_ahead_problem(daily, weather)
    assert features.shape == (2, 15)
    for column in soil.FORECAST_COLUMNS:
        assert features.iloc[1][f"{column}_previous_day"] == hourly.iloc[:24][column].mean()


def test_clean_checkout_downloads_every_input_and_can_repeat_offline(
        monkeypatch, tmp_path, short_forecast_archive):
    _, _ = short_forecast_archive
    urls = []
    def fetch(url, **kwargs):
        urls.append(url)
        query = parse_qs(urlparse(url).query)
        dates = pd.date_range(query["start_date"][0],
                              pd.Timestamp(query["end_date"][0]) + pd.Timedelta(hours=23), freq="h")
        variables = query["hourly"][0].split(",")
        payload = {"hourly": {"time": dates.strftime("%Y-%m-%dT%H:%M").tolist(),
                              **{name: np.full(len(dates), 0.3 if name == soil.TARGET else 1.).tolist()
                                 for name in variables}}}
        return BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr(soil, "urlopen", fetch)
    args = soil.parse_args(["--prepare-only", "--data-dir", str(tmp_path / "observations"),
                            "--forecast-dir", str(tmp_path / "forecasts"),
                            "--control-dir", str(tmp_path / "control"),
                            "--output-dir", str(tmp_path / "run")])
    assert args.download_control  # A plain invocation includes the paper's control.
    assert not args.data_dir.exists()
    soil.run(args)
    assert len(urls) == 12  # 3 input products at each of four sites in this shortened fixture.
    products = {}
    for url in urls:
        query = parse_qs(urlparse(url).query)
        key = (urlparse(url).netloc, query["models"][0], len(query["hourly"][0].split(",")))
        products[key] = products.get(key, 0) + 1
    assert products == {
        ("archive-api.open-meteo.com", "ecmwf_ifs", 9): 4,
        ("historical-forecast-api.open-meteo.com", "ecmwf_ifs", 5): 4,
        ("archive-api.open-meteo.com", "era5_land", 1): 4,
    }
    metadata = json.loads((args.output_dir / "metadata.json").read_text())
    assert metadata["status"] == "inputs_ready"
    assert metadata["regressors_per_model"] == 15
    assert metadata["deviations"] == []
    assert len(metadata["inputs"]) == 24  # CSV and provenance for every site/product.
    assert {value["model"] for value in metadata["observation_sources"].values()} == {"ecmwf_ifs"}
    assert {value["model"] for value in metadata["control_sources"].values()} == {"era5_land"}
    def no_network(*args, **kwargs):
        pytest.fail("The prepared clean-checkout cache should work offline")
    monkeypatch.setattr(soil, "urlopen", no_network)
    args.offline = True
    soil.run(args)
    args.download_control = False
    with pytest.warns(UserWarning, match="control disabled"):
        soil.run(args)
    metadata = json.loads((args.output_dir / "metadata.json").read_text())
    assert metadata["control_sources"] == {}


def test_observation_download_fails_clearly_when_offline_and_empty(
        monkeypatch, tmp_path, short_forecast_archive):
    def no_network(*args, **kwargs):
        pytest.fail("Offline mode attempted a network call")
    monkeypatch.setattr(soil, "urlopen", no_network)
    args = soil.parse_args(["--prepare-only", "--offline", "--data-dir", str(tmp_path / "empty"),
                            "--output-dir", str(tmp_path / "run")])
    with pytest.raises(ValueError, match="Missing ecmwf_ifs inputs"):
        soil.run(args)


def test_rate_limit_retries_respect_server_delay(monkeypatch):
    calls, pauses = [], []
    def fetch(url, **kwargs):
        calls.append(url)
        if len(calls) < 3:
            raise HTTPError(url, 429, "Too Many Requests", {"Retry-After": "1"},
                            BytesIO(b'{"reason":"Minutely API request limit exceeded"}'))
        return BytesIO(b'{"ok":true}')
    monkeypatch.setattr(soil, "urlopen", fetch)
    monkeypatch.setattr(soil.time, "sleep", pauses.append)
    assert soil.request_json("https://example.test/data") == {"ok": True}
    assert len(calls) == 3
    assert pauses == [1., 1.]


def test_day_ahead_features_use_previous_calendar_days_only():
    dates = pd.date_range("2020-01-01", periods=5)
    daily = pd.DataFrame({soil.TARGET: [10., 20., 30., 40., 50.],
                          "rain": [1., 2., 3., 4., 5.]}, index=dates)
    features, outcome = soil.day_ahead_problem(daily, ["rain"])
    assert features.loc[dates[2]].tolist() == [20., 10., 2.]
    assert outcome.loc[dates[2]] == 30.
    changed = daily.copy()
    changed.loc[dates[2]:] = 1000
    updated, _ = soil.day_ahead_problem(changed, ["rain"])
    pd.testing.assert_series_equal(features.loc[dates[2]], updated.loc[dates[2]])


def test_hourly_input_rejects_duplicate_or_missing_hours(tmp_path):
    path = tmp_path / "A.csv"
    pd.DataFrame({"datetime": ["2020-01-01T00:00Z"] * 2, soil.TARGET: [0.2, 0.2]}).to_csv(
        path, index=False)
    with pytest.raises(ValueError, match="contiguous hourly UTC"):
        soil.read_hourly(path)


def test_daily_means_reject_partial_days():
    hourly = pd.DataFrame({soil.TARGET: np.ones(23)},
                          index=pd.date_range("2020-01-01", periods=23, freq="h"))
    with pytest.raises(ValueError, match="24 finite"):
        soil.daily_means(hourly, [soil.TARGET])


def test_episode_separation_is_strictly_more_than_30_days():
    dates = pd.date_range("2020-01-01", soil.MONITOR_END)
    gap = pd.Series(0., index=dates)
    below = [soil.UPGRADE, soil.UPGRADE + pd.Timedelta(days=30),
             soil.UPGRADE + pd.Timedelta(days=61)]
    gap.loc[below] = -1
    episodes, _ = soil.define_episodes({"A": gap}, smoothing=1, percentile=1)
    assert episodes.onset.tolist() == [below[0], below[2]]
    # Post-change extremes must not affect the pre-change threshold.
    assert episodes.threshold.tolist() == [0., 0.]


def test_episode_matching_is_inclusive_site_specific_and_counts_alarm_union():
    onset = pd.Timestamp("2025-01-01")
    episodes = pd.DataFrame({"site": ["A", "A"],
                             "onset": [onset, onset + pd.Timedelta(days=10)]})
    alarms = pd.DataFrame({"target": ["ifs"] * 4, "algorithm": ["kswin"] * 4,
                          "site": ["A", "A", "A", "B"],
                          "date": [onset - pd.Timedelta(days=1),
                                   onset + pd.Timedelta(days=42),
                                   onset + pd.Timedelta(days=53), onset]})
    score, detail = soil.score_episodes(episodes, alarms, "ifs", "kswin")
    assert score["detected_episodes"] == 2
    assert score["post_alarms_in_windows"] == 1
    assert [row["delay_days"] for row in detail] == [42, 32]


def test_matching_windows_clip_at_upgrade_and_monitoring_end():
    episodes = pd.DataFrame({"site": ["A", "A"], "onset": [soil.UPGRADE, soil.MONITOR_END]})
    alarms = pd.DataFrame(columns=["target", "algorithm", "site", "date"])
    alarms["date"] = pd.to_datetime(alarms.date)
    score, detail = soil.score_episodes(episodes, alarms, "ifs", "ph", lead=30, lag=90)
    assert score["detected_episodes"] == 0
    assert detail[0]["window_start"] == soil.UPGRADE
    assert detail[1]["window_end"] == soil.MONITOR_END


def test_kswin_excludes_impossible_exact_pvalues(monkeypatch):
    seen = []
    def runner(series, **kwargs):
        seen.append(kwargs)
        return []
    monkeypatch.setattr(ev, "_run_kswin", runner)
    best, _, table = ev._calibrate_kswin(
        [np.ones(100)], {"window_size": (21, 21), "alpha": (0.001, 0.01), "stat_size": (5, 5)},
        starting_idx=0, n_trials=20, verbose=False, stream_backend="threading",
        require_attainable_alpha=True, false_alarm_budget_per_1000=100)
    floor = 2 / math.comb(10, 5)
    assert 0 < len(table) < 20
    assert table.alpha.min() >= floor
    assert best["alpha"] >= floor
    assert seen and all(call["alpha"] >= floor for call in seen)


def test_widening_preserves_soil_window_limits(monkeypatch):
    streams = [np.random.default_rng(0).exponential(size=300)]
    # Force a request to extend both window ends in every round.
    original = ev.widened_bounds
    def widen(algorithm, bounds, selected, max_warmup_len):
        result = original(algorithm, bounds, selected, max_warmup_len)
        result["window_size"] = (5, 100)
        return result
    monkeypatch.setattr(ev, "widened_bounds", widen)
    params, table, bounds = ev.calibrate_detectors_with_widening(
        streams, algorithms=["kswin"], n_trials=3, max_rounds=1, verbose=False,
        kswin_bounds={"window_size": (21, 42), "alpha": (0.05, 0.1), "stat_size": (5, 10)},
        window_size_limits=(21, 42), false_alarm_budget_per_1000=100)
    assert bounds["kswin"]["window_size"] == (21, 42)
    assert table.window_size.between(21, 42).all()
    assert 21 <= params["kswin"]["window_size"] <= 42


def test_no_widening_reports_only_the_evaluated_bounds():
    bounds = {"delta": (1e-5, 0.1)}
    _, _, final = ev.calibrate_detectors_with_widening(
        [np.ones(100)], algorithms=["adwin"], n_trials=1, max_rounds=0,
        verbose=False, adwin_bounds=bounds, false_alarm_budget_per_1000=100)
    assert final["adwin"] == bounds


def test_monitor_preserves_continuous_stream_and_warmup_exposure(monkeypatch):
    dates = pd.date_range(soil.MONITOR_START, soil.MONITOR_END)
    residual = pd.DataFrame({"normalized_squared_error": np.ones(len(dates))}, index=dates)
    calls = []
    def runner(algorithm, series, params):
        calls.append(len(series))
        return [680, 681]
    monkeypatch.setattr(soil, "run_detector", runner)
    params = {"wasserstein": {"window_size": 42}, "kswin": {"window_size": 41},
              "adwin": {"delta": 0.001}, "ph": {"delta": 1, "threshold": 50}}
    rows, alarms = soil.monitor("ifs", {"A": residual}, params)
    assert calls == [1095] * 4
    assert rows[0]["pre_exposure"] == 681 - 126
    assert rows[0]["post_exposure"] == 414
    assert all(row["pre_alarms"] == row["post_alarms"] == 1 for row in rows)
    assert alarms[1]["date"] == soil.UPGRADE
