"""Soil-case figure script, budget option and paper checker."""

import numpy as np
import pandas as pd
import pytest

from experiments import check_soil_moisture_paper as check
from experiments import plot_soil_moisture as figure
from experiments import soil_moisture as soil


def test_budget_defaults_to_paper_value_and_is_passed_to_calibration(monkeypatch):
    assert soil.parse_args([]).budget == soil.BUDGET == 2.38
    assert soil.parse_args(["--budget", "5"]).budget == 5.0
    with pytest.raises(SystemExit):
        soil.parse_args(["--budget", "0"])
    seen = {}

    def fake(streams, **kwargs):
        seen.update(kwargs)
        return {}, pd.DataFrame(), {}

    monkeypatch.setattr(soil, "calibrate_detectors_with_widening", fake)
    soil.calibrate([np.ones(10)], 1, 42, 0, budget=7.5)
    assert seen["false_alarm_budget_per_1000"] == 7.5


def test_coordinates_round_half_up_as_printed_in_the_paper():
    assert figure.coordinate(11.985) == "11.99"
    assert figure.coordinate(42.9415) == "42.94"


def test_shading_spans_only_post_upgrade_contiguous_below_threshold_days():
    days = pd.date_range("2024-11-01", periods=40, freq="D")
    below = np.zeros(40, dtype=bool)
    below[[2, 3, 15, 16, 17, 30]] = True  # 2 and 3 precede the upgrade on 12 November
    gaps = pd.DataFrame(dict(site="A", date=days, below_threshold=below))
    spans = figure.below_threshold_spans(gaps, "A")
    assert spans == [(days[15], days[17] + pd.Timedelta(days=1)),
                     (days[30], days[30] + pd.Timedelta(days=1))]


def test_figure_is_drawn_from_run_outputs(tmp_path):
    pytest.importorskip("matplotlib")
    dates = pd.date_range(soil.MONITOR_START - pd.Timedelta(days=40), soil.MONITOR_END, freq="D")
    rng = np.random.default_rng(0)
    for site in soil.SITES:
        pd.DataFrame({"normalized_squared_error": rng.gamma(1, 1, len(dates))},
                     index=dates).to_csv(tmp_path / f"ifs_{site}_residuals.csv", index_label="date")
    pd.DataFrame([dict(target="ifs", site="B", algorithm="kswin", index=700,
                       date="2024-12-01")]).to_csv(tmp_path / "alarms.csv", index=False)
    monitored = dates[dates >= soil.MONITOR_START]
    pd.DataFrame(dict(site="A", date=monitored, below_threshold=monitored >= soil.UPGRADE)
                 ).to_csv(tmp_path / "target_gaps.csv", index=False)
    output = tmp_path / "figure.pdf"
    figure.plot(tmp_path, output)
    assert output.read_bytes().startswith(b"%PDF")


def test_checker_covers_every_algorithm_and_target():
    pairs = {(t, a) for t in ("ifs", "era5_land") for a in soil.ALGORITHMS}
    assert set(check.MONITORING) == set(check.PARAMETERS) == set(check.UCB) == pairs
    assert [row[0] for row in check.MATCHING] == ["wasserstein", "kswin", "adwin", "ph"]
    assert check.agrees("1233.4608", "1233.5") and not check.agrees("1233.4", "1233.5")
