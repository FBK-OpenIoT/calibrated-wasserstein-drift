"""Draw the soil-case monitoring figure (paper Figure 4) from a completed run.

From the repository root:
    uv run --extra research --extra figures python -m experiments.plot_soil_moisture \
        --run-dir data/soil_moisture/forecast_run --output data/soil_moisture/soil_upstream_change.pdf

Reads only the run's outputs: per-site IFS residuals, alarms and target gaps.
Each panel shows the trailing 30-day median of the normalized squared residual
(squared error divided by the pooled calibration median), shading for
post-upgrade days on which the smoothed IFS-ERA5-Land difference is below its
episode threshold, and one row of ticks per detector for IFS alarms.
"""

from __future__ import annotations

import argparse
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

from experiments.soil_moisture import ALGORITHMS, MONITOR_END, MONITOR_START, SITES, UPGRADE

LABELS = {"wasserstein": "Wasserstein", "kswin": "KSWIN", "adwin": "ADWIN", "ph": "Page-Hinkley"}
COLOURS = {"wasserstein": "#1f5fa8", "kswin": "#2c9a3c", "adwin": "#c8332e", "ph": "#8e5bb5"}
SHADE = "#f6c9a4"
SMOOTHING_DAYS = 30
# Log-scale y range: the curve stays above 0.04, so alarm ticks get the strip below it.
Y_LIMITS = (0.002, 400)
TICK_ROW, TICK_HEIGHT = 0.055, 0.045


def coordinate(value: float) -> str:
    """Two decimals, rounding halves up as in the paper (11.985 -> 11.99)."""
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def below_threshold_spans(gaps: pd.DataFrame, site: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous post-upgrade runs of below-threshold days, as [start, end + 1 day)."""
    days = gaps[(gaps.site == site) & (gaps.date >= UPGRADE) & gaps.below_threshold].date
    spans, start, previous = [], None, None
    for day in days:
        if previous is None or (day - previous).days > 1:
            if start is not None:
                spans.append((start, previous + pd.Timedelta(days=1)))
            start = day
        previous = day
    if start is not None:
        spans.append((start, previous + pd.Timedelta(days=1)))
    return spans


def plot(run_dir: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    alarms = pd.read_csv(run_dir / "alarms.csv", parse_dates=["date"])
    alarms = alarms[alarms.target == "ifs"]
    gaps = pd.read_csv(run_dir / "target_gaps.csv", parse_dates=["date"])
    gaps["below_threshold"] = gaps.below_threshold.astype(str).str.lower() == "true"

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9})
    figure, axes = plt.subplots(len(SITES), 1, figsize=(7.0, 8.6), sharex=True)
    for ax, (site, (_, latitude, longitude)) in zip(axes, SITES.items()):
        residuals = pd.read_csv(run_dir / f"ifs_{site}_residuals.csv", index_col=0, parse_dates=True)
        smoothed = (residuals["normalized_squared_error"]
                    .rolling(SMOOTHING_DAYS, min_periods=SMOOTHING_DAYS).median()
                    .loc[MONITOR_START:MONITOR_END])
        for start, end in below_threshold_spans(gaps, site):
            ax.axvspan(start, end, color=SHADE, alpha=0.8, linewidth=0)
        ax.plot(smoothed.index, smoothed.to_numpy(), color="black", linewidth=0.9)
        ax.axvline(UPGRADE, color="0.45", linestyle="--", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_ylim(*Y_LIMITS)
        ax.yaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_ylabel("× calib. median")
        ax.set_title(f"Site {site} ({coordinate(latitude)} N, {coordinate(longitude)} E)")
        ax.grid(axis="y", color="0.9", linewidth=0.5)
        transform = ax.get_xaxis_transform()
        for row, algorithm in enumerate(ALGORITHMS):
            dates = alarms[(alarms.site == site) & (alarms.algorithm == algorithm)].date
            low = 0.01 + TICK_ROW * row
            ax.vlines(dates, low, low + TICK_HEIGHT, transform=transform,
                      color=COLOURS[algorithm], linewidth=1.4)
    axes[-1].set_xlim(MONITOR_START, MONITOR_END + pd.Timedelta(days=1))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    handles = [Line2D([], [], color="black", linewidth=0.9, label="30-day median squared residual"),
               Line2D([], [], color="0.45", linestyle="--", linewidth=0.8, label="IFS upgrade"),
               Patch(color=SHADE, label="IFS−ERA5-Land difference below threshold")]
    handles += [Line2D([], [], color=COLOURS[a], linewidth=1.4, label=f"{LABELS[a]} alarm")
                for a in ALGORITHMS]
    figure.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
                  bbox_to_anchor=(0.5, 1.0))
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("data/soil_moisture/forecast_run"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/soil_moisture/soil_upstream_change.pdf"))
    args = parser.parse_args(argv)
    try:
        plot(args.run_dir, args.output)
    except ImportError as exc:
        raise SystemExit("Plotting needs matplotlib: add --extra figures") from exc
    except (OSError, KeyError, ValueError) as exc:
        raise SystemExit(f"Cannot draw soil figure from {args.run_dir}: {exc}") from exc
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
