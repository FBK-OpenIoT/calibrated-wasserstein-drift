"""Compare a completed soil run with the values printed in the paper.

Run from the repository root with --run-dir and optionally --output. This reads
outputs only; it never changes data, parameters or detector results. Exit codes:
0 = all compared values agree, 1 = numerical differences, 2 = invalid inputs.
The paper's soil case uses the archived inputs retrieved on 5 November 2026;
see REPRODUCIBILITY.md.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


# Values printed in Tables 7--9 and G.1 and Appendix G of the paper. Numeric
# strings keep the paper's precision; integers require exact counts.
MONITORING = {
    ("ifs", "wasserstein"): (1, 3, "0.45", "1.81", "0.21", 2, 2),
    ("ifs", "kswin"): (0, 6, "0.00", "3.62", "0.003", 3, 3),
    ("ifs", "adwin"): (3, 6, "1.12", "3.62", "0.08", 1, 1),
    ("ifs", "ph"): (0, 10, "0.00", "6.04", "<0.001", 2, 3),
    ("era5_land", "wasserstein"): (5, 3, "2.25", "1.81", "0.74", None, None),
    ("era5_land", "kswin"): (2, 1, "0.78", "0.60", "0.78", None, None),
    ("era5_land", "adwin"): (5, 4, "1.86", "2.42", "0.47", None, None),
    ("era5_land", "ph"): (4, 2, "1.54", "1.21", "0.75", None, None),
}
PARAMETERS = {
    ("ifs", "wasserstein"): {"window_size": 42, "threshold_drift_score": "8.73"},
    ("ifs", "kswin"): {"window_size": 29, "alpha": "1.17e-4", "stat_size": 12},
    ("ifs", "adwin"): {"delta": "1.49e-5"},
    ("ifs", "ph"): {"threshold": "1233.5", "delta": "6.62e-8"},
    ("era5_land", "wasserstein"): {"window_size": 42, "threshold_drift_score": "16.84"},
    ("era5_land", "kswin"): {"window_size": 40, "alpha": "4.39e-7", "stat_size": 19},
    ("era5_land", "adwin"): {"delta": "3.20e-11"},
    ("era5_land", "ph"): {"threshold": "2280.6", "delta": "1.36e-12"},
}
# Table 8 upper confidence bounds per 1,000 calibration days.
UCB = {
    ("ifs", "wasserstein"): "1.96", ("ifs", "kswin"): "2.25",
    ("ifs", "adwin"): "2.19", ("ifs", "ph"): "2.25",
    ("era5_land", "wasserstein"): "1.96", ("era5_land", "kswin"): "2.28",
    ("era5_land", "adwin"): "2.69", ("era5_land", "ph"): "2.25",
}
GAPS = {
    "A": ("-0.033", "-0.033", "-0.078", "-0.067"),
    "B": ("-0.103", "-0.103", "-0.148", "-0.151"),
    "C": ("0.005", "0.006", "-0.003", "0.004"),
    "D": ("-0.041", "-0.031", "-0.062", "-0.037"),
}
EPISODES = [
    ("A", "2024-12-05"), ("A", "2025-05-06"), ("A", "2025-09-15"),
    ("B", "2025-06-21"), ("B", "2025-09-19"), ("D", "2025-06-14"),
]
DEFINITIONS = {
    (15, 0.5): (7, 2, 3, 2, 3), (15, 1): (7, 2, 3, 2, 3),
    (15, 2): (8, 2, 3, 2, 3), (30, 0.5): (5, 2, 3, 1, 2),
    (30, 1): (6, 2, 3, 1, 2), (30, 2): (6, 2, 3, 1, 2),
    (60, 0.5): (4, 0, 2, 1, 3), (60, 1): (4, 0, 2, 1, 3),
    (60, 2): (4, 0, 2, 1, 3),
}
# Appendix G alarm-matching grid: minimum, maximum and 30-day-lead/42-day-lag count.
MATCHING = (("wasserstein", 2, 2, 2), ("kswin", 1, 4, 3), ("adwin", 1, 3, 3), ("ph", 2, 5, 3))


def agrees(observed: str, expected: str | int) -> bool:
    if isinstance(expected, int):
        return float(observed) == expected
    if expected.startswith("<"):
        return float(observed) < float(expected[1:])
    mantissa, _, exponent = expected.lower().partition("e")
    decimals = len(mantissa.partition(".")[2])
    tolerance = 0.5 * 10 ** (int(exponent or 0) - decimals)
    return abs(float(observed) - float(expected)) <= tolerance + tolerance * 1e-10


def compare(run_dir: Path) -> list[dict]:
    checks = []

    def read(name):
        with (run_dir / name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def add(source, key, metric, observed, expected, *, literal=False):
        checks.append(dict(source=source, key=key, metric=metric, paper=expected,
                           observed=observed, agrees=(observed == expected if literal
                                                     else agrees(str(observed), expected))))

    summary = {(row["target"], row["algorithm"]): row for row in read("summary.csv")}
    metrics = ("pre_alarms", "post_alarms", "pre_rate_per_1000", "post_rate_per_1000",
               "post_rate_increase_pvalue", "detected_episodes", "post_alarms_in_windows")
    for key, expected in MONITORING.items():
        for metric, value in zip(metrics, expected):
            if value is not None:
                add("Table 9", "/".join(key), metric, summary[key][metric], value)
        add("Section 7.3", "/".join(key), "episodes", summary[key]["episodes"], 6)

    for target in ("ifs", "era5_land"):
        selected = {row["algorithm"]: row for row in read(f"{target}_selection.csv")}
        for (expected_target, algorithm), parameters in PARAMETERS.items():
            if expected_target != target:
                continue
            key = f"{target}/{algorithm}"
            for metric, expected in parameters.items():
                add("Table 8", key, metric, selected[algorithm][metric], expected)
            expected_count = 1 if algorithm == "wasserstein" else (
                3 if target == "era5_land" and algorithm == "adwin" else 2)
            add("Table 8", key, "n_false_alarms", selected[algorithm]["n_false_alarms"],
                expected_count)
            add("Table 8", key, "false_alarm_ucb_per_1000",
                selected[algorithm]["false_alarm_ucb_per_1000"], UCB[(target, algorithm)])

    gaps = {(row["site"], row["period"]): row for row in read("target_gap_summary.csv")}
    for site, expected in GAPS.items():
        for (period, metric), value in zip(
                [(p, m) for p in ("before", "after") for m in ("mean", "median")], expected):
            add("Table 7", f"{site}/{period}", metric, gaps[(site, period)][metric], value)
        for period, days in (("before", 681), ("after", 414)):
            add("Table 7", f"{site}/{period}", "days", gaps[(site, period)]["days"], days)

    actual_episodes = sorted((row["site"], row["onset"][:10]) for row in read("episodes.csv"))
    add("Section 7.3", "all sites", "episode_onsets", str(actual_episodes),
        str(EPISODES), literal=True)
    definitions = {(int(row["smoothing_days"]), float(row["percentile"]), row["algorithm"]): row
                   for row in read("episode_definition_sensitivity.csv")}
    for (smoothing, percentile), expected in DEFINITIONS.items():
        for algorithm, detected in zip(("wasserstein", "kswin", "adwin", "ph"), expected[1:]):
            row = definitions[(smoothing, percentile, algorithm)]
            key = f"L={smoothing}/p={percentile}/{algorithm}"
            add("Table G.1", key, "episodes", row["episodes"], expected[0])
            add("Table G.1", key, "detected_episodes", row["detected_episodes"], detected)

    matching = read("alarm_matching_sensitivity.csv")
    for algorithm, low, high, lead30 in MATCHING:
        rows = [row for row in matching if row["algorithm"] == algorithm]
        counts = [int(float(row["detected_episodes"])) for row in rows]
        add("Appendix G", algorithm, "matching_min", min(counts), low)
        add("Appendix G", algorithm, "matching_max", max(counts), high)
        row = next(row for row in rows if int(row["lead_days"]) == 30
                   and int(row["lag_days"]) == 42)
        add("Appendix G", algorithm, "lead30_lag42_detected", row["detected_episodes"], lead30)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        checks = compare(args.run_dir)
    except (OSError, KeyError, ValueError, StopIteration) as exc:
        parser.exit(2, f"Cannot compare soil outputs: {exc}\n")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(checks[0]))
            writer.writeheader()
            writer.writerows(checks)
    differences = sum(not row["agrees"] for row in checks)
    print(f"Compared {len(checks)} printed values/claims; {differences} differ. "
          "See REPRODUCIBILITY.md.")
    raise SystemExit(1 if differences else 0)


if __name__ == "__main__":
    main()
