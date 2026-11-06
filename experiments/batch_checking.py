"""Main benchmark with residuals that arrive in batches.

Every detector reads every residual but may alarm only on the last sample of a
batch of ``batch_size`` samples: the Wasserstein distance is computed once per
batch, ADWIN tests for a cut once per batch (River's ``clock``), and KSWIN and
Page-Hinkley apply their tests once per batch
(``wasserstein_score.batch_detectors``). Each batch size is a separate
calibration under the main protocol (same streams, same false-alarm budget per
1,000 samples, same horizon), since a detector that tests less often admits a
lower threshold at the same budget. ``batch_size=1`` reproduces the main
benchmark.

Reading the results:
  - delays are in samples and include the wait for the end of the batch, so
    the shortest possible delay is ``batch_size - 1``;
  - the Wasserstein fence keeps its time span (``window_size`` samples of
    history), so it holds ``window_size / batch_size`` distances;
  - KSWIN compares the latest ``stat_size`` samples (20-80) with its window, so
    at batch sizes above ``stat_size`` some samples never enter the tested
    segment.

Usage, from the repository root:
    python experiments/batch_checking.py --batch-size 10   # one batch size
    python experiments/batch_checking.py --summarize       # combine the runs
Each run writes to data/batch_checking/batch_<B>/.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from wasserstein_score.benchmark_protocol import DETECTION_HORIZON, DRIFT_AT, N_TEST_STREAMS, STARTING_IDX
from wasserstein_score import helper_functions as md
from wasserstein_score.helper_functions import benchmark_datasets, held_out_null_streams

import run_all_tests as rat

BATCH_SIZES = (1, 10, 50, 100)
OUTPUT_DIR = Path("data") / "batch_checking"

SUMMARY_METRICS = (
    "detection_rate",
    "placebo_detection_rate",
    "median_first_delay",
    "mean_capped_delay",
    "mean_false_positives_common",
)


def run_batch_size(batch_size: int, rescale: bool = rat.RESCALE_BY_CALIBRATION_MEDIAN) -> None:
    if (DRIFT_AT - STARTING_IDX) % batch_size or DETECTION_HORIZON % batch_size:
        raise ValueError(f"batch_size={batch_size} does not divide the drift onset and horizon")
    md.h1(f"Batch checking, batch_size={batch_size}")
    results = [
        rat.run_dataset(
            label,
            *generator(**kwargs),
            drift_at=DRIFT_AT,
            null_data=held_out_null_streams(generator, N_TEST_STREAMS, **kwargs),
            batch_size=batch_size,
            rescale=rescale,
        )
        for label, generator, kwargs in benchmark_datasets()
    ]
    root = OUTPUT_DIR / "median_rescaled" if rescale else OUTPUT_DIR
    output_dir = root / f"batch_{batch_size}"
    provenance = rat.collect_git_provenance(Path(__file__).resolve().parents[1])
    rat.save_benchmark_outputs(results, output_dir, provenance)

    metadata_path = output_dir / "benchmark_run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["experiment"] = "batch_checking"
    metadata["configuration"]["batch_size"] = batch_size
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize(rescale: bool = rat.RESCALE_BY_CALIBRATION_MEDIAN) -> pd.DataFrame:
    frames = []
    root = OUTPUT_DIR / "median_rescaled" if rescale else OUTPUT_DIR
    for batch_dir in sorted(root.glob("batch_*")):
        path = batch_dir / "test_summary_results.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "batch_size", int(batch_dir.name.removeprefix("batch_")))
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No completed runs under {root}")
    summary = pd.concat(frames, ignore_index=True).sort_values(["dataset", "scenario", "batch_size"])
    summary.to_csv(root / "batch_checking_summary.csv", index=False)

    md.h1("Batch checking: detection and delay against batch size")
    md.para(f"Completed batch sizes: {sorted(int(b) for b in summary['batch_size'].unique())}")
    for scenario, group in summary.groupby("scenario", sort=False):
        md.h2(f"{group['dataset'].iloc[0]} / {scenario}")
        for metric in SUMMARY_METRICS:
            md.h3(metric)
            md.table(group.pivot(index="batch_size", columns="algorithm", values=metric).round(3))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--batch-size", type=int, help="run the benchmark at this batch size")
    group.add_argument("--all", action="store_true", help=f"run every batch size in {BATCH_SIZES}, then summarize")
    group.add_argument("--summarize", action="store_true", help="combine the completed runs")
    parser.add_argument("--rescale", action=argparse.BooleanOptionalAction,
                        default=rat.RESCALE_BY_CALIBRATION_MEDIAN,
                        help="divide squared errors by the pooled calibration median (default: off)")
    args = parser.parse_args()
    if args.batch_size is not None:
        run_batch_size(args.batch_size, rescale=args.rescale)
    elif args.all:
        for batch_size in BATCH_SIZES:
            run_batch_size(batch_size, rescale=args.rescale)
        summarize(rescale=args.rescale)
    else:
        summarize(rescale=args.rescale)


if __name__ == "__main__":
    main()
