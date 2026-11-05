"""Statistical significance testing across test streams.

Every detector is evaluated on the *same* test streams, so all comparisons
here are paired by (scenario, stream_id) and consume the per-stream results
DataFrame returned by ``calibrate_and_test`` (one row per
scenario x algorithm x stream).

Tests provided
--------------
- ``wilcoxon_vs_reference`` : paired Wilcoxon signed-rank on a per-stream
  metric (detection delay, common-window false-positive count) of each algorithm against a
  reference algorithm, per scenario, with Holm correction across the whole
  table. Delay comparisons are restricted to streams where *both* detectors
  fired; detection ability itself is tested separately (McNemar), which is the
  standard way to handle the censoring of undetected streams.
- ``mcnemar_vs_reference`` : exact McNemar test on the paired
  detected/missed outcomes per scenario, Holm-corrected.
- ``friedman_nemenyi`` : omnibus Friedman test over all algorithms within one
  scenario, using independent streams as blocks, plus average ranks and the
  Nemenyi critical difference (Demsar 2006). Scenarios may reuse a stream
  seed and must not be pooled as independent blocks. For delays, streams an
  algorithm missed are assigned the worst rank in their block
  (censored-as-worst); blocks missed by every algorithm carry no information
  and are dropped. ``significance_report`` applies Holm correction to the
  scenario-level omnibus tests and Bonferroni correction to Nemenyi families
  across scenarios.
- ``significance_report`` : runs all of the above and pretty-prints them.

References: Demsar (2006) JMLR; Benavoli, Corani & Mangili (2016) JMLR
(pairwise Wilcoxon + Holm preferred over Nemenyi for one-vs-all comparisons).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from wasserstein_score import helper_functions as md


_COMMON_FP_METRIC = "n_false_positives_common"


def _validate_common_fp_exposure(per_stream: pd.DataFrame) -> None:
    """Ensure every paired detector comparison uses the same FP exposure."""
    required = {
        _COMMON_FP_METRIC,
        "common_exposure_start",
        "common_exposure_samples",
    }
    missing = required - set(per_stream.columns)
    if missing:
        raise ValueError(
            "Common-exposure false-positive inference requires columns: "
            + ", ".join(sorted(missing))
        )
    if (per_stream["common_exposure_samples"] <= 0).any():
        raise ValueError("common_exposure_samples must be positive")

    exposure_variants = per_stream.groupby(
        ["scenario", "stream_id"], observed=True
    )[["common_exposure_start", "common_exposure_samples"]].nunique(dropna=False)
    if (exposure_variants > 1).any(axis=None):
        raise ValueError(
            "Common false-positive exposure differs across algorithms within "
            "at least one paired stream."
        )


_FP_SHARED_COLUMNS = [
    _COMMON_FP_METRIC, "common_exposure_start", "common_exposure_samples",
]


def shared_false_positive_groups(per_stream: pd.DataFrame) -> list[list[str]]:
    """Scenarios whose common-window false-positive data coincide.

    Scenarios generated from the same pre-drift streams and calibrated
    together (the River Friedman scenarios, for instance) produce identical
    pre-drift alarms for every detector and stream. Testing their false
    positives once per scenario would repeat one test and inflate its weight;
    each group returned here is tested once. Scenarios are only grouped within
    the same ``dataset`` when that column is present. Groups keep first-seen
    scenario order.
    """
    groups: dict[tuple, list[str]] = {}
    for scenario, block in per_stream.groupby("scenario", sort=False, observed=True):
        frame = (
            block.sort_values(["algorithm", "stream_id"])
            [["algorithm", "stream_id", *_FP_SHARED_COLUMNS]]
            .reset_index(drop=True)
        )
        dataset = tuple(block["dataset"].unique()) if "dataset" in block else ()
        key = (dataset, frame.to_csv(index=False))
        groups.setdefault(key, []).append(scenario)
    return list(groups.values())


def _false_positive_data(per_stream: pd.DataFrame) -> pd.DataFrame:
    """One copy of the per-stream rows of each shared false-positive group,
    labelled with the group's scenarios joined by ``" | "``."""
    frames = []
    for group in shared_false_positive_groups(per_stream):
        frames.append(
            per_stream.loc[per_stream["scenario"] == group[0]]
            .assign(scenario=" | ".join(group))
        )
    return pd.concat(frames, ignore_index=True)


def holm_correction(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (NaNs are passed through untested)."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    pv = p[valid]
    m = len(pv)
    if m == 0:
        return adjusted
    order = np.argsort(pv)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * pv[idx])
        adj[idx] = min(1.0, running_max)
    adjusted[valid] = adj
    return adjusted


def _paired_pivot(per_stream: pd.DataFrame, metric: str) -> pd.DataFrame:
    """(scenario, stream_id) x algorithm matrix for one per-stream metric."""
    return per_stream.pivot_table(
        index=["scenario", "stream_id"], columns="algorithm", values=metric,
        aggfunc="first", dropna=False,
    )


def _safe_wilcoxon(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank robust to degenerate inputs.

    zero_method='zsplit' keeps zero differences (splitting their ranks), which
    matters for zero-inflated metrics like false-positive counts. Returns
    (nan, 1.0) when there are no pairs or every difference is zero.
    """
    diffs = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if len(diffs) == 0 or np.all(diffs == 0):
        return np.nan, 1.0
    try:
        res = stats.wilcoxon(x, y, zero_method="zsplit", method="auto")
        return float(res.statistic), float(res.pvalue)
    except ValueError:
        return np.nan, np.nan


def wilcoxon_vs_reference(
    per_stream: pd.DataFrame,
    metric: str = "first_delay",
    reference: str = "wasserstein",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank of each algorithm vs `reference`, per scenario.

    For ``metric="first_delay"`` only streams where both detectors fired are
    compared (see module docstring). Holm correction is applied across every
    (scenario, algorithm) row of the returned table, so the family-wise error
    rate is controlled over the full set of comparisons reported here.

    Returns one row per (scenario, algorithm != reference) with medians, the
    median paired difference (algorithm - reference; for delays negative means
    the algorithm is faster), raw and Holm-adjusted p-values.
    """
    wide = _paired_pivot(per_stream, metric)
    if reference not in wide.columns:
        raise ValueError(f"Reference algorithm '{reference}' not in results.")

    rows = []
    for scenario in wide.index.get_level_values("scenario").unique():
        block = wide.loc[scenario]
        for algo in wide.columns:
            if algo == reference:
                continue
            paired = block[[reference, algo]].dropna()
            n_pairs = len(paired)
            if n_pairs == 0:
                stat, p = np.nan, np.nan
                med_ref = med_algo = med_diff = np.nan
            else:
                stat, p = _safe_wilcoxon(paired[algo].values, paired[reference].values)
                med_ref = float(paired[reference].median())
                med_algo = float(paired[algo].median())
                med_diff = float((paired[algo] - paired[reference]).median())
            rows.append({
                "scenario": scenario,
                "algorithm": algo,
                "reference": reference,
                "metric": metric,
                "n_pairs": n_pairs,
                f"median_{metric}_ref": med_ref,
                f"median_{metric}_algo": med_algo,
                "median_diff": med_diff,
                "statistic": stat,
                "p_value": p,
            })

    result = pd.DataFrame(rows)
    result["p_holm"] = holm_correction(result["p_value"].values)
    result["significant"] = result["p_holm"] < alpha
    return result


def mcnemar_vs_reference(
    per_stream: pd.DataFrame,
    reference: str = "wasserstein",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Exact McNemar test on paired detected/missed outcomes, per scenario.

    ``ref_only``/``algo_only`` are the discordant counts (streams detected by
    exactly one of the two). The exact test is a two-sided binomial test on
    the discordant pairs; Holm correction across all rows.
    """
    wide = _paired_pivot(per_stream, "detected")
    if reference not in wide.columns:
        raise ValueError(f"Reference algorithm '{reference}' not in results.")

    rows = []
    for scenario in wide.index.get_level_values("scenario").unique():
        block = wide.loc[scenario]
        for algo in wide.columns:
            if algo == reference:
                continue
            paired = block[[reference, algo]].dropna().astype(bool)
            ref_only = int((paired[reference] & ~paired[algo]).sum())
            algo_only = int((~paired[reference] & paired[algo]).sum())
            n_disc = ref_only + algo_only
            p = 1.0 if n_disc == 0 else float(
                stats.binomtest(ref_only, n_disc, 0.5).pvalue
            )
            rows.append({
                "scenario": scenario,
                "algorithm": algo,
                "reference": reference,
                "n_streams": len(paired),
                "rate_ref": float(paired[reference].mean()),
                "rate_algo": float(paired[algo].mean()),
                "ref_only": ref_only,
                "algo_only": algo_only,
                "p_value": p,
            })

    result = pd.DataFrame(rows)
    result["p_holm"] = holm_correction(result["p_value"].values)
    result["significant"] = result["p_holm"] < alpha
    return result


def friedman_nemenyi(
    per_stream: pd.DataFrame,
    metric: str = "first_delay",
    alpha: float = 0.05,
    lower_is_better: bool = True,
) -> dict:
    """Friedman omnibus test + Nemenyi critical difference for one scenario.

    Each independent stream is one block. For ``first_delay``, missed streams
    are censored-as-worst (+inf before ranking); blocks where every algorithm
    missed are dropped. The caller must not pool scenarios: their streams can
    be generated from the same seeds and even have identical pre-drift values.

    Returns a dict with the Friedman statistic/p-value, average ranks (rank 1
    = best), the Nemenyi critical difference at `alpha`, and a boolean matrix
    of pairwise differences exceeding it.
    """
    scenarios = per_stream["scenario"].unique()
    if len(scenarios) != 1:
        raise ValueError("Friedman test requires exactly one scenario at a time.")

    wide = _paired_pivot(per_stream, metric)
    k = wide.shape[1]
    if k < 3:
        raise ValueError("Friedman test needs at least 3 algorithms.")

    matrix = wide.to_numpy(dtype=float)
    if metric == "first_delay":
        matrix = np.where(np.isnan(matrix), np.inf, matrix)
    else:
        matrix = matrix[~np.any(np.isnan(matrix), axis=1)]
    # Blocks where every algorithm ties (all missed, all zero FPs, ...) carry
    # no rank information and make scipy's tie correction degenerate.
    matrix = matrix[~np.all(matrix == matrix[:, [0]], axis=1)]
    if not lower_is_better:
        matrix = -matrix
    n_blocks = matrix.shape[0]
    if n_blocks < 2:
        raise ValueError(f"Only {n_blocks} usable (non-tied) blocks for metric '{metric}'.")

    statistic, p_value = stats.friedmanchisquare(*matrix.T)
    ranks = stats.rankdata(matrix, axis=1)
    avg_ranks = pd.Series(ranks.mean(axis=0), index=wide.columns, name="avg_rank")
    avg_ranks = avg_ranks.sort_values()

    # Demsar (2006): CD = q_alpha * sqrt(k(k+1)/(6N)) with q_alpha = SR/sqrt(2),
    # equivalently CD = SR * sqrt(k(k+1)/(12N)). Dividing SR by sqrt(2) AND
    # using 12N double-counts the sqrt(2), shrinking CD by ~29%.
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf)
    cd = float(q * np.sqrt(k * (k + 1) / (12 * n_blocks)))

    diff = np.abs(avg_ranks.values[:, None] - avg_ranks.values[None, :])
    significant_pairs = pd.DataFrame(
        diff > cd, index=avg_ranks.index, columns=avg_ranks.index
    )

    return {
        "scenario": scenarios[0],
        "metric": metric,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_blocks": int(n_blocks),
        "k": int(k),
        "avg_ranks": avg_ranks,
        "critical_difference": cd,
        "significant_pairs": significant_pairs,
    }


def significance_report(
    per_stream: pd.DataFrame,
    reference: str = "wasserstein",
    alpha: float = 0.05,
    verbose: bool = True,
) -> dict:
    """Run the full significance suite and (optionally) pretty-print it.

    False-positive tests use ``n_false_positives_common``: raw counts from the
    common post-maximum-warm-up window. Detector-specific active-exposure FP
    metrics remain descriptive and are deliberately excluded from inference.
    Scenarios with identical false-positive data (shared pre-drift streams,
    see :func:`shared_false_positive_groups`) are tested once, under a label
    joining their names, so one comparison is neither repeated in the Holm
    family nor reported as several confirmations.

    Returns a dict with keys ``delay_wilcoxon``, ``fp_wilcoxon``,
    ``detection_mcnemar``, ``delay_friedman``, ``fp_friedman``. The Friedman
    values map each scenario name to its test result (or ``None`` if there
    are too few informative blocks). Omnibus p-values receive Holm correction
    across scenarios for each metric. Nemenyi uses ``alpha / n_scenarios``
    and reports pairwise significance only after the corrected omnibus test
    rejects, controlling comparisons across scenario families.
    """
    _validate_common_fp_exposure(per_stream)
    fp_data = _false_positive_data(per_stream)
    results = {
        "delay_wilcoxon": wilcoxon_vs_reference(
            per_stream, "first_delay", reference, alpha
        ),
        "fp_wilcoxon": wilcoxon_vs_reference(
            fp_data, _COMMON_FP_METRIC, reference, alpha
        ),
        "detection_mcnemar": mcnemar_vs_reference(per_stream, reference, alpha),
    }

    for key, metric, data in (("delay_friedman", "first_delay", per_stream),
                              ("fp_friedman", _COMMON_FP_METRIC, fp_data)):
        family: dict[str, dict | None] = {}
        scenarios = data["scenario"].unique()
        pairwise_alpha = alpha / len(scenarios)
        for scenario in scenarios:
            scenario_data = data.loc[data["scenario"] == scenario]
            if scenario_data["algorithm"].nunique() < 3:
                family[scenario] = None
                continue
            try:
                family[scenario] = friedman_nemenyi(
                    scenario_data, metric, pairwise_alpha
                )
            except ValueError as exc:
                family[scenario] = None
                if verbose:
                    md.bullet(f"Friedman on {metric} for {scenario} skipped: {exc}")

        tested = [fr for fr in family.values() if fr is not None]
        adjusted = holm_correction(np.array([fr["p_value"] for fr in tested]))
        for fr, p_holm in zip(tested, adjusted):
            fr["p_holm"] = float(p_holm)
            fr["significant"] = bool(p_holm < alpha)
            fr["pairwise_alpha"] = pairwise_alpha
            # Pairwise claims require both the corrected omnibus rejection
            # and the cross-scenario-adjusted Nemenyi critical difference.
            fr["significant_pairs"] &= fr["significant"]
        results[key] = family

    if verbose:
        md.h2(f"Statistical significance (reference: {reference}, alpha={alpha})")

        md.h3("Detection delay — paired Wilcoxon (jointly detected streams), Holm-corrected")
        md.table(results["delay_wilcoxon"].round(4))

        md.h3(
            "False positives on common exposure window — paired Wilcoxon, "
            "Holm-corrected"
        )
        for group in shared_false_positive_groups(per_stream):
            if len(group) > 1:
                md.bullet(
                    f"{', '.join(group)} share their pre-drift streams and "
                    "calibration (identical false-positive data): tested once."
                )
        md.table(results["fp_wilcoxon"].round(4))

        md.h3("Detection rate — exact McNemar on paired outcomes, Holm-corrected")
        md.table(results["detection_mcnemar"].round(4))

        for key, label in (("delay_friedman", "detection delay (missed = worst rank)"),
                           ("fp_friedman", "false positives (common exposure)")):
            md.h3(f"Friedman + Nemenyi — {label}")
            for scenario, fr in results[key].items():
                if fr is None:
                    md.bullet(f"{scenario}: insufficient informative blocks or algorithms")
                    continue
                md.bullet(
                    f"{scenario}: Friedman chi2={fr['statistic']:.2f}, "
                    f"p={fr['p_value']:.2e}, Holm p={fr['p_holm']:.2e} "
                    f"({fr['n_blocks']} informative independent streams x "
                    f"{fr['k']} algorithms)"
                )
                md.bullet(
                    "Average ranks (1 = best): "
                    + ", ".join(f"{a}={r:.2f}" for a, r in fr["avg_ranks"].items())
                )
                md.bullet(
                    f"Nemenyi critical difference at alpha={fr['pairwise_alpha']:.4g}: "
                    f"{fr['critical_difference']:.3f}; pairwise claims require "
                    "the Holm-corrected Friedman test to reject"
                )

    return results
