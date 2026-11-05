import numpy as np
import pandas as pd
import pytest

from wasserstein_score.significance import (
    friedman_nemenyi,
    holm_correction,
    mcnemar_vs_reference,
    shared_false_positive_groups,
    significance_report,
)


def test_holm_correction_controls_monotonic_adjusted_values():
    adjusted = holm_correction(np.array([0.01, 0.04, 0.03, np.nan]))

    np.testing.assert_allclose(adjusted[:3], [0.03, 0.06, 0.06])
    assert np.isnan(adjusted[3])


def test_mcnemar_uses_paired_detection_outcomes():
    frame = pd.DataFrame(
        {
            "scenario": ["s"] * 8,
            "stream_id": [0, 1, 2, 3] * 2,
            "algorithm": ["wasserstein"] * 4 + ["adwin"] * 4,
            "detected": [True, True, False, False, True, False, False, False],
        }
    )

    result = mcnemar_vs_reference(frame, reference="wasserstein")

    assert len(result) == 1
    assert result.loc[0, "ref_only"] == 1
    assert result.loc[0, "algo_only"] == 0
    assert result.loc[0, "n_streams"] == 4


def _common_exposure_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario": ["s"] * 8,
            "stream_id": [0, 1, 2, 3] * 2,
            "algorithm": ["wasserstein"] * 4 + ["adwin"] * 4,
            "first_delay": [1.0] * 8,
            "detected": [True] * 8,
            "n_false_positives_common": [0] * 4 + [1] * 4,
            # Deliberately points in the opposite direction: inference must
            # use the common-window count, not this legacy active count.
            "n_false_positives": [5] * 4 + [0] * 4,
            "common_exposure_start": [100] * 8,
            "common_exposure_samples": [900] * 8,
        }
    )


def test_significance_report_uses_common_exposure_false_positive_counts():
    results = significance_report(_common_exposure_frame(), verbose=False)
    fp_result = results["fp_wilcoxon"].iloc[0]

    assert fp_result["metric"] == "n_false_positives_common"
    assert fp_result["median_n_false_positives_common_ref"] == 0
    assert fp_result["median_n_false_positives_common_algo"] == 1


def test_significance_report_rejects_mismatched_common_exposure():
    frame = _common_exposure_frame()
    frame.loc[frame["algorithm"] == "adwin", "common_exposure_start"] = 101

    with pytest.raises(ValueError, match="differs across algorithms"):
        significance_report(frame, verbose=False)


def _reused_stream_scenarios() -> pd.DataFrame:
    rows = []
    for scenario in ("river_abrupt", "river_gradual"):
        for stream_id in range(6):
            for rank, algorithm in enumerate(("wasserstein", "adwin", "kswin")):
                rows.append({
                    "scenario": scenario,
                    "stream_id": stream_id,
                    "algorithm": algorithm,
                    "first_delay": float(stream_id + rank + 1),
                    "detected": True,
                    # The same pre-drift stream is reused across scenarios.
                    "n_false_positives_common": rank,
                    "common_exposure_start": 100,
                    "common_exposure_samples": 900,
                })
    return pd.DataFrame(rows)


def test_false_positive_groups_require_identical_data_within_a_dataset():
    frame = _reused_stream_scenarios()
    assert shared_false_positive_groups(frame) == [["river_abrupt", "river_gradual"]]
    # One differing count separates the scenarios.
    changed = frame.copy()
    changed.loc[changed.index[-1], "n_false_positives_common"] += 1
    assert shared_false_positive_groups(changed) == [["river_abrupt"], ["river_gradual"]]
    # Identical data from different datasets are not pooled.
    split = frame.assign(dataset=np.where(frame["scenario"] == "river_abrupt", "a", "b"))
    assert shared_false_positive_groups(split) == [["river_abrupt"], ["river_gradual"]]


def test_friedman_rejects_pooled_scenarios_with_reused_streams():
    with pytest.raises(ValueError, match="exactly one scenario"):
        friedman_nemenyi(_reused_stream_scenarios())


def test_friedman_report_uses_independent_streams_within_each_scenario(capsys):
    results = significance_report(_reused_stream_scenarios(), verbose=True)
    report = capsys.readouterr().out
    assert "river_abrupt: Friedman" in report
    assert "river_gradual: Friedman" in report

    # Delays differ by scenario; the reused pre-drift streams give identical
    # false-positive data, which is tested once.
    pooled = "river_abrupt | river_gradual"
    assert "share their pre-drift streams" in report
    assert set(results["fp_wilcoxon"]["scenario"]) == {pooled}
    for key, scenarios in (("delay_friedman", {"river_abrupt", "river_gradual"}),
                           ("fp_friedman", {pooled})):
        family = results[key]
        assert set(family) == scenarios
        assert all(fr["n_blocks"] == 6 for fr in family.values())
        assert all(fr["pairwise_alpha"] == pytest.approx(0.05 / len(scenarios))
                   for fr in family.values())
        raw = np.array([fr["p_value"] for fr in family.values()])
        adjusted = holm_correction(raw)
        np.testing.assert_allclose(
            [fr["p_holm"] for fr in family.values()], adjusted
        )
        for fr in family.values():
            assert not fr["significant_pairs"].to_numpy().any() or fr["significant"]
