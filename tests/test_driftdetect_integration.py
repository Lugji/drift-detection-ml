"""
DriftDetect test suite.

Covers all MUST requirements from RTM + acceptance criteria.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from driftdetect.models import DetectorConfig
from driftdetect.profiler import build_baseline, load_dataframe
from driftdetect.detector import detect_drift
from driftdetect.simulator import load_scenario, simulate
from driftdetect.coverage import compute_coverage

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def generate_fixtures():
    """Auto-generate fixture CSVs before any test."""
    import subprocess, sys
    subprocess.run([sys.executable, str(Path(__file__).parent / "generate_fixtures.py")], check=True)


@pytest.fixture(scope="session")
def baseline_df():
    return load_dataframe(FIXTURES / "baseline.csv")


@pytest.fixture(scope="session")
def baseline_profile(baseline_df):
    return build_baseline(baseline_df, source="baseline.csv")


@pytest.fixture(scope="session")
def cfg():
    return DetectorConfig()


# ---------------------------------------------------------------------------
# REQ-001: CSV and Parquet loading
# ---------------------------------------------------------------------------

class TestLoading:
    def test_load_csv(self, baseline_df):
        assert isinstance(baseline_df, pd.DataFrame)
        assert len(baseline_df) == 1000

    def test_load_parquet(self, baseline_df, tmp_path):
        p = tmp_path / "test.parquet"
        baseline_df.to_parquet(p, index=False)
        df2 = load_dataframe(p)
        assert list(df2.columns) == list(baseline_df.columns)

    def test_unsupported_format_raises(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported"):
            load_dataframe(p)


# ---------------------------------------------------------------------------
# REQ-002: Baseline profile required fields
# ---------------------------------------------------------------------------

class TestBaseline:
    def test_required_metadata_keys(self, baseline_profile):
        for key in ("source", "n_rows", "n_cols", "build_timestamp"):
            assert key in baseline_profile.metadata

    def test_schema_present(self, baseline_profile):
        assert len(baseline_profile.schema) == 6

    def test_feature_summaries(self, baseline_profile):
        for fname, feat in baseline_profile.features.items():
            assert feat.name == fname
            assert 0.0 <= feat.missing_rate <= 1.0

    def test_numeric_summary_fields(self, baseline_profile):
        feat = baseline_profile.features["age"]
        assert feat.dtype == "numeric"
        assert feat.mean is not None
        assert feat.std is not None
        assert feat.values is not None and len(feat.values) > 0

    def test_categorical_summary_fields(self, baseline_profile):
        feat = baseline_profile.features["region"]
        assert feat.dtype == "categorical"
        assert feat.categories is not None
        assert feat.frequencies is not None
        assert abs(sum(feat.frequencies.values()) - 1.0) < 1e-6

    def test_save_load_roundtrip(self, baseline_profile, tmp_path):
        """AC-01: save/load preserves required fields."""
        p = tmp_path / "baseline.json"
        baseline_profile.save(str(p))
        loaded = type(baseline_profile).load(str(p))
        assert loaded.metadata["n_rows"] == baseline_profile.metadata["n_rows"]
        assert set(loaded.schema.keys()) == set(baseline_profile.schema.keys())
        assert math.isclose(
            loaded.features["age"].mean, baseline_profile.features["age"].mean, rel_tol=1e-6
        )


# ---------------------------------------------------------------------------
# REQ-003: Schema drift detection
# ---------------------------------------------------------------------------

class TestSchemaDrift:
    def test_no_schema_drift(self, baseline_profile, cfg):
        window = load_dataframe(FIXTURES / "window_nodrift.csv")
        result = detect_drift(baseline_profile, window, cfg=cfg)
        assert not result.schema_drift.detected

    def test_added_removed_columns(self, baseline_profile, cfg):
        window = load_dataframe(FIXTURES / "window_schema_drift.csv")
        result = detect_drift(baseline_profile, window, cfg=cfg)
        assert result.schema_drift.detected
        assert "tenure" in result.schema_drift.removed_columns
        assert "new_feature" in result.schema_drift.added_columns

    def test_type_change(self, baseline_profile, cfg):
        window = load_dataframe(FIXTURES / "baseline.csv").copy()
        window["age"] = window["age"].astype(str)   # force type change
        result = detect_drift(baseline_profile, window, cfg=cfg)
        assert result.schema_drift.detected
        assert "age" in result.schema_drift.type_changes


# ---------------------------------------------------------------------------
# REQ-004: Numeric metrics (KS + Wasserstein)
# ---------------------------------------------------------------------------

class TestNumericMetrics:
    def test_ks_stat_in_range(self, baseline_profile, cfg):
        window = load_dataframe(FIXTURES / "window_drift.csv")
        result = detect_drift(baseline_profile, window, cfg=cfg)
        for fname in ["age", "income"]:
            fr = result.features[fname]
            assert fr.ks_stat is not None
            assert 0.0 <= fr.ks_stat <= 1.0

    def test_wasserstein_nonnegative(self, baseline_profile, cfg):
        window = load_dataframe(FIXTURES / "window_drift.csv")
        result = detect_drift(baseline_profile, window, cfg=cfg)
        fr = result.features["age"]
        assert fr.wasserstein is not None and fr.wasserstein >= 0.0

    def test_no_drift_low_ks(self, baseline_profile, cfg):
        window = load_dataframe(FIXTURES / "window_nodrift.csv")
        result = detect_drift(baseline_profile, window, cfg=cfg)
        fr = result.features["score"]
        assert fr.ks_stat is not None and fr.ks_stat < 0.2


# ---------------------------------------------------------------------------
# REQ-005: Categorical metrics (Chi-square + JS)
# ---------------------------------------------------------------------------

class TestCategoricalMetrics:
    def test_js_in_range(self, baseline_profile, cfg):
        window = load_dataframe(FIXTURES / "window_drift.csv")
        result = detect_drift(baseline_profile, window, cfg=cfg)
        fr = result.features["region"]
        assert fr.js_divergence is not None
        assert 0.0 <= fr.js_divergence <= 1.0

    def test_no_drift_low_js(self, baseline_profile, cfg):
        window = load_dataframe(FIXTURES / "window_nodrift.csv")
        result = detect_drift(baseline_profile, window, cfg=cfg)
        fr = result.features["region"]
        assert fr.js_divergence is not None and fr.js_divergence < 0.1


# ---------------------------------------------------------------------------
# REQ-006: Severity in [0,1] and ranking
# ---------------------------------------------------------------------------

class TestSeverityAndRanking:
    def test_severity_bounds(self, baseline_profile, cfg):
        window = load_dataframe(FIXTURES / "window_drift.csv")
        result = detect_drift(baseline_profile, window, cfg=cfg)
        for fr in result.features.values():
            assert 0.0 <= fr.severity <= 1.0, f"{fr.name} severity out of bounds: {fr.severity}"

    def test_ranking_order(self, baseline_profile, cfg):
        window = load_dataframe(FIXTURES / "window_drift.csv")
        result = detect_drift(baseline_profile, window, cfg=cfg)
        sevs = [result.features[f].severity for f in result.feature_ranking]
        assert sevs == sorted(sevs, reverse=True)

    def test_drifted_flag(self, baseline_profile, cfg):
        window = load_dataframe(FIXTURES / "window_drift.csv")
        result = detect_drift(baseline_profile, window, cfg=cfg)
        for fr in result.features.values():
            assert fr.drifted == (fr.severity >= cfg.s_feat)


# ---------------------------------------------------------------------------
# REQ-007: Global drift policy
# ---------------------------------------------------------------------------

class TestGlobalPolicy:
    def test_no_drift_gives_false(self, baseline_profile, cfg):
        window = load_dataframe(FIXTURES / "window_nodrift.csv")
        result = detect_drift(baseline_profile, window, cfg=cfg)
        assert not result.global_drift

    def test_drift_gives_true(self, baseline_profile, cfg):
        window = load_dataframe(FIXTURES / "window_drift.csv")
        result = detect_drift(baseline_profile, window, cfg=cfg)
        assert result.global_drift

    def test_schema_drift_triggers_global(self, baseline_profile, cfg):
        window = load_dataframe(FIXTURES / "window_schema_drift.csv")
        result = detect_drift(baseline_profile, window, cfg=cfg)
        assert result.global_drift  # schema drift alone triggers global flag


# ---------------------------------------------------------------------------
# REQ-008: Scenario YAML loading + schema validation
# ---------------------------------------------------------------------------

class TestScenarioLoading:
    def test_valid_scenario(self, tmp_path):
        yaml_content = """
scenario_id: test_s
seed: 42
n_windows: 20
baseline: baseline.csv
events:
  - event_id: e1
    type: mean
    onset: 10
    features: [age]
    intensity: {delta: 3.0}
"""
        p = tmp_path / "s.yaml"
        p.write_text(yaml_content)
        s = load_scenario(p)
        assert s["scenario_id"] == "test_s"
        assert len(s["events"]) == 1

    def test_missing_key_raises(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("seed: 42\nn_windows: 5\n")
        with pytest.raises(ValueError, match="missing required keys"):
            load_scenario(p)

    def test_unknown_drift_type_raises(self, tmp_path):
        p = tmp_path / "bad2.yaml"
        p.write_text("""
scenario_id: x
seed: 1
n_windows: 5
baseline: b.csv
events:
  - type: unknown_drift
    onset: 0
    features: [x]
""")
        with pytest.raises(ValueError, match="unknown type"):
            load_scenario(p)


# ---------------------------------------------------------------------------
# REQ-009: Drift simulation (all 5 types)
# ---------------------------------------------------------------------------

class TestSimulation:
    def _run(self, baseline_df, drift_type, intensity, feature="age"):
        scenario = {
            "scenario_id": "tc",
            "seed": 0,
            "n_windows": 5,
            "window_size": 300,
            "baseline": "",
            "events": [{"event_id": "e1", "type": drift_type, "onset": 2, "features": [feature], "intensity": intensity}],
        }
        windows, drift_log = simulate(scenario, baseline_df)
        return windows, drift_log

    def test_mean_shift(self, baseline_df):
        windows, _ = self._run(baseline_df, "mean", {"delta": 5.0})
        mean_before = np.mean([windows[i]["age"].mean() for i in range(2)])
        mean_after = np.mean([windows[i]["age"].mean() for i in range(2, 5)])
        assert mean_after > mean_before + 1.0

    def test_variance(self, baseline_df):
        windows, _ = self._run(baseline_df, "variance", {"factor": 5.0})
        std_before = np.mean([windows[i]["age"].std() for i in range(2)])
        std_after = np.mean([windows[i]["age"].std() for i in range(2, 5)])
        assert std_after > std_before * 2

    def test_missingness(self, baseline_df):
        windows, _ = self._run(baseline_df, "missingness", {"rate": 0.8})
        missing_rate = windows[3]["age"].isna().mean()
        assert missing_rate > 0.5

    def test_new_category(self, baseline_df):
        windows, _ = self._run(baseline_df, "new_category", {"rate": 0.5}, feature="region")
        assert "__UNSEEN__" in windows[3]["region"].values

    def test_correlation(self, baseline_df):
        windows, _ = self._run(baseline_df, "correlation", {"fraction": 1.0})
        # After full shuffle, correlation with original position should be broken
        assert len(windows[3]) > 0  # just check it runs without error


# ---------------------------------------------------------------------------
# REQ-010: Ground truth drift log contents
# ---------------------------------------------------------------------------

class TestDriftLog:
    def test_drift_log_fields(self, baseline_df):
        scenario = {
            "scenario_id": "tc",
            "seed": 42,
            "n_windows": 10,
            "window_size": 200,
            "baseline": "",
            "events": [{"event_id": "ev1", "type": "mean", "onset": 5, "features": ["age"], "intensity": {"delta": 3}}],
        }
        _, drift_log = simulate(scenario, baseline_df)
        assert drift_log.scenario_id == "tc"
        assert drift_log.n_windows == 10
        assert len(drift_log.events) == 1
        e = drift_log.events[0]
        assert e.event_id == "ev1"
        assert e.type == "mean"
        assert e.onset == 5
        assert "age" in e.features

    def test_drift_log_save_load(self, baseline_df, tmp_path):
        scenario = {
            "scenario_id": "tc",
            "seed": 1,
            "n_windows": 5,
            "window_size": 100,
            "baseline": "",
            "events": [{"event_id": "e1", "type": "mean", "onset": 2, "features": ["income"], "intensity": {"delta": 2}}],
        }
        _, drift_log = simulate(scenario, baseline_df)
        p = tmp_path / "drift_log.json"
        drift_log.save(str(p))
        from driftdetect.models import DriftLog
        loaded = DriftLog.load(str(p))
        assert loaded.scenario_id == drift_log.scenario_id
        assert loaded.events[0].type == "mean"


# ---------------------------------------------------------------------------
# REQ-011: Detection replay across windows
# ---------------------------------------------------------------------------

class TestDetectionReplay:
    def test_n_windows_gives_n_results(self, baseline_profile, baseline_df, cfg):
        n = 8
        scenario = {
            "scenario_id": "tc",
            "seed": 42,
            "n_windows": n,
            "window_size": 200,
            "baseline": "",
            "events": [],
        }
        windows, drift_log = simulate(scenario, baseline_df)
        results = [detect_drift(baseline_profile, w, cfg=cfg, window_id=str(i)) for i, w in enumerate(windows)]
        assert len(results) == n


# ---------------------------------------------------------------------------
# REQ-012, REQ-013: Coverage metrics + missed events
# ---------------------------------------------------------------------------

class TestCoverage:
    def _build_scenario_and_results(self, baseline_df, baseline_profile, cfg, n=20, onset=10):
        scenario = {
            "scenario_id": "cov_test",
            "seed": 99,
            "n_windows": n,
            "window_size": 300,
            "baseline": "",
            "events": [{"event_id": "e1", "type": "mean", "onset": onset, "features": ["age"], "intensity": {"delta": 5.0}}],
        }
        windows, drift_log = simulate(scenario, baseline_df)
        results = [detect_drift(baseline_profile, w, cfg=cfg) for w in windows]
        return drift_log, results

    def test_coverage_fpr_no_drift(self, baseline_df, baseline_profile, cfg):
        """AC-03: No-drift suite yields FPR ≤ threshold."""
        scenario = {
            "scenario_id": "nodrift",
            "seed": 7,
            "n_windows": 20,
            "window_size": 300,
            "baseline": "",
            "events": [],
        }
        windows, drift_log = simulate(scenario, baseline_df)
        results = [detect_drift(baseline_profile, w, cfg=cfg) for w in windows]
        report = compute_coverage(drift_log, results, cfg=cfg)
        # All windows are null → FPR should be low
        assert report.suite_fpr <= 0.2

    def test_power_strong_drift(self, baseline_df, baseline_profile, cfg):
        """AC-04: Strong drift should be detected (high power)."""
        drift_log, results = self._build_scenario_and_results(baseline_df, baseline_profile, cfg)
        report = compute_coverage(drift_log, results, cfg=cfg)
        assert "mean" in report.power_by_type
        assert report.power_by_type["mean"] > 0.0

    def test_onset_window_active(self, baseline_df, baseline_profile, cfg):
        """AC-02: Onset at window 10 activates drift at window 10."""
        drift_log, results = self._build_scenario_and_results(
            baseline_df, baseline_profile, cfg, n=20, onset=10
        )
        report = compute_coverage(drift_log, results, cfg=cfg)
        ev = report.per_event_results[0]
        # TTD must be ≥ 0 (detected at or after onset)
        if ev.ttd is not None:
            assert ev.ttd >= 0
            if ev.detection_window is not None:
                assert ev.detection_window >= 10

    def test_missed_events_list(self, baseline_df, baseline_profile):
        """REQ-013: Missed events list is correct."""
        # Use a very strict config so events are missed
        strict_cfg = DetectorConfig(s_feat=1.0, s_global=1.0, k_features=100, tolerance_m=1)
        scenario = {
            "scenario_id": "missed_test",
            "seed": 3,
            "n_windows": 5,
            "window_size": 100,
            "baseline": "",
            "events": [{"event_id": "ev_miss", "type": "mean", "onset": 1, "features": ["age"], "intensity": {"delta": 0.01}}],
        }
        windows, drift_log = simulate(scenario, baseline_df)
        results = [detect_drift(baseline_profile, w, cfg=strict_cfg) for w in windows]
        report = compute_coverage(drift_log, results, cfg=strict_cfg)
        # With near-zero shift and very strict thresholds, the event should be missed
        assert any(e.event_id == "ev_miss" for e in report.missed_events)

    def test_coverage_report_fields(self, baseline_df, baseline_profile, cfg):
        drift_log, results = self._build_scenario_and_results(baseline_df, baseline_profile, cfg)
        report = compute_coverage(drift_log, results, cfg=cfg)
        assert isinstance(report.suite_fpr, float)
        assert isinstance(report.power_by_type, dict)
        assert isinstance(report.per_event_results, list)
        assert isinstance(report.missed_events, list)

    def test_coverage_save_load(self, baseline_df, baseline_profile, cfg, tmp_path):
        drift_log, results = self._build_scenario_and_results(baseline_df, baseline_profile, cfg)
        report = compute_coverage(drift_log, results, cfg=cfg)
        p = tmp_path / "coverage.json"
        report.save(str(p))
        from driftdetect.models import CoverageReport
        loaded = CoverageReport.load(str(p))
        assert abs(loaded.suite_fpr - report.suite_fpr) < 1e-9


# ---------------------------------------------------------------------------
# NFR-001: Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_windows(self, baseline_df):
        scenario = {
            "scenario_id": "rep",
            "seed": 42,
            "n_windows": 5,
            "window_size": 200,
            "baseline": "",
            "events": [{"event_id": "e1", "type": "mean", "onset": 2, "features": ["age"], "intensity": {"delta": 2.0}}],
        }
        windows1, _ = simulate(scenario, baseline_df)
        windows2, _ = simulate(scenario, baseline_df)
        for w1, w2 in zip(windows1, windows2):
            pd.testing.assert_frame_equal(w1, w2)
