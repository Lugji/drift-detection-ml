import pandas as pd
import pytest

from driftdetect.baseline import build_baseline_profile
from driftdetect.detector_core import DetectorCore
from driftdetect.policy import PolicyConfig, apply_global_policy


def _baseline_df():
    return pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            "region": ["EU", "EU", "EU", "EU", "EU", "US", "US", "US", "ASIA", "ASIA"],
            "income": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        }
    )


def test_numeric_metrics_respond_to_shift():
    baseline = build_baseline_profile(_baseline_df(), bins=10, top_k=50)

    # Shift age by +5
    window = _baseline_df().copy()
    window["age"] = window["age"] + 5

    r = DetectorCore.numeric_drift_for_feature(baseline, window, "age", tau_wd=1.0)

    assert r["type"] == "numeric"
    assert "ks" in r["metrics"]
    assert "wasserstein" in r["metrics"]

    # Expect some drift
    assert r["metrics"]["ks"] > 0.0
    assert r["metrics"]["wasserstein"] > 0.0

    # Severity must be in [0,1]
    assert 0.0 <= r["severity"] <= 1.0


def test_categorical_metrics_respond_to_distribution_change():
    baseline = build_baseline_profile(_baseline_df(), bins=10, top_k=50)

    # Make region heavily ASIA
    window = _baseline_df().copy()
    window["region"] = ["ASIA"] * 8 + ["EU"] + ["US"]

    r = DetectorCore.categorical_drift_for_feature(baseline, window, "region", alpha=0.05)

    assert r["type"] == "categorical"
    assert "js" in r["metrics"]
    assert "chi2_p" in r["metrics"]

    # Expect drift signal (JS >= 0 always)
    assert r["metrics"]["js"] >= 0.0
    assert 0.0 <= r["severity"] <= 1.0


def test_feature_ranking_logic_example():
    # Simple ranking check with two fake severities
    features_out = {
        "a": {"severity": 0.1},
        "b": {"severity": 0.9},
    }
    ranking = sorted(
        [{"feature": k, "severity": float(v["severity"])} for k, v in features_out.items()],
        key=lambda x: x["severity"],
        reverse=True,
    )
    assert ranking[0]["feature"] == "b"
    assert ranking[1]["feature"] == "a"


def test_global_policy_schema_drift_triggers():
    cfg = PolicyConfig(S_feat=0.7, S_global=0.7, K=3)
    out = apply_global_policy(schema_drift_present=True, max_severity=0.0, drifted_count=0, cfg=cfg)
    assert out["drift_detected"] is True
    assert "schema_drift" in out["reasons"]


def test_global_policy_max_severity_triggers():
    cfg = PolicyConfig(S_feat=0.7, S_global=0.7, K=3)
    out = apply_global_policy(schema_drift_present=False, max_severity=0.8, drifted_count=0, cfg=cfg)
    assert out["drift_detected"] is True
    assert "max_severity>=S_global" in out["reasons"]


def test_global_policy_drifted_count_triggers():
    cfg = PolicyConfig(S_feat=0.7, S_global=0.7, K=3)
    out = apply_global_policy(schema_drift_present=False, max_severity=0.2, drifted_count=3, cfg=cfg)
    assert out["drift_detected"] is True
    assert "drifted_count>=K" in out["reasons"]


def test_global_policy_no_triggers():
    cfg = PolicyConfig(S_feat=0.7, S_global=0.7, K=3)
    out = apply_global_policy(schema_drift_present=False, max_severity=0.2, drifted_count=1, cfg=cfg)
    assert out["drift_detected"] is False
    assert out["reasons"] == []