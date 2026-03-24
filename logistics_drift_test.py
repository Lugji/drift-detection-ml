"""
logistics_drift_test.py

Connects the prof's order simulator to DriftDetect.
Three drift scenarios on realistic logistics data:
  1. Weight mean shift  — supplier changes product line (CUST_A heavier packages)
  2. New unit category  — new fleet vehicle type appears (CUST_B unit_type 80)
  3. Day pattern shift  — CUST_A moves preferred delivery day Tue→Fri

Requires: order_simulator.py in the same folder (prof's file).

Usage:
    python logistics_drift_test.py
"""

from __future__ import annotations

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── DriftDetect imports (new combined API) ────────────────────────────────────
from driftdetect.baseline import build_baseline_profile
from driftdetect.detector_facade import DetectorConfig, DetectorFacade
from driftdetect.coverage import compute_coverage

# ── Prof's simulator ──────────────────────────────────────────────────────────
try:
    from order_simulator import generate_order_df
except ImportError:
    raise ImportError(
        "Place the prof's 'order_simulator.py' next to this script and try again."
    )

# ── Simulator constants ───────────────────────────────────────────────────────
ORDER_TYPE_SPORADIC = 0
ORDER_TYPE_WEEKLY   = 1
ORDER_TYPE_BIWEEKLY = 2
ORDER_FREQ_MAIN_DAY = 1
ORDER_FREQ_TWO_DAYS = 2
ORDER_FREQ_ALL_DAYS = 3
UNIT_TYPES = [10, 18, 20, 22, 24, 30, 40, 80]


# ── Customer profiles ─────────────────────────────────────────────────────────

def make_stable_profiles(seed: int = 0) -> list[dict]:
    np.random.seed(seed)
    return [
        {
            "Customer_ID": "CUST_A",
            "Total_Volume": 200,
            "Number_of_Connections": 1,
            "Connections": [{
                "Connection_ID": "CONN_A1",
                "Order_Distribution": ORDER_FREQ_MAIN_DAY,
                "Order_Frequency": ORDER_TYPE_WEEKLY,
                "Order_Probability": 1.0,
                "Preferred_Days": 2,          # Tuesday
                "Max_Orders_Per_Day": 8,
                "Unit_Type": 20,
                "Unit_Type_Std_Dev": 0.5,
                "Weight_Distribution_Components": [(15.0, 2.0, 0.7), (25.0, 3.0, 0.3)],
                "Arrival_Hour": 9,
                "Delivery_Deadline": 17,
            }],
        },
        {
            "Customer_ID": "CUST_B",
            "Total_Volume": 150,
            "Number_of_Connections": 1,
            "Connections": [{
                "Connection_ID": "CONN_B1",
                "Order_Distribution": ORDER_FREQ_ALL_DAYS,
                "Order_Frequency": ORDER_TYPE_WEEKLY,
                "Order_Probability": 1.0,
                "Preferred_Days": 3,
                "Max_Orders_Per_Day": 6,
                "Unit_Type": 30,
                "Unit_Type_Std_Dev": 1.0,
                "Weight_Distribution_Components": [(20.0, 3.0, 1.0)],
                "Arrival_Hour": 10,
                "Delivery_Deadline": 16,
            }],
        },
    ]


def make_drifted_weight(seed: int = 0) -> list[dict]:
    """CUST_A switches to heavier packages (supplier change)."""
    p = make_stable_profiles(seed)
    p[0]["Connections"][0]["Weight_Distribution_Components"] = [
        (35.0, 4.0, 0.6), (50.0, 5.0, 0.4),
    ]
    return p


def make_drifted_unit_type(seed: int = 0) -> list[dict]:
    """CUST_B starts using unit type 80 (new fleet vehicle)."""
    p = make_stable_profiles(seed)
    p[1]["Connections"][0]["Unit_Type"] = 80
    p[1]["Connections"][0]["Unit_Type_Std_Dev"] = 0.0
    return p


def make_drifted_day(seed: int = 0) -> list[dict]:
    """CUST_A shifts preferred delivery day Tue→Fri."""
    p = make_stable_profiles(seed)
    p[0]["Connections"][0]["Preferred_Days"] = 5   # Friday
    return p


# ── Simulation helpers ────────────────────────────────────────────────────────

def simulate_weeks(profiles: list[dict], n_weeks: int, seed: int) -> pd.DataFrame:
    np.random.seed(seed)
    return generate_order_df(
        profiles=profiles,
        UT_list=UNIT_TYPES,
        sim_weeks=n_weeks,
        sample_time=False,
        sample_from_connections=False,
    )


def weeks_to_windows(df: pd.DataFrame, window_size_weeks: int = 2) -> list[pd.DataFrame]:
    """Merge weeks into batches to reduce per-window sampling noise."""
    all_weeks = sorted(df["Week"].unique())
    windows = []
    for i in range(0, len(all_weeks), window_size_weeks):
        batch = all_weeks[i:i + window_size_weeks]
        chunk = df[df["Week"].isin(batch)].drop(columns=["Week"]).reset_index(drop=True)
        if len(chunk) > 0:
            windows.append(chunk)
    return windows


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_experiment(
    name: str,
    drifted_profiles: list[dict],
    drift_feature: str,
    drift_type: str,
    baseline: object,
    cfg: DetectorConfig,
    n_pre: int = 5,
    n_post: int = 10,
    M: int = 2,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    stable_df  = simulate_weeks(make_stable_profiles(), n_pre,  seed=1)
    drifted_df = simulate_weeks(drifted_profiles,       n_post, seed=2)
    drifted_df["Week"] = drifted_df["Week"] + n_pre

    combined = pd.concat([stable_df, drifted_df], ignore_index=True)
    windows  = weeks_to_windows(combined)
    n_total  = len(windows)

    # Run detection on every window
    det = DetectorFacade(baseline=baseline, cfg=cfg)
    g_flags = []
    all_results = []
    import datetime
    history = {"windows": []}

    for i, w in enumerate(windows):
        result = det.detect_window(w, window_id=f"week_{i:02d}")
        detected = result["global"]["drift_detected"]
        g_flags.append(detected)
        all_results.append(result)
        flag = "⚠ DRIFT" if detected else "  ok   "
        ranking = result.get("feature_ranking", [])
        top = ranking[0]["feature"] if ranking else "—"
        top_sev = result["features"].get(top, {}).get("severity", 0.0)
        print(f"  Week {i:02d} [{flag}]  top: {top:15s}  sev={top_sev:.3f}")

        # Append to history for dashboard
        history["windows"].append({
            "window_id": f"week_{i:02d}",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "global": result["global"],
            "features": {
                k: {"severity": v.get("severity", 0.0), "drifted": v.get("drifted", False)}
                for k, v in result["features"].items()
            },
        })

    # Save artifacts for dashboard
    out_dir = Path(f"logistics_sim_{name}")
    out_dir.mkdir(exist_ok=True)
    Path(out_dir / "drift_latest.json").write_text(
        json.dumps(all_results[-1], indent=2), encoding="utf-8"
    )
    Path(out_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    # Build ground truth drift log for coverage
    drift_log = [{
        "event_id":          f"e_{name}",
        "type":              drift_type,
        "start_window":      n_pre // 2,  # adjusted for 2-week windows
        "affected_features": [drift_feature],
        "intensity":         {},
    }]

    # Compute coverage
    cov = compute_coverage(drift_log=drift_log, g_flags=g_flags, M=M)
    m = cov["metrics"]

    Path(out_dir / "coverage.json").write_text(
        json.dumps(cov, indent=2), encoding="utf-8"
    )
    print(f"  Saved artifacts → {out_dir}/")

    print(f"\n  FPR    : {m['FPR']:.3f}")
    print(f"  Power  : {m['power_overall']:.3f}")
    ttd = cov["per_event"][0]["TTD"]
    print(f"  TTD    : {ttd if ttd is not None else '∞'} weeks")
    if cov["missed_events"]:
        print(f"  ⚠ MISSED — peak severity did not cross threshold")

    return cov


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = DetectorConfig(S_feat=0.7, S_global=0.7, K=2)

    # Build baseline from 20 stable weeks (more data = more stable baseline)
    print("Building baseline from 20 stable weeks...")
    baseline_df = simulate_weeks(make_stable_profiles(), n_weeks=20, seed=0)
    baseline = build_baseline_profile(baseline_df.drop(columns=["Week"]))
    print(f"  Features : {list(baseline.features.keys())}")
    print(f"  Rows     : {baseline.metadata.get('rows', '?')}")

    # Save baseline for dashboard use
    Path("logistics_baseline.json").write_text(
        json.dumps(baseline.model_dump(by_alias=True), indent=2), encoding="utf-8"
    )
    print("  Saved  → logistics_baseline.json")

    # Experiment 1: Weight mean shift
    run_experiment(
        name="weight_mean_shift",
        drifted_profiles=make_drifted_weight(),
        drift_feature="Weight",
        drift_type="numeric_shift",
        baseline=baseline,
        cfg=cfg,
        n_pre=6, n_post=10,
    )

    # Experiment 2: New unit type
    run_experiment(
        name="new_unit_type",
        drifted_profiles=make_drifted_unit_type(),
        drift_feature="Unit_Type",
        drift_type="new_category",
        baseline=baseline,
        cfg=cfg,
        n_pre=6, n_post=10,
    )

    # Experiment 3: Day pattern shift
    run_experiment(
        name="day_pattern_shift",
        drifted_profiles=make_drifted_day(),
        drift_feature="Day",
        drift_type="numeric_shift",
        baseline=baseline,
        cfg=cfg,
        n_pre=6, n_post=10,
    )

    print("\n✅ All experiments complete.")