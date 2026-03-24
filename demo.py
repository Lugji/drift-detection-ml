"""
demo.py — end-to-end sanity check for DriftDetect (combined version).

Runs the full pipeline:
  1. Generate synthetic baseline and window data
  2. Build baseline profile
  3. Run detection (no-drift window + drift window)
  4. Simulate a scenario
  5. Run coverage
  6. Write HTML report

All output goes to demo_output/.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from driftdetect.baseline import build_baseline_profile
from driftdetect.detector_facade import DetectorConfig, DetectorFacade
from driftdetect.reporter import save_report_from_files

OUT = Path("demo_output")
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir()

rng = np.random.default_rng(42)

# ── 1. Generate data ──────────────────────────────────────────────────────────
print("Generating data...")

n = 500
baseline_df = pd.DataFrame({
    "age":    rng.normal(35, 8, n).clip(18, 80),
    "income": rng.normal(50000, 12000, n).clip(10000, None),
    "score":  rng.uniform(0, 1, n),
    "tenure": rng.integers(0, 20, n).astype(float),
    "region": rng.choice(["North", "South", "East", "West"], n),
    "product": rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
})

# No-drift window: same distribution
nodrift_df = pd.DataFrame({
    "age":    rng.normal(35, 8, n).clip(18, 80),
    "income": rng.normal(50000, 12000, n).clip(10000, None),
    "score":  rng.uniform(0, 1, n),
    "tenure": rng.integers(0, 20, n).astype(float),
    "region": rng.choice(["North", "South", "East", "West"], n),
    "product": rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
})

# Drift window: mean shift on age + income, new product category
drift_df = pd.DataFrame({
    "age":    rng.normal(50, 8, n).clip(18, 80),      # +15 mean shift
    "income": rng.normal(70000, 12000, n).clip(10000, None),  # +20k shift
    "score":  rng.uniform(0, 1, n),
    "tenure": rng.integers(0, 20, n).astype(float),
    "region": rng.choice(["North", "South", "East", "West"], n),
    "product": rng.choice(["A", "B", "C", "D"], n, p=[0.3, 0.2, 0.2, 0.3]),  # new category D
})

baseline_df.to_csv(OUT / "baseline.csv", index=False)
nodrift_df.to_csv(OUT / "window_nodrift.csv", index=False)
drift_df.to_csv(OUT / "window_drift.csv", index=False)

# ── 2. Build baseline ─────────────────────────────────────────────────────────
print("Building baseline...")
baseline = build_baseline_profile(baseline_df)
baseline_path = OUT / "baseline.json"
baseline_path.write_text(json.dumps(baseline.model_dump(by_alias=True), indent=2), encoding="utf-8")

# ── 3. Detect ─────────────────────────────────────────────────────────────────
cfg = DetectorConfig()
det = DetectorFacade.from_baseline_json(str(baseline_path), cfg)

print("Running no-drift detection...")
r_nodrift = det.detect_window(nodrift_df, window_id="window_nodrift", out_path=str(OUT / "drift_nodrift.json"))
print(f"  global drift: {r_nodrift['global']['drift_detected']}")

print("Running drift detection...")
r_drift = det.detect_window(drift_df, window_id="window_drift", out_path=str(OUT / "drift_drift.json"))
print(f"  global drift: {r_drift['global']['drift_detected']}")
top = r_drift["feature_ranking"][:3]
print(f"  top features: {[x['feature'] for x in top]}")

# ── 4. HTML report ────────────────────────────────────────────────────────────
print("Writing HTML report...")
save_report_from_files(
    str(OUT / "report.html"),
    drift_path=str(OUT / "drift_drift.json"),
)
print(f"  report: {OUT / 'report.html'}")

# ── 5. Summary ────────────────────────────────────────────────────────────────
print("\n✅ Demo complete. Files in demo_output/:")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name}")

assert r_nodrift["global"]["drift_detected"] == False, "No-drift window should not trigger"
assert r_drift["global"]["drift_detected"] == True, "Drift window should trigger"
print("\n✅ All assertions passed.")
