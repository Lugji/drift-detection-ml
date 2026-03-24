"""Generate test fixture datasets for DriftDetect tests."""
import numpy as np
import pandas as pd
from pathlib import Path

out = Path(__file__).parent / "fixtures"
out.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

# Baseline: 1000 rows, 4 numeric + 2 categorical
n = 1000
df_baseline = pd.DataFrame({
    "age": rng.normal(35, 10, n).clip(18, 80),
    "income": rng.lognormal(10, 1, n),
    "score": rng.uniform(0, 100, n),
    "tenure": rng.integers(0, 20, n).astype(float),
    "region": rng.choice(["EU", "US", "APAC"], n),
    "product": rng.choice(["A", "B", "C", "D"], n, p=[0.4, 0.3, 0.2, 0.1]),
})
df_baseline.to_csv(out / "baseline.csv", index=False)

# No-drift window: same distribution
df_nodrift = pd.DataFrame({
    "age": rng.normal(35, 10, 300).clip(18, 80),
    "income": rng.lognormal(10, 1, 300),
    "score": rng.uniform(0, 100, 300),
    "tenure": rng.integers(0, 20, 300).astype(float),
    "region": rng.choice(["EU", "US", "APAC"], 300),
    "product": rng.choice(["A", "B", "C", "D"], 300, p=[0.4, 0.3, 0.2, 0.1]),
})
df_nodrift.to_csv(out / "window_nodrift.csv", index=False)

# Drift window: mean shift + new category
df_drift = df_nodrift.copy()
df_drift["age"] = df_drift["age"] + 15        # mean shift
df_drift["income"] = df_drift["income"] * 3   # variance + mean
mask = rng.random(300) < 0.4
df_drift.loc[mask, "region"] = "UNKNOWN"       # new category
df_drift.to_csv(out / "window_drift.csv", index=False)

# Schema drift window: missing column + new column
df_schema = df_nodrift.copy()
df_schema = df_schema.drop(columns=["tenure"])
df_schema["new_feature"] = rng.random(300)
df_schema.to_csv(out / "window_schema_drift.csv", index=False)

print("Fixtures generated:", list(out.glob("*.csv")))
