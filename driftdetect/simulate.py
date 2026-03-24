from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from driftdetect.scenario import DriftEvent, Scenario

# =============================================================================
# simulate.py  (REQ-009 / REQ-010)
#
# Purpose:
# - Generate synthetic "windows" (batches) of data.
# - Inject controlled drift events starting at specific window indices.
# - Write:
#   - windows/window_###.csv (or .parquet)
#   - drift_log.json (GROUND TRUTH event list)
#   - manifest.json (where everything is)
#
# Ground truth:
# - drift_log.json is the ground truth log.
# - The detector later produces drift_###.json files.
# - coverage compares detector flags vs drift_log.json.
# =============================================================================


# Gaussian mixture component: (mean, std, weight)
GMMComponent = Tuple[float, float, float]

DEFAULT_ALLOWED_UNIT_TYPES = [10, 18, 20, 22, 24, 30, 40, 80]
DEFAULT_WEIGHT_GMM: List[GMMComponent] = [
    (15.0, 2.0, 0.6),
    (25.0, 3.0, 0.4),
]

# Canonical drift types (these should match what scenario.py outputs after normalization)
CANONICAL_TYPES = {"numeric_shift", "variance_shift", "missingness", "new_category", "correlation_drift"}

# Defensive alias mapping (so old YAMLs don’t silently do nothing)
DRIFT_TYPE_ALIASES: Dict[str, str] = {
    "numeric_shift": "numeric_shift",
    "mean_shift": "numeric_shift",
    "mean": "numeric_shift",
    "variance_shift": "variance_shift",
    "variance": "variance_shift",
    "missingness": "missingness",
    "missing": "missingness",
    "missingness_shift": "missingness",
    "categorical_shift": "new_category",
    "new_category": "new_category",
    "newcategory": "new_category",
    "correlation_drift": "correlation_drift",
    "correlation": "correlation_drift",
}


def _canonicalize_drift_type(dt: str) -> str:
    key = str(dt).lower().strip()
    if key not in DRIFT_TYPE_ALIASES:
        allowed = ", ".join(sorted(DRIFT_TYPE_ALIASES.keys()))
        raise ValueError(f"unsupported drift_type '{dt}'. Allowed: {allowed}")
    return DRIFT_TYPE_ALIASES[key]


def _as_float(x: Any, what: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError) as ex:
        raise ValueError(f"{what} must be numeric, got {x!r}") from ex


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# =============================================================================
# 1) Base data generator (professor-inspired ideas)
# =============================================================================

def _sample_from_gmm(rng: np.random.Generator, components: Sequence[GMMComponent], n: int) -> np.ndarray:
    """Sample numeric values from a Gaussian Mixture Model."""
    weights = np.array([c[2] for c in components], dtype=float)
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(components)) / len(components)

    idx = rng.choice(np.arange(len(components)), size=n, p=weights)
    means = np.array([components[i][0] for i in idx], dtype=float)
    stds = np.array([components[i][1] for i in idx], dtype=float)
    return rng.normal(means, stds)


def _sample_discrete_normal(rng: np.random.Generator, mean: float, std: float, n: int) -> np.ndarray:
    """Sample from a normal distribution and round to int."""
    return np.rint(rng.normal(mean, std, size=n)).astype(int)


def _snap_to_allowed(values: np.ndarray, allowed: Sequence[int]) -> np.ndarray:
    """Snap sampled integer values to the nearest allowed unit type."""
    allowed_arr = np.array(list(allowed), dtype=int)
    out = np.empty_like(values, dtype=int)
    for i, v in enumerate(values):
        j = int(np.argmin(np.abs(allowed_arr - int(v))))
        out[i] = int(allowed_arr[j])
    return out


def _day_probs(preferred_day: int) -> np.ndarray:
    """Probability over days 1..7 with a peak at preferred_day."""
    days = np.arange(1, 8)
    dist = np.abs(days - preferred_day)
    w = 1.0 / (1.0 + dist)
    return w / w.sum()


def _generate_base_window(
    rng: np.random.Generator,
    n_rows: int,
    allowed_unit_types: Sequence[int] = DEFAULT_ALLOWED_UNIT_TYPES,
    weight_gmm: Sequence[GMMComponent] = DEFAULT_WEIGHT_GMM,
) -> pd.DataFrame:
    """
    Generate ONE synthetic window (one batch).

    Columns:
      - customer_id: categorical string
      - day: integer 1..7 (kept numeric; if you want categorical, convert to str like unit_type)
      - unit_type: categorical string (IMPORTANT: kept as str so it's treated as categorical)
      - weight: numeric float
      - price: numeric float correlated with weight and unit_type
    """
    customer_ids = np.array(["CUST001", "CUST002", "CUST003", "CUST004", "CUST005"])
    cust = rng.choice(customer_ids, size=n_rows, replace=True)

    preferred = rng.integers(1, 8, size=n_rows)
    day = np.array([rng.choice(np.arange(1, 8), p=_day_probs(int(p))) for p in preferred], dtype=int)

    unit_raw = _sample_discrete_normal(rng, mean=20.0, std=2.0, n=n_rows)
    unit_type = _snap_to_allowed(unit_raw, allowed_unit_types)

    weight = _sample_from_gmm(rng, weight_gmm, n_rows)
    weight = np.clip(weight, 3.0, None)

    price = (weight * (unit_type / 10.0)) + rng.normal(0.0, 1.5, size=n_rows)

    return pd.DataFrame(
        {
            "customer_id": cust,
            "day": day,
            # Make unit_type categorical by storing as string (so baseline infers categorical)
            "unit_type": pd.Series(unit_type.astype(int)).map(lambda x: f"UT_{x}"),
            "weight": weight.astype(float),
            "price": price.astype(float),
        }
    )


# =============================================================================
# Income-themed base data generator
# =============================================================================

INCOME_OCCUPATIONS = [
    "Tech", "Healthcare", "Education", "Finance",
    "Retail", "Manufacturing", "Admin", "Construction",
]
INCOME_MARITAL = ["Married", "Single", "Divorced", "Widowed"]
INCOME_EDUCATION = ["HS-grad", "Some-college", "Bachelors", "Masters", "Doctorate"]


def _generate_income_window(
    rng: np.random.Generator,
    n_rows: int,
    age_mean: float = 38.0,
    age_std: float = 13.0,
    income_gmm: List[GMMComponent] = None,
    hours_mean: float = 40.0,
    hours_std: float = 10.0,
) -> pd.DataFrame:
    """
    Generate ONE synthetic income/demographic window.

    Features:
      - age: numeric, roughly working-age population
      - income: numeric, Gaussian mixture (lower + higher earners)
      - hours_per_week: numeric, centered around full-time hours
      - education_num: numeric, years of education (8-16)
      - occupation: categorical
      - marital_status: categorical

    This mirrors the Adult Income dataset structure but is fully synthetic
    with controllable distribution parameters.
    """
    if income_gmm is None:
        income_gmm = [(32000.0, 8000.0, 0.55), (78000.0, 22000.0, 0.45)]

    age = rng.normal(age_mean, age_std, n_rows).clip(18, 65)
    income = _sample_from_gmm(rng, income_gmm, n_rows).clip(10000, 250000)
    hours = rng.normal(hours_mean, hours_std, n_rows).clip(20, 80)
    edu_num = rng.integers(8, 17, size=n_rows).astype(float)  # 8=HS, 16=Doctorate
    occupation = rng.choice(INCOME_OCCUPATIONS, size=n_rows)
    marital = rng.choice(
        INCOME_MARITAL, size=n_rows,
        p=[0.48, 0.35, 0.12, 0.05],
    )

    return pd.DataFrame({
        "age": age,
        "income": income,
        "hours_per_week": hours,
        "education_num": edu_num,
        "occupation": occupation,
        "marital_status": marital,
    })



# =============================================================================
# 2) Drift injection primitives (REQ-009)
# =============================================================================

def _apply_numeric_shift(df: pd.DataFrame, features: List[str], mean_shift: float) -> pd.DataFrame:
    """Add mean_shift to numeric columns."""
    for f in features:
        if f in df.columns:
            df[f] = df[f].astype(float) + float(mean_shift)
    return df


def _apply_variance_shift(df: pd.DataFrame, features: List[str], std_mult: float) -> pd.DataFrame:
    """Multiply deviation-from-mean by std_mult (variance change)."""
    for f in features:
        if f in df.columns:
            x = df[f].astype(float)
            mu = float(x.mean())
            df[f] = mu + (x - mu) * float(std_mult)
    return df


def _apply_missingness(df: pd.DataFrame, rng: np.random.Generator, features: List[str], rate: float) -> pd.DataFrame:
    """Replace values with NaN with probability=rate."""
    rate = _clip01(rate)
    for f in features:
        if f in df.columns:
            mask = rng.random(len(df)) < rate
            df.loc[mask, f] = np.nan
    return df


def _apply_new_category(
    df: pd.DataFrame,
    rng: np.random.Generator,
    features: List[str],
    category: Any,
    rate: float,
) -> pd.DataFrame:
    """Overwrite a fraction of rows with a new category value."""
    rate = _clip01(rate)
    for f in features:
        if f in df.columns:
            mask = rng.random(len(df)) < rate
            raw = str(category)
            # Keep unit_type's synthetic naming convention while preserving explicit labels.
            if f == "unit_type" and not raw.startswith("UT_"):
                value = f"UT_{raw}"
            else:
                value = raw
            df.loc[mask, f] = value
    return df


def _apply_correlation_shuffle(df: pd.DataFrame, rng: np.random.Generator, features: List[str]) -> pd.DataFrame:
    """
    Correlation drift: break the relationship between two features (keep marginals similar).
    If features has at least 2, shuffle the second one.
    """
    if len(features) < 2:
        return df
    a, b = features[0], features[1]
    if a in df.columns and b in df.columns:
        df[b] = df[b].to_numpy()[rng.permutation(len(df))]
    return df


def _first_present(d: Dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def _float_intensity(event: DriftEvent, intensity: Dict[str, Any], keys: Sequence[str], default: float, what: str) -> float:
    raw = _first_present(intensity, keys)
    if raw is None:
        return float(default)
    return _as_float(raw, f"event '{event.event_id}': intensity.{what}")


def _compute_strength(window_idx: int, event: DriftEvent) -> float:
    intensity = dict(event.intensity or {})
    mode = str(intensity.get("mode", "abrupt")).lower().strip()
    if window_idx < int(event.start_window):
        return 0.0
    if mode == "gradual":
        ramp = max(1, int(intensity.get("ramp_windows", 4)))
        elapsed = window_idx - int(event.start_window)
        return float(min(elapsed / ramp, 1.0))
    return 1.0


def apply_drift_event(
    df: pd.DataFrame,
    rng: np.random.Generator,
    event: DriftEvent,
    strength: float = 1.0,
) -> pd.DataFrame:
    if strength <= 0.0:
        return df
    dt = _canonicalize_drift_type(event.drift_type)
    intensity = dict(event.intensity or {})

    if dt == "numeric_shift":
        mean_shift = _float_intensity(event, intensity, ["mean_shift", "delta"], 2.0, "mean_shift")
        return _apply_numeric_shift(df, event.features, mean_shift=mean_shift * strength)

    if dt == "variance_shift":
        std_mult = _float_intensity(event, intensity, ["std_mult"], 1.5, "std_mult")
        if std_mult <= 0:
            raise ValueError(f"event '{event.event_id}': intensity.std_mult must be > 0")
        effective_mult = 1.0 + (std_mult - 1.0) * strength
        return _apply_variance_shift(df, event.features, std_mult=effective_mult)

    if dt == "missingness":
        rate = _float_intensity(event, intensity,
            ["missing_rate", "missing_rate_delta", "rate", "fraction"], 0.2, "missing_rate")
        return _apply_missingness(df, rng, event.features, rate=rate * strength)

    if dt == "new_category":
        category = _first_present(intensity, ["category", "value"]) or "NEW_CAT"
        rate = _float_intensity(event, intensity, ["rate", "fraction"], 0.3, "rate")
        return _apply_new_category(df, rng, event.features, category=category, rate=rate * strength)

    if dt == "correlation_drift":
        if len(event.features) < 2:
            return df
        a, b = event.features[0], event.features[1]
        if a in df.columns and b in df.columns:
            n_shuffle = max(2, int(len(df) * strength))
            idx = rng.choice(len(df), size=n_shuffle, replace=False)
            df = df.copy()
            b_vals = df[b].to_numpy().copy()
            b_vals[idx] = b_vals[rng.permutation(idx)]
            df[b] = b_vals
        return df

    raise ValueError(f"event '{event.event_id}': unsupported drift_type '{event.drift_type}'")


# =============================================================================
# 3) Scenario simulation (REQ-010: drift_log.json ground truth)
# =============================================================================

def simulate_scenario(scenario: Scenario, outdir: str, overwrite: bool = False) -> dict:
    out = Path(outdir)
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{outdir} exists. Use --overwrite to replace it.")
        shutil.rmtree(out)

    windows_dir = out / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(scenario.seed))

    # dataset field selects base generator (extra field, defaults to "logistics")
    dataset = str(getattr(scenario, "dataset", None) or "logistics").lower().strip()
    if dataset == "income":
        def base_gen(rng, n):
            return _generate_income_window(rng=rng, n_rows=n)
    else:
        def base_gen(rng, n):
            return _generate_base_window(rng=rng, n_rows=n)

    drift_log = [
        {
            "event_id": e.event_id,
            "type": _canonicalize_drift_type(e.drift_type),
            "start_window": int(e.start_window),
            "affected_features": list(e.features),
            "intensity": dict(e.intensity or {}),
        }
        for e in scenario.events
    ]

    ext = "csv" if str(scenario.output_format).lower() == "csv" else "parquet"

    windows_entries = []
    for i in range(int(scenario.window_count)):
        df = base_gen(rng, int(scenario.window_size))

        for e in scenario.events:
            strength = _compute_strength(i, e)
            if strength > 0.0:
                df = apply_drift_event(df, rng, e, strength=strength)

        wname = f"window_{i:03d}.{ext}"
        wpath = windows_dir / wname

        if ext == "csv":
            df.to_csv(wpath, index=False)
        else:
            try:
                df.to_parquet(wpath, index=False, engine="pyarrow")
            except Exception as ex:
                raise RuntimeError("Failed to write Parquet. Install pyarrow.") from ex

        windows_entries.append({"index": i, "path": f"windows/{wname}"})

    (out / "drift_log.json").write_text(json.dumps(drift_log, indent=2), encoding="utf-8")

    manifest = {
        "scenario_id": scenario.scenario_id or "scenario",
        "seed": int(scenario.seed),
        "window_count": int(scenario.window_count),
        "window_size": int(scenario.window_size),
        "output_format": str(scenario.output_format).lower(),
        "dataset": dataset,
        "windows_dir": "windows",
        "windows": windows_entries,
        "drift_log_path": "drift_log.json",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest

