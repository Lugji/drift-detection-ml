from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from driftdetect.models import BaselineFeature, BaselineProfile, Histogram, NumericStats


def _infer_type(s: pd.Series, name: str, force_categorical: Set[str]) -> str:
    """
    Decide whether a column should be treated as numeric or categorical.

    MVP rule:
    - If column name is in force_categorical -> categorical
    - bool -> categorical (pandas treats bool as numeric otherwise)
    - else if pandas thinks it's numeric -> numeric
    - else -> categorical
    """
    if name in force_categorical:
        return "categorical"
    if pd.api.types.is_bool_dtype(s):
        return "categorical"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    return "categorical"


def _numeric_stats(s: pd.Series) -> NumericStats:
    """
    Numeric summary for baseline.json.
    """
    x = s.dropna().astype(float)
    if len(x) == 0:
        return NumericStats(mean=0.0, std=0.0, min=0.0, max=0.0, q05=0.0, q50=0.0, q95=0.0)

    return NumericStats(
        mean=float(x.mean()),
        std=float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        min=float(x.min()),
        max=float(x.max()),
        q05=float(x.quantile(0.05)),
        q50=float(x.quantile(0.50)),
        q95=float(x.quantile(0.95)),
    )


def _numeric_hist(s: pd.Series, bins: int) -> Histogram:
    """
    Baseline histogram (edges + probs).

    This is used later for approximate KS / Wasserstein without storing raw baseline data.
    """
    x = s.dropna().astype(float).to_numpy()
    if len(x) == 0:
        edges = np.linspace(0.0, 1.0, bins + 1)
        probs = np.ones(bins, dtype=float) / float(bins)
        return Histogram(edges=edges.tolist(), probs=probs.tolist())

    counts, edges = np.histogram(x, bins=bins)
    total = float(counts.sum())
    probs = (counts.astype(float) / total) if total > 0 else (np.ones_like(counts, dtype=float) / float(len(counts)))

    return Histogram(edges=edges.astype(float).tolist(), probs=probs.tolist())


def _categorical_freq(s: pd.Series, top_k: int) -> Dict[str, float]:
    """
    Categorical distribution (proportions).
    Keeps top_k and folds the rest into "__OTHER__".
    """
    x = s.dropna().astype(str)
    if len(x) == 0:
        return {}

    vc = x.value_counts(normalize=True)

    if len(vc) <= top_k:
        return {str(k): float(v) for k, v in vc.items()}

    head = vc.head(top_k)
    other = float(1.0 - head.sum())

    d = {str(k): float(v) for k, v in head.items()}
    d["__OTHER__"] = max(other, 0.0)
    return d


def build_baseline_profile(
    df: pd.DataFrame,
    *,
    bins: int = 20,
    top_k: int = 50,
    force_categorical: Optional[List[str]] = None,
) -> BaselineProfile:
    """
    Build the full baseline profile from a dataset.

    force_categorical:
    - list of columns that should be treated as categorical even if pandas infers numeric
      (useful for IDs, codes, day-of-week, etc.)
    """
    schema: List[Dict[str, str]] = []
    features: Dict[str, BaselineFeature] = {}

    force_set: Set[str] = set(force_categorical or [])

    for col in df.columns:
        name = str(col)
        ftype = _infer_type(df[col], name=name, force_categorical=force_set)
        schema.append({"name": name, "type": ftype})

        missing_rate = float(df[col].isna().mean())

        if ftype == "numeric":
            stats = _numeric_stats(df[col])
            hist = _numeric_hist(df[col], bins=bins)
            features[name] = BaselineFeature(
                type="numeric",
                missing_rate=missing_rate,
                stats=stats,
                hist=hist,
            )
        else:
            freq = _categorical_freq(df[col], top_k=top_k)
            features[name] = BaselineFeature(
                type="categorical",
                missing_rate=missing_rate,
                freq=freq,
            )

    meta = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "bins": int(bins),
        "top_k": int(top_k),
        "force_categorical": sorted(list(force_set)),
    }

    return BaselineProfile(metadata=meta, schema_=schema, features=features)