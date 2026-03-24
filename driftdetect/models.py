from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

# FeatureType restricts feature "type" to only two allowed values.
# This prevents typos and keeps logic consistent everywhere (baseline/detect/dashboard).
FeatureType = Literal["numeric", "categorical"]


class Histogram(BaseModel):
    """
    A compact summary of a numeric distribution.

    Why this exists:
    - We want to compare baseline vs window numeric distributions without storing all raw baseline values.
    - The baseline stores a histogram once; later we compute a window histogram using the same bin edges.

    Fields:
    - edges: bin boundaries (length = bins + 1)
    - probs: bin probabilities (length = bins, sums to ~1)
    """
    edges: List[float]
    probs: List[float]


class NumericStats(BaseModel):
    """
    Human-readable numeric summary of one numeric feature.

    Why this exists:
    - Makes drift explainable (mean/std/quantiles are easier than KS/Wasserstein).
    - Also used for normalization in severity mapping (e.g., baseline std).

    Fields:
    - mean/std/min/max: basic summary
    - q05/q50/q95: quantiles (robust view of distribution shape)
    """
    mean: float
    std: float
    min: float
    max: float
    q05: float
    q50: float
    q95: float


class BaselineFeature(BaseModel):
    """
    Baseline information for ONE feature (column).

    This is the per-feature part of baseline.json.

    Fields:
    - type: "numeric" or "categorical"
    - missing_rate: how many values are missing (NaN) in the baseline
    - stats: numeric stats (only for numeric features)
    - hist: numeric histogram (only for numeric features)
    - freq: category frequencies/proportions (only for categorical features)

    Design idea:
    One unified object for both numeric and categorical features.
    Fields that don't apply are simply None.
    """

    model_config = ConfigDict(extra="forbid")

    type: FeatureType
    missing_rate: float = Field(ge=0.0, le=1.0)

    stats: Optional[NumericStats] = None
    hist: Optional[Histogram] = None
    freq: Optional[Dict[str, float]] = None


class BaselineProfile(BaseModel):
    """
    The full baseline artifact: baseline.json.

    Why this exists:
    - Baseline is our definition of the "past population" (P0).
    - Every drift comparison uses this profile as reference.

    Fields:
    - metadata: general info (rows, cols, bins, timestamps, etc.)
    - schema_: list of column name/type pairs (used for schema drift detection)
    - features: mapping feature_name -> BaselineFeature

    Important detail: schema_ vs "schema"
    - Pydantic BaseModel already has a method called schema().
    - To avoid clashes and warnings, we name the Python attribute schema_.
    - But in JSON we still want the key to be "schema".
    - Field(alias="schema") + populate_by_name=True makes that work.
    """


    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Stored in Python as schema_, written to JSON as "schema".
    schema_: List[Dict[str, str]] = Field(alias="schema")

    features: Dict[str, BaselineFeature]
