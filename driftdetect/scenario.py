from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator


# Canonical drift types used across the project (especially coverage.json "power_by_type")
CANONICAL_DRIFT_TYPES = {
    "numeric_shift",
    "variance_shift",
    "missingness",
    "new_category",
    "correlation_drift",
}

# Aliases accepted from scenario YAML (user-friendly / legacy names)
DRIFT_TYPE_ALIASES: Dict[str, str] = {
    # mean / numeric shift
    "numeric_shift": "numeric_shift",
    "mean_shift": "numeric_shift",
    "mean": "numeric_shift",

    # variance shift
    "variance_shift": "variance_shift",
    "variance": "variance_shift",

    # missingness
    "missingness": "missingness",
    "missing": "missingness",
    "missingness_shift": "missingness",

    # categorical / new category
    "new_category": "new_category",
    "newcategory": "new_category",
    "categorical_shift": "new_category",

    # correlation drift
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


class DriftEvent(BaseModel):
    """
    One drift event from scenario YAML.

    Notes:
    - drift_type is normalized to a CANONICAL drift type (so coverage "power_by_type" is stable).
    - intensity remains flexible (extra keys allowed), but we validate obvious numeric parameters.
    """
    model_config = ConfigDict(extra="allow")

    event_id: str
    drift_type: str
    start_window: int
    features: List[str] = Field(default_factory=list)
    intensity: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("drift_type")
    @classmethod
    def _normalize_drift_type(cls, v: str) -> str:
        return _canonicalize_drift_type(v)

    @model_validator(mode="after")
    def _validate_event(self) -> "DriftEvent":
        # Basic sanity: usually drift events should target at least one feature
        if not self.features:
            raise ValueError(f"event '{self.event_id}': features must be a non-empty list")

        # Minimal intensity validation (only when relevant keys are present)
        inten = dict(self.intensity or {})

        if self.drift_type == "numeric_shift":
            if "mean_shift" in inten:
                _as_float(inten["mean_shift"], f"event '{self.event_id}': intensity.mean_shift")
            if "delta" in inten:
                _as_float(inten["delta"], f"event '{self.event_id}': intensity.delta")

        elif self.drift_type == "variance_shift":
            if "std_mult" in inten:
                std_mult = _as_float(inten["std_mult"], f"event '{self.event_id}': intensity.std_mult")
                if std_mult <= 0:
                    raise ValueError(f"event '{self.event_id}': intensity.std_mult must be > 0")

        elif self.drift_type == "missingness":
            # rate-like values should be in [0,1] if provided
            for k in ("missing_rate", "missing_rate_delta", "rate", "fraction"):
                if k in inten:
                    r = _as_float(inten[k], f"event '{self.event_id}': intensity.{k}")
                    if not (0.0 <= r <= 1.0):
                        raise ValueError(f"event '{self.event_id}': intensity.{k} must be in [0,1]")

        elif self.drift_type == "new_category":
            for k in ("rate", "fraction"):
                if k in inten:
                    r = _as_float(inten[k], f"event '{self.event_id}': intensity.{k}")
                    if not (0.0 <= r <= 1.0):
                        raise ValueError(f"event '{self.event_id}': intensity.{k} must be in [0,1]")

        elif self.drift_type == "correlation_drift":
            # correlation drift needs at least two features to shuffle one against another
            if len(self.features) < 2:
                raise ValueError(f"event '{self.event_id}': correlation_drift requires at least 2 features")

        return self


class Scenario(BaseModel):
    """
    Scenario definition loaded from YAML (REQ-008).

    Matches your pytest assumptions:
    - scenario_id optional
    - output_format default csv
    - events can be empty, but usually not
    """
    model_config = ConfigDict(extra="allow")

    scenario_id: Optional[str] = None
    seed: int
    window_count: int
    window_size: int
    output_format: str = "csv"
    events: List[DriftEvent] = Field(default_factory=list)

    @field_validator("output_format")
    @classmethod
    def _normalize_output_format(cls, v: str) -> str:
        fmt = str(v).lower().strip()
        if fmt not in {"csv", "parquet"}:
            raise ValueError("output_format must be 'csv' or 'parquet'")
        return fmt

    @model_validator(mode="after")
    def _validate_scenario(self) -> "Scenario":
        if self.window_count <= 0:
            raise ValueError("window_count must be > 0")
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")

        # start_window range check (pytest expects these exact messages)
        for e in self.events:
            if e.start_window < 0:
                raise ValueError("start_window must be >= 0")
            if e.start_window >= self.window_count:
                raise ValueError("start_window must be < window_count")

        # uniqueness check (pytest expects this exact message)
        ids = [e.event_id for e in self.events]
        if len(set(ids)) != len(ids):
            raise ValueError("event_id values must be unique")

        return self


def load_scenario(path: str) -> Scenario:
    """
    Load and validate a scenario YAML file (REQ-008).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Scenario YAML must be a mapping at the top level")

    return Scenario.model_validate(raw)