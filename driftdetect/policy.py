from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass(frozen=True)
class PolicyConfig:
    """
    Global drift policy configuration (REQ-007).

    - S_feat: per-feature threshold for drifted=True
    - S_global: threshold on max severity to trigger global drift
    - K: trigger global drift if >=K features are drifted
    """
    S_feat: float = 0.7
    S_global: float = 0.7
    K: int = 3


def apply_global_policy(
    *,
    schema_drift_present: bool,
    max_severity: float,
    drifted_count: int,
    cfg: PolicyConfig,
) -> Dict[str, Any]:
    """
    Compute global drift decision g(w_i) and provide reasons (transparent output).

    Returns a dict like:
      {
        "drift_detected": True/False,
        "reasons": ["schema_drift", "max_severity>=S_global", ...],
        "max_severity": ...,
        "drifted_count": ...,
        "schema_drift_present": ...
      }
    """
    reasons: List[str] = []

    if schema_drift_present:
        reasons.append("schema_drift")

    if max_severity >= cfg.S_global:
        reasons.append("max_severity>=S_global")

    if drifted_count >= cfg.K:
        reasons.append("drifted_count>=K")

    drift_detected = len(reasons) > 0

    return {
        "drift_detected": drift_detected,
        "reasons": reasons,
        "max_severity": float(max_severity),
        "drifted_count": int(drifted_count),
        "schema_drift_present": bool(schema_drift_present),
        "S_feat": float(cfg.S_feat),
        "S_global": float(cfg.S_global),
        "K": int(cfg.K),
    }
