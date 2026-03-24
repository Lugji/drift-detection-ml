from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# coverage.py  (REQ-012 / REQ-013)
#
# Purpose:
# - Compare simulator ground truth (drift_log.json) against detector outputs
#   (drift_###.json global flags) and compute monitoring coverage metrics:
#     * FPR (False Positive Rate)
#     * Power (overall + by drift type)
#     * TTD (Time-to-Detect) summary
#     * Missed events list
#
# Key design decision (matches your SRS):
# - Coverage uses ONLY the global window-level drift flag g(w_i),
#   not per-feature drifted flags.
# =============================================================================


# ---------------------------------------------------------------------
# Helper: read JSON from disk
# ---------------------------------------------------------------------
def _read_json(path: str) -> Any:
    """Read a JSON file and return the parsed Python object."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------
# Ground truth loader
# ---------------------------------------------------------------------
def _load_drift_log(drift_log_path: str) -> List[Dict[str, Any]]:
    """
    Load the simulator "ground truth" drift log.

    drift_log.json is produced by simulate.py and is a LIST of event objects.
    Each event explains:
      - event_id
      - type (drift type name)
      - start_window (when drift starts)
      - affected_features
      - intensity parameters (how strong the drift is)
    """
    raw = _read_json(drift_log_path)
    if not isinstance(raw, list):
        raise ValueError("drift_log.json must be a JSON list of events")
    return raw


# ---------------------------------------------------------------------
# Drift result loaders (detector outputs)
# ---------------------------------------------------------------------
def _parse_index_from_filename(p: Path) -> int:
    """
    Convert a drift result filename into an integer window index.

    Expected format: drift_003.json -> 3
    """
    stem = p.stem  # "drift_003"
    if "_" not in stem:
        raise ValueError(f"Invalid drift filename (expected drift_###.json): {p.name}")
    tail = stem.split("_")[-1]
    if not tail.isdigit():
        raise ValueError(f"Invalid drift filename index (expected digits): {p.name}")
    return int(tail)


def _load_drift_results(drifts_dir: str) -> List[Dict[str, Any]]:
    """
    Load all drift_###.json files created by detect-suite.

    Returns:
      - list of drift dicts sorted by window index

    Also validates that indices are contiguous:
      drift_000.json, drift_001.json, ..., drift_(N-1).json
    """
    p = Path(drifts_dir)
    if not p.exists():
        raise FileNotFoundError(str(p))

    files = sorted(p.glob("drift_*.json"))
    files = [f for f in files if f.name != "drift_log.json"]  # defensive

    if not files:
        raise ValueError(f"No drift_*.json files found in {drifts_dir}")

    parsed: List[Tuple[int, Dict[str, Any]]] = []
    for f in files:
        idx = _parse_index_from_filename(f)
        obj = _read_json(str(f))
        if not isinstance(obj, dict):
            raise ValueError(f"{f.name} must be a JSON object")
        parsed.append((idx, obj))

    parsed.sort(key=lambda t: t[0])

    indices = [i for i, _ in parsed]
    if indices[0] != 0:
        raise ValueError(f"Drift files must start at drift_000.json, found drift_{indices[0]:03d}.json")

    for expected, actual in enumerate(indices):
        if expected != actual:
            raise ValueError(
                f"Missing drift file: expected drift_{expected:03d}.json but found drift_{actual:03d}.json"
            )

    return [obj for _, obj in parsed]


def _extract_global_flags(drift_results: List[Dict[str, Any]]) -> List[bool]:
    """
    Extract global drift decisions g(w_i) from drift_###.json.

    We read:
      drift["global"]["drift_detected"]  -> True/False
    """
    flags: List[bool] = []
    for d in drift_results:
        g = d.get("global", {})
        if not isinstance(g, dict):
            g = {}
        flags.append(bool(g.get("drift_detected", False)))
    return flags


# ---------------------------------------------------------------------
# Core coverage computation (REQ-012 / REQ-013)
# ---------------------------------------------------------------------
def compute_coverage(
    drift_log: List[Dict[str, Any]],
    g_flags: List[bool],
    M: int,
) -> Dict[str, Any]:
    """
    Compute monitoring coverage from:
      - drift_log: ground truth injected events (simulator output)
      - g_flags: global drift decisions per window from detector outputs
      - M: tolerance (max allowed detection delay in windows)
    """
    if M < 0:
        raise ValueError("M must be >= 0")

    N = len(g_flags)
    if N == 0:
        raise ValueError("Need at least 1 window to compute coverage")

    starts: List[int] = []
    for e in drift_log:
        s = int(e.get("start_window", 0))
        if s < 0:
            raise ValueError(f"drift_log event '{e.get('event_id','')}' has start_window < 0")
        if s >= N:
            raise ValueError(
                f"drift_log event '{e.get('event_id','')}' has start_window={s} but only N={N} windows"
            )
        starts.append(s)

    # SRS MVP rule: once an event starts, it remains active until the end.
    if starts:
        min_start = min(starts)
        no_drift_idx = list(range(0, min_start))
        drift_idx = list(range(min_start, N))
    else:
        no_drift_idx = list(range(N))
        drift_idx = []

    if no_drift_idx:
        fp = sum(1 for i in no_drift_idx if g_flags[i])
        fpr: Optional[float] = fp / float(len(no_drift_idx))
    else:
        fpr = None

    per_event: List[Dict[str, Any]] = []
    missed: List[Dict[str, Any]] = []
    ttd_finite: List[int] = []

    for e in drift_log:
        event_id = str(e.get("event_id", ""))
        dtype = str(e.get("type", e.get("drift_type", "")))
        s = int(e.get("start_window", 0))
        feats = e.get("affected_features", e.get("features", []))
        intensity = e.get("intensity", {})

        first_detect: Optional[int] = None
        for i in range(s, N):
            if g_flags[i]:
                first_detect = i
                break

        if first_detect is None:
            ttd: Optional[int] = None
            detected_within_M = False
        else:
            ttd = int(first_detect - s)
            detected_within_M = (ttd <= M)
            ttd_finite.append(ttd)

        row = {
            "event_id": event_id,
            "type": dtype,
            "start_window": s,
            "affected_features": list(feats) if isinstance(feats, list) else feats,
            "first_detection_window": first_detect,
            "detected_within_M": bool(detected_within_M),
            "TTD": ttd,
            "M": int(M),
            "intensity": intensity if isinstance(intensity, dict) else {},
        }
        per_event.append(row)

        if not detected_within_M:
            missed.append(row)

    power_by_type: Dict[str, float] = {}
    for t in sorted({r["type"] for r in per_event}):
        ev = [r for r in per_event if r["type"] == t]
        power_by_type[t] = sum(1 for r in ev if r["detected_within_M"]) / float(len(ev))

    overall_power: Optional[float] = (
        sum(1 for r in per_event if r["detected_within_M"]) / float(len(per_event))
        if per_event
        else None
    )

    if ttd_finite:
        arr = np.array(ttd_finite, dtype=float)
        ttd_summary = {
            "count_detected": int(len(ttd_finite)),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(np.max(arr)),
        }
    else:
        ttd_summary = {"count_detected": 0, "median": None, "p90": None, "max": None}

    return {
        "metadata": {
            "windows_total": int(N),
            "no_drift_windows": int(len(no_drift_idx)),
            "drift_windows": int(len(drift_idx)),
            "M": int(M),
            "note": (
                "Coverage uses the global drift flag g(w_i). "
                "Overlapping events may inflate power for later events."
            ),
        },
        "metrics": {
            "FPR": fpr,
            "power_overall": overall_power,
            "power_by_type": power_by_type,
            "TTD_summary": ttd_summary,
        },
        "per_event": per_event,
        "missed_events": missed,
    }


# ---------------------------------------------------------------------
# Convenience wrapper used by your CLI command "coverage"
# ---------------------------------------------------------------------
def coverage_from_simdir(
    manifest_path: str,
    drifts_dir: str,
    M: int,
    out_path: str,
) -> Dict[str, Any]:
    """
    Convenience wrapper:
      - reads manifest.json to find drift_log.json
      - reads drift_###.json results from drifts_dir
      - computes coverage and writes coverage.json
    """
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")

    drift_log_path = manifest.get("drift_log_path")
    if not drift_log_path:
        raise ValueError("manifest.json missing drift_log_path")

    # drift_log_path can be relative to manifest.json directory -> resolve it
    manifest_dir = Path(manifest_path).resolve().parent
    drift_log_path_resolved = Path(str(drift_log_path))
    if not drift_log_path_resolved.is_absolute():
        drift_log_path_resolved = (manifest_dir / drift_log_path_resolved).resolve()

    drift_log = _load_drift_log(str(drift_log_path_resolved))
    drift_results = _load_drift_results(drifts_dir)
    flags = _extract_global_flags(drift_results)

    cov = compute_coverage(drift_log=drift_log, g_flags=flags, M=M)

    cov["metadata"]["scenario_id"] = manifest.get("scenario_id")
    cov["metadata"]["manifest_path"] = str(Path(manifest_path).resolve())
    cov["metadata"]["drift_log_path"] = str(drift_log_path_resolved)
    cov["metadata"]["drifts_dir"] = str(Path(drifts_dir).resolve())

    Path(out_path).write_text(json.dumps(cov, indent=2), encoding="utf-8")
    return cov