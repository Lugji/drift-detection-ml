from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from driftdetect.io import load_dataset
from driftdetect.models import BaselineProfile
from driftdetect.detector_core import DetectorCore
from driftdetect.policy import PolicyConfig, apply_global_policy


@dataclass(frozen=True)
class DetectorConfig:
    """
    DetectorConfig

    All parameters that define drift decisions.
    Stored into drift.json for reproducibility.
    """
    alpha: float = 0.05
    S_feat: float = 0.7
    S_global: float = 0.7
    K: int = 3
    tau_wd: float = 1.0
    tau_mu: float = 1.0
    tau_sigma: float = 0.5
    tau_miss: float = 0.10  # 10% missing-rate change -> severity 1 (tune)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": float(self.alpha),
            "S_feat": float(self.S_feat),
            "S_global": float(self.S_global),
            "K": int(self.K),
            "tau_wd": float(self.tau_wd),
            "tau_mu": float(self.tau_mu),
            "tau_sigma": float(self.tau_sigma),
            "tau_miss": float(self.tau_miss),
        }

    def config_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]


class DetectorFacade:
    """
    driftdetect.detector_facade

    Purpose:
    - Hold BaselineProfile + DetectorConfig.
    - Provide detect_window() and detect_suite_from_manifest().
    - Handle I/O + artifact writing.
    - Provide sanity_check_from_manifest() for measurability.

    Notes:
    - Core computations live in DetectorCore.
    - Global policy is owned by policy.py (single source of truth).
    """

    def __init__(self, baseline: BaselineProfile, cfg: DetectorConfig, core: Any = None) -> None:
        self.baseline = baseline
        self.cfg = cfg
        self.core = core or DetectorCore

    # ----------------------------
    # constructors / IO
    # ----------------------------

    @classmethod
    def from_baseline_json(cls, baseline_path: str, cfg: DetectorConfig, core: Any = None) -> "DetectorFacade":
        obj = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
        baseline = BaselineProfile.model_validate(obj)
        return cls(baseline=baseline, cfg=cfg, core=core)

    @staticmethod
    def _write_json(path: Path, obj: Any) -> None:
        path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    # ----------------------------
    # window detection
    # ----------------------------

    def detect_window(
        self,
        window_df: pd.DataFrame,
        window_id: str,
        *,
        out_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        # (REQ-003) schema drift
        sd = self.core.schema_drift(self.baseline, window_df)
        sd_present = bool(sd.get("added") or sd.get("removed") or sd.get("type_changed"))

        type_changed = {x.get("name") for x in sd.get("type_changed", []) if isinstance(x, dict)}

        features_out: Dict[str, Any] = {}

        for feat, bfeat in self.baseline.features.items():
            if feat not in window_df.columns:
                continue

            if feat in type_changed:
                f: Dict[str, Any] = {
                    "type": "type_changed",
                    "metrics": {},
                    "severity": 1.0,
                    "drifted": True,
                    "baseline_summary": {"type": str(bfeat.type)},
                    "current_summary": {"type": "changed"},
                    "note": "feature type changed; metrics not comparable",
                }
            else:
                if str(bfeat.type) == "numeric":
                    f = self.core.numeric_drift_for_feature(
                        self.baseline,
                        window_df,
                        feat,
                        tau_wd=self.cfg.tau_wd,
                        tau_mu=self.cfg.tau_mu,
                        tau_sigma=self.cfg.tau_sigma,
                        tau_miss=self.cfg.tau_miss,
                    )
                else:
                    f = self.core.categorical_drift_for_feature(
                        self.baseline,
                        window_df,
                        feat,
                        alpha=self.cfg.alpha,
                        tau_miss=self.cfg.tau_miss,
                    )

                f["drifted"] = bool(float(f.get("severity", 0.0)) >= float(self.cfg.S_feat))

            features_out[feat] = f

        ranking = sorted(
            [{"feature": k, "severity": float(v.get("severity", 0.0))} for k, v in features_out.items()],
            key=lambda x: x["severity"],
            reverse=True,
        )

        max_sev = float(max((x["severity"] for x in ranking), default=0.0))
        drifted_count = int(sum(1 for v in features_out.values() if bool(v.get("drifted"))))

        # global policy (single source of truth in policy.py)
        policy_cfg = PolicyConfig(S_feat=self.cfg.S_feat, S_global=self.cfg.S_global, K=self.cfg.K)
        global_block = apply_global_policy(
            schema_drift_present=sd_present,
            max_severity=max_sev,
            drifted_count=drifted_count,
            cfg=policy_cfg,
        )

        global_block["config"] = self.cfg.to_dict()
        global_block["config_hash"] = self.cfg.config_hash()

        drift_result = {
            "metadata": {
                "window_id": str(window_id),
                "rows": int(len(window_df)),
                "cols": int(window_df.shape[1]),
            },
            "global": global_block,
            "schema_drift": sd,
            "features": features_out,
            "feature_ranking": ranking,
        }

        if out_path is not None:
            self._write_json(Path(out_path), drift_result)

        return drift_result

    # ----------------------------
    # suite detection
    # ----------------------------

    @staticmethod
    def _resolve_window_path(manifest_path: str, p: str) -> str:
        base = Path(manifest_path).resolve().parent
        pp = Path(p)

        # absolute paths are fine
        if pp.is_absolute():
            return str(pp.resolve())

        # 1) normal case: path is relative to manifest directory (e.g. "windows/window_000.csv")
        cand1 = (base / pp).resolve()
        if cand1.exists():
            return str(cand1)

        # 2) common case: manifest stored "sim_out/windows/..." while manifest is already in sim_out/
        #    -> drop the duplicated leading folder name
        if pp.parts and pp.parts[0] == base.name:
            cand2 = (base / Path(*pp.parts[1:])).resolve()
            if cand2.exists():
                return str(cand2)

        # 3) fallback: interpret p relative to current working directory
        cand3 = pp.resolve()
        if cand3.exists():
            return str(cand3)

        # last resort (keeps error message useful)
        return str(cand1)

    @staticmethod
    def _normalize_drift_log_events(raw: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        out: List[Dict[str, Any]] = []
        for e in raw:
            if not isinstance(e, dict):
                continue
            onset = e.get("onset", e.get("start_window", 0))
            feats = e.get("features", e.get("affected_features", []))

            ev = dict(e)
            ev["onset"] = int(onset)
            ev["features"] = list(feats) if isinstance(feats, list) else []
            ev["start_window"] = int(onset)
            ev["affected_features"] = list(ev["features"])
            out.append(ev)

        return out

    def detect_suite_from_manifest(
        self,
        manifest_path: str,
        outdir: str,
        *,
        overwrite: bool = False,
        run_sanity: bool = False,
        sanity_sample_rows: Optional[int] = None,
        sanity_seed: int = 0,
    ) -> Dict[str, Any]:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)

        drifts_dir = out / "drifts"
        drifts_dir.mkdir(exist_ok=True)

        existing_drift_files = sorted(drifts_dir.glob("drift_*.json"))
        if existing_drift_files and not overwrite:
            raise FileExistsError(
                f"{drifts_dir} contains existing drift_*.json files. "
                "Use --overwrite to replace them."
            )

        if overwrite:
            for p in existing_drift_files:
                p.unlink()

        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        windows = manifest.get("windows", [])
        if not isinstance(windows, list) or not windows:
            raise ValueError("manifest.json must contain a non-empty 'windows' list")

        windows_sorted = sorted(
            windows,
            key=lambda w: int(w["index"]) if isinstance(w, dict) and "index" in w else 10**9,
        )

        drift_results: List[Dict[str, Any]] = []

        for w in windows_sorted:
            if not isinstance(w, dict) or "index" not in w or "path" not in w:
                raise ValueError("Each window entry must have 'index' and 'path'")

            idx = int(w["index"])
            wpath = self._resolve_window_path(manifest_path, str(w["path"]))
            df = load_dataset(wpath)

            drift = self.detect_window(df, window_id=Path(wpath).name)

            out_path = drifts_dir / f"drift_{idx:03d}.json"
            self._write_json(out_path, drift)

            drift_results.append({"index": idx, "window_path": wpath, "drift_path": str(out_path)})

        # drift_log normalization (optional copy)
        normalized_drift_log_path = None
        drift_log_path = manifest.get("drift_log_path")
        if isinstance(drift_log_path, str) and drift_log_path:
            raw_path = self._resolve_window_path(manifest_path, drift_log_path)
            raw_log = json.loads(Path(raw_path).read_text(encoding="utf-8"))
            norm_log = self._normalize_drift_log_events(raw_log)
            normalized_drift_log_path = str((out / "drift_log_normalized.json").resolve())
            self._write_json(Path(normalized_drift_log_path), norm_log)

        detect_manifest: Dict[str, Any] = {
            "manifest_path": str(Path(manifest_path).resolve()),
            "drifts_dir": str(drifts_dir),
            "window_count": int(len(drift_results)),
            "drift_results": drift_results,
            "config": self.cfg.to_dict(),
            "config_hash": self.cfg.config_hash(),
        }
        if normalized_drift_log_path:
            detect_manifest["drift_log_normalized_path"] = normalized_drift_log_path

        if run_sanity and normalized_drift_log_path:
            rep = self.sanity_check_from_manifest(
                manifest_path,
                sample_rows=sanity_sample_rows,
                seed=sanity_seed,
            )
            sanity_path = out / "sanity_check.json"
            self._write_json(sanity_path, rep)
            detect_manifest["sanity_check_path"] = str(sanity_path)

        self._write_json(out / "detect_manifest.json", detect_manifest)
        return detect_manifest

    # ----------------------------
    # drift semantics for sanity checks
    # ----------------------------

    @staticmethod
    def _sigmoid(t: int, p: float, w: float) -> float:
        if w <= 0:
            return 1.0 if t >= p else 0.0
        return float(1.0 / (1.0 + math.exp(-4.0 * (float(t) - float(p)) / float(w))))

    def _event_strength(self, event: Mapping[str, Any], *, t: int) -> float:
        onset = int(event.get("onset", event.get("start_window", 0)))
        endw = event.get("end_window", None)
        endw = int(endw) if endw is not None else None

        if t < onset:
            return 0.0
        if endw is not None and t > endw:
            return 0.0

        intensity = dict(event.get("intensity", {}) or {})
        mode = str(intensity.get("mode", "abrupt")).lower().strip()

        if mode != "gradual":
            return 1.0

        width = int(intensity.get("width", 5))
        p = float(intensity.get("position", onset + width / 2.0))
        return float(self._sigmoid(t, p=p, w=float(width)))

    # ----------------------------
    # sanity check (measured vs expected)
    # ----------------------------

    def sanity_check_from_manifest(
        self,
        manifest_path: str,
        *,
        sample_rows: Optional[int] = None,
        seed: int = 0,
        atol_mean: float = 0.25,
        rtol_mean: float = 0.15,
        atol_rate: float = 0.03,
        rtol_std: float = 0.15,
        corr_drop_min: float = 0.20,
    ) -> Dict[str, Any]:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        windows = manifest.get("windows", [])
        if not isinstance(windows, list):
            raise ValueError("manifest.json missing windows list")

        windows_sorted = sorted(windows, key=lambda w: int(w["index"]))
        dfs: List[pd.DataFrame] = []
        for w in windows_sorted:
            wpath = self._resolve_window_path(manifest_path, str(w["path"]))
            dfs.append(load_dataset(wpath))

        drift_log_path = manifest.get("drift_log_path")
        if not drift_log_path:
            return {"summary": {"events_total": 0, "checks_total": 0, "checks_ok": 0, "checks_failed": 0}, "events": []}

        raw_log = json.loads(Path(self._resolve_window_path(manifest_path, drift_log_path)).read_text(encoding="utf-8"))
        events = self._normalize_drift_log_events(raw_log)

        def subsample(df: pd.DataFrame, key: List[int]) -> pd.DataFrame:
            if sample_rows is None or len(df) <= int(sample_rows):
                return df
            ss = np.random.SeedSequence(key)
            rng = np.random.default_rng(ss)
            idx = rng.choice(len(df), size=int(sample_rows), replace=False)
            return df.iloc[idx].reset_index(drop=True)

        out: Dict[str, Any] = {"events": []}

        for e in events:
            event_id = str(e.get("event_id", "event"))
            dt = str(e.get("type", ""))
            onset = int(e.get("onset", 0))
            feats = list(e.get("features", []))
            intensity = dict(e.get("intensity", {}) or {})

            if onset <= 0 or onset >= len(dfs):
                out["events"].append({"event_id": event_id, "type": dt, "skipped": True})
                continue

            strength = float(self._event_strength(e, t=onset))

            df0 = subsample(dfs[onset - 1], [seed, onset - 1])
            df1 = subsample(dfs[onset], [seed, onset])

            ev: Dict[str, Any] = {"event_id": event_id, "type": dt, "onset": onset, "strength": strength, "checks": []}

            if dt == "numeric_shift":
                mean_shift = float(intensity.get("mean_shift", intensity.get("delta", 2.0)))
                expected = mean_shift * strength
                for f in feats:
                    if f not in df0.columns or f not in df1.columns:
                        continue
                    m0 = pd.to_numeric(df0[f], errors="coerce").mean()
                    m1 = pd.to_numeric(df1[f], errors="coerce").mean()
                    measured = float(m1 - m0)
                    ok = abs(measured - expected) <= (atol_mean + rtol_mean * abs(expected))
                    ev["checks"].append({"feature": f, "metric": "mean_delta", "measured": measured, "expected": expected, "ok": ok})

            elif dt == "variance_shift":
                std_mult = float(intensity.get("std_mult", 1.5))
                expected = 1.0 + (std_mult - 1.0) * strength
                for f in feats:
                    if f not in df0.columns or f not in df1.columns:
                        continue
                    s0 = pd.to_numeric(df0[f], errors="coerce").std(ddof=0)
                    s1 = pd.to_numeric(df1[f], errors="coerce").std(ddof=0)
                    measured = float(s1 / s0) if s0 and s0 > 0 else float("nan")
                    ok = math.isfinite(measured) and abs(measured - expected) <= (rtol_std * abs(expected))
                    ev["checks"].append({"feature": f, "metric": "std_ratio", "measured": measured, "expected": expected, "ok": ok})

            elif dt == "missingness":
                rate = float(intensity.get("missing_rate", intensity.get("rate", 0.2)))
                expected = float(max(0.0, min(1.0, rate * strength)))
                for f in feats:
                    if f not in df1.columns:
                        continue
                    measured = float(df1[f].isna().mean())
                    ok = abs(measured - expected) <= atol_rate
                    ev["checks"].append({"feature": f, "metric": "missing_rate", "measured": measured, "expected": expected, "ok": ok})

            elif dt == "new_category":
                rate = float(intensity.get("rate", 0.3))
                expected = float(max(0.0, min(1.0, rate * strength)))
                for f in feats:
                    if f not in df1.columns:
                        continue
                    prev = set(df0[f].dropna().astype(str).unique()) if f in df0.columns else set()
                    now = df1[f].dropna().astype(str)
                    measured = float((~now.isin(prev)).mean())
                    ok = abs(measured - expected) <= atol_rate
                    ev["checks"].append({"feature": f, "metric": "novel_value_rate(any)", "measured": measured, "expected": expected, "ok": ok})

            elif dt == "correlation_drift":
                if len(feats) >= 2:
                    a, b = feats[0], feats[1]
                    if a in df0.columns and b in df0.columns and a in df1.columns and b in df1.columns:
                        c0 = float(pd.to_numeric(df0[a], errors="coerce").corr(pd.to_numeric(df0[b], errors="coerce")))
                        c1 = float(pd.to_numeric(df1[a], errors="coerce").corr(pd.to_numeric(df1[b], errors="coerce")))
                        drop = abs(c0) - abs(c1)
                        ok = math.isfinite(drop) and drop >= corr_drop_min * strength
                        ev["checks"].append({"pair": [a, b], "metric": "abs_corr_drop", "baseline": c0, "after": c1, "measured": drop, "expected_min": corr_drop_min * strength, "ok": ok})

            out["events"].append(ev)

        checks = [c for ev in out["events"] for c in (ev.get("checks") or [])]
        out["summary"] = {
            "events_total": len(out["events"]),
            "checks_total": len(checks),
            "checks_ok": int(sum(1 for c in checks if c.get("ok") is True)),
            "checks_failed": int(sum(1 for c in checks if c.get("ok") is False)),
        }
        return out
