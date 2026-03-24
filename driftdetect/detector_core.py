# driftdetect/detector_core.py
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency

from driftdetect.models import BaselineProfile


class DetectorCore:
    """
    driftdetect.detector_core

    Pure drift evidence:
    - schema drift
    - numeric drift metrics/severity
    - categorical drift metrics/severity
    - missingness drift severity (baseline missing_rate vs window missing rate)

    No I/O. No policy. No mutation.
    """

    # ----------------------------
    # schema drift
    # ----------------------------

    @staticmethod
    def infer_feature_type(series: pd.Series) -> str:
        # must match baseline inference rule:
        # - bool is categorical (pandas treats bool as numeric otherwise)
        if pd.api.types.is_bool_dtype(series):
            return "categorical"
        return "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical"

    @classmethod
    def window_schema_map(cls, df: pd.DataFrame) -> Dict[str, str]:
        return {str(c): cls.infer_feature_type(df[c]) for c in df.columns}

    @staticmethod
    def _baseline_schema_from_profile(baseline: BaselineProfile) -> Dict[str, str]:
        # single source of truth: baseline.features
        return {str(name): str(feat.type) for name, feat in baseline.features.items()}

    @classmethod
    def schema_drift(cls, baseline: BaselineProfile, window_df: pd.DataFrame) -> Dict[str, Any]:
        baseline_schema = cls._baseline_schema_from_profile(baseline)
        window_schema = cls.window_schema_map(window_df)

        base_cols = set(baseline_schema.keys())
        win_cols = set(window_schema.keys())

        added = sorted(win_cols - base_cols)
        removed = sorted(base_cols - win_cols)

        type_changed: List[Dict[str, str]] = []
        for c in sorted(base_cols & win_cols):
            if baseline_schema[c] != window_schema[c]:
                type_changed.append({"name": c, "baseline": baseline_schema[c], "window": window_schema[c]})

        return {"added": added, "removed": removed, "type_changed": type_changed}

    # ----------------------------
    # helpers
    # ----------------------------

    @staticmethod
    def _clip01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    @staticmethod
    def _missingness_severity(mr_base: float, mr_win: float, tau_miss: float, eps: float = 1e-9) -> float:
        delta = abs(float(mr_win) - float(mr_base))
        return DetectorCore._clip01(delta / max(float(tau_miss), eps))

    @staticmethod
    def _model_dump(obj: Any) -> Dict[str, Any]:
        if obj is None:
            return {}
        if hasattr(obj, "model_dump"):
            return dict(obj.model_dump())
        if hasattr(obj, "dict"):
            return dict(obj.dict())
        return {}

    @staticmethod
    def _safe_probs(p: np.ndarray) -> np.ndarray:
        if p.size == 0:
            return p
        s = float(p.sum())
        if s <= 0:
            return np.ones_like(p, dtype=float) / float(len(p))
        return p / s

    # ----------------------------
    # numeric drift
    # ----------------------------

    @staticmethod
    def _window_hist_probs(window_series: pd.Series, edges: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Returns (q, has_evidence)

        has_evidence=False means: no non-null values in window_series.
        In that case the caller should not invent drift; use q=p (no evidence).
        """
        edges = np.asarray(edges, dtype=float)
        bins = int(len(edges) - 1)
        if bins <= 0:
            return np.array([], dtype=float), False

        x = window_series.dropna().astype(float).to_numpy()
        if len(x) == 0:
            return np.array([], dtype=float), False

        lo = float(edges[0])
        hi = float(edges[-1])

        # numpy.histogram ignores values outside range; clip so mass does not disappear
        x = np.clip(x, lo, hi)

        counts, _ = np.histogram(x, bins=edges)
        total = float(counts.sum())
        if total <= 0:
            return np.array([], dtype=float), False

        return (counts.astype(float) / total), True

    @staticmethod
    def _ks_from_probs(p: np.ndarray, q: np.ndarray) -> float:
        if p.size == 0 or q.size == 0:
            return 0.0
        return float(np.max(np.abs(np.cumsum(p) - np.cumsum(q))))

    @staticmethod
    def _wasserstein_from_probs(p: np.ndarray, q: np.ndarray, edges: np.ndarray) -> float:
        if p.size == 0 or q.size == 0:
            return 0.0
        widths = np.diff(np.asarray(edges, dtype=float))
        return float(np.sum(np.abs(np.cumsum(p) - np.cumsum(q)) * widths))

    @classmethod
    def numeric_drift_for_feature(
        cls,
        baseline: BaselineProfile,
        window_df: pd.DataFrame,
        feature: str,
        *,
        tau_wd: float = 1.0,
        tau_mu: float = 1.0,
        tau_sigma: float = 0.5,
        tau_miss: float = 0.10,
        eps: float = 1e-9,
    ) -> Dict[str, Any]:
        bfeat = baseline.features[feature]

        if getattr(bfeat, "hist", None) is None or getattr(bfeat, "stats", None) is None:
            return {
                "type": "numeric",
                "metrics": {},
                "severity": 0.0,
                "drifted": False,
                "baseline_summary": cls._model_dump(getattr(bfeat, "stats", None)),
                "current_summary": {},
                "note": "missing baseline hist/stats",
            }

        # missingness drift (baseline vs window)
        mr_base = float(getattr(bfeat, "missing_rate", 0.0))
        mr_win = float(window_df[feature].isna().mean())
        s_miss = cls._missingness_severity(mr_base, mr_win, tau_miss=tau_miss, eps=eps)

        edges = np.asarray(bfeat.hist.edges, dtype=float)
        p = cls._safe_probs(np.asarray(bfeat.hist.probs, dtype=float))

        q, has_evidence = cls._window_hist_probs(window_df[feature], edges)
        if not has_evidence:
            # no values -> don't invent distribution drift
            q = p.copy()

        ks = cls._ks_from_probs(p, q)
        wd = cls._wasserstein_from_probs(p, q, edges)

        s_ks = cls._clip01(float(ks))

        b_mean = float(bfeat.stats.mean)
        b_std = float(bfeat.stats.std)

        q05 = float(getattr(bfeat.stats, "q05", b_mean))
        q95 = float(getattr(bfeat.stats, "q95", b_mean))
        robust_std = (q95 - q05) / 3.2898 if (q95 > q05) else 0.0

        baseline_scale = max(b_std, robust_std, eps)

        wd_star = float(wd) / baseline_scale
        s_wd = cls._clip01(wd_star / max(float(tau_wd), eps))

        cur = window_df[feature].dropna().astype(float)
        if len(cur) == 0:
            cur_mean = None
            cur_std = None
            s_mean = 0.0
            s_std = 0.0
            cur_summary: Dict[str, Any] = {}
        else:
            cur_mean = float(cur.mean())
            cur_std = float(cur.std(ddof=1)) if len(cur) > 1 else 0.0

            z_mu = abs(cur_mean - b_mean) / baseline_scale
            s_mean = cls._clip01(z_mu / max(float(tau_mu), eps))

            r_sigma = abs(cur_std - b_std) / baseline_scale
            s_std = cls._clip01(r_sigma / max(float(tau_sigma), eps))

            cur_summary = {"mean": cur_mean, "std": cur_std}

        severity = float(max(s_ks, s_wd, s_mean, s_std, s_miss))

        # small sample of window values for dashboard distribution plot (max 300)
        _win_vals = window_df[feature].dropna().astype(float).tolist()
        window_sample = _win_vals[:300] if len(_win_vals) > 300 else _win_vals

        return {
            "type": "numeric",
            "window_sample": window_sample,
            "metrics": {
                "ks": float(ks),
                "wasserstein": float(wd),
                "s_KS": float(s_ks),
                "s_WD": float(s_wd),
                "mean_shift_z": None if cur_mean is None else abs(cur_mean - b_mean) / baseline_scale,
                "std_change_rel": None if cur_std is None else abs(cur_std - b_std) / baseline_scale,
                "s_mean": float(s_mean),
                "s_std": float(s_std),
                "missing_rate_baseline": float(mr_base),
                "missing_rate_window": float(mr_win),
                "missing_rate_delta": float(abs(mr_win - mr_base)),
                "s_missing": float(s_miss),
                "baseline_scale": float(baseline_scale),
                "tau_wd": float(tau_wd),
                "tau_mu": float(tau_mu),
                "tau_sigma": float(tau_sigma),
                "tau_miss": float(tau_miss),
            },
            "severity": float(severity),
            "drifted": False,
            "baseline_summary": cls._model_dump(bfeat.stats),
            "current_summary": cur_summary,
        }

    # ----------------------------
    # categorical drift
    # ----------------------------

    @staticmethod
    def _aligned_category_probs(
        baseline_freq: Mapping[str, float],
        window_series: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
        w = window_series.dropna().astype(str)
        w_counts = w.value_counts()
        w_total = int(w_counts.sum()) if len(w_counts) else 0

        cats = list(baseline_freq.keys())
        if "__OTHER__" not in cats:
            cats.append("__OTHER__")
        if len(cats) == 0:
            return np.array([], dtype=float), np.array([], dtype=float), [], w_total

        p = np.array([float(baseline_freq.get(c, 0.0)) for c in cats], dtype=float)
        p = DetectorCore._safe_probs(p)

        if w_total == 0:
            q = p.copy()
            return p, q, cats, w_total

        q_counts = {c: 0 for c in cats}
        for cat, cnt in w_counts.items():
            cat = str(cat)
            if cat in q_counts:
                q_counts[cat] += int(cnt)
            else:
                q_counts["__OTHER__"] += int(cnt)

        q = np.array([q_counts[c] for c in cats], dtype=float) / float(w_total)
        q = DetectorCore._safe_probs(q)

        return p, q, cats, w_total

    @staticmethod
    def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        if p.size == 0 or q.size == 0:
            return 0.0
        d = float(jensenshannon(p, q, base=2.0))
        if not np.isfinite(d):
            return 0.0
        return float(max(0.0, d * d))

    @staticmethod
    def _chi_square_pvalue_from_probs(p: np.ndarray, q: np.ndarray, n_ref: int, n_win: int) -> float:
        if p.size == 0 or q.size == 0 or n_ref <= 0 or n_win <= 0:
            return 1.0

        ref = np.rint(p * n_ref).astype(int)
        win = np.rint(q * n_win).astype(int)

        ref[-1] += n_ref - int(ref.sum())
        win[-1] += n_win - int(win.sum())

        table = np.vstack([ref, win])

        keep = table.sum(axis=0) > 0
        table = table[:, keep]
        if table.shape[1] < 2:
            return 1.0

        try:
            _, pval, _, _ = chi2_contingency(table, correction=False)
            return float(pval)
        except ValueError:
            table = table + 1
            _, pval, _, _ = chi2_contingency(table, correction=False)
            return float(pval)

    @classmethod
    def categorical_drift_for_feature(
        cls,
        baseline: BaselineProfile,
        window_df: pd.DataFrame,
        feature: str,
        *,
        alpha: float = 0.05,
        tau_miss: float = 0.10,
    ) -> Dict[str, Any]:
        bfeat = baseline.features[feature]
        baseline_freq = getattr(bfeat, "freq", None) or {}

        # missingness drift (baseline vs window)
        mr_base = float(getattr(bfeat, "missing_rate", 0.0))
        mr_win = float(window_df[feature].isna().mean())
        s_miss = cls._missingness_severity(mr_base, mr_win, tau_miss=tau_miss)

        p, q, _, w_total = cls._aligned_category_probs(baseline_freq, window_df[feature])
        js = cls._js_divergence(p, q)

        n_ref = int(getattr(baseline, "metadata", {}).get("rows", 1000))
        n_win = int(w_total)
        pval = cls._chi_square_pvalue_from_probs(p, q, n_ref=n_ref, n_win=n_win)

        s_js = cls._clip01(js)
        s_p = cls._clip01((float(alpha) - float(pval)) / float(alpha)) if float(alpha) > 0 else 0.0
        severity = float(max(s_js, s_p, s_miss))

        cur_freq = window_df[feature].dropna().astype(str).value_counts(normalize=True).to_dict()

        return {
            "type": "categorical",
            "metrics": {
                "js": float(js),
                "chi2_p": float(pval),
                "s_JS": float(s_js),
                "s_p": float(s_p),
                "missing_rate_baseline": float(mr_base),
                "missing_rate_window": float(mr_win),
                "missing_rate_delta": float(abs(mr_win - mr_base)),
                "s_missing": float(s_miss),
                "tau_miss": float(tau_miss),
            },
            "severity": float(severity),
            "drifted": False,
            "baseline_summary": {"freq": dict(baseline_freq)},
            "current_summary": {"freq": cur_freq},
        }