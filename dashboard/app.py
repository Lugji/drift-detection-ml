"""
DriftDetect Streamlit Dashboard – REQ-017, REQ-018, REQ-019, REQ-020.

Artifact-only: reads baseline.json, drift.json, history.json, coverage.json.
No drift computation is performed here (REQ-020 / AC-07).
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Compatibility helpers (supports both our old format and friend's format) ─
def get_global_drift(drift_data: dict) -> bool:
    """Read drift_detected from either {global_drift: bool} or {global: {drift_detected: bool}}"""
    if "global" in drift_data and isinstance(drift_data["global"], dict):
        return bool(drift_data["global"].get("drift_detected", False))
    return bool(drift_data.get("global_drift", False))

def get_ranking(drift_data: dict) -> list:
    """Return feature ranking as a list of feature name strings, regardless of format."""
    raw = drift_data.get("feature_ranking", [])
    if raw and isinstance(raw[0], dict):
        return [r["feature"] for r in raw]
    return raw if raw else list(drift_data.get("features", {}).keys())

st.set_page_config(
    page_title="DriftDetect",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .alert-red {
    background: #fef2f2; border: 1.5px solid #fca5a5; border-radius: 10px;
    padding: 14px 18px; display: flex; align-items: flex-start; gap: 12px;
  }
  .alert-green {
    background: #f0fdf4; border: 1.5px solid #86efac; border-radius: 10px;
    padding: 14px 18px; display: flex; align-items: flex-start; gap: 12px;
  }
  .alert-icon { font-size: 1.4rem; line-height: 1; }
  .alert-title { font-weight: 700; font-size: 1rem; margin-bottom: 2px; }
  .alert-body  { font-size: 0.85rem; color: #555; }
  .metric-card {
    background: white; border-radius: 10px; padding: 16px 20px;
    border: 1px solid #e5e7eb; text-align: center;
  }
  .metric-label { font-size: 0.78rem; color: #6b7280; text-transform: uppercase;
                  letter-spacing: .05em; }
  .metric-value { font-size: 1.6rem; font-weight: 700; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def sev_color(v: float) -> str:
    if v >= 0.7:  return "#ef4444"
    if v >= 0.4:  return "#f59e0b"
    return "#22c55e"


def alert_box(drifted: bool, extra: str = "") -> None:
    if drifted:
        st.markdown(f"""
        <div class="alert-red">
          <div class="alert-icon">🔴</div>
          <div>
            <div class="alert-title">Drift Detected</div>
            <div class="alert-body">Significant distribution change found.{' ' + extra if extra else ''}</div>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-green">
          <div class="alert-icon">🟢</div>
          <div>
            <div class="alert-title">No Drift</div>
            <div class="alert-body">Distribution within expected range.{' ' + extra if extra else ''}</div>
          </div>
        </div>""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────

# Resolve project root (two levels up from dashboard/app.py)
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _p(*parts):
    """Build absolute path from project root."""
    return os.path.join(_ROOT, *parts)

EXPERIMENTS = {
    "Income Drift (Simulation)": {
        "baseline": _p("income_baseline.json"),
        "drift":    _p("income_sim", "drift_w8.json"),
        "history":  _p("income_sim", "history.json"),
        "coverage": _p("income_sim", "coverage.json"),
    },
    "Logistics Experiment": {
        "baseline": _p("logistics_baseline.json"),
        "drift":    _p("logistics_sim_weight_mean_shift", "drift_latest.json"),
        "history":  _p("logistics_sim_weight_mean_shift", "history.json"),
        "coverage": _p("logistics_sim_day_pattern_shift", "coverage.json"),
    },
    "Demo (Synthetic)": {
        "baseline": _p("demo_output", "baseline.json"),
        "drift":    _p("demo_output", "drift_drift.json"),
        "history":  _p("demo_output", "history.json"),
        "coverage": _p("demo_output", "sim", "coverage.json"),
    },
    "Custom": {
        "baseline": "",
        "drift":    "",
        "history":  "",
        "coverage": "",
    },
}

with st.sidebar:
    st.markdown("## 🔍 Drift Monitoring")
    st.markdown("---")

    experiment = st.selectbox(
        "Experiment",
        options=list(EXPERIMENTS.keys()),
        index=0,
    )

    defaults = EXPERIMENTS[experiment]
    custom = experiment == "Custom"

    st.markdown("**Artifact paths**")
    baseline_path = st.text_input("baseline.json",  value=defaults["baseline"], disabled=not custom)
    drift_path    = st.text_input("drift.json",     value=defaults["drift"],    disabled=not custom)
    history_path  = st.text_input("history.json",   value=defaults["history"],  disabled=not custom)
    coverage_path = st.text_input("coverage.json",  value=defaults["coverage"], disabled=not custom)

    st.markdown("---")
    st.markdown("**Thresholds**")
    thresh_warn  = st.slider("Warning (amber)", 0.0, 1.0, 0.4, 0.05)
    thresh_alert = st.slider("Alert (red)",     0.0, 1.0, 0.7, 0.05)
    st.caption("Matches `s_feat` in DetectorConfig")


# ── Load artifacts ────────────────────────────────────────────────────────────

baseline_data = load_json(baseline_path)
drift_data    = load_json(drift_path)
history_data  = load_json(history_path)
cov_data      = load_json(coverage_path)

tabs = st.tabs(["📊 Monitoring", "🧪 Coverage", "📋 Raw Artifacts"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – Monitoring  (REQ-018)
# ════════════════════════════════════════════════════════════════════════════

with tabs[0]:

    # ── Top row: alert + key metrics ─────────────────────────────────────────
    col_alert, col_m1, col_m2, col_m3 = st.columns([2, 1, 1, 1])

    with col_alert:
        if drift_data:
            alert_box(get_global_drift(drift_data))
        else:
            st.info("No drift.json found. Run `driftdetect detect` first.")

    if drift_data:
        meta     = drift_data.get("metadata", {})
        features = drift_data.get("features", {})
        ranking  = get_ranking(drift_data)
        top_sev  = features[ranking[0]]["severity"] if ranking else 0.0
        n_drifted = sum(1 for f in features.values() if f.get("drifted"))

        with col_m1:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-label">Max Severity</div>
              <div class="metric-value" style="color:{sev_color(top_sev)}">{top_sev:.2f}</div>
            </div>""", unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-label">Drifted Features</div>
              <div class="metric-value">{n_drifted}/{len(features)}</div>
            </div>""", unsafe_allow_html=True)
        with col_m3:
            sd = drift_data.get("schema_drift", {})
            schema_ok = not sd.get("detected", False)
            st.markdown(f"""<div class="metric-card">
              <div class="metric-label">Schema</div>
              <div class="metric-value" style="color:{'#22c55e' if schema_ok else '#ef4444'}">
                {'✓ OK' if schema_ok else '⚠ Changed'}
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Severity Over Time chart ─────────────────────────────────────────────
    st.subheader("Severity Over Time")

    if history_data and history_data.get("windows"):
        windows_hist = history_data["windows"]
        all_features = sorted({f for w in windows_hist for f in w.get("features", {})})

        selected_feature = st.selectbox(
            "Feature",
            options=all_features,
            index=0 if all_features else 0,
        )

        # Build time-series DataFrame
        rows = []
        for i, w in enumerate(windows_hist):
            feat_data = w.get("features", {}).get(selected_feature, {})
            rows.append({
                "Window":   w.get("window_id", f"w{i}"),
                "Severity": feat_data.get("severity", 0.0),
                "Drifted":  feat_data.get("drifted", False),
                "GlobalDrift": get_global_drift(w),
            })
        df_hist = pd.DataFrame(rows)

        # Try plotly for the nice band chart; fall back to st.line_chart
        try:
            import plotly.graph_objects as go

            fig = go.Figure()

            # Threshold bands (drawn first so they sit behind the line)
            fig.add_hrect(y0=thresh_alert, y1=1.05,
                          fillcolor="#fee2e2", opacity=0.35, line_width=0,
                          annotation_text="Drift zone", annotation_position="top right",
                          annotation_font_color="#b91c1c")
            fig.add_hrect(y0=thresh_warn, y1=thresh_alert,
                          fillcolor="#fef9c3", opacity=0.45, line_width=0,
                          annotation_text="Warning", annotation_position="top right",
                          annotation_font_color="#854d0e")
            fig.add_hrect(y0=0, y1=thresh_warn,
                          fillcolor="#dcfce7", opacity=0.35, line_width=0,
                          annotation_text="Stable", annotation_position="top right",
                          annotation_font_color="#166534")

            # Threshold lines
            fig.add_hline(y=thresh_alert, line_dash="dash", line_color="#ef4444",
                          line_width=1.2, opacity=0.7)
            fig.add_hline(y=thresh_warn,  line_dash="dash", line_color="#f59e0b",
                          line_width=1.2, opacity=0.7)

            # Severity line
            point_colors = [sev_color(v) for v in df_hist["Severity"]]
            fig.add_trace(go.Scatter(
                x=df_hist["Window"],
                y=df_hist["Severity"],
                mode="lines+markers",
                line=dict(color="#2563eb", width=2.5),
                marker=dict(size=8, color=point_colors, line=dict(width=1.5, color="white")),
                hovertemplate="<b>%{x}</b><br>Severity: %{y:.3f}<extra></extra>",
                name=selected_feature,
            ))

            fig.update_layout(
                title=dict(text=f"<b>{selected_feature}</b> — Severity Over Time",
                           font=dict(size=15)),
                yaxis=dict(range=[0, 1.05], title="Severity", tickformat=".1f",
                           gridcolor="#f3f4f6"),
                xaxis=dict(title="Window", gridcolor="#f3f4f6"),
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(l=40, r=40, t=50, b=40),
                height=340,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            # Fallback: plain line chart
            st.line_chart(df_hist.set_index("Window")["Severity"])
            st.caption(f"Install plotly for threshold bands: `pip install plotly`")

    else:
        # No history yet — show placeholder with demo data
        st.caption("No history.json found. "
                   "Add `--append-history history.json` to your `driftdetect detect` calls to build up a time series.")

        if drift_data and features:
            # Show at least the current window as a single point
            st.info("Showing current window only. Run detection on multiple windows to see a trend.")
            ranking = get_ranking(drift_data)
            selected_feature = st.selectbox("Feature", options=ranking, key="trend_feature")
            sev = features.get(selected_feature, {}).get("severity", 0.0)
            st.metric(f"{selected_feature} severity (current window)", f"{sev:.3f}")

    st.markdown("---")

    # ── Distribution comparison ───────────────────────────────────────────────
    if drift_data and baseline_data:
        feat_list = get_ranking(drift_data)
        # Use the feature selected above if it exists, otherwise let user pick
        if "selected_feature" not in dir() or selected_feature not in feat_list:
            selected_feature = feat_list[0] if feat_list else None

        if selected_feature:
            st.subheader(f"{selected_feature} — Distribution")

            feat_info     = drift_data["features"].get(selected_feature, {})
            baseline_feat = (baseline_data.get("features") or {}).get(selected_feature, {})

            col_dist, col_stats = st.columns([3, 1])

            with col_dist:
                # her JSON uses "type", old code used "dtype"
                dtype = feat_info.get("type") or feat_info.get("dtype")

                if dtype == "numeric":
                    # Reconstruct baseline samples from histogram edges+probs
                    baseline_vals = []
                    hist = baseline_feat.get("hist") or {}
                    edges = hist.get("edges", [])
                    probs = hist.get("probs", [])
                    if edges and probs:
                        import numpy as np
                        centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges)-1)]
                        total = 300
                        for c, p in zip(centers, probs):
                            baseline_vals.extend([c] * max(1, int(round(p * total))))

                    window_vals = feat_info.get("window_sample") or []

                    if baseline_vals or window_vals:
                        try:
                            import plotly.graph_objects as go

                            fig2 = go.Figure()
                            if baseline_vals:
                                fig2.add_trace(go.Histogram(
                                    x=baseline_vals, nbinsx=30,
                                    name="Baseline",
                                    marker_color="#93c5fd", opacity=0.75,
                                    histnorm="probability",
                                ))
                            if window_vals:
                                fig2.add_trace(go.Histogram(
                                    x=window_vals, nbinsx=30,
                                    name="Current window",
                                    marker_color="#f97316", opacity=0.75,
                                    histnorm="probability",
                                ))
                            fig2.update_layout(
                                barmode="overlay",
                                title=dict(text=f"<b>{selected_feature}</b> distribution",
                                           font=dict(size=14)),
                                xaxis_title=selected_feature,
                                yaxis_title="Proportion",
                                plot_bgcolor="white",
                                paper_bgcolor="white",
                                legend=dict(orientation="h", y=1.12),
                                height=300,
                                margin=dict(l=40, r=20, t=50, b=40),
                            )
                            st.plotly_chart(fig2, use_container_width=True)

                        except ImportError:
                            if baseline_vals:
                                st.bar_chart(pd.Series(baseline_vals).value_counts(bins=20).sort_index())
                    else:
                        st.info("No sample values stored. Re-run detect to populate window_sample.")

                elif dtype == "categorical":
                    base_freqs   = baseline_feat.get("freq") or baseline_feat.get("frequencies") or {}
                    window_freqs = (feat_info.get("current_summary") or {}).get("freq") or {}

                    if base_freqs or window_freqs:
                        # Build combined DataFrame with all categories from both
                        all_cats = sorted(set(list(base_freqs.keys()) + list(window_freqs.keys())))
                        df_cat = pd.DataFrame({
                            "Category":       all_cats,
                            "Baseline":       [base_freqs.get(c, 0.0)   for c in all_cats],
                            "Current window": [window_freqs.get(c, 0.0) for c in all_cats],
                        }).sort_values("Baseline", ascending=False)

                        try:
                            import plotly.graph_objects as go
                            fig_cat = go.Figure()
                            fig_cat.add_trace(go.Bar(
                                x=df_cat["Category"], y=df_cat["Baseline"],
                                name="Baseline", marker_color="#93c5fd", opacity=0.85,
                            ))
                            fig_cat.add_trace(go.Bar(
                                x=df_cat["Category"], y=df_cat["Current window"],
                                name="Current window", marker_color="#f97316", opacity=0.85,
                            ))
                            fig_cat.update_layout(
                                barmode="group",
                                title=dict(text=f"<b>{selected_feature}</b> category frequencies",
                                           font=dict(size=14)),
                                xaxis_title="Category",
                                yaxis_title="Proportion",
                                plot_bgcolor="white",
                                paper_bgcolor="white",
                                legend=dict(orientation="h", y=1.12),
                                height=300,
                                margin=dict(l=40, r=20, t=50, b=60),
                            )
                            st.plotly_chart(fig_cat, use_container_width=True)
                        except ImportError:
                            st.bar_chart(df_cat.set_index("Category")[["Baseline", "Current window"]])
                    else:
                        st.info("No categorical frequency data available.")

            with col_stats:
                st.markdown("**Baseline**")
                stats = baseline_feat.get("stats") or {}
                if stats.get("mean") is not None:
                    st.metric("Mean",   f"{stats['mean']:.2f}")
                    st.metric("Std",    f"{stats['std']:.2f}")
                    st.metric("Median", f"{stats.get('q50', 0):.2f}")

                st.markdown("**Current window**")
                metrics = feat_info.get("metrics") or {}
                if metrics.get("ks") is not None:
                    st.metric("KS stat",  f"{metrics['ks']:.3f}")
                if metrics.get("wasserstein") is not None:
                    st.metric("WD",       f"{metrics['wasserstein']:.3f}")
                if metrics.get("js") is not None:
                    st.metric("JS div",   f"{metrics['js']:.3f}")
                sev_val = feat_info.get("severity", 0.0)
                st.markdown(f"**Severity:** <span style='color:{sev_color(sev_val)};font-weight:700'>"
                            f"{sev_val:.3f}</span>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Schema drift expander ────────────────────────────────────────────────
    if drift_data:
        sd = drift_data.get("schema_drift", {})
        sd_detected = bool(sd.get("added") or sd.get("removed") or sd.get("type_changed"))
        with st.expander("Schema Drift Details", expanded=sd_detected):
            if sd_detected:
                if sd.get("added"):
                    st.warning(f"Added columns: {sd['added']}")
                if sd.get("removed"):
                    st.error(f"Removed columns: {sd['removed']}")
                if sd.get("type_changed"):
                    st.warning(f"Type changes: {sd['type_changed']}")
            else:
                st.success("No schema changes detected.")

    # ── Full feature table ────────────────────────────────────────────────────
    if drift_data:
        with st.expander("All Features", expanded=False):
            features = drift_data.get("features", {})
            ranking  = get_ranking(drift_data)
            rows = []
            for fname in ranking:
                f = features.get(fname, {})
                m = f.get("metrics") or {}
                ftype = f.get("type") or f.get("dtype") or ""
                rows.append({
                    "Feature":  fname,
                    "Type":     ftype,
                    "Severity": round(f.get("severity", 0.0), 6),
                    "Drifted":  "✅" if f.get("drifted") else "—",
                    "KS":       round(m["ks"], 3)           if m.get("ks")           is not None else None,
                    "WD":       round(m["wasserstein"], 3)  if m.get("wasserstein")  is not None else None,
                    "JS":       round(m["js"], 3)           if m.get("js")           is not None else None,
                })
            if rows:
                df_f = pd.DataFrame(rows)
                st.dataframe(
                    df_f.style.background_gradient(subset=["Severity"], cmap="RdYlGn_r"),
                    use_container_width=True,
                )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – Coverage  (REQ-019)
# ════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.header("Coverage Report")

    if not cov_data:
        st.info("No coverage.json found. Run `driftdetect coverage` first.")
    else:
        meta = cov_data.get("metadata", {})
        st.caption(f"Suite: **{meta.get('suite_id','—')}** | "
                   f"Tolerance M = {meta.get('tolerance_m','—')} windows | "
                   f"Config hash: `{meta.get('config_hash','—')}`")

        # Top metrics
        c1, c2, c3, c4 = st.columns(4)
        fpr  = cov_data.get("suite_fpr", 0.0)
        ttd  = cov_data.get("ttd_stats") or {}
        per  = cov_data.get("per_event_results", [])
        missed = cov_data.get("missed_events", [])
        power_vals = list(cov_data.get("power_by_type", {}).values())
        avg_power = sum(power_vals) / len(power_vals) if power_vals else 0.0

        c1.metric("Suite FPR",   f"{fpr:.3f}",      help="False positive rate — lower is better")
        c2.metric("Avg Power",   f"{avg_power:.3f}", help="Mean detection power across drift types")
        c3.metric("TTD mean",    f"{ttd.get('mean',0):.1f} w" if ttd else "—",
                  help="Mean time-to-detect in windows")
        c4.metric("Missed",      f"{len(missed)} / {len(per)}",
                  delta="all caught" if not missed else f"{len(missed)} missed",
                  delta_color="normal" if not missed else "inverse")

        st.markdown("---")

        # Power bar chart
        st.subheader("Detection Power by Drift Type")
        power = cov_data.get("power_by_type", {})
        if power:
            try:
                import plotly.graph_objects as go
                fig3 = go.Figure(go.Bar(
                    x=list(power.keys()),
                    y=list(power.values()),
                    marker_color=["#22c55e" if v >= 0.8 else "#f59e0b" if v >= 0.5 else "#ef4444"
                                  for v in power.values()],
                    text=[f"{v:.0%}" for v in power.values()],
                    textposition="outside",
                ))
                fig3.update_layout(
                    yaxis=dict(range=[0, 1.1], title="Power", tickformat=".0%"),
                    xaxis_title="Drift type",
                    plot_bgcolor="white", paper_bgcolor="white",
                    height=280, margin=dict(l=40, r=20, t=20, b=40),
                )
                fig3.add_hline(y=0.8, line_dash="dot", line_color="#6b7280",
                               annotation_text="80% target", annotation_position="right")
                st.plotly_chart(fig3, use_container_width=True)
            except ImportError:
                df_p = pd.DataFrame([{"Type": k, "Power": v} for k, v in power.items()])
                st.bar_chart(df_p.set_index("Type"))

        # TTD histogram
        finite_ttds = [e["ttd"] for e in per if e.get("ttd") is not None]
        if finite_ttds:
            st.subheader("Time-to-Detect Distribution")
            try:
                import plotly.graph_objects as go
                fig4 = go.Figure(go.Histogram(
                    x=finite_ttds, nbinsx=max(len(set(finite_ttds)), 1),
                    marker_color="#2563eb", opacity=0.8,
                ))
                fig4.update_layout(
                    xaxis_title="Windows after onset",
                    yaxis_title="Count",
                    plot_bgcolor="white", paper_bgcolor="white",
                    height=220, margin=dict(l=40, r=20, t=20, b=40),
                )
                st.plotly_chart(fig4, use_container_width=True)
            except ImportError:
                st.bar_chart(pd.Series(finite_ttds).value_counts().sort_index())

        # Missed events
        st.subheader("Missed Events")
        if missed:
            st.error(f"{len(missed)} event(s) not detected within tolerance M = {meta.get('tolerance_m','?')}")
            st.dataframe(pd.DataFrame(missed), use_container_width=True)
        else:
            st.success("All events detected within tolerance window.")

        # Per-event table
        with st.expander("All per-event results"):
            if per:
                df_pe = pd.DataFrame(per)
                st.dataframe(df_pe, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 – Raw artifacts
# ════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.header("Raw Artifacts")
    for label, path in [
        ("baseline.json",  baseline_path),
        ("drift.json",     drift_path),
        ("history.json",   history_path),
        ("coverage.json",  coverage_path),
    ]:
        with st.expander(label, expanded=False):
            if Path(path).exists():
                st.json(load_json(path))
            else:
                st.caption(f"Not found: `{path}`")