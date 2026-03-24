from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from driftdetect.io import load_dataset
from driftdetect.baseline import build_baseline_profile
from driftdetect.scenario import load_scenario
from driftdetect.simulate import simulate_scenario
from driftdetect.detector_facade import DetectorConfig, DetectorFacade
from driftdetect.coverage import coverage_from_simdir
from driftdetect.reporter import save_report_from_files

app = typer.Typer(help="DriftDetect CLI")


@app.command("baseline-build")
def baseline_build(
    data: str = typer.Option(..., "--data", help="Path to baseline dataset (.csv or .parquet)"),
    out: str = typer.Option("baseline.json", "--out", help="Output baseline JSON file"),
    top_k: int = typer.Option(50, "--top-k", help="Top-k categories stored for categorical features"),
):
    """
    Build baseline.json from a dataset file (REQ-001/REQ-002).
    """
    df = load_dataset(data)
    baseline = build_baseline_profile(df, top_k=top_k)
    payload = baseline.model_dump(by_alias=True)

    Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote baseline profile to {out}")


@app.command("scenario-validate")
def scenario_validate(
    path: str = typer.Option(..., "--path", help="Path to scenario YAML file"),
):
    """
    Validate a scenario YAML file (REQ-008).
    """
    scenario = load_scenario(path)
    print(
        "Loaded scenario: "
        f"scenario_id={scenario.scenario_id}, "
        f"seed={scenario.seed}, "
        f"window_count={scenario.window_count}, "
        f"window_size={scenario.window_size}, "
        f"output_format={scenario.output_format}, "
        f"events={len(scenario.events)}"
    )


@app.command("simulate")
def simulate_cmd(
    scenario_path: str = typer.Option(..., "--scenario", help="Path to scenario YAML file"),
    outdir: str = typer.Option("sim_out", "--outdir", help="Output directory"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite output directory if it exists"),
):
    """
    Generate synthetic windows + drift_log.json + manifest.json from scenario YAML (REQ-009/REQ-010).
    """
    scenario = load_scenario(scenario_path)
    manifest = simulate_scenario(scenario, outdir=outdir, overwrite=overwrite)

    manifest_path = Path(outdir) / "manifest.json"
    drifts_dir_default = Path(outdir) / "drifts"
    coverage_out_default = Path(outdir) / "coverage.json"

    print("Simulation complete.")
    print(f"- windows: {manifest['window_count']}  (in {Path(outdir) / 'windows'})")
    print(f"- drift_log: {manifest['drift_log_path']}")
    print(f"- manifest: {manifest_path}")

    # Helpful “next steps” for demo / professor
    print("\nNext commands:")
    print(f"  python -m driftdetect.cli detect-suite --baseline baseline.json --manifest {manifest_path} --outdir {outdir}")
    print(f"  python -m driftdetect.cli coverage --manifest {manifest_path} --drifts-dir {drifts_dir_default} --out {coverage_out_default}")


@app.command("detect")
def detect_cmd(
    baseline_path: str = typer.Option(..., "--baseline", help="Path to baseline.json"),
    window_path: str = typer.Option(..., "--window", help="Path to one window dataset (.csv/.parquet)"),
    window_id: Optional[str] = typer.Option(
        None, "--window-id", help="Optional custom id stored in drift.json (defaults to filename)"
    ),
    out: str = typer.Option("drift.json", "--out", help="Output drift JSON file"),
    alpha: float = typer.Option(0.05, "--alpha", help="Significance level for categorical p-value mapping"),
    S_feat: float = typer.Option(0.7, "--S-feat", help="Feature drift threshold (severity >= S_feat)"),
    S_global: float = typer.Option(0.7, "--S-global", help="Global max severity threshold"),
    K: int = typer.Option(3, "--K", help="Global count rule: drift if >= K features are drifted"),
    tau_wd: float = typer.Option(1.0, "--tau-wd", help="Wasserstein severity scaling parameter"),
    tau_mu: float = typer.Option(1.0, "--tau-mu", help="Mean-shift severity scaling parameter"),
    tau_sigma: float = typer.Option(0.5, "--tau-sigma", help="Std-change severity scaling parameter"),
    tau_miss: float = typer.Option(0.10, "--tau-miss", help="Missing-rate severity scaling parameter"),
    append_history: str = typer.Option("", "--append-history", help="Path to history.json to append result to"),
    ci: bool = typer.Option(False, "--ci", help="Exit with code 1 if drift detected (for CI pipelines)"),
):
    """
    Detect drift for ONE window and write drift.json (production mode).
    """
    cfg = DetectorConfig(
        alpha=alpha,
        S_feat=S_feat,
        S_global=S_global,
        K=K,
        tau_wd=tau_wd,
        tau_mu=tau_mu,
        tau_sigma=tau_sigma,
        tau_miss=tau_miss,
    )

    det = DetectorFacade.from_baseline_json(baseline_path, cfg)

    window_df = load_dataset(window_path)
    wid = window_id or Path(window_path).name

    drift_result = det.detect_window(window_df, window_id=wid, out_path=out)

    # Append to history.json for dashboard time-series chart
    if append_history:
        import datetime
        hist_path = Path(append_history)
        history = {"windows": []}
        if hist_path.exists():
            try:
                history = json.loads(hist_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        entry = {
            "window_id": wid,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "global": drift_result.get("global", {}),
            "features": {
                k: {"severity": v.get("severity", 0.0), "drifted": v.get("drifted", False)}
                for k, v in drift_result.get("features", {}).items()
            },
        }
        history.setdefault("windows", []).append(entry)
        hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"Wrote drift result to {out}")

    if ci and drift_result.get("global", {}).get("drift_detected", False):
        print("CI GATE FAIL: drift_detected=true")
        raise typer.Exit(code=1)


@app.command("detect-suite")
def detect_suite_cmd(
    baseline_path: str = typer.Option(..., "--baseline", help="Path to baseline.json"),
    manifest_path: str = typer.Option(..., "--manifest", help="Path to sim manifest.json"),
    outdir: Optional[str] = typer.Option(
        None, "--outdir", help="Output directory (default: same folder as manifest.json)"
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing drift_*.json files"),
    alpha: float = typer.Option(0.05, "--alpha", help="Significance level for categorical p-value mapping"),
    S_feat: float = typer.Option(0.7, "--S-feat", help="Feature drift threshold (severity >= S_feat)"),
    S_global: float = typer.Option(0.7, "--S-global", help="Global max severity threshold"),
    K: int = typer.Option(3, "--K", help="Global count rule: drift if >= K features are drifted"),
    tau_wd: float = typer.Option(1.0, "--tau-wd", help="Wasserstein severity scaling parameter"),
    tau_mu: float = typer.Option(1.0, "--tau-mu", help="Mean-shift severity scaling parameter"),
    tau_sigma: float = typer.Option(0.5, "--tau-sigma", help="Std-change severity scaling parameter"),
    tau_miss: float = typer.Option(0.10, "--tau-miss", help="Missing-rate severity scaling parameter"),
    run_sanity: bool = typer.Option(False, "--run-sanity", help="Run sanity check report after detection"),
):
    """
    Run detection across all simulated windows listed in manifest.json.
    Writes drift_###.json files to outdir/drifts/ and a detect_manifest.json.
    """
    cfg = DetectorConfig(
        alpha=alpha,
        S_feat=S_feat,
        S_global=S_global,
        K=K,
        tau_wd=tau_wd,
        tau_mu=tau_mu,
        tau_sigma=tau_sigma,
        tau_miss=tau_miss,
    )

    resolved_outdir = outdir or str(Path(manifest_path).parent)

    det = DetectorFacade.from_baseline_json(baseline_path, cfg)

    dm = det.detect_suite_from_manifest(
        manifest_path=manifest_path,
        outdir=resolved_outdir,
        overwrite=overwrite,
        run_sanity=run_sanity,
    )

    print("Detection complete.")
    print(f"- drifts_dir: {dm['drifts_dir']}")
    print(f"- windows: {len(dm['drift_results'])}")


@app.command("coverage")
def coverage_cmd(
    manifest_path: str = typer.Option(..., "--manifest", help="Path to sim manifest.json"),
    drifts_dir: Optional[str] = typer.Option(
        None, "--drifts-dir", help="Directory containing drift_###.json (default: manifest_dir/drifts)"
    ),
    M: int = typer.Option(3, "--M", help="Tolerance (max allowed detection delay) in windows"),
    out: Optional[str] = typer.Option(
        None, "--out", help="Output coverage.json path (default: next to manifest.json)"
    ),
):
    """
    Compute coverage metrics + missed events from drift_log.json and drift_###.json (REQ-012/REQ-013).
    """
    manifest_dir = Path(manifest_path).parent
    resolved_drifts_dir = drifts_dir or str(manifest_dir / "drifts")
    resolved_out = out or str(manifest_dir / "coverage.json")

    cov = coverage_from_simdir(
        manifest_path=manifest_path,
        drifts_dir=resolved_drifts_dir,
        M=M,
        out_path=resolved_out,
    )

    print("Coverage complete.")
    print(f"- wrote: {resolved_out}")
    print(f"- FPR: {cov['metrics']['FPR']}")
    print(f"- power_overall: {cov['metrics']['power_overall']}")
    print(f"- missed_events: {len(cov['missed_events'])}")

@app.command("ci-gate")
def ci_gate_cmd(
    drift_path: str = typer.Option(..., "--drift", help="Path to drift.json"),
):
    obj = json.loads(Path(drift_path).read_text(encoding="utf-8"))
    detected = bool(obj.get("global", {}).get("drift_detected", False))

    if detected:
        print("CI GATE FAIL: drift_detected=true")
        raise typer.Exit(code=1)

    print("CI GATE PASS: drift_detected=false")
    raise typer.Exit(code=0)


@app.command("report")
def report_cmd(
    drift_path: Optional[str] = typer.Option(None, "--drift", help="Path to drift.json"),
    coverage_path: Optional[str] = typer.Option(None, "--coverage", help="Path to coverage.json"),
    out: str = typer.Option("report.html", "--out", help="Output HTML report path"),
):
    """
    Generate an HTML report from drift.json and/or coverage.json artifacts.
    """
    if not drift_path and not coverage_path:
        print("Error: provide at least --drift or --coverage")
        raise typer.Exit(code=1)

    save_report_from_files(out, drift_path=drift_path, coverage_path=coverage_path)
    print(f"Wrote report to {out}")


if __name__ == "__main__":
    app()
