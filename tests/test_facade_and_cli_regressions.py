import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from driftdetect.baseline import build_baseline_profile
from driftdetect.cli import app
from driftdetect.detector_facade import DetectorConfig, DetectorFacade
from driftdetect.scenario import DriftEvent, Scenario
from driftdetect.simulate import simulate_scenario


def _write_baseline_json(df: pd.DataFrame, path) -> None:
    baseline = build_baseline_profile(df, bins=10, top_k=50)
    path.write_text(json.dumps(baseline.model_dump(by_alias=True), indent=2), encoding="utf-8")


def _simulate_two_windows(tmp_path):
    scenario = Scenario(
        scenario_id="suite",
        seed=123,
        window_count=2,
        window_size=300,
        output_format="csv",
        events=[
            DriftEvent(
                event_id="e1",
                drift_type="mean_shift",
                start_window=1,
                features=["weight"],
                intensity={"delta": 2.0},
            )
        ],
    )
    outdir = tmp_path / "sim"
    simulate_scenario(scenario, outdir=str(outdir), overwrite=False)
    return outdir, outdir / "manifest.json"


def test_detect_suite_does_not_overwrite_without_flag(tmp_path):
    outdir, manifest_path = _simulate_two_windows(tmp_path)

    baseline_df = pd.read_csv(outdir / "windows" / "window_000.csv")
    baseline_path = tmp_path / "baseline.json"
    _write_baseline_json(baseline_df, baseline_path)

    det = DetectorFacade.from_baseline_json(str(baseline_path), DetectorConfig())
    det.detect_suite_from_manifest(str(manifest_path), str(outdir), overwrite=False)

    drift0 = outdir / "drifts" / "drift_000.json"
    drift0.write_text('{"sentinel": true}', encoding="utf-8")

    with pytest.raises(FileExistsError, match="Use --overwrite to replace"):
        det.detect_suite_from_manifest(str(manifest_path), str(outdir), overwrite=False)

    assert json.loads(drift0.read_text(encoding="utf-8")) == {"sentinel": True}


def test_cli_detect_accepts_tau_miss_option(tmp_path):
    baseline_df = pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24, 25],
            "region": ["EU", "US", "EU", "ASIA", "US", "EU"],
            "income": [1000, 1100, 1200, 1300, 1400, 1500],
        }
    )
    baseline_path = tmp_path / "baseline.json"
    _write_baseline_json(baseline_df, baseline_path)

    window_df = baseline_df.copy()
    window_df.loc[[0, 1, 2], "income"] = None
    window_path = tmp_path / "window.csv"
    window_df.to_csv(window_path, index=False)

    out_path = tmp_path / "drift.json"
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "detect",
            "--baseline",
            str(baseline_path),
            "--window",
            str(window_path),
            "--out",
            str(out_path),
            "--tau-miss",
            "0.33",
        ],
    )
    assert res.exit_code == 0, res.output

    obj = json.loads(out_path.read_text(encoding="utf-8"))
    assert obj["global"]["config"]["tau_miss"] == pytest.approx(0.33)


def test_cli_detect_suite_accepts_tau_miss_option(tmp_path):
    outdir, manifest_path = _simulate_two_windows(tmp_path)

    baseline_df = pd.read_csv(outdir / "windows" / "window_000.csv")
    baseline_path = tmp_path / "baseline.json"
    _write_baseline_json(baseline_df, baseline_path)

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "detect-suite",
            "--baseline",
            str(baseline_path),
            "--manifest",
            str(manifest_path),
            "--outdir",
            str(outdir),
            "--overwrite",
            "--tau-miss",
            "0.27",
        ],
    )
    assert res.exit_code == 0, res.output

    detect_manifest = json.loads((outdir / "detect_manifest.json").read_text(encoding="utf-8"))
    drift0 = json.loads((outdir / "drifts" / "drift_000.json").read_text(encoding="utf-8"))

    assert detect_manifest["config"]["tau_miss"] == pytest.approx(0.27)
    assert drift0["global"]["config"]["tau_miss"] == pytest.approx(0.27)
