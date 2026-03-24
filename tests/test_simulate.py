import pandas as pd
import pytest

from driftdetect.scenario import DriftEvent, Scenario
from driftdetect.simulate import simulate_scenario


def _read_window(outdir, idx):
    return pd.read_csv(outdir / "windows" / f"window_{idx:03d}.csv")


def test_mean_shift_delta_alias_applied(tmp_path):
    scenario = Scenario(
        scenario_id="sim-mean",
        seed=123,
        window_count=2,
        window_size=12000,
        output_format="csv",
        events=[
            DriftEvent(
                event_id="e1",
                drift_type="mean_shift",
                start_window=1,
                features=["weight"],
                intensity={"delta": 5.0},
            )
        ],
    )

    outdir = tmp_path / "sim_mean"
    simulate_scenario(scenario, outdir=str(outdir), overwrite=False)

    w0 = _read_window(outdir, 0)["weight"].mean()
    w1 = _read_window(outdir, 1)["weight"].mean()

    # Large window_size keeps this stable; expect ~ +5 shift.
    assert 4.5 <= (w1 - w0) <= 5.5


def test_categorical_value_fraction_alias_applied(tmp_path):
    scenario = Scenario(
        scenario_id="sim-cat",
        seed=123,
        window_count=2,
        window_size=12000,
        output_format="csv",
        events=[
            DriftEvent(
                event_id="e1",
                drift_type="new_category",
                start_window=1,
                features=["unit_type"],
                intensity={"value": "UT_SPECIAL", "fraction": 0.25},
            )
        ],
    )

    outdir = tmp_path / "sim_cat"
    simulate_scenario(scenario, outdir=str(outdir), overwrite=False)

    w0 = _read_window(outdir, 0)
    w1 = _read_window(outdir, 1)

    p0 = (w0["unit_type"] == "UT_SPECIAL").mean()
    p1 = (w1["unit_type"] == "UT_SPECIAL").mean()

    assert p0 == 0.0
    assert 0.20 <= p1 <= 0.30


def test_missingness_shift_alias_applied(tmp_path):
    scenario = Scenario(
        scenario_id="sim-missing",
        seed=123,
        window_count=2,
        window_size=12000,
        output_format="csv",
        events=[
            DriftEvent(
                event_id="e1",
                drift_type="missingness_shift",
                start_window=1,
                features=["weight"],
                intensity={"missing_rate_delta": 0.5},
            )
        ],
    )

    outdir = tmp_path / "sim_missing"
    simulate_scenario(scenario, outdir=str(outdir), overwrite=False)

    w0 = _read_window(outdir, 0)["weight"].isna().mean()
    w1 = _read_window(outdir, 1)["weight"].isna().mean()

    assert w0 < 0.01
    assert 0.45 <= w1 <= 0.55


def test_unknown_drift_type_rejected_by_scenario():
    with pytest.raises(ValueError, match="unsupported drift_type"):
        Scenario(
            scenario_id="bad",
            seed=1,
            window_count=1,
            window_size=10,
            output_format="csv",
            events=[
                DriftEvent(
                    event_id="bad",
                    drift_type="totally_unknown",
                    start_window=0,
                    features=["weight"],
                    intensity={},
                )
            ],
        )