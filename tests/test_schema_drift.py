import pandas as pd

from driftdetect.baseline import build_baseline_profile
from driftdetect.detector_core import DetectorCore


def _baseline_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [20, 21, 22],
            "region": ["EU", "US", "EU"],
            "income": [1000, 1100, 1200],
        }
    )


def test_schema_drift_added_column():
    baseline = build_baseline_profile(_baseline_df(), bins=10, top_k=50)

    window = _baseline_df().copy()
    window["new_col"] = ["X", "Y", "Z"]

    sd = DetectorCore.schema_drift(baseline, window)
    assert sd["added"] == ["new_col"]
    assert sd["removed"] == []
    assert sd["type_changed"] == []


def test_schema_drift_removed_column():
    baseline = build_baseline_profile(_baseline_df(), bins=10, top_k=50)

    window = _baseline_df()[["age", "region"]].copy()  # drop income
    sd = DetectorCore.schema_drift(baseline, window)

    assert sd["added"] == []
    assert sd["removed"] == ["income"]
    assert sd["type_changed"] == []


def test_schema_drift_type_changed_numeric_to_categorical():
    baseline = build_baseline_profile(_baseline_df(), bins=10, top_k=50)

    window = _baseline_df().copy()
    # Turn numeric income into string => should be inferred as categorical
    window["income"] = window["income"].astype(str)

    sd = DetectorCore.schema_drift(baseline, window)

    assert sd["added"] == []
    assert sd["removed"] == []
    assert len(sd["type_changed"]) == 1
    assert sd["type_changed"][0]["name"] == "income"
    assert sd["type_changed"][0]["baseline"] == "numeric"
    assert sd["type_changed"][0]["window"] == "categorical"