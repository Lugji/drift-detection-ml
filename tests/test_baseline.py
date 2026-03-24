import json
import pandas as pd
import pytest

from driftdetect.io import load_dataset
from driftdetect.baseline import build_baseline_profile


def _make_df():
    # Small deterministic dataset
    return pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            "region": ["EU", "EU", "EU", "EU", "EU", "US", "US", "US", "ASIA", "ASIA"],
            "income": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        }
    )


def test_load_dataset_csv_and_build_baseline(tmp_path):
    df = _make_df()
    csv_path = tmp_path / "baseline.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_dataset(str(csv_path))
    assert loaded.shape == (10, 3)

    baseline = build_baseline_profile(loaded, bins=10, top_k=50)

    # metadata
    assert baseline.metadata["rows"] == 10
    assert baseline.metadata["cols"] == 3

    # schema types
    schema_map = {x["name"]: x["type"] for x in baseline.schema_}
    assert schema_map["age"] == "numeric"
    assert schema_map["income"] == "numeric"
    assert schema_map["region"] == "categorical"

    # features: presence + missing rates
    assert baseline.features["age"].missing_rate == 0.0
    assert baseline.features["region"].missing_rate == 0.0

    # numeric stats exist for numeric features
    assert baseline.features["age"].stats is not None
    assert baseline.features["income"].stats is not None

    # categorical freq exists for categorical features
    assert baseline.features["region"].freq is not None
    freq = baseline.features["region"].freq
    assert abs(sum(freq.values()) - 1.0) < 1e-6
    assert freq["EU"] == pytest.approx(0.5)


def test_load_dataset_parquet_and_build_baseline(tmp_path):
    # Skip test if pyarrow not installed in venv
    pytest.importorskip("pyarrow")

    df = _make_df()
    pq_path = tmp_path / "baseline.parquet"
    df.to_parquet(pq_path, index=False)

    loaded = load_dataset(str(pq_path))
    assert loaded.shape == (10, 3)

    baseline = build_baseline_profile(loaded, bins=10, top_k=50)

    schema_map = {x["name"]: x["type"] for x in baseline.schema_}
    assert schema_map["age"] == "numeric"
    assert schema_map["region"] == "categorical"


def test_baseline_serialization_uses_schema_alias(tmp_path):
    df = _make_df()
    baseline = build_baseline_profile(df, bins=10, top_k=50)

    # by_alias=True should export schema_ as "schema"
    payload = baseline.model_dump(by_alias=True)
    assert "schema" in payload
    assert "schema_" not in payload

    # Ensure JSON roundtrip works
    j = json.dumps(payload)
    obj = json.loads(j)
    assert "schema" in obj
