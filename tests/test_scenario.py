import pytest
from driftdetect.scenario import load_scenario


def test_load_scenario_minimal(tmp_path):
    content = """
scenario_id: demo
seed: 42
window_count: 5
window_size: 100
output_format: csv
events:
  - event_id: age_shift
    drift_type: mean_shift
    start_window: 1
    features: [age]
    intensity:
      delta: 2.0
  - event_id: region_shift
    drift_type: new_category
    start_window: 3
    features: [region]
    intensity:
      value: NEW_REGION
      fraction: 0.2
"""
    path = tmp_path / "scenario.yaml"
    path.write_text(content, encoding="utf-8")

    scenario = load_scenario(str(path))

    assert scenario.scenario_id == "demo"
    assert scenario.seed == 42
    assert scenario.window_count == 5
    assert scenario.window_size == 100
    assert scenario.output_format == "csv"
    assert len(scenario.events) == 2
    assert scenario.events[0].event_id == "age_shift"
    assert scenario.events[0].start_window == 1
    assert scenario.events[0].features == ["age"]


def test_event_start_window_out_of_range_raises(tmp_path):
    content = """
scenario_id: demo
seed: 7
window_count: 2
window_size: 50
events:
  - event_id: event1
    drift_type: mean_shift
    start_window: 2
    features: [age]
    intensity:
      delta: 1.0
"""
    path = tmp_path / "scenario.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="start_window must be < window_count"):
        load_scenario(str(path))


def test_duplicate_event_ids_raise(tmp_path):
    content = """
scenario_id: demo
seed: 7
window_count: 3
window_size: 50
events:
  - event_id: event1
    drift_type: mean_shift
    start_window: 0
    features: [age]
    intensity:
      delta: 1.0
  - event_id: event1
    drift_type: missingness_shift
    start_window: 1
    features: [income]
    intensity:
      missing_rate_delta: 0.1
"""
    path = tmp_path / "scenario.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="event_id values must be unique"):
        load_scenario(str(path))
