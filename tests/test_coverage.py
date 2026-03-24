from driftdetect.coverage import compute_coverage


def test_coverage_basic():
    # 5 windows, event starts at window 2, detection at window 2 => TTD=0
    drift_log = [
        {"event_id": "e1", "type": "numeric_shift", "start_window": 2, "affected_features": ["weight"], "intensity": {}}
    ]
    g = [False, False, True, True, True]

    cov = compute_coverage(drift_log=drift_log, g_flags=g, M=1)

    assert cov["metrics"]["FPR"] == 0.0  # windows 0,1 are no-drift; both False
    assert cov["per_event"][0]["TTD"] == 0
    assert cov["per_event"][0]["detected_within_M"] is True
    assert len(cov["missed_events"]) == 0


def test_coverage_missed_event():
    drift_log = [
        {"event_id": "e1", "type": "numeric_shift", "start_window": 1, "affected_features": ["x"], "intensity": {}}
    ]
    g = [False, False, False]

    cov = compute_coverage(drift_log=drift_log, g_flags=g, M=2)

    assert cov["per_event"][0]["TTD"] is None
    assert cov["per_event"][0]["detected_within_M"] is False
    assert len(cov["missed_events"]) == 1
