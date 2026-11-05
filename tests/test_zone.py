import json

import pytest

from arten.postprocess import Detection
from arten.zone import count_detections_in_zone, load_zone_profile, summarise_zone


def test_load_zone_profile_and_count_bottom_center(tmp_path):
    path = tmp_path / "zone.json"
    path.write_text(json.dumps({"name": "crossing", "points": [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]}))
    profile = load_zone_profile(path)
    detections = [
        Detection(0, "person", 0.9, (40, 20, 60, 70)),
        Detection(2, "car", 0.9, (40, 20, 60, 70)),
        Detection(0, "person", 0.9, (2, 2, 10, 10)),
    ]

    assert count_detections_in_zone(detections, profile, width=100, height=100) == {
        "person": 1,
        "vehicle": 1,
    }


def test_zone_profile_rejects_out_of_range_points(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"name": "bad", "points": [[0, 0], [2, 0], [0, 1]]}))
    with pytest.raises(ValueError, match="from 0 to 1"):
        load_zone_profile(path)


def test_zone_summary_reports_co_presence_without_risk_claim():
    rows = [
        {"source_frame_idx": 0, "person": 1, "vehicle": 1},
        {"source_frame_idx": 1, "person": 2, "vehicle": 1},
        {"source_frame_idx": 2, "person": 0, "vehicle": 1},
        {"source_frame_idx": 3, "person": 1, "vehicle": 1},
    ]
    result = summarise_zone(rows, source_fps=2, stride=1, is_video=True)

    assert result["frames_with_people_and_vehicles"] == 3
    assert result["co_presence_sample_seconds"] == 1.5
    assert result["longest_co_presence_sample_seconds"] == 1.0
    assert "not a conflict" in result["interpretation"]
