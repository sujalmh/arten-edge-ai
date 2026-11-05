import numpy as np

from namma_drishti.attributes import AttributeAccumulator, estimate_vehicle_color
from namma_drishti.postprocess import Detection
from namma_drishti.tracker import IoUTracker


def solid_frame(bgr):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20:80, 20:80] = bgr
    return frame


def test_color_estimates_common_vehicle_paints():
    box = (20, 20, 80, 80)
    assert estimate_vehicle_color(solid_frame((0, 0, 255)), box) == "red"
    assert estimate_vehicle_color(solid_frame((255, 0, 0)), box) == "blue"
    assert estimate_vehicle_color(solid_frame((0, 255, 0)), box) == "green"
    assert estimate_vehicle_color(solid_frame((0, 255, 255)), box) == "yellow"
    assert estimate_vehicle_color(solid_frame((255, 255, 255)), box) == "white"
    assert estimate_vehicle_color(solid_frame((0, 0, 0)), box) == "black"
    assert estimate_vehicle_color(solid_frame((128, 128, 128)), box) == "gray/silver"


def test_color_returns_unknown_for_tiny_crop():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert estimate_vehicle_color(frame, (0, 0, 5, 5)) == "unknown"


def test_tracker_attaches_direction_speed_and_dwell():
    tracker = IoUTracker(min_hits=1)
    tracker.update([Detection(0, "person", 0.9, (10, 10, 50, 100))], 0)
    tracker.update([Detection(0, "person", 0.9, (20, 10, 60, 100))], 1)
    detection = tracker.update([Detection(0, "person", 0.9, (30, 10, 70, 100))], 2)[0]
    tracker.annotate_motion_attributes([detection], source_fps=10, stride=1)
    attrs = detection.attributes
    assert attrs["direction"] == "right"
    assert attrs["speed_px_s"] > 0
    assert attrs["dwell_s"] == 0.2
    assert attrs["hits"] >= 2


def test_accumulator_deduplicates_tracks_and_summarises():
    acc = AttributeAccumulator()
    det = Detection(
        2,
        "car",
        0.9,
        (10, 10, 50, 50),
        track_id=7,
        attributes={"direction": "right", "speed_px_s": 12.5, "dwell_s": 1.2, "hits": 3, "color_estimate": "white"},
    )
    acc.update([det], 0)
    acc.update([det], 1)
    summary = acc.summary()
    assert summary["total_tracks"] == 1
    car = summary["by_class"]["car"]
    assert car["tracks"] == 1
    assert car["directions"] == {"right": 1}
    assert car["colors"] == {"white": 1}
    assert car["mean_dwell_s"] == 1.2


def test_detection_dict_includes_attributes():
    det = Detection(2, "car", 0.9, (1, 2, 3, 4), track_id=1, attributes={"direction": "left"})
    payload = det.to_dict()
    assert payload["attributes"] == {"direction": "left"}
    assert payload["track_id"] == 1
