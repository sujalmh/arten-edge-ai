import numpy as np

from arten.postprocess import count_by_class, decode, nms
from arten.preprocess import LetterboxMeta


def identity_meta(size=640):
    return LetterboxMeta(
        scale=1.0, pad_x=0, pad_y=0, orig_w=size, orig_h=size
    )


def make_output(rows, n_classes=80):
    """rows: list of (cx, cy, w, h, class_id, score)."""
    out = np.zeros((1, 4 + n_classes, len(rows)), dtype=np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(rows):
        out[0, 0, i] = cx
        out[0, 1, i] = cy
        out[0, 2, i] = w
        out[0, 3, i] = h
        out[0, 4 + cls, i] = score
    return out


def test_decode_single_person():
    out = make_output([(320, 320, 100, 100, 0, 0.9)])
    dets, stats = decode(out, identity_meta(), classes=["person"], conf_threshold=0.3)
    assert len(dets) == 1
    det = dets[0]
    assert det.class_name == "person"
    assert abs(det.box[0] - 270) < 1e-3
    assert abs(det.box[2] - 370) < 1e-3
    assert stats.after_nms == 1


def test_nms_suppresses_duplicate_person():
    out = make_output(
        [
            (320, 320, 100, 100, 0, 0.9),
            (322, 322, 100, 100, 0, 0.8),
            (100, 100, 50, 50, 2, 0.7),
        ]
    )
    dets, _ = decode(out, identity_meta(), classes=["person", "car"], conf_threshold=0.3)
    assert len(dets) == 2
    assert {d.class_name for d in dets} == {"person", "car"}


def test_class_filter_and_counts():
    out = make_output(
        [
            (320, 320, 100, 100, 0, 0.9),
            (100, 100, 50, 50, 2, 0.8),
            (200, 200, 50, 50, 2, 0.6),
        ]
    )
    dets, _ = decode(out, identity_meta(), classes=["car"], conf_threshold=0.3)
    assert len(dets) == 2
    assert all(d.class_name == "car" for d in dets)
    counts = count_by_class(dets, ["person", "car"])
    assert counts == {"person": 0, "car": 2}


def test_yolov5_objectness_layout():
    out = np.zeros((1, 85, 1), dtype=np.float32)
    out[0, :4, 0] = [320, 320, 100, 100]
    out[0, 4, 0] = 0.8  # objectness
    out[0, 5 + 0, 0] = 0.9  # person
    dets, _ = decode(out, identity_meta(), classes=["person"], conf_threshold=0.5)
    assert len(dets) == 1
    assert abs(dets[0].score - 0.72) < 1e-4


def test_rescale_to_original_coordinates():
    meta = LetterboxMeta(scale=0.5, pad_x=0, pad_y=0, orig_w=1280, orig_h=1280)
    out = make_output([(160, 160, 80, 80, 2, 0.9)])
    dets, _ = decode(out, meta, classes=["car"], conf_threshold=0.3)
    assert len(dets) == 1
    assert np.allclose(dets[0].box, (240, 240, 400, 400), atol=1e-3)


def test_low_confidence_returns_empty():
    out = make_output([(320, 320, 100, 100, 0, 0.2)])
    dets, stats = decode(out, identity_meta(), classes=["person"], conf_threshold=0.5)
    assert dets == []
    assert stats.after_confidence == 0


def test_nms_empty_and_non_overlapping():
    assert nms(np.empty((0, 4), dtype=np.float32), np.empty((0,)), 0.5).size == 0
    boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    assert nms(boxes, scores, 0.5).tolist() == [0, 1]
