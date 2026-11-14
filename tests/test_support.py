import numpy as np

from arten.support import Segmenter, classify_posture, is_rider


def _keypoints(**overrides):
    points = [(0.0, 0.0, 0.0)] * 17
    for index, value in overrides.items():
        points[int(index[1:])] = value
    return points


def test_posture_standing():
    points = _keypoints(
        k5=(100, 100, 0.9), k6=(110, 100, 0.9),
        k11=(100, 150, 0.9), k12=(110, 150, 0.9),
        k15=(100, 200, 0.9), k16=(110, 200, 0.9),
    )
    assert classify_posture(points, (80, 90, 130, 210)) == "standing"


def test_posture_walking():
    points = _keypoints(
        k5=(100, 100, 0.9), k6=(110, 100, 0.9),
        k11=(100, 150, 0.9), k12=(110, 150, 0.9),
        k15=(80, 200, 0.9), k16=(130, 200, 0.9),
    )
    assert classify_posture(points, (60, 90, 150, 210)) == "walking"


def test_posture_fallen():
    points = _keypoints(
        k5=(100, 100, 0.9), k6=(200, 100, 0.9),
        k11=(200, 105, 0.9), k12=(300, 105, 0.9),
    )
    assert classify_posture(points, (80, 90, 320, 210)) == "fallen"


def test_posture_unknown_and_rider():
    assert classify_posture(_keypoints(), (0, 0, 10, 10)) == "unknown"
    assert classify_posture(_keypoints(), (0, 0, 10, 10), rider=True) == "riding"


def test_surface_at():
    mask = np.full((100, 100), 3, dtype=np.uint8)
    assert Segmenter.surface_at(mask, (10, 10, 50, 50)) == "road"
    mask[:] = 4
    assert Segmenter.surface_at(mask, (10, 10, 50, 50)) == "sidewalk"
    mask[:] = 0
    assert Segmenter.surface_at(mask, (10, 10, 50, 50)) == "other"
    assert Segmenter.surface_at(None, (10, 10, 50, 50)) == "unknown"


def test_is_rider():
    assert is_rider((10, 10, 50, 50), [(5, 20, 60, 60)])
    assert not is_rider((10, 10, 50, 50), [(100, 100, 200, 200)])
