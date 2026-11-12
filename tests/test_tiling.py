from arten.postprocess import Detection
from arten.tiling import merge_detections, tile_origins


def _det(class_id: int, score: float, box: tuple[float, float, float, float]) -> Detection:
    return Detection(class_id=class_id, class_name=str(class_id), score=score, box=box)


def test_origins_2x2_overlap():
    assert tile_origins(1040, 780, 2) == [(0, 0), (400, 0), (0, 140), (400, 140)]


def test_origins_off():
    assert tile_origins(1040, 780, 0) == [(0, 0)]
    assert tile_origins(1040, 780, 1) == [(0, 0)]


def test_origins_small_image():
    assert tile_origins(320, 240, 2) == [(0, 0)]


def test_origins_grid_three():
    origins = tile_origins(1040, 780, 3)
    assert len(origins) == 9
    assert origins[0] == (0, 0)
    assert origins[-1] == (400, 140)


def test_merge_same_box():
    detections = [_det(0, 0.9, (0, 0, 10, 10)), _det(0, 0.8, (0, 0, 10, 10))]
    assert len(merge_detections(detections)) == 1


def test_merge_keeps_different_classes():
    detections = [_det(0, 0.9, (0, 0, 10, 10)), _det(2, 0.8, (0, 0, 10, 10))]
    assert len(merge_detections(detections)) == 2


def test_merge_contained_box():
    detections = [_det(0, 0.9, (0, 0, 10, 10)), _det(0, 0.7, (1, 1, 9, 9))]
    assert len(merge_detections(detections)) == 1


def test_merge_disjoint_boxes():
    detections = [_det(0, 0.9, (0, 0, 10, 10)), _det(0, 0.8, (20, 20, 30, 30))]
    assert len(merge_detections(detections)) == 2
