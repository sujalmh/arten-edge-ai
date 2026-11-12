"""Tiled inference for dense or high-resolution scenes.

A single 640x640 pass shrinks a 1040x780 frame to 0.62x, which loses small
pedestrians and two-wheelers. Tiling runs the same model on native-resolution
640x640 crops and merges the boxes back into image coordinates. No retraining
is involved; the cost is one model pass per tile.
"""

from __future__ import annotations

from .constants import INPUT_SIZE
from .postprocess import Detection


def tile_origins(
    width: int,
    height: int,
    grid: int,
    tile_size: int | None = None,
) -> list[tuple[int, int]]:
    """Return crop origins for an NxN overlapping grid.

    ``grid <= 1`` returns a single full-frame origin. Images smaller than the
    tile size also return one origin so the normal letterbox handles padding.
    """

    if grid <= 1:
        return [(0, 0)]

    tile = tile_size or INPUT_SIZE[0]

    def axis(length: int) -> list[int]:
        if length <= tile:
            return [0]
        step = (length - tile) / (grid - 1)
        return sorted({int(round(index * step)) for index in range(grid)})

    return [(x, y) for y in axis(height) for x in axis(width)]


def _iou_and_containment(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float]:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0:
        return 0.0, 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    iou = intersection / union if union > 0 else 0.0
    contained = intersection / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0
    return iou, contained


def merge_detections(
    detections: list[Detection],
    iou_threshold: float = 0.5,
    containment_threshold: float = 0.8,
) -> list[Detection]:
    """Merge per-tile detections, class by class.

    A lower score box is dropped when it overlaps a higher score box of the
    same class by more than ``iou_threshold`` or when most of it is contained
    in the other box. Different classes are never merged.
    """

    kept: list[Detection] = []
    for class_id in sorted({detection.class_id for detection in detections}):
        group = sorted(
            (d for d in detections if d.class_id == class_id),
            key=lambda d: d.score,
            reverse=True,
        )
        kept_class: list[Detection] = []
        for detection in group:
            duplicate = False
            for other in kept_class:
                iou, contained = _iou_and_containment(detection.box, other.box)
                if iou > iou_threshold or contained > containment_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept_class.append(detection)
        kept.extend(kept_class)
    return kept
