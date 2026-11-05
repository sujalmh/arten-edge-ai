"""Decode YOLO output, filter categories, run NMS, map back to pixels.

Deliberately implemented in NumPy rather than torchvision so the Raspberry Pi
install stays small and the decode logic is easy to audit. The model output is
the standard YOLO11 layout: [1, 84, 8400] = 4 xywh + 80 class scores, with no
separate objectness score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import CLASS_TO_ID, COCO_CLASSES
from .preprocess import LetterboxMeta, unletterbox_boxes


@dataclass
class Detection:
    class_id: int
    class_name: str
    score: float
    box: tuple[float, float, float, float]
    track_id: int | None = None
    attributes: dict[str, str] = field(default_factory=dict)

    def to_dict(self, ndigits: int = 2) -> dict:
        return {
            "class_id": int(self.class_id),
            "class": self.class_name,
            "score": round(float(self.score), 4),
            "box": [round(float(v), ndigits) for v in self.box],
            **({"attributes": dict(self.attributes)} if self.attributes else {}),
            **({"track_id": self.track_id} if self.track_id is not None else {}),
        }


@dataclass
class DecodeStats:
    candidates: int = 0
    after_confidence: int = 0
    after_nms: int = 0
    discarded_small: int = 0
    extra: dict = field(default_factory=dict)


def _as_nx84(raw: np.ndarray) -> tuple[np.ndarray, bool]:
    """Normalise raw prediction shapes to (N, C) and report objectness presence."""

    arr = np.asarray(raw)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"expected batch size 1, got {arr.shape}")
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D prediction array, got {arr.shape}")

    # YOLO11: (84, N) -> (N, 84). YOLOv5: (85, N) -> (N, 85).
    if arr.shape[0] in (84, 85) and arr.shape[1] > arr.shape[0]:
        arr = arr.T
    # Already (N, 84/85). Some exports emit (1, 84, 8400); handled above.
    if arr.shape[1] not in (84, 85):
        # Last-resort transpose for unusual exports.
        if arr.shape[0] in (84, 85):
            arr = arr.T
        else:
            raise ValueError(f"unsupported prediction shape after squeeze: {arr.shape}")

    has_objectness = arr.shape[1] == 85
    return np.ascontiguousarray(arr, dtype=np.float32), has_objectness


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression. Returns indices of kept boxes."""

    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)


def decode(
    raw: np.ndarray,
    meta: LetterboxMeta,
    classes: list[str] | tuple[str, ...] | None = None,
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.7,
    max_detections: int = 300,
    max_candidates: int = 30_000,
    min_box_px: float = 2.0,
) -> tuple[list[Detection], DecodeStats]:
    """Decode raw model output into detections in original image coordinates.

    Args:
        raw: model output, accepted as (1, 84, N), (84, N), (N, 84), or YOLOv5
            style (N, 85) with objectness.
        meta: letterbox transform used for this frame.
        classes: category names to keep; None means all 80 COCO classes.
        conf_threshold: minimum score after class/objectness multiplication.
        iou_threshold: NMS IoU threshold.
        max_detections: cap after NMS.
        max_candidates: cap before NMS (highest scores first).
        min_box_px: drop boxes thinner than this after rescaling.
    """

    pred, has_objectness = _as_nx84(raw)
    stats = DecodeStats(candidates=int(pred.shape[0]))

    n_classes = pred.shape[1] - 5 if has_objectness else pred.shape[1] - 4
    if n_classes <= 0:
        raise ValueError(f"prediction has no class columns: {pred.shape}")

    selected = list(range(n_classes)) if classes is None else [
        CLASS_TO_ID[name] for name in classes if name in CLASS_TO_ID
    ]
    if not selected:
        raise ValueError(f"none of the requested classes exist: {classes}")

    boxes_xywh = pred[:, :4]
    if has_objectness:
        class_scores = pred[:, 5 : 5 + n_classes] * pred[:, 4:5]
    else:
        class_scores = pred[:, 4 : 4 + n_classes]

    # Restrict to requested classes, then take the best class per row.
    selected_scores = class_scores[:, selected]
    best_local = selected_scores.argmax(axis=1)
    best_scores = selected_scores[np.arange(pred.shape[0]), best_local]
    keep = best_scores >= conf_threshold
    stats.after_confidence = int(keep.sum())
    if not keep.any():
        return [], stats

    boxes_xywh = boxes_xywh[keep]
    best_scores = best_scores[keep]
    best_local = best_local[keep]
    class_ids = np.asarray(selected, dtype=np.int64)[best_local]

    # xywh (centre) -> xyxy.
    cx, cy, w, h = boxes_xywh.T
    boxes = np.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), axis=1)

    if boxes.shape[0] > max_candidates:
        top = np.argsort(best_scores)[::-1][:max_candidates]
        boxes, best_scores, class_ids = boxes[top], best_scores[top], class_ids[top]

    # Per-class NMS. Keeps a car and a person that overlap, suppresses duplicates.
    kept_parts: list[np.ndarray] = []
    for class_id in np.unique(class_ids):
        idx = np.flatnonzero(class_ids == class_id)
        local_keep = nms(boxes[idx], best_scores[idx], iou_threshold)
        kept_parts.append(idx[local_keep])
    if kept_parts:
        kept = np.concatenate(kept_parts)
    else:
        kept = np.empty((0,), dtype=np.int64)

    if kept.size:
        order = np.argsort(best_scores[kept])[::-1][:max_detections]
        kept = kept[order]
    boxes, best_scores, class_ids = boxes[kept], best_scores[kept], class_ids[kept]

    boxes = unletterbox_boxes(boxes, meta)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    valid = (widths >= min_box_px) & (heights >= min_box_px)
    stats.discarded_small = int((~valid).sum())
    boxes, best_scores, class_ids = boxes[valid], best_scores[valid], class_ids[valid]

    detections = [
        Detection(
            class_id=int(cid),
            class_name=COCO_CLASSES[int(cid)],
            score=float(score),
            box=tuple(float(v) for v in box),
        )
        for box, score, cid in zip(boxes, best_scores, class_ids)
    ]
    detections.sort(key=lambda d: d.score, reverse=True)
    stats.after_nms = len(detections)
    return detections, stats


def count_by_class(
    detections: list[Detection],
    classes: list[str] | tuple[str, ...] | None = None,
) -> dict[str, int]:
    """Count detections per requested class, including zeros."""

    names = list(classes) if classes is not None else list(COCO_CLASSES)
    counts = {name: 0 for name in names}
    for det in detections:
        if det.class_name in counts:
            counts[det.class_name] += 1
    return counts
