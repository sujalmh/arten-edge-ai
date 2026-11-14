"""Support models: top-down pose and semantic segmentation.

These are optional detail models around the detector:

* Lite-HRNet18 (Apache-2.0) takes a person crop from the detector and returns
  17 COCO keypoints. Used for posture and a rider/fall hint.
* DDRNet23-slim (MIT) returns an 11-class road-scene mask. Used for the surface
  under each detection (road, sidewalk, other) instead of a hand-drawn polygon.

Both models are optional and run on a cadence; the detector remains the
per-frame path.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

POSE_INPUT = (384, 288)  # height, width
POSE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
POSE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
POSE_SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)
KEYPOINT_CONFIDENCE = 0.2

SEG_INPUT = (720, 960)  # height, width
SEG_CLASSES = (
    "sky", "building", "pole", "road", "sidewalk", "tree", "sign", "fence",
    "car", "pedestrian", "bicyclist",
)
SEG_ROAD = 3
SEG_SIDEWALK = 4


def _load_method(model_path: str | Path):
    import torch
    from executorch.runtime import Runtime

    program = Runtime.get().load_program(str(model_path))
    return torch, program.load_method("forward")


class PoseEstimator:
    """Top-down COCO pose for one detector box."""

    def __init__(self, model_path: str | Path) -> None:
        self._torch, self._method = _load_method(model_path)
        self._method.execute([self._torch.zeros((1, 3, *POSE_INPUT))])

    def infer(self, frame_bgr: np.ndarray, box: tuple[float, float, float, float]):
        x1, y1, x2, y2 = (float(value) for value in box)
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        width, height = x2 - x1, y2 - y1
        aspect = POSE_INPUT[1] / POSE_INPUT[0]
        if width > aspect * height:
            height = width / aspect
        else:
            width = height * aspect
        width *= 1.25
        height *= 1.25
        scale_x = POSE_INPUT[1] / width
        scale_y = POSE_INPUT[0] / height
        matrix = np.array(
            [
                [scale_x, 0.0, POSE_INPUT[1] / 2.0 - scale_x * center_x],
                [0.0, scale_y, POSE_INPUT[0] / 2.0 - scale_y * center_y],
            ],
            dtype=np.float32,
        )
        crop = cv2.warpAffine(frame_bgr, matrix, (POSE_INPUT[1], POSE_INPUT[0]))
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - POSE_MEAN) / POSE_STD
        tensor = self._torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0)

        started = time.perf_counter()
        heatmaps = self._method.execute([tensor])[0]
        latency_ms = (time.perf_counter() - started) * 1000.0

        array = heatmaps.detach().cpu().numpy()[0] if hasattr(heatmaps, "detach") else np.asarray(heatmaps)[0]
        inverse = cv2.invertAffineTransform(matrix)
        points: list[tuple[float, float, float]] = []
        for channel in range(array.shape[0]):
            heatmap = array[channel]
            index = int(np.argmax(heatmap))
            y, x = divmod(index, heatmap.shape[1])
            dx = 0.25 * np.sign(heatmap[y, min(x + 1, 71)] - heatmap[y, max(x - 1, 0)]) if 0 < x < 71 else 0.0
            dy = 0.25 * np.sign(heatmap[min(y + 1, 95), x] - heatmap[max(y - 1, 0), x]) if 0 < y < 95 else 0.0
            px, py = (x + dx) * 4.0, (y + dy) * 4.0
            original = inverse @ np.array([px, py, 1.0])
            points.append((float(original[0]), float(original[1]), float(heatmap[y, x])))
        return points, latency_ms


def classify_posture(
    keypoints: list[tuple[float, float, float]],
    box: tuple[float, float, float, float],
    rider: bool = False,
) -> str:
    """Heuristic posture from COCO keypoints. Indicative, not a fall detector."""

    if rider:
        return "riding"

    def point(index: int):
        x, y, confidence = keypoints[index]
        return (x, y) if confidence >= KEYPOINT_CONFIDENCE else None

    left_shoulder, right_shoulder = point(5), point(6)
    left_hip, right_hip = point(11), point(12)
    if not all((left_shoulder, right_shoulder, left_hip, right_hip)):
        return "unknown"

    height = max(1.0, float(box[3] - box[1]))
    shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
    hip_x = (left_hip[0] + right_hip[0]) / 2.0
    hip_y = (left_hip[1] + right_hip[1]) / 2.0
    horizontal = abs(shoulder_x - hip_x)
    vertical = abs(shoulder_y - hip_y)
    if horizontal > 0.6 * height and vertical < 0.3 * height:
        return "fallen"

    left_ankle, right_ankle = point(15), point(16)
    if left_ankle and right_ankle and abs(left_ankle[0] - right_ankle[0]) > 0.25 * height:
        return "walking"
    return "standing"


class Segmenter:
    """11-class road-scene mask for surface attribution."""

    def __init__(self, model_path: str | Path) -> None:
        self._torch, self._method = _load_method(model_path)
        self._method.execute([self._torch.zeros((1, 3, *SEG_INPUT))])

    def infer(self, frame_bgr: np.ndarray):
        resized = cv2.resize(frame_bgr, (SEG_INPUT[1], SEG_INPUT[0]))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) - 128.0
        tensor = self._torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0)

        started = time.perf_counter()
        logits = self._method.execute([tensor])[0]
        latency_ms = (time.perf_counter() - started) * 1000.0

        upsampled = self._torch.nn.functional.interpolate(
            logits, size=SEG_INPUT, mode="bilinear", align_corners=False
        )
        mask = upsampled.argmax(1)[0].detach().cpu().numpy().astype(np.uint8)
        full = cv2.resize(
            mask, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        return full, latency_ms

    @staticmethod
    def surface_at(
        mask: np.ndarray | None,
        box: tuple[float, float, float, float],
    ) -> str:
        if mask is None:
            return "unknown"
        x1, y1, x2, y2 = box
        x = int(np.clip((x1 + x2) / 2.0, 0, mask.shape[1] - 1))
        y = int(np.clip(y2 - 0.02 * max(1.0, y2 - y1), 0, mask.shape[0] - 1))
        patch = mask[max(0, y - 8) : y + 9, max(0, x - 8) : x + 9]
        values, counts = np.unique(patch, return_counts=True)
        shares = {int(value): int(count) / patch.size for value, count in zip(values, counts)}
        # Prefer road/sidewalk when either occupies a meaningful share of the
        # patch; a vehicle body or a rider can occlude the exact foot pixel.
        for class_id, name in ((SEG_ROAD, "road"), (SEG_SIDEWALK, "sidewalk")):
            if shares.get(class_id, 0.0) >= 0.15:
                return name
        class_id = int(values[int(np.argmax(counts))])
        if class_id == SEG_ROAD:
            return "road"
        if class_id == SEG_SIDEWALK:
            return "sidewalk"
        return "other"


def is_rider(
    person_box: tuple[float, float, float, float],
    motorcycle_boxes: list[tuple[float, float, float, float]],
    overlap_threshold: float = 0.3,
) -> bool:
    """A person whose box is mostly inside a motorcycle box is a rider."""

    px1, py1, px2, py2 = person_box
    person_area = max(1.0, (px2 - px1) * (py2 - py1))
    for bx1, by1, bx2, by2 in motorcycle_boxes:
        ix1, iy1 = max(px1, bx1), max(py1, by1)
        ix2, iy2 = min(px2, bx2), min(py2, by2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if intersection / person_area > overlap_threshold:
            return True
    return False
