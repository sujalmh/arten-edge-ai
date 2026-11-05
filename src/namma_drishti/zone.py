"""Configurable, privacy-preserving zone exposure metrics.

This module reports where people and vehicles are simultaneously observed. It
does not estimate collision risk, speed, distance, or near misses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .constants import VEHICLE_CLASSES
from .postprocess import Detection


@dataclass(frozen=True)
class ZoneProfile:
    name: str
    points: tuple[tuple[float, float], ...]


def load_zone_profile(path: str | Path) -> ZoneProfile:
    """Load a polygon whose vertices are normalized to [0, 1] image space."""

    config_path = Path(path).expanduser()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    name = payload.get("name")
    raw_points = payload.get("points")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("zone config needs a non-empty 'name'")
    if not isinstance(raw_points, list) or len(raw_points) < 3:
        raise ValueError("zone config 'points' must contain at least 3 [x, y] vertices")

    points: list[tuple[float, float]] = []
    for point in raw_points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("each zone point must be an [x, y] pair")
        x, y = float(point[0]), float(point[1])
        if not np.isfinite(x) or not np.isfinite(y) or not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("zone coordinates must be finite values from 0 to 1")
        points.append((x, y))

    contour = np.asarray(points, dtype=np.float32)
    if abs(float(cv2.contourArea(contour))) < 1e-6:
        raise ValueError("zone polygon must have non-zero area")
    return ZoneProfile(name=name.strip(), points=tuple(points))


def pixel_points(profile: ZoneProfile, width: int, height: int) -> np.ndarray:
    return np.asarray(
        [[round(x * (width - 1)), round(y * (height - 1))] for x, y in profile.points],
        dtype=np.int32,
    )


def count_detections_in_zone(
    detections: list[Detection], profile: ZoneProfile, width: int, height: int
) -> dict[str, int]:
    """Count detections whose bottom-center point falls inside the zone."""

    contour = pixel_points(profile, width, height).reshape((-1, 1, 2))
    counts = {"person": 0, "vehicle": 0}
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        anchor = ((x1 + x2) / 2.0, y2)
        if cv2.pointPolygonTest(contour, anchor, False) < 0:
            continue
        if detection.class_name == "person":
            counts["person"] += 1
        elif detection.class_name in VEHICLE_CLASSES:
            counts["vehicle"] += 1
    return counts


def summarise_zone(
    rows: list[dict], *, source_fps: float, stride: int, is_video: bool
) -> dict:
    """Summarize per-sample counts without turning co-presence into a risk score."""

    interval = stride / source_fps if is_video and source_fps > 0 else None
    people = [int(row["person"]) for row in rows]
    vehicles = [int(row["vehicle"]) for row in rows]
    both = [p > 0 and v > 0 for p, v in zip(people, vehicles)]

    longest_run = current_run = 0
    previous_index: int | None = None
    for row, present in zip(rows, both):
        source_index = int(row["source_frame_idx"])
        if present:
            if previous_index is None or source_index - previous_index == stride:
                current_run += 1
            else:
                current_run = 1
            longest_run = max(longest_run, current_run)
            previous_index = source_index
        else:
            current_run = 0
            previous_index = None

    frames = len(rows)
    both_frames = sum(both)
    summary = {
        "frames": frames,
        "peak_people_per_frame": max(people, default=0),
        "peak_vehicles_per_frame": max(vehicles, default=0),
        "frames_with_people_and_vehicles": both_frames,
        "share_frames_with_people_and_vehicles": round(both_frames / frames, 3) if frames else 0.0,
        "interpretation": "descriptive co-presence in a manually selected image zone; not a conflict or crash-risk measure",
    }
    if interval is not None:
        summary["person_sample_seconds"] = round(sum(people) * interval, 2)
        summary["vehicle_sample_seconds"] = round(sum(vehicles) * interval, 2)
        summary["co_presence_sample_seconds"] = round(both_frames * interval, 2)
        summary["longest_co_presence_sample_seconds"] = round(longest_run * interval, 2)
    return summary
