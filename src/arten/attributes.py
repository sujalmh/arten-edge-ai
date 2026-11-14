"""Lightweight, non-learned appearance attributes for detected objects."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .postprocess import Detection


def estimate_vehicle_color(frame_bgr: np.ndarray, box: tuple[float, float, float, float]) -> str:
    """Estimate a coarse vehicle color from the center of its detection crop.

    This is a classical HSV heuristic, not a trained attribute model. It is
    intentionally conservative for tiny/washed-out crops.
    """
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    # Trim edges to reduce road/background pixels; vehicle bodies tend to occupy
    # the middle portion of a detection, though this remains viewpoint-sensitive.
    left = max(0, int(x1 + (x2 - x1) * 0.15))
    right = min(width, int(x2 - (x2 - x1) * 0.15))
    top = max(0, int(y1 + (y2 - y1) * 0.40))
    bottom = min(height, int(y2 - (y2 - y1) * 0.08))
    crop = frame_bgr[top:bottom, left:right]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return "unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)  # OpenCV hue range: 0..179.
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    value = hsv[:, :, 2].astype(np.float32) / 255.0

    neutral = saturation < 0.25
    # Do not let large neutral windows, reflections, or road background hide a
    # clearly dominant paint hue; only call achromatic when neutral pixels win.
    if float(neutral.mean()) >= 0.62:
        # Use brighter neutral pixels: shadows and glass otherwise drag white
        # paint down into the gray/silver bucket.
        neutral_v = value[neutral]
        bright_neutral = float(np.percentile(neutral_v, 70))
        if bright_neutral < 0.24:
            return "black"
        if bright_neutral > 0.72:
            return "white"
        return "gray/silver"

    chromatic = (saturation >= 0.23) & (value >= 0.16)
    if not chromatic.any():
        return "unknown"

    # Broad sectors with the red hue wrap handled explicitly. These labels are
    # still affected by glass, reflections, lighting, and background pixels.
    hues = hue[chromatic]
    weights = saturation[chromatic] * value[chromatic]
    masks = (
        (hues < 10) | (hues >= 168),
        (hues >= 10) & (hues < 25),
        (hues >= 25) & (hues < 38),
        (hues >= 38) & (hues < 85),
        (hues >= 85) & (hues < 125),
        (hues >= 125) & (hues < 168),
    )
    labels = ("red", "orange/brown", "yellow", "green", "blue", "purple")
    totals = [float(weights[mask].sum()) for mask in masks]
    return labels[int(np.argmax(totals))]


@dataclass
class AttributeAccumulator:
    """Collect per-track attributes without double-counting frames.

    Motion attributes come from the lightweight tracker; colour is the optional
    classical-CV estimate. Rows are keyed by track id for video, and by frame
    index for independent images.
    """

    tracks: dict[str, dict] = field(default_factory=dict)

    def update(self, detections: list[Detection], frame_idx: int) -> None:
        for index, detection in enumerate(detections):
            if not detection.attributes:
                continue
            if detection.track_id is not None:
                key = f"track:{detection.track_id}"
            else:
                key = f"frame:{frame_idx}:{index}"
            row = self.tracks.setdefault(
                key,
                {
                    "class": detection.class_name,
                    "track_id": detection.track_id,
                    "color": None,
                    "direction": None,
                    "speed_px_s": None,
                    "dwell_s": None,
                    "hits": 0,
                    "score": 0.0,
                    "surface": None,
                    "posture": None,
                    "rider": None,
                },
            )
            row["class"] = detection.class_name
            row["track_id"] = detection.track_id
            row["score"] = max(float(row["score"]), float(detection.score))
            for attr in (
                "color_estimate", "direction", "speed_px_s", "dwell_s", "hits",
                "surface", "posture", "rider",
            ):
                if attr in detection.attributes:
                    row["color" if attr == "color_estimate" else attr] = detection.attributes[attr]

    def summary(self) -> dict | None:
        if not self.tracks:
            return None
        grouped: dict[str, dict] = {}
        for row in self.tracks.values():
            entry = grouped.setdefault(
                row["class"],
                {
                    "tracks": 0,
                    "directions": {},
                    "colors": {},
                    "surfaces": {},
                    "postures": {},
                    "riders": {},
                    "dwell_s": [],
                    "speed_px_s": [],
                },
            )
            entry["tracks"] += 1
            direction = row.get("direction")
            if direction:
                entry["directions"][direction] = entry["directions"].get(direction, 0) + 1
            color = row.get("color")
            if color and color != "unknown":
                entry["colors"][color] = entry["colors"].get(color, 0) + 1
            surface = row.get("surface")
            if surface and surface != "unknown":
                entry["surfaces"][surface] = entry["surfaces"].get(surface, 0) + 1
            posture = row.get("posture")
            if posture and posture != "unknown":
                entry["postures"][posture] = entry["postures"].get(posture, 0) + 1
            rider = row.get("rider")
            if rider:
                entry["riders"][rider] = entry["riders"].get(rider, 0) + 1
            if row.get("dwell_s") is not None:
                entry["dwell_s"].append(float(row["dwell_s"]))
            if row.get("speed_px_s") is not None:
                entry["speed_px_s"].append(float(row["speed_px_s"]))

        by_class: dict[str, dict] = {}
        for class_name, entry in grouped.items():
            dwell = entry["dwell_s"]
            speed = entry["speed_px_s"]
            by_class[class_name] = {
                "tracks": entry["tracks"],
                "directions": dict(sorted(entry["directions"].items())),
                "colors": dict(sorted(entry["colors"].items())),
                "surfaces": dict(sorted(entry["surfaces"].items())),
                "postures": dict(sorted(entry["postures"].items())),
                "riders": dict(sorted(entry["riders"].items())),
                "mean_dwell_s": round(sum(dwell) / len(dwell), 2) if dwell else None,
                "mean_speed_px_s": round(sum(speed) / len(speed), 2) if speed else None,
            }
        return {
            "note": (
                "Image-plane motion, coarse colour, surface and posture are derived "
                "attributes, not trained model outputs. Speed is uncalibrated "
                "pixels/second; surface and posture are sampled on a cadence."
            ),
            "by_class": by_class,
            "total_tracks": len(self.tracks),
        }

    def rows(self) -> list[dict]:
        return sorted(
            self.tracks.values(),
            key=lambda row: (row["class"], -1 if row["track_id"] is None else row["track_id"]),
        )
