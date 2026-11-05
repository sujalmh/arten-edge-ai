"""A tiny IoU/velocity tracker for indicative unique counts.

Tracking is *not* required by the hackathon brief, and this is not a
production MOT system. It exists so a 15-second clip can answer "how many
distinct pedestrians/vehicles were seen?" in addition to the mandatory
per-frame counts. The README and report label these numbers as indicative.

Design: greedy one-to-one matching per class, constant-velocity prediction,
a short coast window for occlusions, and confirmation after N hits to suppress
one-frame false positives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .postprocess import Detection


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    areas_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = area_a + areas_b - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _centre(box: np.ndarray) -> np.ndarray:
    return np.array(
        [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32
    )


@dataclass
class Track:
    track_id: int
    class_id: int
    class_name: str
    box: np.ndarray
    score: float
    velocity: np.ndarray = field(
        default_factory=lambda: np.zeros((4,), dtype=np.float32)
    )
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    first_frame: int = 0
    last_frame: int = 0

    @property
    def area(self) -> float:
        return float(max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1]))


class IoUTracker:
    """Greedy per-class tracker with a constant-velocity motion model."""

    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_misses: int = 5,
        min_hits: int = 2,
        velocity_smoothing: float = 0.5,
        distance_fallback: float = 0.35,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.velocity_smoothing = velocity_smoothing
        self.distance_fallback = distance_fallback
        self.tracks: list[Track] = []
        self._next_id = 1
        self.unique_counts: dict[str, int] = {}
        self.seen_track_ids: set[int] = set()

    def _new_track(self, det: Detection, frame_idx: int) -> Track:
        track = Track(
            track_id=self._next_id,
            class_id=det.class_id,
            class_name=det.class_name,
            box=np.asarray(det.box, dtype=np.float32),
            score=det.score,
            first_frame=frame_idx,
            last_frame=frame_idx,
        )
        self._next_id += 1
        self.tracks.append(track)
        return track

    def _confirm(self, track: Track) -> None:
        if track.confirmed:
            return
        track.confirmed = True
        if track.track_id not in self.seen_track_ids:
            self.seen_track_ids.add(track.track_id)
            self.unique_counts[track.class_name] = (
                self.unique_counts.get(track.class_name, 0) + 1
            )

    def update(self, detections: list[Detection], frame_idx: int) -> list[Detection]:
        """Update tracks and attach track IDs to detections in place."""

        # Predict where each existing track should be this frame.
        predicted: dict[int, np.ndarray] = {}
        for track in self.tracks:
            predicted[track.track_id] = np.clip(
                track.box + track.velocity, 0.0, 1e9
            ).astype(np.float32)

        matched_track_ids: set[int] = set()
        matched_det_ids: set[int] = set()

        # Pass 1: greedy IoU matching, independently per class.
        for class_name in sorted({d.class_name for d in detections}):
            track_indices = [
                i
                for i, t in enumerate(self.tracks)
                if t.class_name == class_name and t.track_id not in matched_track_ids
            ]
            det_indices = [
                j
                for j, d in enumerate(detections)
                if d.class_name == class_name and j not in matched_det_ids
            ]
            if not track_indices or not det_indices:
                continue

            pairs: list[tuple[float, int, int]] = []
            for ti in track_indices:
                track = self.tracks[ti]
                for dj in det_indices:
                    iou = float(
                        _iou_one_to_many(
                            predicted[track.track_id],
                            np.asarray([detections[dj].box], dtype=np.float32),
                        )[0]
                    )
                    if iou >= self.iou_threshold:
                        pairs.append((iou, ti, dj))

            for _, ti, dj in sorted(pairs, reverse=True):
                track = self.tracks[ti]
                if track.track_id in matched_track_ids or dj in matched_det_ids:
                    continue
                self._update_track(track, detections[dj], frame_idx)
                matched_track_ids.add(track.track_id)
                matched_det_ids.add(dj)

        # Pass 2: centre-distance fallback for fast motion / low overlap.
        # Restricted to same class and a generous size-ratio check to avoid
        # swapping identities when two people cross.
        for ti, track in enumerate(self.tracks):
            if track.track_id in matched_track_ids or track.misses > 1:
                continue
            best: tuple[float, int] | None = None
            for dj, det in enumerate(detections):
                if dj in matched_det_ids or det.class_name != track.class_name:
                    continue
                det_box = np.asarray(det.box, dtype=np.float32)
                det_area = max(
                    0.0, det_box[2] - det_box[0]
                ) * max(0.0, det_box[3] - det_box[1])
                if track.area <= 0 or det_area <= 0:
                    continue
                ratio = det_area / track.area
                if not (0.4 <= ratio <= 2.5):
                    continue
                dist = float(
                    np.linalg.norm(_centre(predicted[track.track_id]) - _centre(det_box))
                )
                diag = float(
                    np.hypot(
                        predicted[track.track_id][2] - predicted[track.track_id][0],
                        predicted[track.track_id][3] - predicted[track.track_id][1],
                    )
                )
                if diag <= 0:
                    continue
                normalised = dist / diag
                if normalised <= self.distance_fallback:
                    if best is None or normalised < best[0]:
                        best = (normalised, dj)
            if best is not None:
                dj = best[1]
                self._update_track(track, detections[dj], frame_idx)
                matched_track_ids.add(track.track_id)
                matched_det_ids.add(dj)

        # Coast unmatched tracks, then retire them.
        for track in list(self.tracks):
            if track.track_id in matched_track_ids:
                continue
            track.misses += 1
            track.box = predicted[track.track_id]
            track.last_frame = frame_idx
            if track.misses > self.max_misses:
                self.tracks.remove(track)

        # Births for unmatched detections.
        for dj, det in enumerate(detections):
            if dj in matched_det_ids:
                continue
            track = self._new_track(det, frame_idx)
            if track.hits >= self.min_hits:
                self._confirm(track)
            det.track_id = track.track_id

        # Track IDs are attached by _update_track for matches and by _new_track
        # for births, so detections are ready to serialise here.
        return detections

    def _update_track(self, track: Track, det: Detection, frame_idx: int) -> None:
        new_box = np.asarray(det.box, dtype=np.float32)
        delta = new_box - track.box
        track.velocity = (
            (1.0 - self.velocity_smoothing) * track.velocity
            + self.velocity_smoothing * delta
        ).astype(np.float32)
        track.box = new_box
        track.score = det.score
        track.hits += 1
        track.misses = 0
        track.last_frame = frame_idx
        det.track_id = track.track_id
        if track.hits >= self.min_hits:
            self._confirm(track)

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1
        self.unique_counts.clear()
        self.seen_track_ids.clear()

    def unique_counts_for(self, classes: list[str] | tuple[str, ...]) -> dict[str, int]:
        return {name: int(self.unique_counts.get(name, 0)) for name in classes}

    def annotate_motion_attributes(
        self, detections: list[Detection], source_fps: float, stride: int = 1
    ) -> list[Detection]:
        """Attach image-plane direction, relative speed and dwell time.

        These are derived from the tracker's smoothed bbox velocity, not from a
        calibrated camera. Speed is pixels/second and is useful for comparing
        motion within a clip, not for km/h claims.
        """

        tracks_by_id = {track.track_id: track for track in self.tracks}
        seconds_per_processed_frame = stride / source_fps if source_fps > 0 else 0.0
        for detection in detections:
            if detection.track_id is None:
                continue
            track = tracks_by_id.get(detection.track_id)
            if track is None or not track.confirmed:
                continue
            vx, vy = float(track.velocity[0]), float(track.velocity[1])
            speed_px_frame = float(np.hypot(vx, vy))
            if speed_px_frame < 0.75:
                direction = "stationary"
            elif abs(vx) >= abs(vy):
                direction = "right" if vx > 0 else "left"
            else:
                direction = "down" if vy > 0 else "up"
            speed_px_s = (
                speed_px_frame / seconds_per_processed_frame
                if seconds_per_processed_frame > 0
                else 0.0
            )
            dwell_s = max(0, track.last_frame - track.first_frame) * seconds_per_processed_frame
            detection.attributes.update(
                {
                    "direction": direction,
                    "speed_px_s": round(speed_px_s, 1),
                    "dwell_s": round(dwell_s, 2),
                    "hits": track.hits,
                }
            )
        return detections
