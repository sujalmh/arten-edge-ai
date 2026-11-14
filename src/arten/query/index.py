"""Build a SQLite evidence index from an Arten run directory.

The index is intentionally boring: three tables, a handful of indexes, and a
meta file that records exactly what is and is not available. That meta file is
what lets the query layer say "not in this index" instead of guessing.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass
class _TrackAcc:
    first_frame: int
    last_frame: int
    frames: int
    first_cx: float
    first_cy: float
    last_cx: float
    last_cy: float
    score_sum: float
    score_n: int
    direction: str | None = None
    speed_px_s: float | None = None
    dwell_s: float | None = None
    colour: str | None = None
    hits: int | None = None

    def update(self, frame_idx: int, box: list[float], score: float, attrs: dict[str, Any]) -> None:
        cx, cy = _centre(box)
        self.last_frame = frame_idx
        self.frames += 1
        self.last_cx, self.last_cy = cx, cy
        self.score_sum += float(score or 0.0)
        self.score_n += 1
        if attrs.get("direction"):
            self.direction = str(attrs["direction"])
        if attrs.get("speed_px_s") is not None:
            self.speed_px_s = float(attrs["speed_px_s"])
        if attrs.get("dwell_s") is not None:
            self.dwell_s = float(attrs["dwell_s"])
        if attrs.get("color_estimate"):
            self.colour = str(attrs["color_estimate"])
        if attrs.get("hits") is not None:
            self.hits = int(attrs["hits"])


def build_index(
    run_dir: str | Path,
    out_path: str | Path | None = None,
    *,
    start_time: str | None = None,
    force: bool = False,
) -> Path:
    """Build ``evidence.db`` from a run directory and return its path."""
    run_dir = Path(run_dir).expanduser().resolve()
    detections_path = run_dir / "detections.jsonl"
    if not detections_path.exists():
        raise FileNotFoundError(f"missing {detections_path}")

    out = Path(out_path).expanduser().resolve() if out_path else run_dir / "evidence.db"
    if out.exists():
        if not force:
            raise FileExistsError(f"{out} already exists; pass --force to rebuild")
        out.unlink()

    summary = _read_json(run_dir / "summary.json") or {}
    run = summary.get("run") or {}
    model = summary.get("model") or {}
    latency = summary.get("latency") or {}

    source_size = run.get("source_size") or [None, None]
    width = float(source_size[0]) if source_size and source_size[0] else None
    height = float(source_size[1]) if source_size and source_size[1] else None
    annotate_every = int(run.get("annotate_every") or 1)

    conn = sqlite3.connect(out)
    try:
        # DELETE mode (the SQLite default) keeps the index a single file so
        # LAN transfer is `evidence.db + evidence.db.meta.json` with no WAL
        # sidecars. WAL would leave -shm/-wal files that are easy to forget.
        conn.execute("PRAGMA journal_mode=DELETE")
        _create_schema(conn)

        frames = 0
        detections = 0
        max_ts = 0.0
        max_x2 = 0.0
        max_y2 = 0.0
        observed_classes: set[str] = set()
        has_colour = False
        has_attributes = False
        tracks: dict[tuple[int, str], _TrackAcc] = {}

        conn.execute("BEGIN")
        with detections_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                frame_idx = int(record.get("frame_idx", frames))
                timestamp_s = float(record.get("timestamp_s", 0.0))
                counts = record.get("counts") or {}
                max_ts = max(max_ts, timestamp_s)
                frames += 1

                conn.execute(
                    "INSERT INTO frames(frame_idx, source_frame_idx, timestamp_s, inference_ms, counts_json, ood) "
                    "VALUES (?, ?, ?, ?, ?, 0)",
                    (
                        frame_idx,
                        record.get("source_frame_idx"),
                        timestamp_s,
                        record.get("inference_ms"),
                        json.dumps(counts, separators=(",", ":")),
                    ),
                )

                for det in record.get("detections") or []:
                    box = det.get("box") or [0.0, 0.0, 0.0, 0.0]
                    if len(box) < 4:
                        box = list(box) + [0.0] * (4 - len(box))
                    x1, y1, x2, y2 = (float(v) for v in box[:4])
                    max_x2 = max(max_x2, x2)
                    max_y2 = max(max_y2, y2)
                    cls = str(det.get("class") or "")
                    observed_classes.add(cls)
                    attrs = det.get("attributes") or {}
                    if attrs.get("color_estimate"):
                        has_colour = True
                    if any(k in attrs for k in ("direction", "speed_px_s", "dwell_s")):
                        has_attributes = True
                    track_id = det.get("track_id")
                    conn.execute(
                        "INSERT INTO detections(frame_idx, timestamp_s, class_id, class, score, "
                        "x1, y1, x2, y2, track_id, direction, speed_px_s, dwell_s, hits, colour, zone) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                        (
                            frame_idx,
                            timestamp_s,
                            det.get("class_id"),
                            cls,
                            float(det.get("score") or 0.0),
                            x1,
                            y1,
                            x2,
                            y2,
                            track_id,
                            attrs.get("direction"),
                            attrs.get("speed_px_s"),
                            attrs.get("dwell_s"),
                            attrs.get("hits"),
                            attrs.get("color_estimate"),
                        ),
                    )
                    detections += 1

                    if track_id is not None:
                        key = (int(track_id), cls)
                        cx, cy = _centre([x1, y1, x2, y2])
                        acc = tracks.get(key)
                        if acc is None:
                            tracks[key] = _TrackAcc(
                                first_frame=frame_idx,
                                last_frame=frame_idx,
                                frames=1,
                                first_cx=cx,
                                first_cy=cy,
                                last_cx=cx,
                                last_cy=cy,
                                score_sum=float(det.get("score") or 0.0),
                                score_n=1,
                                direction=attrs.get("direction"),
                                speed_px_s=attrs.get("speed_px_s"),
                                dwell_s=attrs.get("dwell_s"),
                                colour=attrs.get("color_estimate"),
                                hits=attrs.get("hits"),
                            )
                        else:
                            acc.update(frame_idx, [x1, y1, x2, y2], float(det.get("score") or 0.0), attrs)

        if frames == 0:
            raise ValueError("detections.jsonl contains no frames")

        csv_tracks = _read_track_csv(run_dir / "attribute_tracks.csv")
        _insert_tracks(conn, tracks, csv_tracks, width or max_x2 or 1.0, height or max_y2 or 1.0)
        conn.commit()

        classes = sorted(set(run.get("classes") or []) | observed_classes)
        meta = {
            "schema_version": SCHEMA_VERSION,
            "source_run": str(run_dir),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rows": {
                "frames": frames,
                "detections": detections,
                "tracks": len(tracks),
            },
            "classes": classes,
            "time_range_s": [0.0, round(max_ts, 3)],
            "duration_s": float(run.get("duration_s") or max_ts),
            "source_size": [width, height],
            "annotate_every": annotate_every,
            "has_zone": (run_dir / "zone_counts.csv").exists(),
            "has_colour": has_colour,
            "has_attributes": has_attributes,
            "start_time": start_time or run.get("start_time"),
            "input_sha256": {
                "detections.jsonl": _sha256(detections_path),
                "attribute_tracks.csv": _sha256(run_dir / "attribute_tracks.csv"),
            },
            "model": {
                "name": model.get("name"),
                "sha256": model.get("sha256"),
                "size_mb": model.get("size_mb"),
            },
            "latency": {
                "model_ms": (latency.get("model_inference") or {}).get("mean_ms"),
                "pipeline_ms": (latency.get("pipeline_total") or {}).get("mean_ms"),
                "fps": (summary.get("throughput") or {}).get("pipeline_fps_from_mean"),
            },
            "unsupported": _unsupported_from_meta(
                has_zone=(run_dir / "zone_counts.csv").exists(),
                has_colour=has_colour,
                has_attributes=has_attributes,
                start_time=start_time or run.get("start_time"),
            ),
        }

        _insert_meta(conn, meta)
        conn.commit()

        conn.execute(
            "INSERT INTO runs(run_id, input, model_sha256, created_at, frames, source_fps, "
            "duration_s, pipeline_fps, model_ms, peak_rss_mb) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(run_dir),
                run.get("input"),
                model.get("sha256"),
                meta["created_at"],
                frames,
                run.get("source_fps"),
                meta["duration_s"],
                meta["latency"]["fps"],
                meta["latency"]["model_ms"],
                summary.get("peak_rss_mb"),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    # Sidecar name is derived from the database name so two indexes in the
    # same directory cannot overwrite each other's metadata.
    meta_path = out.with_name(out.name + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            input TEXT,
            model_sha256 TEXT,
            created_at TEXT,
            frames INTEGER,
            source_fps REAL,
            duration_s REAL,
            pipeline_fps REAL,
            model_ms REAL,
            peak_rss_mb REAL
        );
        CREATE TABLE frames (
            frame_idx INTEGER PRIMARY KEY,
            source_frame_idx INTEGER,
            timestamp_s REAL,
            inference_ms REAL,
            counts_json TEXT,
            ood INTEGER DEFAULT 0
        );
        CREATE TABLE detections (
            frame_idx INTEGER,
            timestamp_s REAL,
            class_id INTEGER,
            class TEXT,
            score REAL,
            x1 REAL, y1 REAL, x2 REAL, y2 REAL,
            track_id INTEGER,
            direction TEXT,
            speed_px_s REAL,
            dwell_s REAL,
            hits INTEGER,
            colour TEXT,
            zone TEXT
        );
        CREATE TABLE tracks (
            track_id INTEGER,
            class TEXT,
            first_frame INTEGER,
            last_frame INTEGER,
            frames INTEGER,
            direction TEXT,
            speed_px_s REAL,
            dwell_s REAL,
            hits INTEGER,
            mean_score REAL,
            colour TEXT,
            crossed INTEGER,
            net_dx REAL,
            net_dy REAL,
            PRIMARY KEY (track_id, class)
        );
        CREATE INDEX idx_det_class_frame ON detections(class, frame_idx);
        CREATE INDEX idx_det_track ON detections(track_id);
        CREATE INDEX idx_det_time ON detections(timestamp_s);
        CREATE INDEX idx_tracks_class ON tracks(class);
        """
    )


def _insert_tracks(
    conn: sqlite3.Connection,
    tracks: dict[tuple[int, str], _TrackAcc],
    csv_tracks: dict[tuple[int, str], dict[str, str]],
    width: float,
    height: float,
) -> None:
    rows = []
    keys = set(tracks) | set(csv_tracks)
    for key in keys:
        track_id, cls = key
        acc = tracks.get(key)
        csv_row = csv_tracks.get(key, {})

        first_frame = acc.first_frame if acc else None
        last_frame = acc.last_frame if acc else None
        frames = acc.frames if acc else 0
        mean_score = (acc.score_sum / acc.score_n) if acc and acc.score_n else None
        net_dx = (acc.last_cx - acc.first_cx) if acc else None
        net_dy = (acc.last_cy - acc.first_cy) if acc else None

        direction = csv_row.get("direction") or (acc.direction if acc else None)
        speed = _float_or_none(csv_row.get("speed_px_s"))
        if speed is None and acc:
            speed = acc.speed_px_s
        dwell = _float_or_none(csv_row.get("dwell_s"))
        if dwell is None and acc:
            dwell = acc.dwell_s
        colour = csv_row.get("color") or (acc.colour if acc else None)
        hits = _int_or_none(csv_row.get("hits"))
        if hits is None and acc:
            hits = acc.hits
        score = _float_or_none(csv_row.get("score"))
        if score is None:
            score = mean_score

        crossed = 0
        if net_dx is not None and net_dy is not None and frames >= 3:
            if abs(net_dx) >= 0.25 * width or abs(net_dy) >= 0.25 * height:
                crossed = 1

        rows.append(
            (
                track_id,
                cls,
                first_frame,
                last_frame,
                frames,
                direction,
                speed,
                dwell,
                hits,
                score,
                colour,
                crossed,
                net_dx,
                net_dy,
            )
        )

    conn.executemany(
        "INSERT INTO tracks(track_id, class, first_frame, last_frame, frames, direction, "
        "speed_px_s, dwell_s, hits, mean_score, colour, crossed, net_dx, net_dy) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def _read_track_csv(path: Path) -> dict[tuple[int, str], dict[str, str]]:
    if not path.exists():
        return {}
    out: dict[tuple[int, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                track_id = int(row.get("track_id"))
            except (TypeError, ValueError):
                continue
            cls = str(row.get("class") or "")
            out[(track_id, cls)] = dict(row)
    return out


def _insert_meta(conn: sqlite3.Connection, meta: dict[str, Any]) -> None:
    rows = []
    for key, value in meta.items():
        rows.append((key, json.dumps(value, ensure_ascii=False)))
    rows.append(("schema_version", json.dumps(SCHEMA_VERSION)))
    conn.executemany("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", rows)


def _unsupported_from_meta(
    *,
    has_zone: bool,
    has_colour: bool,
    has_attributes: bool,
    start_time: str | None,
) -> list[str]:
    out = []
    if not has_zone:
        out.append("zone filtering (run with --zone to add it)")
    if not has_colour:
        out.append("colour filtering (run with --estimate-vehicle-colors to add it)")
    if not has_attributes:
        out.append("motion attributes (run without --no-attributes to add them)")
    if not start_time:
        out.append("time-of-day words such as 'morning' (re-index with --start-time to add them)")
    return out


def _centre(box: list[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value in (None, "", "None") else float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return None if value in (None, "", "None") else int(float(value))
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
