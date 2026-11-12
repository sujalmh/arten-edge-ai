"""Video/image processing pipeline: infer, count, annotate, write evidence."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .constants import (
    DEFAULT_CLASSES,
    DEFAULT_MODEL_PATH,
    INPUT_SIZE,
    PROJECT_NAME,
    PROJECT_TAGLINE,
    VEHICLE_CLASSES,
)
from .metrics import (
    environment_info,
    frames_with_both,
    peak_rss_mb,
    person_seconds,
    summarise_latencies,
    write_json,
)
from .attributes import AttributeAccumulator, estimate_vehicle_color
from .model import ExecuTorchDetector
from .postprocess import count_by_class, decode
from .preprocess import letterbox, read_image
from .tiling import merge_detections, tile_origins
from .tracker import IoUTracker
from .visualize import annotate_frame, make_contact_sheet
from .zone import ZoneProfile, count_detections_in_zone, load_zone_profile, summarise_zone

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class FrameRecord:
    frame_idx: int
    source_frame_idx: int
    timestamp_s: float
    counts: dict[str, int]
    tracked_counts: dict[str, int] | None = None
    inference_ms: float = 0.0
    read_ms: float = 0.0
    preprocess_ms: float = 0.0
    postprocess_ms: float = 0.0
    track_ms: float = 0.0
    annotate_ms: float = 0.0
    total_ms: float = 0.0
    detections: list[dict] = field(default_factory=list)
    zone_counts: dict[str, int] | None = None


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def _list_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTENSIONS else []
    return sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _safe_fps(value: float, fallback: float = 10.0) -> float:
    if value is None or not np.isfinite(value) or value <= 0:
        return fallback
    return float(value)


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*codec), fps, size
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"could not open a video writer for {path}")


def _annotate_and_save(
    frame: np.ndarray,
    detections,
    counts: dict[str, int],
    frame_idx: int,
    total_frames: int | None,
    inference_ms: float,
    classes: tuple[str, ...],
    show_track_ids: bool,
    output_dir: Path,
    filename: str,
    zone_profile: ZoneProfile | None = None,
    zone_counts: dict[str, int] | None = None,
    model_tag: str = "ExecuTorch INT8",
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    annotated = annotate_frame(
        frame,
        detections,
        counts,
        frame_idx=frame_idx,
        total_frames=total_frames,
        inference_ms=inference_ms,
        classes=classes,
        show_track_ids=show_track_ids,
        model_tag=model_tag,
        zone_name=zone_profile.name if zone_profile else None,
        zone_points=zone_profile.points if zone_profile else None,
        zone_counts=zone_counts,
    )
    path = output_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return annotated, (time.perf_counter() - started) * 1000.0


def _write_csv(path: Path, records: list[FrameRecord], classes: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        header = [
            "frame_idx",
            "source_frame_idx",
            "timestamp_s",
            *[f"count_{name}" for name in classes],
            "count_total",
            "read_ms",
            "preprocess_ms",
            "inference_ms",
            "postprocess_ms",
            "track_ms",
            "annotate_ms",
            "total_ms",
        ]
        has_zone = any(rec.zone_counts is not None for rec in records)
        if has_zone:
            header.extend(["zone_person", "zone_vehicle", "zone_both"])
        writer.writerow(header)
        for rec in records:
            row = [
                    rec.frame_idx,
                    rec.source_frame_idx,
                    f"{rec.timestamp_s:.3f}",
                    *[rec.counts.get(name, 0) for name in classes],
                    sum(rec.counts.values()),
                    f"{rec.read_ms:.2f}",
                    f"{rec.preprocess_ms:.2f}",
                    f"{rec.inference_ms:.2f}",
                    f"{rec.postprocess_ms:.2f}",
                    f"{rec.track_ms:.2f}",
                    f"{rec.annotate_ms:.2f}",
                    f"{rec.total_ms:.2f}",
                ]
            if has_zone:
                zone = rec.zone_counts or {"person": 0, "vehicle": 0}
                row.extend([zone["person"], zone["vehicle"], int(zone["person"] > 0 and zone["vehicle"] > 0)])
            writer.writerow(row)


def _write_jsonl(path: Path, records: list[FrameRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for rec in records:
            payload = {
                "frame_idx": rec.frame_idx,
                "source_frame_idx": rec.source_frame_idx,
                "timestamp_s": round(rec.timestamp_s, 3),
                "counts": rec.counts,
                "inference_ms": round(rec.inference_ms, 2),
                "detections": rec.detections,
            }
            if rec.zone_counts is not None:
                payload["zone_counts"] = rec.zone_counts
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_zone_csv(path: Path, records: list[FrameRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["frame_idx", "source_frame_idx", "timestamp_s", "zone_person", "zone_vehicle", "zone_both"])
        for rec in records:
            zone = rec.zone_counts or {"person": 0, "vehicle": 0}
            writer.writerow([
                rec.frame_idx,
                rec.source_frame_idx,
                f"{rec.timestamp_s:.3f}",
                zone["person"],
                zone["vehicle"],
                int(zone["person"] > 0 and zone["vehicle"] > 0),
            ])


def _write_attribute_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["class", "track_id", "color", "direction", "speed_px_s", "dwell_s", "hits", "score"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate_counts(
    records: list[FrameRecord],
    classes: tuple[str, ...],
    tracker: IoUTracker | None,
) -> dict[str, dict]:
    per_class: dict[str, dict] = {}
    for name in classes:
        values = [rec.counts.get(name, 0) for rec in records]
        arr = np.asarray(values, dtype=np.float64)
        per_class[name] = {
            "peak": int(arr.max()) if arr.size else 0,
            "mean": round(float(arr.mean()), 2) if arr.size else 0.0,
            "p95": int(np.percentile(arr, 95)) if arr.size else 0,
            "total_detections": int(arr.sum()) if arr.size else 0,
            "unique_tracked": (
                int(tracker.unique_counts.get(name, 0)) if tracker is not None else None
            ),
        }
    return per_class


def _detect_frame(
    detector: ExecuTorchDetector,
    frame: np.ndarray,
    classes: tuple[str, ...],
    conf_threshold: float,
    iou_threshold: float,
    tile_grid: int,
) -> tuple[list, float, float, float]:
    """Detect on a frame, optionally over an NxN grid of 640 crops.

    Returns ``(detections, preprocess_ms, inference_ms, postprocess_ms)``.
    Detections are in original frame coordinates; per-tile duplicates are
    merged by class.
    """

    origins = tile_origins(frame.shape[1], frame.shape[0], tile_grid)
    detections: list = []
    preprocess_ms = 0.0
    inference_ms = 0.0
    postprocess_ms = 0.0

    for x, y in origins:
        crop = frame if tile_grid <= 1 else frame[y : y + INPUT_SIZE[0], x : x + INPUT_SIZE[1]]
        t_pre = time.perf_counter()
        chw, meta = letterbox(crop, INPUT_SIZE)
        preprocess_ms += (time.perf_counter() - t_pre) * 1000.0

        raw, tile_ms = detector.infer(chw)
        inference_ms += tile_ms

        t_post = time.perf_counter()
        found, _ = decode(
            raw,
            meta,
            classes=classes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        postprocess_ms += (time.perf_counter() - t_post) * 1000.0

        if tile_grid > 1:
            for detection in found:
                x1, y1, x2, y2 = detection.box
                detection.box = (x1 + x, y1 + y, x2 + x, y2 + y)
        detections.extend(found)

    if tile_grid > 1:
        t_merge = time.perf_counter()
        detections = merge_detections(detections)
        postprocess_ms += (time.perf_counter() - t_merge) * 1000.0

    return detections, preprocess_ms, inference_ms, postprocess_ms


def run_input(
    input_path: str | Path,
    output_dir: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    classes: tuple[str, ...] | list[str] = DEFAULT_CLASSES,
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.7,
    track: bool = True,
    attributes: bool = True,
    save_video: bool = False,
    annotate_every: int = 10,
    max_frames: int | None = None,
    stride: int = 1,
    threads: int | None = None,
    quiet: bool = False,
    zone_config: str | Path | None = None,
    estimate_vehicle_colors: bool = False,
    tile_grid: int = 0,
) -> dict:
    """Process a video, one image, or a directory of images.

    Returns the summary dict and writes the full evidence bundle to output_dir.
    """

    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = tuple(classes)
    if not classes:
        raise ValueError("at least one category is required")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if annotate_every < 0:
        raise ValueError("annotate_every must be >= 0")
    if tile_grid < 0 or tile_grid > 6:
        raise ValueError("tile grid must be between 0 and 6")
    zone_profile = load_zone_profile(zone_config) if zone_config else None

    is_video = _is_video(input_path)
    image_paths = [] if is_video else _list_images(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")
    if not is_video and not image_paths:
        raise ValueError(
            f"{input_path} is neither a supported video nor an image/directory of images"
        )

    detector = ExecuTorchDetector(model_path, threads=threads, warmup=3)
    tracker = IoUTracker() if (track and is_video) else None
    attribute_acc = AttributeAccumulator() if (attributes or estimate_vehicle_colors) else None
    if not quiet:
        print(f"\nArten · {detector.model_name}")
        print(f"  model     {Path(model_path).name} ({detector.model_size_mb:.2f} MB)")
        print(f"  runtime   ExecuTorch {detector.executorch_version} · torch {detector.torch_version}")
        print(f"  input     {input_path}")
        print(f"  classes   {', '.join(classes)}")
        print(f"  tracking  {'on (indicative unique counts)' if tracker else 'off'}")
        if attribute_acc:
            print("  attrs     on-device motion (direction, relative speed, dwell)" + (" + vehicle colour" if estimate_vehicle_colors else ""))
        if zone_profile:
            print(f"  zone      {zone_profile.name} (manual polygon; co-presence only)")
        if tile_grid > 1:
            print(f"  tiling    {tile_grid}x{tile_grid} ({tile_grid * tile_grid} crops per frame)")
        print(f"  output    {output_dir}\n")

    records: list[FrameRecord] = []
    saved_annotated: list[Path] = []
    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    video_writer: cv2.VideoWriter | None = None
    last_annotated: np.ndarray | None = None
    source_fps = 1.0
    source_size = (0, 0)
    source_frame_count = 0

    frame_iter_started = time.perf_counter()

    if is_video:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {input_path}")
        source_fps = _safe_fps(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_size = (width, height)
        source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if width <= 0 or height <= 0:
            raise RuntimeError("video reports an invalid frame size")

        if save_video:
            video_writer = _open_writer(
                output_dir / "annotated.mp4", source_fps / stride, (width, height)
            )

        raw_idx = 0
        processed = 0
        while True:
            if max_frames is not None and processed >= max_frames:
                break
            read_started = time.perf_counter()
            ok, frame = cap.read()
            read_ms = (time.perf_counter() - read_started) * 1000.0
            if not ok:
                break
            if raw_idx % stride != 0:
                raw_idx += 1
                continue

            t_frame = time.perf_counter()
            detections, preprocess_ms, inference_ms, detect_post_ms = _detect_frame(
                detector, frame, classes, conf_threshold, iou_threshold, tile_grid
            )
            t_post = time.perf_counter()
            if estimate_vehicle_colors:
                for detection in detections:
                    if detection.class_name in VEHICLE_CLASSES:
                        detection.attributes["color_estimate"] = estimate_vehicle_color(
                            frame, detection.box
                        )
            counts = count_by_class(detections, classes)
            zone_counts = (
                count_detections_in_zone(detections, zone_profile, frame.shape[1], frame.shape[0])
                if zone_profile
                else None
            )
            postprocess_ms = detect_post_ms + (time.perf_counter() - t_post) * 1000.0

            track_ms = 0.0
            tracked_counts = None
            if tracker is not None:
                t_track = time.perf_counter()
                tracker.update(detections, processed)
                if attributes:
                    tracker.annotate_motion_attributes(detections, source_fps, stride)
                track_ms = (time.perf_counter() - t_track) * 1000.0
                tracked_counts = tracker.unique_counts_for(classes)
            if attribute_acc is not None:
                attribute_acc.update(detections, processed)

            annotate_ms = 0.0
            annotated_for_video: np.ndarray | None = None
            save_this_frame = annotate_every and (
                processed == 0 or processed % annotate_every == 0
            )
            if save_this_frame:
                last_annotated, annotate_ms = _annotate_and_save(
                    frame,
                    detections,
                    counts,
                    raw_idx,
                    source_frame_count,
                    inference_ms,
                    classes,
                    tracker is not None,
                    annotated_dir,
                    f"frame_{processed:05d}.jpg",
                    zone_profile,
                    zone_counts,
                    detector.model_tag,
                )
                saved_annotated.append(annotated_dir / f"frame_{processed:05d}.jpg")
                annotated_for_video = last_annotated

            if video_writer is not None:
                if annotated_for_video is None:
                    # The HUD must be on every encoded frame, even when we only
                    # JPEG-save every Nth one.
                    t_ann = time.perf_counter()
                    annotated_for_video = annotate_frame(
                        frame,
                        detections,
                        counts,
                        frame_idx=raw_idx,
                        total_frames=source_frame_count,
                        inference_ms=inference_ms,
                        classes=classes,
                        show_track_ids=tracker is not None,
                        model_tag=detector.model_tag,
                        zone_name=zone_profile.name if zone_profile else None,
                        zone_points=zone_profile.points if zone_profile else None,
                        zone_counts=zone_counts,
                    )
                    annotate_ms = (time.perf_counter() - t_ann) * 1000.0
                    last_annotated = annotated_for_video
                video_writer.write(annotated_for_video)

            total_ms = (
                read_ms
                + preprocess_ms
                + inference_ms
                + postprocess_ms
                + track_ms
                + annotate_ms
            )
            records.append(
                FrameRecord(
                    frame_idx=processed,
                    source_frame_idx=raw_idx,
                    timestamp_s=raw_idx / source_fps,
                    counts=counts,
                    tracked_counts=tracked_counts,
                    inference_ms=inference_ms,
                    read_ms=read_ms,
                    preprocess_ms=preprocess_ms,
                    postprocess_ms=postprocess_ms,
                    track_ms=track_ms,
                    annotate_ms=annotate_ms,
                    total_ms=total_ms,
                    detections=[d.to_dict() for d in detections],
                    zone_counts=zone_counts,
                )
            )
            processed += 1
            raw_idx += 1
            if not quiet and processed % 10 == 0:
                print(
                    f"\r  processed {processed} frames · "
                    f"{1000.0 / max(inference_ms, 1e-6):.1f} FPS model · "
                    f"{sum(counts.values())} objects",
                    end="",
                    flush=True,
                )
        cap.release()
        if video_writer is not None:
            video_writer.release()
        duration_s = source_frame_count / source_fps if source_frame_count else raw_idx / source_fps
    else:
        # Image mode: each image is one independent frame. Tracking across
        # unrelated images would be meaningless, so it stays off.
        total_images = len(image_paths)
        if max_frames is not None:
            image_paths = image_paths[:max_frames]
            total_images = len(image_paths)
        duration_s = float(len(image_paths))
        for idx, path in enumerate(image_paths):
            t_frame = time.perf_counter()
            frame = read_image(path)
            read_ms = (time.perf_counter() - t_frame) * 1000.0
            detections, preprocess_ms, inference_ms, detect_post_ms = _detect_frame(
                detector, frame, classes, conf_threshold, iou_threshold, tile_grid
            )
            t_post = time.perf_counter()
            if estimate_vehicle_colors:
                for detection in detections:
                    if detection.class_name in VEHICLE_CLASSES:
                        detection.attributes["color_estimate"] = estimate_vehicle_color(
                            frame, detection.box
                        )
            counts = count_by_class(detections, classes)
            zone_counts = (
                count_detections_in_zone(detections, zone_profile, frame.shape[1], frame.shape[0])
                if zone_profile
                else None
            )
            if attribute_acc is not None:
                attribute_acc.update(detections, idx)
            postprocess_ms = detect_post_ms + (time.perf_counter() - t_post) * 1000.0

            annotate_ms = 0.0
            if annotate_every:
                last_annotated, annotate_ms = _annotate_and_save(
                    frame,
                    detections,
                    counts,
                    idx,
                    total_images,
                    inference_ms,
                    classes,
                    False,
                    annotated_dir,
                    f"{path.stem}.jpg",
                    zone_profile,
                    zone_counts,
                    detector.model_tag,
                )
                saved_annotated.append(annotated_dir / f"{path.stem}.jpg")

            total_ms = read_ms + preprocess_ms + inference_ms + postprocess_ms + annotate_ms
            records.append(
                FrameRecord(
                    frame_idx=idx,
                    source_frame_idx=idx,
                    timestamp_s=float(idx),
                    counts=counts,
                    tracked_counts=None,
                    inference_ms=inference_ms,
                    read_ms=read_ms,
                    preprocess_ms=preprocess_ms,
                    postprocess_ms=postprocess_ms,
                    annotate_ms=annotate_ms,
                    total_ms=total_ms,
                    detections=[d.to_dict() for d in detections],
                    zone_counts=zone_counts,
                )
            )
            if not quiet:
                print(
                    f"\r  {idx + 1}/{total_images} images · "
                    f"{path.name} · {sum(counts.values())} objects",
                    end="",
                    flush=True,
                )
        if not quiet:
            print()

    if not quiet:
        print()

    if not records:
        raise RuntimeError("no frames/images were processed")

    wall_s = time.perf_counter() - frame_iter_started
    per_class = _aggregate_counts(records, classes, tracker)
    inference_latency = summarise_latencies([r.inference_ms for r in records])
    pipeline_latency = summarise_latencies([r.total_ms for r in records])
    preprocess_latency = summarise_latencies([r.preprocess_ms for r in records])
    postprocess_latency = summarise_latencies([r.postprocess_ms for r in records])
    read_latency = summarise_latencies([r.read_ms for r in records])
    annotate_latency = summarise_latencies([r.annotate_ms for r in records])

    person_series = [r.counts.get("person", 0) for r in records]
    vehicle_series = [
        sum(r.counts.get(name, 0) for name in VEHICLE_CLASSES) for r in records
    ]
    exposure = {
        "person_seconds": (
            person_seconds(person_series, source_fps, stride) if is_video else 0.0
        ),
        "vehicle_seconds": (
            person_seconds(vehicle_series, source_fps, stride) if is_video else 0.0
        ),
        "frames_with_person_and_vehicle": frames_with_both(person_series, vehicle_series)["frames"],
        "share_frames_with_person_and_vehicle": frames_with_both(
            person_series, vehicle_series
        )["share"],
    }

    zone_result = None
    if zone_profile:
        zone_rows = [
            {"source_frame_idx": rec.source_frame_idx, **(rec.zone_counts or {"person": 0, "vehicle": 0})}
            for rec in records
        ]
        zone_result = {
            "name": zone_profile.name,
            "points_normalized": [list(point) for point in zone_profile.points],
            "counts": summarise_zone(
                zone_rows, source_fps=source_fps, stride=stride, is_video=is_video
            ),
        }

    model_info = detector.info()
    summary = {
        "project": PROJECT_NAME,
        "tagline": PROJECT_TAGLINE,
        "run": {
            "input": str(input_path),
            "input_type": "video" if is_video else "images",
            "output_dir": str(output_dir),
            "frames_processed": len(records),
            "source_frames": source_frame_count if is_video else len(image_paths),
            "stride": stride,
            "tile_grid": tile_grid,
            "duration_s": round(duration_s, 3),
            "wall_time_s": round(wall_s, 3),
            "source_fps": round(source_fps, 3) if is_video else None,
            "source_size": list(source_size) if is_video else None,
            "classes": list(classes),
            "confidence_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "tracking": tracker is not None,
            "attributes": attribute_acc is not None,
            "threads": threads,
            "annotate_every": annotate_every,
            "save_video": save_video,
            "vehicle_color_estimate": estimate_vehicle_colors,
        },
        "model": model_info["model"],
        "environment": {k: v for k, v in model_info.items() if k not in ("model", "reference")},
        "counts": per_class,
        "latency": {
            "model_inference": inference_latency,
            "preprocess": preprocess_latency,
            "postprocess": postprocess_latency,
            "read": read_latency,
            "annotate": annotate_latency,
            "pipeline_total": pipeline_latency,
        },
        "throughput": {
            "model_fps_from_mean": inference_latency.get("fps_from_mean"),
            "pipeline_fps_from_mean": pipeline_latency.get("fps_from_mean"),
            "processed_frames_per_wall_second": round(len(records) / wall_s, 2)
            if wall_s > 0
            else None,
        },
        "exposure": exposure,
        "zone": zone_result,
        "attributes": attribute_acc.summary() if attribute_acc else None,
        "peak_rss_mb": peak_rss_mb(),
        "reference": detector.model_reference,
        "outputs": {
            "counts_csv": str(output_dir / "counts.csv"),
            "zone_counts_csv": str(output_dir / "zone_counts.csv") if zone_profile else None,
            "attribute_tracks_csv": str(output_dir / "attribute_tracks.csv") if attribute_acc and attribute_acc.tracks else None,
            "attribute_summary_json": str(output_dir / "attribute_summary.json") if attribute_acc and attribute_acc.tracks else None,
            "detections_jsonl": str(output_dir / "detections.jsonl"),
            "summary_json": str(output_dir / "summary.json"),
            "summary_txt": str(output_dir / "summary.txt"),
            "annotated_dir": str(annotated_dir)
            if annotate_every
            else None,
            "annotated_video": str(output_dir / "annotated.mp4")
            if save_video
            else None,
            "contact_sheet": str(output_dir / "contact_sheet.jpg")
            if saved_annotated
            else None,
        },
        "notes": [
            (
                f"Inference ran over a {tile_grid}x{tile_grid} grid of 640 crops per "
                "frame; per-tile boxes were merged by class. Latency includes every tile."
                if tile_grid > 1
                else "Inference ran as a single 640x640 pass per frame."
            ),
            "Counts are per processed frame, not unique objects, unless the "
            "'unique_tracked' field is used.",
            "unique_tracked is indicative: a lightweight IoU/velocity tracker, "
            "not a production multi-object tracking system.",
            "Latency is measured on the device that produced this run. Compare "
            "only with runs on the same hardware and thread count.",
            "The model detects 80 COCO classes. Auto-rickshaws, carts and other "
            "India-specific categories are not guaranteed to be recognised.",
            "No faces, licence plates or identities are extracted. Frames are "
            "processed locally and only the requested evidence is written.",
            "Annotated-video encoding is an output convenience and is not included "
            "in the pipeline latency; model-only and per-stage timings are unaffected.",
            "Direction and speed are image-plane attributes from the tracker. Speed "
            "is uncalibrated pixels/second, not km/h. Colour is an optional HSV "
            "heuristic on vehicle crops, not a trained attribute model.",
        ],
    }

    _write_csv(output_dir / "counts.csv", records, classes)
    if attribute_acc and attribute_acc.tracks:
        _write_attribute_csv(output_dir / "attribute_tracks.csv", attribute_acc.rows())
        write_json(output_dir / "attribute_summary.json", {"summary": attribute_acc.summary(), "tracks": attribute_acc.rows()})
    if zone_profile:
        _write_zone_csv(output_dir / "zone_counts.csv", records)
    _write_jsonl(output_dir / "detections.jsonl", records)
    write_json(output_dir / "summary.json", summary)

    if saved_annotated:
        # Pick up to six evenly spaced annotated frames.
        if len(saved_annotated) <= 6:
            picks = saved_annotated
        else:
            step = max(1, len(saved_annotated) // 6)
            picks = saved_annotated[::step][:6]
        frames = [cv2.imread(str(p)) for p in picks]
        frames = [f for f in frames if f is not None]
        if frames:
            make_contact_sheet(
                frames,
                output_dir / "contact_sheet.jpg",
                title="Arten — per-frame detections",
                subtitle=(
                    f"{Path(input_path).name} · {len(records)} frames · "
                    f"{', '.join(classes)}"
                ),
            )

    from .report import format_summary, write_markdown_report

    summary_text = format_summary(summary)
    (output_dir / "summary.txt").write_text(summary_text + "\n", encoding="utf-8")
    write_markdown_report(summary, output_dir / "REPORT.md")

    if not quiet:
        print(summary_text)
        print(f"\nEvidence written to {output_dir}")

    return summary


def benchmark_model(
    model_path: str | Path,
    input_path: str | Path | None = None,
    runs: int = 50,
    warmup: int = 10,
    threads: int | None = None,
) -> dict:
    """Measure model-only latency on a real frame or a zero input.

    This deliberately excludes decode/NMS/annotation so the number is
    comparable with Arm's published model-card latency.
    """

    detector = ExecuTorchDetector(model_path, threads=threads, warmup=0)
    if input_path:
        path = Path(input_path).expanduser().resolve()
        if path.is_dir():
            candidates = _list_images(path)
            if not candidates:
                raise RuntimeError(f"no images found in {path}")
            path = candidates[0]
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            frame = read_image(path)
        else:
            cap = cv2.VideoCapture(str(path))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError(f"could not read a frame from {path}")
            if frame is None:
                raise RuntimeError(f"could not decode a frame from {path}")
        chw, _ = letterbox(frame)
    else:
        chw = np.zeros((3, INPUT_SIZE[0], INPUT_SIZE[1]), dtype=np.float32)

    for _ in range(warmup):
        detector.infer(chw)
    latencies = []
    for _ in range(runs):
        _, ms = detector.infer(chw)
        latencies.append(ms)

    stats = summarise_latencies(latencies)
    return {
        "model": detector.info()["model"],
        "environment": environment_info(threads),
        "input": str(input_path) if input_path else "zeros",
        "warmup": warmup,
        "runs": runs,
        "latency": stats,
        "peak_rss_mb": peak_rss_mb(),
        "reference": detector.model_reference,
        "note": (
            "Model-only latency; excludes preprocessing, decode, NMS and "
            "annotation. Do not compare across different devices."
        ),
    }
