"""Timing, memory and aggregate metrics with explicit, honest labels."""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from .constants import PI5_REFERENCE


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_size_mb(path: str | Path) -> float:
    return round(Path(path).stat().st_size / (1024 * 1024), 3)


def peak_rss_mb() -> float | None:
    """Peak resident set size for this process, normalised to MiB.

    macOS reports bytes; Linux reports KiB. This is process-wide and includes
    the Python runtime, so it is an upper bound, not model-only memory.
    """

    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return round(value / (1024 * 1024), 1)
        return round(value / 1024, 1)
    except (AttributeError, OSError):
        return None


def summarise_latencies(values: Iterable[float]) -> dict[str, float]:
    data = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if data.size == 0:
        return {}
    mean = float(data.mean())
    return {
        "count": float(data.size),
        "mean_ms": round(mean, 2),
        "p50_ms": round(float(np.percentile(data, 50)), 2),
        "p95_ms": round(float(np.percentile(data, 95)), 2),
        "p99_ms": round(float(np.percentile(data, 99)), 2),
        "min_ms": round(float(data.min()), 2),
        "max_ms": round(float(data.max()), 2),
        "std_ms": round(float(data.std()), 2),
        "fps_from_mean": round(1000.0 / mean, 2) if mean > 0 else 0.0,
    }


def environment_info(threads: int | None = None) -> dict:
    info: dict = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "peak_rss_mb": peak_rss_mb(),
        "threads": threads,
    }
    try:
        import torch

        info["torch"] = torch.__version__
    except Exception:  # pragma: no cover - torch is optional for docs/tests
        info["torch"] = None
    try:
        from importlib.metadata import version

        info["executorch"] = version("executorch")
    except Exception:  # pragma: no cover
        info["executorch"] = None
    try:
        import cv2

        info["opencv"] = cv2.__version__
    except Exception:  # pragma: no cover
        info["opencv"] = None
    return info


def person_seconds(
    per_frame_counts: list[int], source_fps: float, stride: int = 1
) -> float:
    """Observed person-seconds in the processed frames.

    Each processed frame represents ``stride / source_fps`` seconds. This is a
    descriptive exposure measure for the clip, not an estimate for the whole
    day. For images, callers should leave this at 0.
    """

    if source_fps <= 0:
        return 0.0
    seconds_per_processed_frame = stride / source_fps
    return round(sum(per_frame_counts) * seconds_per_processed_frame, 2)


def frames_with_both(
    person_counts: list[int], vehicle_counts: list[int]
) -> dict[str, float | int]:
    total = len(person_counts)
    if total == 0:
        return {"frames": 0, "share": 0.0}
    both = sum(1 for p, v in zip(person_counts, vehicle_counts) if p > 0 and v > 0)
    return {"frames": both, "share": round(both / total, 3)}


def reference_block() -> dict:
    """Arm's published Raspberry Pi 5 reference numbers, clearly labelled."""

    return {
        "label": "Arm model-card reference on Raspberry Pi 5 (not measured here)",
        **PI5_REFERENCE,
    }


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def median(values: list[float]) -> float:
    return round(statistics.median(values), 2) if values else 0.0
