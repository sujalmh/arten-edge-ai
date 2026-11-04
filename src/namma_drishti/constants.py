"""Shared constants: COCO labels, default categories, colours, paths."""

from __future__ import annotations

from pathlib import Path

# The 80 COCO classes, in the exact index order used by YOLO11.
COCO_CLASSES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(COCO_CLASSES)}

# Categories that matter for a street-safety audit.
DEFAULT_CLASSES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
)

VEHICLE_CLASSES: frozenset[str] = frozenset(
    {"bicycle", "car", "motorcycle", "bus", "truck"}
)
PERSON_CLASSES: frozenset[str] = frozenset({"person"})

# Stable BGR colours for the six default categories. Readable on grey roads.
CLASS_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "person": (60, 220, 60),
    "bicycle": (230, 180, 40),
    "car": (50, 160, 255),
    "motorcycle": (255, 120, 40),
    "bus": (230, 60, 230),
    "truck": (40, 220, 230),
}
FALLBACK_COLOR_BGR: tuple[int, int, int] = (200, 200, 200)

# Repository layout. model/ and data/ are resolved from the repo root so the
# CLI works from any current directory after `pip install -e .`.
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "yolo11n_int8_xnnpack.pte"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "samples"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results"

MODEL_NAME = "YOLO11n (COCO) — Arm-optimized INT8"
MODEL_DESCRIPTION = (
    "Ultralytics YOLO11n, 80-class COCO detector, statically quantized to INT8 "
    "by Arm and exported to ExecuTorch .pte with the XNNPACK delegate."
)
MODEL_SOURCE_URL = (
    "https://huggingface.co/Arm/"
    "yolo11n-int8-xnnpack-executorch-raspberrypi5"
)
MODEL_SHA256 = "154c54b52389219ebd982e1003227c1abbc3f29a99bdad241147b684d3041b96"
MODEL_SIZE_MB = 4.73
INPUT_SIZE = (640, 640)  # (height, width), fixed by the .pte
PAD_VALUE = 114  # Ultralytics letterbox grey

# Reference numbers published in Arm's model card for Raspberry Pi 5.
# These are *reference* values, not measurements from this repository.
PI5_REFERENCE = {
    "device": "Raspberry Pi 5 (Cortex-A76, 4 cores @ 2.4 GHz)",
    "runtime": "ExecuTorch 1.1.0, XNNPACK + KleidiAI, 1 thread",
    "p50_latency_ms": 134.10,
    "fps": 7.46,
    "peak_memory_mb": 59.17,
    "map_50_95": 38.90,
    "map_50": 54.66,
    "model_size_mb": 4.73,
    "source": MODEL_SOURCE_URL,
}
