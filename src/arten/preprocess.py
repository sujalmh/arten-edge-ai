"""Letterbox preprocessing that matches the Arm model's export contract.

The .pte was exported with a fixed 640x640 input, RGB order, [0, 1] float32,
no mean/std normalisation, and a 114-grey letterbox pad. Getting this wrong
silently shifts boxes and lowers scores, so the implementation is kept explicit
and unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .constants import INPUT_SIZE, PAD_VALUE


@dataclass(frozen=True)
class LetterboxMeta:
    """Everything needed to map boxes back to original image coordinates."""

    scale: float
    pad_x: int
    pad_y: int
    orig_w: int
    orig_h: int
    input_w: int = INPUT_SIZE[1]
    input_h: int = INPUT_SIZE[0]

    def to_dict(self) -> dict[str, float | int]:
        return {
            "scale": self.scale,
            "pad_x": self.pad_x,
            "pad_y": self.pad_y,
            "orig_w": self.orig_w,
            "orig_h": self.orig_h,
            "input_w": self.input_w,
            "input_h": self.input_h,
        }


def read_image(path: str | Path) -> np.ndarray:
    """Read an image as BGR, honouring EXIF orientation.

    OpenCV ignores EXIF rotation, which would make phone photos from the
    organiser arrive sideways. PIL handles it, then we convert to BGR so the
    rest of the pipeline has one convention.
    """

    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        rgb = np.asarray(img, dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def letterbox(
    image_bgr: np.ndarray,
    size: tuple[int, int] = INPUT_SIZE,
    pad_value: int = PAD_VALUE,
) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize preserving aspect ratio and pad to a fixed square.

    Args:
        image_bgr: HxWx3 uint8 image in BGR order.
        size: (height, width) of the model input.
        pad_value: 0-255 grey level used for padding.

    Returns:
        chw: float32 array of shape (3, H, W), values in [0, 1], RGB order.
        meta: transform needed for the inverse mapping.
    """

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 BGR image, got {image_bgr.shape}")

    target_h, target_w = size
    orig_h, orig_w = image_bgr.shape[:2]
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError("image has zero height or width")

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, round(orig_w * scale))
    new_h = max(1, round(orig_h * scale))

    # Resize the uint8 pixels first (Ultralytics order), then divide by 255.
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    # BGR -> RGB, HWC -> CHW, uint8 -> float32 in [0, 1].
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    meta = LetterboxMeta(
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        orig_w=orig_w,
        orig_h=orig_h,
        input_w=target_w,
        input_h=target_h,
    )
    return chw, meta


def unletterbox_boxes(boxes: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Map xyxy boxes from letterboxed input coordinates to original pixels."""

    boxes = boxes.astype(np.float32, copy=True)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - meta.pad_x) / meta.scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - meta.pad_y) / meta.scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, meta.orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, meta.orig_h)
    return boxes
