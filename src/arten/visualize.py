"""Annotated frames, contact sheets and demo title cards.

The visual language is intentionally simple: one colour per category, a compact
counts panel, and a footer that makes the edge runtime unmistakable. No
dashboard, no browser, no cloud.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .constants import (
    CLASS_COLORS_BGR,
    DIRECTION_ARROWS,
    DEFAULT_CLASSES,
    FALLBACK_COLOR_BGR,
)
from .postprocess import Detection

_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSans-Bold.ttf",
    "DejaVuSans.ttf",
)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _class_color(class_name: str) -> tuple[int, int, int]:
    return CLASS_COLORS_BGR.get(class_name, FALLBACK_COLOR_BGR)


def annotate_frame(
    frame_bgr: np.ndarray,
    detections: list[Detection],
    counts: dict[str, int],
    frame_idx: int = 0,
    total_frames: int | None = None,
    inference_ms: float | None = None,
    model_tag: str = "YOLO11n INT8 · ExecuTorch",
    classes: tuple[str, ...] = DEFAULT_CLASSES,
    show_track_ids: bool = False,
    footer: str = "Local inference · no cloud · no identities",
    zone_name: str | None = None,
    zone_points: tuple[tuple[float, float], ...] | None = None,
    zone_counts: dict[str, int] | None = None,
) -> np.ndarray:
    """Draw boxes, labels and a compact counts panel onto a BGR frame."""

    image = _bgr_to_pil(frame_bgr).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    scale = max(0.65, min(width, height) / 900.0)
    font_size = max(12, int(15 * scale))
    small_size = max(10, int(12 * scale))
    font = _load_font(font_size)
    small = _load_font(small_size)
    line_width = max(2, int(3 * scale))

    if zone_name and zone_points:
        polygon = [(round(x * (width - 1)), round(y * (height - 1))) for x, y in zone_points]
        draw.polygon(polygon, fill=(255, 178, 44, 32), outline=(255, 178, 44, 230), width=line_width)

    # Boxes and labels.
    attributes_present = False
    for det in detections:
        attributes_present = attributes_present or bool(det.attributes)
        x1, y1, x2, y2 = det.box
        color = _class_color(det.class_name)
        rgba = (*reversed(color), 255)  # BGR -> RGB
        draw.rectangle([x1, y1, x2, y2], outline=rgba, width=line_width)
        parts = [det.class_name, f"{det.score:.2f}"]
        if show_track_ids and det.track_id is not None:
            parts.insert(0, f"#{det.track_id}")
        direction = det.attributes.get("direction")
        if direction:
            parts.append(DIRECTION_ARROWS.get(direction, ""))
        color_estimate = det.attributes.get("color_estimate")
        if color_estimate:
            parts.append(f"~{color_estimate}")
        label = " ".join(part for part in parts if part)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_y = max(0, y1 - text_h - 6)
        draw.rectangle(
            [x1, label_y, x1 + text_w + 8, label_y + text_h + 6],
            fill=(*rgba[:3], 210),
        )
        draw.text((x1 + 4, label_y + 2), label, font=font, fill=(255, 255, 255, 255))

    # Counts panel, top-left.
    panel_w = max(250, int(310 * scale))
    row_h = font_size + 6
    panel_h = int(30 * scale) + row_h * (len(classes) + 1) + 24
    draw.rounded_rectangle(
        [12, 12, 12 + panel_w, 12 + panel_h],
        radius=int(10 * scale),
        fill=(12, 18, 24, 205),
        outline=(255, 255, 255, 40),
        width=1,
    )
    draw.text(
        (24, 20),
        "ARTEN",
        font=font,
        fill=(255, 255, 255, 255),
    )
    sub = f"frame {frame_idx + 1}"
    if total_frames:
        sub += f"/{total_frames}"
    if inference_ms is not None:
        sub += f"  ·  {inference_ms:.0f} ms"
    draw.text((24, 20 + font_size + 4), sub, font=small, fill=(170, 190, 200, 255))

    y = 20 + font_size + small_size + 14
    total = 0
    for name in classes:
        count = int(counts.get(name, 0))
        total += count
        color = _class_color(name)
        draw.ellipse(
            [24, y + 4, 24 + int(8 * scale), y + 4 + int(8 * scale)],
            fill=(*reversed(color), 255),
        )
        draw.text((24 + int(14 * scale), y), name, font=small, fill=(235, 240, 245, 255))
        draw.text(
            (12 + panel_w - int(26 * scale), y),
            str(count),
            font=small,
            fill=(255, 255, 255, 255),
            anchor="ra",
        )
        y += row_h
    draw.text(
        (24, y + 2),
        f"total {total}",
        font=small,
        fill=(255, 220, 120, 255),
    )

    if zone_name and zone_counts is not None:
        badge = (
            f"ZONE · {zone_name}  P {zone_counts.get('person', 0)}"
            f"  V {zone_counts.get('vehicle', 0)}"
        )
        bbox = draw.textbbox((0, 0), badge, font=small)
        badge_w = bbox[2] - bbox[0] + 22
        badge_h = bbox[3] - bbox[1] + 14
        left = max(12, width - badge_w - 14)
        draw.rounded_rectangle(
            [left, 14, left + badge_w, 14 + badge_h],
            radius=6,
            fill=(18, 22, 26, 220),
            outline=(255, 178, 44, 255),
            width=2,
        )
        draw.text((left + 11, 20), badge, font=small, fill=(255, 226, 170, 255))

    # Footer.
    footer_text = f"{model_tag}  ·  {footer}"
    if attributes_present:
        footer_text += "  ·  attributes on-device"
    bbox = draw.textbbox((0, 0), footer_text, font=small)
    text_w = bbox[2] - bbox[0]
    draw.rectangle(
        [0, height - int(26 * scale), width, height],
        fill=(10, 14, 18, 190),
    )
    draw.text(
        (max(10, (width - text_w) // 2), height - int(21 * scale)),
        footer_text,
        font=small,
        fill=(200, 210, 218, 255),
    )

    return _pil_to_bgr(Image.alpha_composite(image, overlay))


def make_contact_sheet(
    frames: list[np.ndarray],
    output_path: str | Path,
    title: str = "Arten — sample detections",
    subtitle: str = "",
    columns: int = 3,
    thumb_width: int = 420,
) -> Path:
    """Build a labelled grid of annotated frames for the README/report."""

    if not frames:
        raise ValueError("need at least one frame for a contact sheet")
    columns = max(1, min(columns, len(frames)))
    rows = (len(frames) + columns - 1) // columns

    thumbs = []
    for frame in frames:
        h, w = frame.shape[:2]
        thumb_h = int(round(h * thumb_width / w))
        thumbs.append(cv2.resize(frame, (thumb_width, thumb_h), interpolation=cv2.INTER_AREA))
    cell_h = max(t.shape[0] for t in thumbs)
    gap = 12
    header = 70
    footer = 42
    sheet_w = columns * thumb_width + (columns + 1) * gap
    sheet_h = header + rows * cell_h + (rows - 1) * gap + footer

    sheet = Image.new("RGB", (sheet_w, sheet_h), (16, 20, 26))
    draw = ImageDraw.Draw(sheet)
    title_font = _load_font(24)
    sub_font = _load_font(14)
    draw.text((gap + 4, 16), title, font=title_font, fill=(255, 255, 255))
    if subtitle:
        draw.text((gap + 4, 46), subtitle, font=sub_font, fill=(170, 185, 195))

    for i, thumb in enumerate(thumbs):
        row, col = divmod(i, columns)
        x = gap + col * (thumb_width + gap)
        y = header + row * (cell_h + gap)
        sheet.paste(Image.fromarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)), (x, y))

    draw.text(
        (gap + 4, sheet_h - 28),
        "Generated locally with ExecuTorch · counts are per frame · no cloud",
        font=sub_font,
        fill=(140, 160, 172),
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return output_path


def make_title_card(
    output_path: str | Path,
    title: str,
    subtitle: str,
    footer: str,
    width: int = 1280,
    height: int = 720,
) -> Path:
    """A dark title/end card for the demo video."""

    image = Image.new("RGB", (width, height), (12, 16, 22))
    draw = ImageDraw.Draw(image)
    # Subtle accent bars.
    draw.rectangle([0, 0, width, 8], fill=(60, 220, 60))
    draw.rectangle([0, height - 8, width, height], fill=(50, 160, 255))
    title_font = _load_font(56)
    sub_font = _load_font(26)
    foot_font = _load_font(18)
    draw.text((80, 190), title, font=title_font, fill=(255, 255, 255))
    draw.text((80, 280), subtitle, font=sub_font, fill=(180, 200, 210))
    draw.text((80, height - 120), footer, font=foot_font, fill=(130, 150, 165))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)
    return output_path
