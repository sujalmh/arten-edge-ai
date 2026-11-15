# 16-image validation matrices

Raw measurements from the Raspberry Pi 5, before the board was returned. These
files are kept because the board is no longer available and the runs cannot be
repeated locally.

## Setup

- Board: Raspberry Pi 5 Model B, 8 GB, Bookworm 64-bit.
- Runtime: ExecuTorch 1.1.0 + torch 2.10, prebuilt Arm INT8 models.
- Inputs: 16 images covering dense and sparse traffic, day, night, rain, fog,
  blur, low resolution, crowd scenes and four frames from the pedestrian clip.
- Grid: 3 detectors × 2 modes (full frame, 2×2 tiles) × 16 images = 96 runs per
  file. `peak` counts are per frame; confidence is 0.15 for YOLO11n and 0.2 for
  YOLOv5s/YOLOv8s.
- NMS: the `iou07` file used the old 0.7 threshold; the `iou05` file used 0.5,
  which is now the shipped default. The `iou_sweep` file compares 0.7, 0.5 and
  0.45 for YOLOv5s only.

## Finding

Lowering NMS from 0.7 to 0.5 mainly removes duplicate boxes in full-frame mode:
mean people per image falls from 11.5 to 9.0 for YOLOv5s and 8.4 to 7.0 for
YOLO11n. Tiled mode is essentially unchanged (YOLOv5s 14.4 → 14.3 people,
YOLOv8s 14.6 → 14.4), so the accuracy-oriented mode keeps its recall. Totals
across all 96 runs: person 1141 → 1065, motorcycle 298 → 274, car 1349 → 1303.

The practical reading: full-frame counts are conservative in dense scenes but
cleaner in sparse ones; use 2×2 tiling when counting accuracy matters, and
report bus/truck counts as unreliable in tiled mode.

## Files

- `matrix_iou07_16images.json` — 96 runs at NMS 0.7.
- `matrix_iou05_16images.json` — 96 runs at NMS 0.5.
- `v5s_iou_sweep_16images.json` — YOLOv5s at NMS 0.7, 0.5 and 0.45.

Each record contains image, model, mode, confidence, NMS threshold, per-class
peak counts, model p50/mean latency and pipeline mean latency.
