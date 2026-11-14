# Model

## Primary artifact

| Field | Value |
|---|---|
| File | `yolo11n_int8_xnnpack.pte` |
| Size | 4.73 MB |
| Architecture | Ultralytics YOLO11n, 80-class COCO detector |
| Optimization | Static INT8 post-training quantization by Arm; stem convolution and detection head remain FP32 |
| Runtime | ExecuTorch 1.1.0 (pinned on Pi), XNNPACK + KleidiAI |
| Input | `[1, 3, 640, 640]` float32, RGB, `[0, 1]`, letterbox pad 114 |
| Output | `[1, 84, 8400]` = 4 xywh centre + 80 class scores, no objectness |
| Source | [Arm/yolo11n-int8-xnnpack-executorch-raspberrypi5](https://huggingface.co/Arm/yolo11n-int8-xnnpack-executorch-raspberrypi5) |
| SHA-256 | `154c54b52389219ebd982e1003227c1abbc3f29a99bdad241147b684d3041b96` |
| Licence | AGPL-3.0 (Ultralytics YOLO11n) |

The source URL is the Arm model card. The `.pte` is committed for offline
reproducibility; `scripts/download_model.sh` re-downloads it and verifies the
same SHA-256.

## Why this model

- **Small enough for the edge:** 4.73 MB, 59.17 MB peak memory on Pi 5.
- **Fast enough for a 15-second clip:** 7.46 FPS on Pi 5, so 150 frames take
  about 20 seconds of model time — comfortably inside a one-day hackathon demo.
- **Accurate enough for counting:** 38.90% mAP@0.5:0.95 / 54.66% mAP@0.5 on
  COCO val2017, only 0.4 pp below the FP32 baseline.
- **Arm-verified on the target:** p50 latency, p90/p99, model load time, peak
  memory and delegation are published in the model card, so the proposal does
  not rely on a laptop benchmark.

Reference numbers from the model card (Raspberry Pi 5, Cortex-A76, 4 cores
@ 2.4 GHz, Raspberry Pi OS 64-bit Trixie, ExecuTorch 1.1.0, XNNPACK, 1 thread):

| Metric | Value |
|---|---:|
| p50 latency | 134.10 ms |
| p90 / p99 latency | 134.68 / 135.06 ms |
| Throughput | 7.46 FPS |
| Model load | 8.82 ms |
| Model size | 4.73 MB |
| Peak memory | 59.17 MB |
| mAP@0.5:0.95 | 38.90% |
| mAP@0.5 | 54.66% |

These are **reference** values, not measurements from this repository. Run
`arten benchmark` on the actual Pi and replace the dev-host row in the
report.

## I/O contract

The pipeline follows `model_io.yaml` exactly:

1. Read image as RGB.
2. Letterbox-resize to 640×640 with 114-grey padding, centre-anchored.
3. Resize uint8 pixels first, then divide by 255 (no mean/std).
4. Run `forward` with a `[1, 3, 640, 640]` float32 tensor.
5. Transpose output to `[8400, 84]`; score = max class score; xywh → xyxy;
   per-class NMS (IoU 0.7); top 300; inverse letterbox.

## Fallback export

`scripts/export_model.py` exports `yolo11n.pt` to ExecuTorch FP32 using
Ultralytics. That path is for reproducibility if the prebuilt INT8 model is
unavailable; it is larger and slower on the Pi. The deployable artifact remains
the Arm INT8 `.pte`.

```bash
pip install ultralytics executorch torch
python scripts/export_model.py --weights yolo11n.pt
```

## Licence note

YOLO11n is AGPL-3.0. The Arm-optimized `.pte` is distributed under the same
terms. This repository is AGPL-3.0 so the code and model can be shared together.
If you need a permissive licence, replace the detector with a permissively
licensed model and re-export; the CLI accepts any YOLO-style `.pte` with an
`[N, 84]` or `[N, 85]` output.

## Optional accuracy detector

| File | Task | Size | Licence | SHA-256 |
|---|---|---|---:|---|
| `yolov5s_int8_xnnpack.pte` | YOLOv5s, 80-class COCO | 7.13 MB | AGPL-3.0 | `40763869f4cd0dfb41b7d13e2d1f5c1b5f446dda34719e75d09bb87a352306c9` |

On the dense Bengaluru image this model measured **29 people, 26 cars, 15
motorcycles** at **149.74 ms** full-frame on the Pi 5, and **29/41/25** with
2×2 tiling at **601.73 ms**. It is faster and has higher person/two-wheeler
recall than YOLOv8s on this scene, despite a lower COCO mAP (36.09). Use it with
`--model models/yolov5s_int8_xnnpack.pte`. Bus/truck counts are unreliable in
tiled mode because large vehicles produce partial boxes across crops.

## Support models (optional)

Two small models add detail around the detector. Both are committed, run on a
cadence, and are off unless their CLI flags are given.

| File | Task | Size | Input | Output | Licence | SHA-256 |
|---|---|---|---:|---|---|---|---|
| `lite_hrnet18_int8_xnnpack.pte` | 17-keypoint pose | 2.82 MB | `[1,3,384,288]`, ImageNet norm | `[1,17,96,72]` heatmaps | Apache-2.0 | `95b9d1de60a0bb22a84bacc01f658b2f26479bd3aec4f423b594f74ec7cd9017` |
| `ddrnet23_slim_int8_xnnpack.pte` | 11-class road segmentation | 5.64 MB | `[1,3,720,960]`, RGB − 128 | `[1,11,90,120]` logits | MIT | `d59ba585166a31b852aa695297f34f70998ad3366a261e84a071cdf6f6dc452b` |

Sources:
[Arm/lite-hrnet18-int8-xnnpack-executorch-raspberrypi5](https://huggingface.co/Arm/lite-hrnet18-int8-xnnpack-executorch-raspberrypi5),
[Arm/ddrnet23-slim-int8-xnnpack-executorch-raspberrypi5](https://huggingface.co/Arm/ddrnet23-slim-int8-xnnpack-executorch-raspberrypi5).

Usage:

```bash
arten run -i clip.mp4 \
  --pose-model models/lite_hrnet18_int8_xnnpack.pte --pose-every 10 \
  --seg-model models/ddrnet23_slim_int8_xnnpack.pte --seg-every 10
```

- Pose runs on the largest person boxes, up to `--pose-max-persons` (default 5),
  and adds `posture` (`standing`, `walking`, `fallen`, `riding`, `unknown`) and
  keypoints to the annotation.
- Segmentation samples the surface at each detection's foot and adds `surface`
  (`road`, `sidewalk`, `other`).
- A person box that overlaps a motorcycle box is tagged `rider`.
- Surface and posture are sampled, indicative attributes, not trained risk
  classifiers.
