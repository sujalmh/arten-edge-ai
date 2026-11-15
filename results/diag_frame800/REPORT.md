# Arten — Edge Object Detection, Counting & Attributes

**Run report**

Offline pedestrian and vehicle detection, counting and attribute analysis with ExecuTorch on a Raspberry Pi. This report was generated from the run's `summary.json`.

## Run

- **Input:** `/home/pi/frame800.png`
- **Frames processed:** 1
- **Categories:** person, bicycle, car, motorcycle, bus, truck
- **Thresholds:** confidence 0.1, IoU 0.5
- **Tracking:** off
- **Device:** aarch64 · Linux-6.12.25+rpt-rpi-2712-aarch64-with-glibc2.36
- **Peak process RSS:** 497.5 MB

## Model

- **Name:** YOLOv5s (COCO) — Arm-optimized INT8
- **Format:** ExecuTorch .pte, 7.13 MB
- **Backend:** XNNPACK (CPU)
- **Quantization:** INT8 static post-training quantization; stem conv and detection head remain FP32 (per Arm model card)
- **SHA-256:** `40763869f4cd0dfb41b7d13e2d1f5c1b5f446dda34719e75d09bb87a352306c9`
- **Source:** https://huggingface.co/Arm/yolov5s-int8-xnnpack-executorch
- **Reference accuracy (COCO val2017):** mAP@0.5:0.95 36.09%, mAP@0.5 54.97%

## Counts per frame

| Category | Peak/frame | Mean/frame | p95 | Total detections | Unique (indicative) |
|---|---:|---:|---:|---:|---:|
| person | 1 | 1.00 | 1 | 1 | — |
| bicycle | 0 | 0.00 | 0 | 0 | — |
| car | 12 | 12.00 | 12 | 12 | — |
| motorcycle | 0 | 0.00 | 0 | 0 | — |
| bus | 3 | 3.00 | 3 | 3 | — |
| truck | 0 | 0.00 | 0 | 0 | — |

> Counts are per processed frame. `Unique` comes from a lightweight IoU/velocity tracker and is indicative, not a production MOT count.

## Latency and throughput

| Stage | Mean (ms) | p50 (ms) | p95 (ms) | FPS from mean |
|---|---:|---:|---:|---:|
| Model inference | 149.34 | 149.34 | 149.34 | 6.70 |
| Preprocess | 3.82 | 3.82 | 3.82 | 261.53 |
| Decode + NMS | 5.23 | 5.23 | 5.23 | 191.17 |
| Frame read | 18.42 | 18.42 | 18.42 | 54.28 |
| Annotate/save | 17.84 | 17.84 | 17.84 | 56.05 |
| Pipeline total | 194.66 | 194.66 | 194.66 | 5.14 |

- **Model-only:** 6.70 FPS
- **Pipeline:** 5.14 FPS
- **Wall time:** 0.20 s

## Exposure signals

- **Observed person-seconds:** 0.00
- **Observed vehicle-seconds:** 0.00
- **Frames with both people and vehicles:** 1 (100.0%)

> These are descriptive measures for the supplied clip, not extrapolations to a full day. A 15-second clip at school dismissal is a sample, not a traffic census.

## Evidence

![Contact sheet](contact_sheet.jpg)

![frame800.jpg](annotated/frame800.jpg)

- `counts.csv` — per-frame category counts and stage timings
- `detections.jsonl` — boxes, scores and optional track IDs
- `summary.json` — machine-readable summary
- `annotated/` — sample annotated frames


## Reference benchmark (not this run)

Arm's model card reports **258.07 ms p50 / 3.88 FPS** on a Raspberry Pi 5 (ExecuTorch 1.1.0, XNNPACK + KleidiAI), with **109.44 MB** peak memory. Those numbers are a reference for the prebuilt model; the table above is the only latency measured in this run.

## Notes and limitations

- Inference ran as a single 640x640 pass per frame.
- Counts are per processed frame, not unique objects, unless the 'unique_tracked' field is used.
- unique_tracked is indicative: a lightweight IoU/velocity tracker, not a production multi-object tracking system.
- Latency is measured on the device that produced this run. Compare only with runs on the same hardware and thread count.
- The model detects 80 COCO classes. Auto-rickshaws, carts and other India-specific categories are not guaranteed to be recognised.
- No faces, licence plates or identities are extracted. Frames are processed locally and only the requested evidence is written.
- Annotated-video encoding is an output convenience and is not included in the pipeline latency; model-only and per-stage timings are unaffected.
- Direction and speed are image-plane attributes from the tracker. Speed is uncalibrated pixels/second, not km/h. Colour is an optional HSV heuristic on vehicle crops, not a trained attribute model.
