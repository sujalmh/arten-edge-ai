# Arten — Edge Object Detection, Counting & Attributes

**Run report**

Offline pedestrian and vehicle detection, counting and attribute analysis with ExecuTorch on a Raspberry Pi. This report was generated from the run's `summary.json`.

## Run

- **Input:** `/home/pi/left800.png`
- **Frames processed:** 1
- **Categories:** person, bicycle, car, motorcycle, bus, truck
- **Thresholds:** confidence 0.1, IoU 0.5
- **Tracking:** off
- **Device:** aarch64 · Linux-6.12.25+rpt-rpi-2712-aarch64-with-glibc2.36
- **Peak process RSS:** 494.7 MB

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
| person | 0 | 0.00 | 0 | 0 | — |
| bicycle | 0 | 0.00 | 0 | 0 | — |
| car | 4 | 4.00 | 4 | 4 | — |
| motorcycle | 0 | 0.00 | 0 | 0 | — |
| bus | 0 | 0.00 | 0 | 0 | — |
| truck | 0 | 0.00 | 0 | 0 | — |

> Counts are per processed frame. `Unique` comes from a lightweight IoU/velocity tracker and is indicative, not a production MOT count.

## Latency and throughput

| Stage | Mean (ms) | p50 (ms) | p95 (ms) | FPS from mean |
|---|---:|---:|---:|---:|
| Model inference | 150.00 | 150.00 | 150.00 | 6.67 |
| Preprocess | 4.82 | 4.82 | 4.82 | 207.47 |
| Decode + NMS | 5.03 | 5.03 | 5.03 | 198.75 |
| Frame read | 7.11 | 7.11 | 7.11 | 140.57 |
| Annotate/save | 9.40 | 9.40 | 9.40 | 106.37 |
| Pipeline total | 176.37 | 176.37 | 176.37 | 5.67 |

- **Model-only:** 6.67 FPS
- **Pipeline:** 5.67 FPS
- **Wall time:** 0.18 s

## Exposure signals

- **Observed person-seconds:** 0.00
- **Observed vehicle-seconds:** 0.00
- **Frames with both people and vehicles:** 0 (0.0%)

> These are descriptive measures for the supplied clip, not extrapolations to a full day. A 15-second clip at school dismissal is a sample, not a traffic census.

## Evidence

![Contact sheet](contact_sheet.jpg)

![left800.jpg](annotated/left800.jpg)

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
