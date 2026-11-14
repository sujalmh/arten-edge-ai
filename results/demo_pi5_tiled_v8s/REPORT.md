# Arten — Edge Object Detection, Counting & Attributes

**Run report**

Offline pedestrian and vehicle detection, counting and attribute analysis with ExecuTorch on a Raspberry Pi. This report was generated from the run's `summary.json`.

## Run

- **Input:** `/home/pi/arten/data/samples/bengaluru_traffic.jpg`
- **Frames processed:** 1
- **Categories:** person, bicycle, car, motorcycle, bus, truck
- **Thresholds:** confidence 0.2, IoU 0.7
- **Tracking:** off
- **Device:** aarch64 · Linux-6.12.25+rpt-rpi-2712-aarch64-with-glibc2.36
- **Peak process RSS:** 482.1 MB

## Model

- **Name:** YOLOv8s (COCO) — Arm-optimized INT8
- **Format:** ExecuTorch .pte, 13.24 MB
- **Backend:** XNNPACK (CPU)
- **Quantization:** INT8 static post-training quantization; stem conv and detection head remain FP32 (per Arm model card)
- **SHA-256:** `7a8a56b82280a4032f3a9da69fe2599c1eb140bcd143413076120b2c1047a578`
- **Source:** https://huggingface.co/Arm/yolov8s-int8-xnnpack-executorch
- **Reference accuracy (COCO val2017):** mAP@0.5:0.95 43.86%, mAP@0.5 60.39%

## Counts per frame

| Category | Peak/frame | Mean/frame | p95 | Total detections | Unique (indicative) |
|---|---:|---:|---:|---:|---:|
| person | 27 | 27.00 | 27 | 27 | — |
| bicycle | 0 | 0.00 | 0 | 0 | — |
| car | 41 | 41.00 | 41 | 41 | — |
| motorcycle | 19 | 19.00 | 19 | 19 | — |
| bus | 4 | 4.00 | 4 | 4 | — |
| truck | 3 | 3.00 | 3 | 3 | — |

> Counts are per processed frame. `Unique` comes from a lightweight IoU/velocity tracker and is indicative, not a production MOT count.

## Latency and throughput

| Stage | Mean (ms) | p50 (ms) | p95 (ms) | FPS from mean |
|---|---:|---:|---:|---:|
| Model inference | 935.22 | 935.22 | 935.22 | 1.07 |
| Preprocess | 14.04 | 14.04 | 14.04 | 71.24 |
| Decode + NMS | 22.24 | 22.24 | 22.24 | 44.96 |
| Frame read | 17.98 | 17.98 | 17.98 | 55.60 |
| Annotate/save | 0.00 | 0.00 | 0.00 | 0.00 |
| Pipeline total | 989.48 | 989.48 | 989.48 | 1.01 |

- **Model-only:** 1.07 FPS
- **Pipeline:** 1.01 FPS
- **Wall time:** 0.99 s

## Exposure signals

- **Observed person-seconds:** 0.00
- **Observed vehicle-seconds:** 0.00
- **Frames with both people and vehicles:** 1 (100.0%)

> These are descriptive measures for the supplied clip, not extrapolations to a full day. A 15-second clip at school dismissal is a sample, not a traffic census.

## Evidence

- `counts.csv` — per-frame category counts and stage timings
- `detections.jsonl` — boxes, scores and optional track IDs
- `summary.json` — machine-readable summary
- `annotated/` — sample annotated frames


## Reference benchmark (not this run)

Arm's model card reports **457.03 ms p50 / 2.19 FPS** on a Raspberry Pi 5 (ExecuTorch 1.1.0, XNNPACK + KleidiAI), with **90.28 MB** peak memory. Those numbers are a reference for the prebuilt model; the table above is the only latency measured in this run.

## Notes and limitations

- Inference ran over a 2x2 grid of 640 crops per frame; per-tile boxes were merged by class. Latency includes every tile.
- Counts are per processed frame, not unique objects, unless the 'unique_tracked' field is used.
- unique_tracked is indicative: a lightweight IoU/velocity tracker, not a production multi-object tracking system.
- Latency is measured on the device that produced this run. Compare only with runs on the same hardware and thread count.
- The model detects 80 COCO classes. Auto-rickshaws, carts and other India-specific categories are not guaranteed to be recognised.
- No faces, licence plates or identities are extracted. Frames are processed locally and only the requested evidence is written.
- Annotated-video encoding is an output convenience and is not included in the pipeline latency; model-only and per-stage timings are unaffected.
- Direction and speed are image-plane attributes from the tracker. Speed is uncalibrated pixels/second, not km/h. Colour is an optional HSV heuristic on vehicle crops, not a trained attribute model.
