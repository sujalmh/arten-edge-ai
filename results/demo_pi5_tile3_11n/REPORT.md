# Arten — Edge Object Detection, Counting & Attributes

**Run report**

Offline pedestrian and vehicle detection, counting and attribute analysis with ExecuTorch on a Raspberry Pi. This report was generated from the run's `summary.json`.

## Run

- **Input:** `/home/pi/arten/data/samples/bengaluru_traffic.jpg`
- **Frames processed:** 1
- **Categories:** person, bicycle, car, motorcycle, bus, truck
- **Thresholds:** confidence 0.15, IoU 0.7
- **Tracking:** off
- **Device:** aarch64 · Linux-6.12.25+rpt-rpi-2712-aarch64-with-glibc2.36
- **Peak process RSS:** 449.9 MB

## Model

- **Name:** YOLO11n (COCO) — Arm-optimized INT8
- **Format:** ExecuTorch .pte, 4.73 MB
- **Backend:** XNNPACK (CPU)
- **Quantization:** INT8 static post-training quantization; stem conv and detection head remain FP32 (per Arm model card)
- **SHA-256:** `154c54b52389219ebd982e1003227c1abbc3f29a99bdad241147b684d3041b96`
- **Source:** https://huggingface.co/Arm/yolo11n-int8-xnnpack-executorch-raspberrypi5
- **Reference accuracy (COCO val2017):** mAP@0.5:0.95 38.90%, mAP@0.5 54.66%

## Counts per frame

| Category | Peak/frame | Mean/frame | p95 | Total detections | Unique (indicative) |
|---|---:|---:|---:|---:|---:|
| person | 30 | 30.00 | 30 | 30 | — |
| bicycle | 0 | 0.00 | 0 | 0 | — |
| car | 43 | 43.00 | 43 | 43 | — |
| motorcycle | 14 | 14.00 | 14 | 14 | — |
| bus | 5 | 5.00 | 5 | 5 | — |
| truck | 5 | 5.00 | 5 | 5 | — |

> Counts are per processed frame. `Unique` comes from a lightweight IoU/velocity tracker and is indicative, not a production MOT count.

## Latency and throughput

| Stage | Mean (ms) | p50 (ms) | p95 (ms) | FPS from mean |
|---|---:|---:|---:|---:|
| Model inference | 1225.71 | 1225.71 | 1225.71 | 0.82 |
| Preprocess | 27.77 | 27.77 | 27.77 | 36.01 |
| Decode + NMS | 47.87 | 47.87 | 47.87 | 20.89 |
| Frame read | 17.60 | 17.60 | 17.60 | 56.82 |
| Annotate/save | 0.00 | 0.00 | 0.00 | 0.00 |
| Pipeline total | 1318.95 | 1318.95 | 1318.95 | 0.76 |

- **Model-only:** 0.82 FPS
- **Pipeline:** 0.76 FPS
- **Wall time:** 1.32 s

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

Arm's model card reports **134.10 ms p50 / 7.46 FPS** on a Raspberry Pi 5 (ExecuTorch 1.1.0, XNNPACK + KleidiAI, 1 thread), with **59.17 MB** peak memory. Those numbers are a reference for the prebuilt model; the table above is the only latency measured in this run.

## Notes and limitations

- Inference ran over a 3x3 grid of 640 crops per frame; per-tile boxes were merged by class. Latency includes every tile.
- Counts are per processed frame, not unique objects, unless the 'unique_tracked' field is used.
- unique_tracked is indicative: a lightweight IoU/velocity tracker, not a production multi-object tracking system.
- Latency is measured on the device that produced this run. Compare only with runs on the same hardware and thread count.
- The model detects 80 COCO classes. Auto-rickshaws, carts and other India-specific categories are not guaranteed to be recognised.
- No faces, licence plates or identities are extracted. Frames are processed locally and only the requested evidence is written.
- Annotated-video encoding is an output convenience and is not included in the pipeline latency; model-only and per-stage timings are unaffected.
- Direction and speed are image-plane attributes from the tracker. Speed is uncalibrated pixels/second, not km/h. Colour is an optional HSV heuristic on vehicle crops, not a trained attribute model.
