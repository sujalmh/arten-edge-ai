# Namma Drishti — judge brief

**Offline edge detection and counting for safer streets**

Namma AI Hackathon 2026 · Problem Statement 2 · [Team name] · [College / organisation]

---

## One sentence

Namma Drishti counts pedestrians and vehicles offline on a sub-₹10,000
Raspberry Pi, so a school can justify a crossing guard without buying a
surveillance system.

## The problem

- Road traffic injuries are the **leading cause of death for children and young
  adults aged 5–29** (WHO).
- India recorded **1,68,491 road deaths in 2022**; pedestrians were **19.5%**,
  two-wheeler riders **44.5%** (MoRTH).
- The missing data is simple: how many people cross, when, and alongside how much
  traffic. That data is usually locked behind expensive cameras or cloud
  subscriptions.

## What it does

1. Takes a **10–20 second clip** or **10–20 images**.
2. Detects **person, bicycle, car, motorcycle, bus, truck**.
3. Prints **per-frame counts** and writes `counts.csv` / `detections.jsonl`.
4. Saves **annotated frames, a contact sheet and an optional annotated video**.
5. Reports **p50/p95 latency, throughput, model size and memory**.
6. Adds **on-device attributes**: image-plane direction, dwell, relative speed
   and optional coarse vehicle colour.
7. Runs entirely locally: **no cloud, no identities**.

## Numbers that matter

| Metric | Value | Basis |
|---|---:|---|
| Model | YOLO11n INT8, 4.73 MB | Arm-optimized ExecuTorch `.pte` |
| Pi 5 reference latency | 134.10 ms p50 / 7.46 FPS | Arm model card |
| Pi 5 reference memory | 59.17 MB peak | Arm model card |
| COCO accuracy | 38.90% mAP@0.5:0.95 | Arm model card |
| Dev-host model latency | 20.34 ms mean / 49.16 FPS | This repo, labelled dev |
| Sample clip | 150 frames, peak 9 people/frame, 18 tracked people | This repo, indicative |
| Clip attributes | 9 left / 4 right / 2 stationary / 2 down / 1 up; mean dwell 4.5 s | This repo, derived |
| Traffic colours | 17 cars: 9 white, 4 gray/silver, 2 blue, 1 red, 1 orange/brown | This repo, HSV heuristic |
| Tests | 29 passing | This repo |

## Mandatory deliverables

| Requirement | Evidence |
|---|---|
| Clip/images on ExecuTorch + Pi | `data/samples/pedestrians_15s.mp4`; `scripts/setup_pi.sh` |
| Per-frame counts | `results/demo_pedestrians/counts.csv`; HUD |
| Category + counts + 2 samples | Contact sheet + two annotated Bengaluru images |
| Latency, model, throughput | `summary.json`; `benchmark_dev.json` |
| 2-minute explanation | `docs/pitch-2min.md` |

## Why it is credible

- The model is a **target-verified Arm artifact**, not a laptop script.
- Every stage is timed: preprocess, inference, decode/NMS, tracking, annotation.
- The report separates **Pi reference** from **dev-host measurement**.
- The COCO/India gap is named: auto-rickshaws are not a class; dense-scene
  motorcycles are missed.
- The attribute layer is **derived, not learned**: image-plane direction,
  uncalibrated speed, coarse HSV colour. The caveats are printed with every run.
- The repo ships **29 tests**, a one-command Pi setup, a checksum-verified model
  and a reproducible evidence bundle.

## Practical application

A school records three 15-second clips: morning drop-off, afternoon dismissal
and a quiet hour. It runs them on the Pi, offline, and compares peak pedestrian
counts, vehicle counts and observed person-seconds. A high pedestrian/vehicle
overlap at dismissal, plus direction and dwell attributes, is evidence for a
crossing guard, a zebra crossing or a no-parking zone at the gate. One Pi. No
subscription.

## Demo commands

```bash
namma-drishti doctor
namma-drishti run -i data/samples/pedestrians_15s.mp4 \
  -o results/demo --save-video --estimate-vehicle-colors
namma-drishti benchmark --input data/samples/bengaluru_traffic.jpg --runs 50
```

## Links

- GitHub: https://github.com/sujalmh/namma-drishti
- Proposal: `docs/proposal.md`
- Technical report: `docs/report.md`
- Two-minute pitch: `docs/pitch-2min.md`
- Demo video: `results/demo.mp4`

---

**Count movement, not people.**
