# Research notes — Namma AI Hackathon, Problem Statement 2

**Prepared:** 4 November 2026
**Purpose:** shortlisting research and technical decisions for the Namma Drishti
submission. This is a working document, not the submission itself.

---

## 1. The official programme

Source: <https://www.coe-iot.com/namma-ai-hackathon/>

| Item | Detail |
|---|---|
| Programme | Namma AI Initiative — Arm + IIIT Bangalore + NASSCOM CoE IoT, supported by the UK Government (TSI) |
| Format | Online proposal screening, then a one-day in-person hackathon |
| Launch | 29 October 2026 |
| Proposal deadline | **6 November 2026** |
| Finalists announced | By **11 November 2026** (top ~30–40 teams, ~80–100 members) |
| Live hackathon | **14 November 2026**, NASSCOM office, Bengaluru |
| Winners announced | 19 November 2026 |
| Team size | 1–3; team leader registers; students from the same college / professionals from the same organisation |
| Project hosting | Must be on GitHub and submitted via the challenge form |
| Finalist support | Focused virtual technical training before the event |
| Rewards | Cash award by IIIT Bangalore; certificates for completed projects |

**Selected problem statement:** Problem Statement 2 — *AI for Safer Communities:
Edge-Based Object Detection and Counting*.

**Mandatory deliverables (verbatim intent):**

1. Process one organiser-provided 10–20 second clip or 10–20 sample images using
   an approved pretrained object-detection model on ExecuTorch and Raspberry Pi.
2. Detect at least one predefined category and display counts for each selected
   image/frame; terminal output or saved annotated images suffice.
3. Submit category and counts plus at least two sample detection outputs.
4. Report average inference time per frame/image, model used and approximate
   throughput where feasible.
5. Explain the societal issue, how detection/counting helps, and one practical
   application (maximum 2 minutes).

**Explicitly not required:** live camera, real-time surveillance, tracking,
complex dashboards, cloud connectivity, mobile applications.

**Implication for strategy:** do not spend the one-day event building a UI or a
tracking platform. Spend it on a reliable detector, honest measurements, clean
evidence and a crisp societal story.

## 2. What the screening likely rewards

The organisers do not publish a scoring rubric. The following signals are
inferred directly from the brief and the programme page; treat them as a
screening lens, not a guaranteed rubric.

| Signal | What it means in practice | How Namma Drishti answers |
|---|---|---|
| Societal relevance | A specific, credible problem — not “AI for good” in general | School-zone pedestrian/vehicle exposure; India road-death data; ward-level decision |
| Technical fit | ExecuTorch on Raspberry Pi, not a laptop or cloud API | Arm-optimized `.pte`, ExecuTorch runtime, CPU-only XNNPACK, Pi setup script |
| Functional proof | It actually detects and counts on the supplied data | Working pipeline, committed sample clip, per-frame CSV, annotated frames |
| Measurable results | Latency, throughput, model, memory, counts | p50/p95, model-only vs pipeline FPS, model card reference, benchmark harness |
| Feasibility | Can a 1–3 person team deliver in 6–8 hours? | Prebuilt pipeline, one-command setup, 6-hour plan, fallbacks |
| Clarity | Can a non-specialist understand the value in 2 minutes? | One sentence, one workflow, one decision; no dashboard |
| Responsibility | Privacy, bias, limitations | No identities, no cloud, COCO/India gap named, aggregate outputs |
| Differentiation | Something beyond the minimum | Edge artifact, full-pipeline measurement, evidence bundle, local application |

**Proposal implication:** lead with the human problem and the working evidence,
not with the model architecture. The model is a commodity; the deployment,
measurement and decision workflow are the differentiators.

## 3. Technical landscape

### 3.1 ExecuTorch on Raspberry Pi

- ExecuTorch is PyTorch's on-device runtime. It ships the `.pte` format, a
  lightweight C++ runtime and delegates such as XNNPACK for CPU inference.
- PyPI `executorch` now publishes Linux aarch64 wheels, so a Pi can install the
  Python runtime without a source build. Earlier versions did not; the 2025
  issue requesting aarch64 wheels is resolved in current releases.
- The Arm model card pins **ExecuTorch 1.1.0**. The repository's development host
  uses 1.5.1 and loads the same `.pte`; the Pi setup pins 1.1.0 for exact parity.
- ExecuTorch 1.1.0's declared Linux dependency set includes `coremltools`, which
  has no aarch64 wheel. Arm's own runtime project excludes it. `setup_pi.sh`
  therefore installs `executorch==1.1.0` with `--no-deps` and installs the
  runtime support packages explicitly (the same working set as Arm's
  `pyproject.toml`). This is a packaging detail, not a runtime fork.
- The XNNPACK delegate is the practical CPU path on Pi. Arm's benchmark uses
  1 thread; more threads can help on some workloads but can also add overhead.
  The CLI exposes `--threads` so the team can measure rather than guess.
- Raspberry Pi 5 (Cortex-A76, 4 cores @ 2.4 GHz) is the reference. Pi 4 will run
  the same code but slower; no Pi 4 number is claimed without measurement.

### 3.2 Object-detection models available for ExecuTorch on Pi 5

Arm's model cards provide comparable measurements. All are on Pi 5, CPU,
ExecuTorch, XNNPACK; the exact thread count and input size are in each card.

| Model | Size | mAP@0.5:0.95 | p50 latency | FPS | Licence | Verdict |
|---|---:|---:|---:|---:|---|---|
| **YOLO11n INT8** | **4.73 MB** | **38.90%** | **134.10 ms** | **7.46** | AGPL-3.0 | **Chosen** — best size/speed/accuracy balance; prebuilt and verified |
| YOLOv5s INT8 | 7.13 MB | 36.09% | 258.07 ms | 3.87 | AGPL-3.0 | Slower, lower mAP, larger |
| YOLOv8s INT8 | 13.24 MB | 43.86% | 457.03 ms | 2.19 | AGPL-3.0 | Better mAP but 3.4× slower; too slow for a snappy demo |
| YOLOX-s INT8 | 14.28 MB | 40.01% | 578.44 ms | 1.73 | Apache-2.0 | Permissive, but slowest of the small models |
| YOLO12l INT8 | 33.81 MB | 52.29% | 2650.58 ms | 0.38 | AGPL-3.0 | Far too heavy for CPU |
| YOLO26n FP32 | 9.4 MB | ~48.0% | ~142 ms | ~7.0 | AGPL-3.0 | Attractive accuracy, but FP32 export is ~2× larger and no INT8 in Ultralytics path |
| DETR ResNet-50 INT8 | — | — | 2123.36 ms | 0.47 | Apache-2.0 | Transformer detector; far too heavy |

**Decision:** YOLO11n INT8. It is the only candidate that combines a sub-5 MB
artifact, >7 FPS on Pi 5, a published accuracy number and a prebuilt,
target-verified `.pte`. YOLO26n FP32 is the fallback if a newer model is
preferred and the extra size/latency is acceptable.

### 3.3 Ultralytics ExecuTorch export

- Ultralytics added native `format="executorch"` export in v8.3.220 (October
  2025). It uses `torch.export`, `XnnpackPartitioner` and
  `to_edge_transform_and_lower`, and writes `model.pte` + `metadata.yaml`.
- Requirements: Python 3.10–3.14, PyTorch ≥ 2.9, `executorch`, `flatbuffers`.
- The export is **FP32 only**; Ultralytics does not apply INT8/FP16 at export
  time. That is why the Arm INT8 artifact is preferred for the Pi.
- The repository includes `scripts/export_model.py` for reproducibility and as
  a fallback. It is not the primary deployment path.

### 3.4 Why not a torchvision detector?

Torchvision's SSDlite/MobileNet is permissively licensed (BSD-3-Clause) and
small, but its ExecuTorch export path is less mature, its COCO accuracy is
lower, and there is no Arm-verified Pi 5 model card with comparable numbers.
For a one-day hackathon, the verified Arm artifact reduces integration risk.

### 3.5 Why not a cloud API?

The brief explicitly values offline operation and avoids continuous cloud
connectivity. A cloud detector would also raise privacy concerns for footage of
children and would fail in low-connectivity deployments. Offline is both a
technical and a trust decision.

## 4. Deployment considerations

| Topic | Finding | Decision |
|---|---|---|
| Pi model | Pi 5 is the reference; Pi 4 runs aarch64 but slower | Target Pi 5; support Pi 4 with honest, lower numbers |
| OS | 64-bit Raspberry Pi OS Trixie/Bookworm | Setup script checks aarch64 and Python ≥ 3.10 |
| Python | ExecuTorch 1.1.0 supports 3.10–3.13 | Pi OS Trixie's Python 3.13 is fine |
| Install | PyPI aarch64 wheels now available; CPU torch index avoids CUDA; coremltools has no aarch64 wheel | `requirements-pi.txt` pins versions; `setup_pi.sh` installs ExecuTorch `--no-deps` and the runtime support set explicitly |
| Exact stack smoke test | ExecuTorch 1.1.0 + torch 2.10.0 + the same `.pte` processed 3 frames end-to-end; 20-run benchmark: 23.83 ms mean / 41.96 FPS on the dev host | Validated before the event; Pi timing still to be measured |
| Model size | 4.73 MB; fits in RAM and on disk | Commit the `.pte`; verify SHA-256 |
| Input | Fixed 640×640; letterbox 114 pad; RGB /255 | Preprocess unit-tested against the contract |
| Output | `[1, 84, 8400]`; 4 xywh + 80 classes; no objectness | NumPy decode + per-class NMS |
| Postprocess | NMS in NumPy avoids torchvision dependency | Smaller install, easier audit |
| Video | OpenCV handles MP4/AVI/MOV; H.264 encode works on the dev host | `--save-video` optional; `--stride` for speed |
| Images | PIL + EXIF transpose handles phone orientation | `read_image` normalises to BGR |
| Throughput | 7.46 FPS reference; 150 frames ≈ 20 s model time | Fine for a 15 s clip; not real-time at 10 FPS |
| Memory | 59.17 MB model-card peak; process RSS includes Python/torch | Report both, never conflate |
| Thermal | Pi 5 can throttle under sustained load | Short clips, active cooler, measure p95 |

## 5. Competitive landscape

Typical hackathon submissions for this brief tend to fall into four patterns:

1. **Laptop YOLO demo.** PyTorch + webcam + a bounding-box window. It may look
   impressive, but it does not demonstrate ExecuTorch or the Pi constraint.
2. **Cloud API wrapper.** Upload frames to a hosted detector and show counts.
   It violates the offline/edge intent and raises privacy issues.
3. **Heavy model on Pi.** A larger YOLO variant at 1–2 FPS; the demo is slow and
   the “real-time” claim collapses under measurement.
4. **Dashboard-first.** A polished UI with weak measurement and no evidence
   bundle; the mandatory counts/latency story gets lost.

Namma Drishti deliberately inverts the priorities:

- **Edge artifact first:** a target-verified INT8 `.pte`, not a laptop script.
- **Measurement first:** model-only and pipeline latency, p50/p95, separate
  throughputs.
- **Attributes, not just boxes:** image-plane direction, dwell, relative speed
  and optional coarse vehicle colour ride along with the same detections, with
  the caveats printed in the evidence.
- **Evidence first:** CSV/JSONL/annotated frames/report, not screenshots.
- **Privacy first:** aggregate counts, no identities, no network calls.
- **Honesty first:** COCO reference accuracy, named failure cases, no invented
  accuracy on the organiser clip.

The result is less flashy than a dashboard and much harder to challenge on the
mandatory deliverables.

## 6. Proposal strategy

### 6.1 What the screen sees

The proposal is a high-level document. It should answer, in order:

1. **What societal problem?** A specific street-safety decision, with a credible
   statistic and a named beneficiary.
2. **What will you build?** A scoped PoC that maps 1:1 to the mandatory
   deliverables.
3. **Why ExecuTorch + Pi?** A target-verified model with numbers, not a promise.
4. **Can you deliver in one day?** Existing repo, tests, setup script, 6-hour
   plan, fallbacks.
5. **What makes it different?** Edge-first, privacy-first, evidence-first,
   honest limits.
6. **What is the impact?** A school-zone audit that leads to one concrete
   intervention.

### 6.2 What to avoid

- “Real-time” without defining FPS and hardware.
- “High accuracy” without a dataset or metric.
- A dashboard mock-up as the centrepiece.
- A generic “smart city” story with no named user or decision.
- Hiding the COCO/India gap; judges from Arm will spot it.
- Claiming Pi numbers measured on a laptop.

### 6.3 The one-sentence test

If a judge remembers only one sentence, it should be:

> **Namma Drishti counts pedestrians and vehicles offline on a sub-₹10,000 Raspberry
> Pi, so a school can justify a crossing guard without buying a surveillance
> system.**

## 7. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Pi 5 unavailable or shared | Medium | High | Pipeline runs on Pi 4/x86_64; pre-generated evidence on USB; arrive early |
| ExecuTorch install fails on Pi | Medium | High | Pin known-good versions; pre-download wheels; fallback to a prebuilt venv tarball |
| `.pte`/runtime mismatch | Low | High | Pin 1.1.0 on Pi; test the exact artifact before the event; keep FP32 export fallback |
| Organiser clip codec unsupported | Low | Medium | OpenCV handles common codecs; convert with ffmpeg; `--stride`/`--max-frames` |
| Model misses objects in the clip | Medium | Medium | Tune confidence on 2–3 frames; choose a close-range clip; report counts honestly |
| Low FPS on Pi 4 | High | Medium | Use stride, shorter clip, smaller output; report measured FPS |
| Thermal throttling | Medium | Medium | Active cooler, short runs, p95 reporting, rest between runs |
| No internet at venue | High | Low | Everything is local; model and samples committed; wheels pre-downloaded |
| Team member unavailable | Low | Medium | Cross-train on the CLI and the report; all commands documented |
| AGPL question from judges | Low | Low | Explain the licence, the model notice, and the permissive-model swap path |
| Counts challenged | Medium | Medium | Show CSV, annotated frames, exact command; separate per-frame from unique; no accuracy claim |

## 8. Open questions to resolve before the event

1. **Which Pi will be provided?** Pi 5 8 GB is assumed. If Pi 4, adjust the
   expected throughput and clip length.
2. **Is YOLO11n an “approved” model?** The brief says “approved pretrained
   object-detection model” but does not publish a list. The CLI accepts any
   YOLO-style `.pte`; confirm with the organisers and swap if needed.
3. **What are the organiser clip properties?** 10–20 s, unknown FPS/resolution.
   The pipeline handles common codecs; test on the day.
4. **Is there a slide deck requirement?** The brief asks for a ≤2-minute
   explanation, not slides. The pitch script is designed to work without slides.
5. **Is internet available at the venue?** Assume no; pre-download everything.
6. **Are power strips/HDMI available?** Bring a USB-C PD supply, HDMI adapter,
   keyboard/mouse and a USB stick with evidence.

## 9. Sources

1. Namma AI Hackathon official page — <https://www.coe-iot.com/namma-ai-hackathon/>
2. Arm YOLO11n INT8 model card —
   <https://huggingface.co/Arm/yolo11n-int8-xnnpack-executorch-raspberrypi5>
3. Arm YOLOv8s INT8 model card —
   <https://huggingface.co/Arm/yolov8s-int8-xnnpack-executorch>
4. Arm YOLOv5s INT8 model card —
   <https://huggingface.co/Arm/yolov5s-int8-xnnpack-executorch>
5. Arm YOLOX-s INT8 model card —
   <https://huggingface.co/Arm/yolox-s-int8-xnnpack-executorch-raspberrypi5>
6. Arm YOLO12l INT8 model card —
   <https://huggingface.co/Arm/yolo12l-int8-xnnpack-executorch-raspberrypi5>
7. Arm DETR ResNet-50 INT8 model page —
   <https://developer.arm.com/ai/models/hugging-face/Arm/detr-resnet-50-int8-xnnpack-executorch-raspberrypi5/detr-resnet-50-int8-pte>
8. Ultralytics ExecuTorch integration —
   <https://docs.ultralytics.com/integrations/executorch>
9. Ultralytics Raspberry Pi benchmarks —
   <https://docs.ultralytics.com/guides/raspberry-pi>
10. ExecuTorch getting started —
    <https://docs.pytorch.org/executorch/main/getting-started.html>
11. ExecuTorch PyPI —
    <https://pypi.org/project/executorch/>
12. Arm ExecuTorch on Arm course —
    <https://learn.arm.com/courses/executorch-on-arm/>
13. WHO road traffic injuries fact sheet —
    <https://www.who.int/news-room/fact-sheets/detail/road-traffic-injuries>
14. MoRTH *Road Accidents in India 2022* —
    <https://morth.gov.in/backend/documents/uploaded/1755600426_RA_2022_30_Oct.pdf>
15. OpenCV `vtest.avi` —
    <https://github.com/opencv/opencv/blob/4.x/samples/data/vtest.avi>
16. Wikimedia Commons Bengaluru images — see `data/README.md` for URLs and
    licences.
