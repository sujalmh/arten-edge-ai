# Copy-paste form answers

The submission portal may use different field names. These drafts cover the
likely fields; trim to the portal's character limits and replace `[bracketed]`
text. Keep the numbers consistent with the repository.

---

## Project title

**Namma Drishti — offline edge detection and counting for safer streets**

Alternative if a shorter title is needed: **Namma Drishti — count movement, not people.**

## Problem statement

**Problem Statement 2 — AI for Safer Communities: Edge-Based Object Detection and Counting**

## 50-word elevator pitch

Namma Drishti turns a 15-second street clip into offline, privacy-preserving
counts of pedestrians and vehicles on a Raspberry Pi. Using an Arm-optimized
YOLO11n INT8 model in ExecuTorch, it runs at 7.46 FPS with no cloud, GPU or
identities — evidence for safer crossings, not surveillance.

## 150-word summary

Road traffic injuries are the leading cause of death for young people aged 5–29.
India recorded 1,68,491 road deaths in 2022; pedestrians were 19.5% and
two-wheeler riders 44.5%. The missing ingredient for safer streets is not more
cameras — it is an affordable way to count movement. Namma Drishti is an offline
edge-AI PoC that processes a 10–20 second clip or images on a Raspberry Pi 5.
A 4.73 MB YOLO11n INT8 model runs through ExecuTorch with XNNPACK at 7.46 FPS,
detecting people, cars, motorcycles, buses, trucks and bicycles. It outputs
per-frame counts, annotated frames, a CSV/JSON evidence bundle and honest
latency percentiles. No footage leaves the device and no identities are
extracted. The practical use case is a school-zone safety audit: compare
dismissal-time clips and make a case for a crossing guard or zebra crossing.

## 300-word solution description

Namma Drishti is a lightweight, offline object-detection and counting PoC built
for the Namma AI Hackathon's safer-communities problem statement. It takes one
10–20 second clip or a folder of 10–20 images and produces per-frame counts for
six categories: person, bicycle, car, motorcycle, bus and truck.

The core is an Arm-optimized YOLO11n INT8 ExecuTorch model (4.73 MB, 80 COCO
classes). On a Raspberry Pi 5 it runs at 134.10 ms p50 / 7.46 FPS with 59.17 MB
peak memory, using the CPU and XNNPACK — no GPU and no cloud. The pipeline
letterboxes each frame to 640×640, runs the `.pte` through the ExecuTorch
runtime, decodes the 4 xywh + 80 class outputs, applies per-class NMS, and maps
boxes back to the original image. Counts are written per frame to CSV and JSONL;
annotated frames, a contact sheet and an optional annotated video make the result
auditable. An optional IoU/velocity tracker adds indicative unique counts, clearly
labelled as such.

The societal application is a school-zone safety audit. A school records three
15-second clips — morning drop-off, afternoon dismissal and a quiet hour — and
runs them locally on the Pi. Comparing peak pedestrian counts, vehicle counts and
observed person-seconds gives a ward officer or school committee concrete
evidence for a crossing guard, a zebra crossing or a no-parking zone. The footage
never leaves the school, and the output is aggregate counts, not identities.

The project is already beyond a concept: the repository contains a working
pipeline, 21 unit tests, a one-command Pi setup, a committed model with checksum
verification, sample evidence bundles and a 21-second demo. At the finals we will
run the organiser clip on the Pi and replace the reference numbers with on-device
measurements.

## 500-word technical approach

**Model and runtime.** We use YOLO11n, a 2.6M-parameter COCO detector, statically
quantized to INT8 by Arm and exported to ExecuTorch `.pte`. The artifact is 4.73
MB; the stem convolution and detection head remain FP32. On Raspberry Pi 5 with
ExecuTorch 1.1.0 and XNNPACK, Arm's model card reports 134.10 ms p50 latency,
7.46 FPS, 59.17 MB peak memory and 38.90% mAP@0.5:0.95 (54.66% mAP@0.5) on COCO
val2017. We chose this verified artifact over a larger FP32 model because the PoC
must run on the target, not just on a laptop.

**Pipeline.** The package is deliberately small and testable:

1. **Input** — OpenCV for video, PIL with EXIF transpose for phone images. The
   CLI accepts a clip, a single image or a directory of images.
2. **Preprocess** — aspect-preserving letterbox to 640×640, RGB, `/255`, 114-grey
   pad. This matches Arm's export contract exactly; the inverse transform is kept
   so boxes map back to original pixels.
3. **Inference** — the ExecuTorch `forward` method is loaded once and reused.
   Model-only latency is timed around `Method.execute`.
4. **Decode** — the raw `[1, 84, 8400]` output is transposed, filtered by
   confidence, converted from xywh to xyxy, and passed through per-class NumPy
   NMS. The decoder also accepts YOLOv5-style `[N, 85]` outputs, so a different
   approved model can be dropped in with `--model`.
5. **Count** — per-category counts per frame. An optional greedy IoU/velocity
   tracker provides indicative unique counts; it is explicitly not a production
   MOT system.
6. **Evidence** — `counts.csv`, `detections.jsonl`, `summary.json`,
   `summary.txt`, `REPORT.md`, annotated frames, a contact sheet and an optional
   `annotated.mp4`.

**Measurement.** The standalone benchmark runs 10 warmups and 50 measured
inferences on a real frame. The pipeline records read, preprocess, inference,
decode/NMS, tracking and annotation times separately, then reports mean, p50,
p95 and p99. Throughput is reported twice — model-only FPS and full-pipeline FPS
— and never conflated. The report distinguishes the Pi 5 reference numbers from
the development-host numbers; the team replaces the dev row with an on-device row
at the finals.

**Reproducibility.** The model is committed and checksum-verified. `setup_pi.sh`
pins ExecuTorch 1.1.0 and CPU PyTorch, creates a venv, installs the package and
runs `doctor`. Twenty-one unit tests cover letterboxing, NMS, decode, tracker
identity and metrics. A fallback export script reproduces the model through
Ultralytics if the prebuilt artifact is unavailable.

**Why edge, why now.** A Raspberry Pi 5 plus a phone clip is within reach of a
school or ward office. The model is small enough to load in under 10 ms, the
pipeline runs offline, and the output is a count table rather than a video feed.
That changes the deployment conversation from "install a surveillance system" to
"run a 15-second audit".

## Innovation / differentiation (150 words)

Most hackathon object-detection demos stop at a laptop and a bounding box. Namma
Drishti is built around the constraints of the target: a 4.73 MB INT8 ExecuTorch
model, CPU-only XNNPACK inference, a one-command Pi setup and a checksum-verified
artifact. It measures the full pipeline, not just the model, and separates p50
from p95. It is privacy-preserving by design: no faces, plates or identities, and
no network calls. It is honest about failure: the dense Bengaluru stress test
misses small motorcycles, and auto-rickshaws are not a COCO class. Finally, it
ships evidence, not screenshots — per-frame CSV, detections JSONL, annotated
frames, a generated report and a reproducible benchmark.

## Feasibility / execution plan (150 words)

The PoC is already working on a development host, with the model, pipeline,
tests, sample evidence and demo video committed. The final is therefore a
deployment and validation exercise. First 45 minutes: verify the Pi, run
`setup_pi.sh` and `doctor`, load the model. Next 45 minutes: run the organiser
clip and inspect counts. Next hour: tune the confidence threshold on two or three
frames and freeze it. Next hour: run the full clip and a 50-run benchmark, save
the evidence bundle. Next hour: fill the report and build the contact sheet.
Final hour: rehearse the two-minute pitch and keep a USB backup of all outputs.
Fallbacks include a prebuilt model, an FP32 export path and pre-generated
evidence if the Pi fails.

## Impact / measurable outcomes (150 words)

The immediate output is measurable: per-frame counts for six categories, peak
and mean counts, observed person-seconds and vehicle-seconds, model latency
(p50/p95), throughput and memory. On the committed 15-second clip, the pipeline
processes 150 frames, finds up to 9 people in a frame and tracks 18 people
(indicative) plus 2 vehicles. The practical impact is a repeatable school-zone
audit: three clips at different times, compared offline, produce evidence for a
crossing guard, zebra crossing or parking restriction. The cost is one
Raspberry Pi and no cloud subscription. The longer-term impact is a template for
wards that cannot afford proprietary traffic-analytics platforms: count
movement, prioritise one intervention, measure again.

## Team roles

| Name | Role | Relevant experience |
|---|---|---|
| [Member 1] | Edge runtime and Pi deployment | [fill] |
| [Member 2] | Computer vision and evaluation | [fill] |
| [Member 3] | Product, societal application and pitch | [fill] |

## GitHub / demo links

- GitHub: https://github.com/sujalmh/namma-drishti
- Demo video: `results/demo.mp4`
- Proposal: `docs/proposal.md`
- Technical report: `docs/report.md`
- Two-minute pitch: `docs/pitch-2min.md`

## Declaration (if asked)

The project is original work built for the Namma AI Hackathon. It uses the
open-source YOLO11n model and Arm's pre-optimized ExecuTorch artifact under
AGPL-3.0; all third-party licences and sample-media attributions are documented
in the repository. No participant materials contain confidential third-party
information.
