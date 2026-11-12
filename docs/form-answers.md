# Copy-paste form answers

The submission portal may use different field names. These drafts cover the
likely fields; trim to the portal's character limits and replace `[bracketed]`
text. Keep the numbers consistent with the repository.

---

## Project title

**Arten — Edge Object Detection, Counting & Attributes**

Alternative if a shorter title is needed: **Arten — count movement, not people.**

## Problem statement

**Problem Statement 2 — AI for Safer Communities: Edge-Based Object Detection and Counting**

## 50-word elevator pitch

Arten turns a 15-second street clip into offline, privacy-preserving
counts and movement attributes of pedestrians and vehicles on a Raspberry Pi.
Using an Arm-optimized YOLO11n INT8 model in ExecuTorch, it runs at a 7.46 FPS Pi 5 reference with no cloud, GPU or
identities — evidence for safer crossings, not surveillance.

## 150-word summary

Road traffic injuries are the leading cause of death for young people aged 5–29.
India recorded 1,68,491 road deaths in 2022; pedestrians were 19.5% and
two-wheeler riders 44.5%. The missing ingredient for safer streets is not more
cameras — it is an affordable way to count movement. Arten is an offline
edge-AI PoC that processes a 10–20 second clip or images on a Raspberry Pi 4 or 5.
A 4.73 MB YOLO11n INT8 model runs through ExecuTorch with XNNPACK (7.46 FPS Pi 5 reference),
detecting people, cars, motorcycles, buses, trucks and bicycles. It outputs
per-frame counts, on-device motion attributes and optional vehicle colour,
annotated frames, a CSV/JSON evidence bundle and honest latency percentiles. No footage leaves the device and no identities are
extracted. The practical use case is a school-zone safety audit: compare
dismissal-time clips and make a case for a crossing guard or zebra crossing.

## 300-word solution description

Arten is a lightweight, offline object-detection and counting PoC for safer
communities. It takes one
10–20 second clip or a folder of 10–20 images and produces per-frame counts for
six categories: person, bicycle, car, motorcycle, bus and truck.

The core is an Arm-optimized YOLO11n INT8 ExecuTorch model (4.73 MB, 80 COCO
classes). On a Raspberry Pi 5 it runs at 134.10 ms p50 / 7.46 FPS with 59.17 MB
peak memory, CPU-only XNNPACK, no GPU or cloud. A source-built
ARMv8.0 runtime brings the same pipeline to Pi 4/3, and a tiled mode runs native
crops for dense scenes at about four times the cost. The pipeline
letterboxes each frame to 640×640, runs the `.pte` through the ExecuTorch
runtime, decodes the 4 xywh + 80 class outputs, applies per-class NMS, and maps
boxes back to the original image. Counts are written per frame to CSV and JSONL;
annotated frames, a contact sheet and an optional annotated video make the result
auditable. An optional IoU/velocity tracker adds indicative unique counts plus image-plane
direction, dwell time and relative speed; an optional HSV pass adds a coarse
vehicle colour. All are derived attributes, not learned predictions.

The societal application is a school-zone safety audit. A school records three
15-second clips — morning drop-off, afternoon dismissal and a quiet hour — and
runs them locally on the Pi. Comparing peak pedestrian counts, vehicle counts and
observed person-seconds gives a ward officer or school committee concrete
evidence for a crossing guard, a zebra crossing or a no-parking zone. The footage
never leaves the school, and the output is aggregate counts, not identities.

The repository contains a working pipeline, 37 tests, a one-command Pi setup, a
checksum-verified model, evidence and a 21-second demo. The organiser clip can
replace the reference numbers with on-device measurements.


## Approach and overall plan (application answer)

Arten is an offline edge-AI system that turns a short street clip or a small
image set into a clear picture of pedestrian and vehicle movement. The problem
is not a lack of cameras; it is a lack of affordable, trustworthy data. A school
or ward office cannot justify an expensive camera platform or cloud subscription
just to count people crossing a road. Arten runs a compact pretrained detector
through ExecuTorch on a low-cost Raspberry Pi. It counts people, bicycles, cars,
motorcycles, buses and trucks per frame, then adds movement context: direction
across the image, dwell time, and relative speed. For vehicles, it also estimates
a coarse colour. The output is a count table and annotated frames a
non-specialist can read.

The novelty is that the count is explained, not just reported. Most demos stop at
boxes and a total. Direction, dwell and speed come from the same lightweight
tracker used for counting, not a second model, so the system stays small and
transparent. It is edge-first: the model is compact, footage stays on the
device, and no identities are recognised. The attribute layer is deliberately
classical, and its limits are stated with the output.

The plan is to build the smallest reliable end-to-end path first: run the
pretrained detector on a sample clip, confirm the counts, then add tracking and
the attribute layer. Next, package the pipeline so the same command handles
video or images, writes counts and annotated evidence, and reports latency.
Then validate it on the supplied material, document the limits and prepare the
short demonstration. A fallback model path and pre-generated evidence keep the
demonstration reproducible if the hardware or the clip changes. The detector now
defaults to a calibrated 0.2 threshold, runs on Pi 4/5 through a source-built
ARMv8.0 runtime, supports tiled native-resolution inference for dense scenes,
and installs offline from a local wheelhouse.

## 500-word technical approach

**Model and runtime.** I use YOLO11n, a 2.6M-parameter COCO detector, statically
quantized to INT8 by Arm and exported to ExecuTorch `.pte`. The artifact is 4.73
MB; the stem convolution and detection head remain FP32. On Raspberry Pi 5 with
ExecuTorch 1.1.0 and XNNPACK, Arm's model card reports 134.10 ms p50 latency,
7.46 FPS, 59.17 MB peak memory and 38.90% mAP@0.5:0.95 (54.66% mAP@0.5) on COCO
val2017. I chose this verified artifact over a larger FP32 model because the PoC
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
5. **Count + attribute** — per-category counts per frame. An optional greedy
   IoU/velocity tracker provides indicative unique counts and image-plane
   direction, dwell and relative speed. An optional HSV pass adds a coarse
   vehicle colour; an optional polygon adds a descriptive zone co-presence
   count. None of these is a production MOT or identity system.
6. **Evidence** — `counts.csv`, `detections.jsonl`, `attribute_tracks.csv`,
   `attribute_summary.json`, `summary.json`, `summary.txt`, `REPORT.md`,
   annotated frames, a contact sheet and an optional `annotated.mp4`.

**Measurement.** The standalone benchmark runs 10 warmups and 50 measured
inferences on a real frame. The pipeline records read, preprocess, inference,
decode/NMS, tracking, attribute extraction and annotation times separately,
then reports mean, p50,
p95 and p99. Throughput is reported twice — model-only FPS and full-pipeline FPS
— and never conflated. The report distinguishes the Pi 5 reference numbers from
the development-host numbers; the final Pi run replaces the dev row with an on-device row
at the event.

**Reproducibility.** The model is committed and checksum-verified. `setup_pi.sh`
pins ExecuTorch 1.1.0 and CPU PyTorch, creates a venv, installs the package and
runs `doctor`. Thirty-seven unit tests cover letterboxing, NMS, decode, tiled merge,
tracker identity, attributes, zone metrics and latency statistics. A fallback export script reproduces the model through
Ultralytics if the prebuilt artifact is unavailable.

**Why edge, why now.** A Raspberry Pi 4 or 5 plus a phone clip is within reach of a
school or ward office. The model is small enough to load in under 10 ms, the
pipeline runs offline, and the output is a count table rather than a video feed.
That changes the deployment conversation from "install a surveillance system" to
"run a 15-second audit".

## Innovation / differentiation (150 words)

Most hackathon object-detection demos stop at a laptop and a bounding box. Arten
is built around the constraints of the target: a 4.73 MB INT8 ExecuTorch
model, CPU-only XNNPACK inference, a one-command Pi setup and a checksum-verified
artifact. It adds derived movement attributes — image-plane direction, dwell and
relative speed — plus optional coarse vehicle colour, all on the same tracker.
It measures the full pipeline, not just the model, and separates p50 from p95.
It is privacy-preserving by design: no faces, plates or identities, and no
network calls. It is honest about failure: the full-frame dense-traffic pass
under-counts people and two-wheelers, auto-rickshaws are not a COCO class, and
tiling improves recall at four times the latency. Finally, it
ships evidence, not screenshots — per-frame CSV, detections JSONL, annotated
frames, a generated report and a reproducible benchmark.

## Feasibility / execution plan (150 words)

The PoC is already working, with the model, pipeline, tests, sample evidence and
demo video committed. Remaining work is validation on the event board: verify
`setup_pi.sh` and `doctor`, run the organiser clip, confirm the calibrated 0.2
threshold and, for dense frames, compare the full-frame and 2×2 tiled counts.
Then capture the benchmark and evidence bundle and rehearse the two-minute
pitch. Fallbacks include the prebuilt model, a 960 export path and the committed
evidence if the board fails.

## Impact / measurable outcomes (150 words)

The immediate output is measurable: per-frame counts for six categories, peak
and mean counts, observed person-seconds and vehicle-seconds, model latency
(p50/p95), throughput and memory. On the committed 15-second clip, the pipeline
processes 150 frames, finds up to 10 people in a frame and produces 33
indicative person tracks plus 2 vehicles. A 2×2 tiled YOLOv8s pass on the dense
image finds 27 people, 41 cars and 19 motorcycles. The practical impact is a repeatable school-zone
audit: three clips at different times, compared offline, produce evidence for a
crossing guard, zebra crossing or parking restriction. The cost is one
Raspberry Pi and no cloud subscription. The longer-term impact is a template for
wards that cannot afford proprietary traffic-analytics platforms: count
movement, prioritise one intervention, measure again.

## Author

| Name | Role |
|---|---|
| **Sujal M H** | Solo participant — edge runtime, computer vision, evaluation and pitch |

## GitHub / demo links

- GitHub: https://github.com/sujalmh/arten-edge-ai
- Demo video: https://github.com/sujalmh/arten-edge-ai/releases/download/v1.0.0/demo.mp4
- Project report: `docs/report.md`
- Two-minute pitch: `docs/pitch-2min.md`

## Declaration (if asked)

The project is original work built for the Namma AI Hackathon. It uses the
open-source YOLO11n model and Arm's pre-optimized ExecuTorch artifact under
AGPL-3.0; all third-party licences and sample-media attributions are documented
in the repository. No participant materials contain confidential third-party
information.
