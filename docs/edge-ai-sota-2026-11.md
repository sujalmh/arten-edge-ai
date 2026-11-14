# Edge AI detection & recognition — what's actually new (Nov 2026 scan)

**Prepared:** 14 November 2026
**Purpose:** competitive/novelty scan for Arten. What is genuinely new in edge detection and
recognition, what is now commodity, what is hype, and which claims in this repo need to change.
**Method:** web scan of 2026 papers, vendor benchmarks and deployment write-ups. Vendor numbers
are labelled as vendor numbers; independent measurements are labelled separately. Nothing here is
a substitute for measuring on the Pi.

---

## 0. The one-paragraph version

The 2024 story was "small YOLO on a Pi". The 2026 story is **NMS-free detectors, open-vocabulary
heads, quantisation-aware compression, and measurement discipline**. Detection accuracy on edge
hardware improved sharply (YOLO26, RF-DETR, D-FINE), but the harder shift is cultural: the
credible 2026 edge paper reports *warm* p99, energy per inference, thermal state and failed/degraded
requests — not a cold mean. Privacy stopped being a nice-to-have: EU AI Act prohibitions have applied
since Feb 2025, and "counting people" is explicitly *not* biometric identification, which is exactly
the line Arten already sits on. Novelty in 2026 is no longer "I ran a CNN on a Pi"; it is
**what you measure, what you refuse to compute, and what decision the output changes**.

---

## 1. Six shifts, one page

| # | Shift | Evidence | Relevance to Arten |
|---|---|---|---|
| 1 | **NMS-free, end-to-end detectors** became the default new architecture | YOLO26 (Sep 2025) removes DFL, adds one-to-one head, ProgLoss, STAL, MuSGD; RF-DETR is first real-time detector >60 COCO AP | Arten still runs NumPy NMS after decode. NMS is cheap per frame at low object counts but it is now an architectural anachronism on CPU |
| 2 | **Open-vocabulary detection reached real-time** | YOLOE / YOLOE-26, YOLO-World v2, OV-DEIM; prompt-free 4,585-name vocabulary | Direct answer to Arten's #1 limitation ("COCO is not India", 0 rickshaws). A text prompt `auto-rickshaw` is now plausible on-device |
| 3 | **Real-time transformer detectors overtook small YOLOs on accuracy** | RF-DETR-N 48.4 AP vs YOLO11-N 37.4 AP (vendor, T4 FP16); D-FINE-N 42.7 AP at 3.8M params | Not deployable on Pi 5 CPU yet at these sizes, but it reframes "YOLO11n is accurate enough" as a *constraint* claim, not a *best-in-class* claim |
| 4 | **Compression got data-aware** | ExecuTorch + OpenVINO/NNCF: AWQ, SmoothQuant, BiasCorrection, Hessian-based mixed INT4/INT8 per layer; Cortex-M backend with CMSIS-NN (Oct 2026) | Arten uses static INT8 PTQ with stem + head in FP32. That is still the right call for a one-day build; mixed precision is the documented next step |
| 5 | **Edge benchmarks got audited and found wanting** | 2026 reproducibility studies: requested power modes not realised, FP16/FP32 batch-1 inversions, 270% latency swings under memory contention, 40% throughput loss from thermals | Arten already separates model vs pipeline latency and reports p50/p95. It does **not** report warm p99, energy, or thermal state — the three things 2026 reviewers now ask for |
| 6 | **Privacy moved from feature to legal baseline** | AI Act Art. 5 prohibitions live since 2 Feb 2025; high-risk Annex III duties deferred to 2 Dec 2027 by Reg (EU) 2026/1744; "counting people does not establish identity" | Arten's aggregate-only design is now the *compliant default*, not a differentiator. The differentiator is proving it (no face/plate tensors, no network calls, retention) |

---

## 2. Detectors: what changed under the hood

### 2.1 YOLO26 is the new reference small detector

Architectural changes (Ultralytics paper + independent benchmarks):

- **Removed Distribution Focal Loss (DFL)** — simpler regression head, better for CPU/NPU lowering.
- **End-to-end / NMS-free** one-to-one head, trained alongside the one-to-many head.
- **ProgLoss** and **STAL** (small-target-aware label assignment) — small-object recall is the headline gain.
- **MuSGD** optimiser for stable convergence.

Published Pi 5 numbers (Ultralytics docs, FP32, 640, `imgsz=640`, inference only, *thread count not stated*):

| Model | Format | Size | mAP@50:95 | ms/img |
|---|---|---:|---:|---:|
| YOLO26n | PyTorch | 5.3 MB | 47.6 | 299.09 |
| YOLO26n | ONNX | 9.5 MB | 47.3 | 125.99 |
| YOLO26n | NCNN | 9.4 MB | 47.8 | **67.03** |
| YOLO26n | ExecuTorch | 9.4 MB | 47.7 | **144.83** |
| YOLO26s | ExecuTorch | 36.5 MB | 56.7 | 376.48 |

Compare to the Arm card Arten uses: **YOLO11n INT8 ExecuTorch, 4.73 MB, 38.90 mAP, 134.10 ms,
1 thread**. So:

- On raw ExecuTorch latency, YOLO11n INT8 is still slightly ahead (134 vs 145 ms), and it is 2× smaller.
- On accuracy per frame, YOLO26n FP32 is ~+9 mAP — a much bigger gap than the 11 ms latency gap.
- The two numbers are **not** directly comparable (different thread counts, precision, and probably
  different post-processing). Do not put them in one table without that caveat.
- NCNN at 67 ms shows the runtime matters more than the architecture. That is also a threat to the
  "ExecuTorch or nothing" framing: ExecuTorch is the *brief's* requirement, not the fastest option.

**Critical deployment detail:** Ultralytics' ExecuTorch export is **FP32 only** — no export-time
INT8/FP16 — and YOLO26 **falls back to the one-to-many (NMS) path on ExecuTorch**, because the
end-to-end operators do not lower. So the headline "NMS-free" benefit is *not available* on Arten's
runtime today. NCNN, OpenVINO, TensorRT and CoreML are where NMS-free actually lands.

### 2.2 Transformer detectors crossed the accuracy frontier

| Model | COCO AP50:95 | Params | Latency (T4, TRT FP16) | Licence |
|---|---:|---:|---:|---|
| YOLO11-N | 37.4 | 2.6 M | 2.5 ms | AGPL-3.0 |
| YOLO26 Nano | ~40.3 | 2.4 M | 1.7 ms | AGPL-3.0 |
| D-FINE-N | 42.7 | 3.8 M | 2.1 ms | Apache-2.0 |
| RF-DETR-N | 48.4 | 30.5 M | 2.3 ms | Apache-2.0 |
| RF-DETR-2XL | **60.1** | 126.9 M | 17.2 ms | PML 1.0 |

Source: Roboflow/RF-DETR benchmarks (vendor, T4 TensorRT FP16). RF-DETR is the first real-time
detector over 60 COCO AP; it is ICLR 2026. But note the parameter counts: RF-DETR-N is 30.5M vs
2.4M for YOLO26n. **At Pi 5 CPU budgets, this family is not yet a drop-in.** It matters as a
citation that the accuracy ceiling moved, not as a model Arten should switch to today.

Independent 2026 benchmark finding worth quoting: on nine YOLO variants across three Rockchip SoCs,
**latency correlated more strongly with mAP than with FLOPs or parameter count**, and multi-core NPU
scheduling gave **<10%** gains because of synchronisation and shared-memory bottlenecks
(*Scientific Reports*, Jan 2026). Translation: architecture-aware benchmarking beats spec-sheet
reasoning, which is exactly what Arten's harness does.

### 2.3 Open-vocabulary is the answer to "COCO is not India"

- **YOLOE / YOLOE-26** (ICCV 2025, extended 2026): text prompts, visual prompts, and a prompt-free
  4,585-name vocabulary; built on YOLOv8/11/26 backbones; near closed-set speed. YOLOE-26x reaches
  40.6 AP on LVIS minival under text prompting.
- **OV-DEIM** (2026): NMS-free DETR-style open-vocabulary detector; beats YOLOE by 1.7–4.6 AP on
  rare LVIS categories.
- **YOLO-World v2**: 20–30 FPS, best prompt robustness in a 2026 comparative study, but weaker than
  YOLOE on unseen-class generalisation after fine-tuning.

Practical caveat for Arten: open-vocabulary heads add a text-embedding path that may not lower to
ExecuTorch/XNNPACK cleanly, and the project has no Pi time to validate it. Treat it as a
**documented fallback and a one-line answer to judges**, not a change to make on event day.

---

## 3. Runtimes: ExecuTorch matured, but not in the direction Arten needs

- **ExecuTorch 1.5** (16 Sep 2026): bounded generation cancellation, batched request scheduling,
  a linkable C++ SDK inside the Python wheel, packaged delegate libraries (Qualcomm, Core ML, MLX,
  CUDA, OpenVINO), schema-version checks on `.pte`. Mostly LLM-serving work.
- **ExecuTorch 1.3.1 + SME2**: Arm SME2 kernels through KleidiAI cut SqueezeSAM INT8 556 → 304 ms
  (1.83×) and FP16 1,163 → 298 ms (3.9×) on one core. Streaming ASR: whole-recording time 9.79 →
  4.61 s (FP16) with SME2.
- **Cortex-M backend** (Oct 2026): INT8 + CMSIS-NN lowering, M0–M85, target-aware weight packing,
  static memory planning. This is the "PyTorch on a microcontroller" story, not a Pi story.
- **OpenVINO backend + NNCF** (2026): AWQ, SmoothQuant, BiasCorrection, Hessian-based mixed
  precision, with an official YOLO26 example. Intel-only.

**Negative finding that saves time:** SME2 requires Armv9.2-class cores. Pi 5 is Cortex-A76
(Armv8.2). **SME2/KleidiAI gains do not apply to the Pi**, so do not chase them. Likewise, the
KleidiAI/XNNPACK wins quoted on phones (vivo X300, Pixel 11) are not transferable evidence.

Also note: Ultralytics pins `executorch<1.5` when PyTorch < 2.13, and ExecuTorch export remains
FP32-only. Arten's decision to use Arm's pre-quantised 1.1.0-parity `.pte` is still the only way to
get INT8 through ExecuTorch on a Pi without doing the quantisation work yourself.

---

## 4. Hardware: the Pi 5 got an NPU, and it mostly doesn't matter for one stream

**Raspberry Pi AI HAT+ 2** (Hailo-10H, 40 TOPS INT4, 8 GB onboard, launched 15 Jan 2026, ~$130):

- Vision performance is "broadly equivalent" to the 26 TOPS Hailo-8 for YOLO-class models, per
  Raspberry Pi's own launch post. The extra TOPS are INT4 for LLMs/VLMs.
- Sixfab's same-Pi bench: YOLOv8l NPU throughput 40.0 FPS async, **11.9 FPS end-to-end** with
  decode. Hailo-8L: 15.0 async, 8.7 end-to-end. **Single-stream, the Pi 5's video decode is the
  bottleneck, not the NPU.**
- A hands-on field test: 10,244 frames in 544 s = **18.8 FPS end-to-end** including JPEG read,
  inference, ByteTrack, SQLite; 50.43 ms per valid frame for detect+track; reported **37.76× faster
  than CPU inference**, 19.64× end-to-end, stable at 51–60 °C under 30 min load.
- Edge-LLM-Bench (MobiSys workshop 2026): Hailo-10H does **not** beat the host CPU on token
  throughput for small LLMs, but cuts total system power 51% and frees the CPU.

**Implication for Arten:** an AI HAT would multiply throughput ~3–5× end-to-end, but (a) it costs
more than the Pi itself, (b) it does not run ExecuTorch `.pte` — it runs HEF via HailoRT — so it
fails the brief's runtime requirement, and (c) at 7.5 FPS Arten is already decode-cheap and
compute-bound. Mention it as the scaling path; do not deploy it.

Other 2026 hardware context: Rockchip RK3588 runs nano YOLOs at 21–30 ms; Intel Core Ultra NPUs run
YOLO26n at ~5 ms INT8 (vendor, OpenVINO 2026.2). None of it is ExecuTorch-on-Pi.

---

## 5. Quantisation: PTQ is fine, QAT is the documented recovery

- Systematic 2026 study (Jetson Nano): INT8 PTQ gave up to **3.8× size reduction and 3× latency
  reduction**, at a cost as low as **3.2 mAP**; QAT recovered YOLOv5s to **45.3 mAP, within 0.9 of
  FP32**. It also proposes **mAP/Joule** as the selection metric.
- YOLO26 quantisation is *not uniformly safe*: on Hailo-8L, accuracy retention was 92.3% (n), 88.9%
  (s), **84.0% (m)**, 87.4% (l). The M variant's loss was traced to early convs with peaked
  activations, feature splitters with weight asymmetry, and bimodal softmax in attention; standard
  uniform INT8 plus post-quantisation fine-tuning recovered **<0.5%**. The fix is mixed precision on
  bottleneck layers, with a latency cost.
- Hybrid INT8 on RK3588 (RKNN): ~21 → ~38 FPS detection, ~4× vs CPU, ~1/4 model size.

**For Arten:** the "stem conv and head stay FP32" choice is defensible and now well-supported by the
2026 mixed-precision literature. If asked "why not full INT8?", the answer is the YOLO26-M result —
uniform INT8 can fail structurally, not just noisily, and the detection head is the sensitive part.

---

## 6. The measurement wave — the most important section for this project

2026 produced a visible backlash against edge-AI benchmark theatre. Concrete findings:

- **Requested power modes are not realised treatments.** A 558-run Jetson Orin Nano audit found all
  126 "fast" runs landed in a clock regime determined by the *previous* run's state, not the request;
  the requested EMC caps were never applied. Reliable claims need state reset + read-back, one time
  boundary per metric, and randomised temporal blocks.
- **Thermals dominate.** A phone benchmark saw latency go 85 → 230 ms (2.7×) after 5 minutes of
  sustained inference as the SoC throttled from 32 → 48 °C. iPhone 16 Pro loses ~40% of peak
  throughput within three iterations.
- **Memory bandwidth, not TOPS, governs robustness.** Under memory load, RV1106 slowed 57% and
  RK3568 **270%**; RK3588 stayed under 5%. Multi-core NPU scaling is <10%.
- **The proposed replacement metrics** are warm-device **p99**, **deadline miss rate**, **useful
  goodput** (completed ∧ quality-gated ∧ on time ∧ no forbidden fallback), **joules per useful
  result**, thermal dwell, and sustained retention first-to-last. A five-minute, 1 req/s phase gives
  ~3 samples in the top 1% — too few for a release p99.
- **"Name the silicon" rule:** a speed multiple is not a specification until device, quantisation and
  thermal envelope are stated. The four numbers that decide a deployment: worst-case p99 latency on
  the oldest supported device after 10 min; peak memory headroom under concurrent load; sustained
  thermal behaviour at 20 min; model rollback time.

**Where Arten stands:** better than most hackathon work (model-only timer around `execute`, separate
pipeline timer, p50/p95, warmups, checksum, environment metadata). Still short of the 2026 bar:
no warm p99 over ≥10 minutes, no energy per inference, no thermal trace, and the headline Pi number
is Arm's reference rather than a local measurement. The report already flags the last one — that
honesty is the asset; the missing p99/energy/thermal columns are the cheap upgrade.

---

## 7. Privacy and regulation: the line Arten already sits on

- **EU AI Act Art. 5** prohibitions apply since **2 Feb 2025**: real-time remote biometric
  identification in public spaces for law enforcement (narrow exceptions), emotion recognition at
  work/in education, untargeted face-scraping, biometric categorisation of protected traits.
- **High-risk Annex III duties** were deferred from 2 Aug 2026 to **2 Dec 2027** by the Digital
  Omnibus, **Regulation (EU) 2026/1744** (in force 27 Jul 2026). The Art. 5 prohibitions did **not**
  move.
- Explicitly: **counting people, measuring density, and monitoring restricted areas are not
  biometric identification.** Identity is the line, not observation. Fines for prohibited practice:
  up to €35M or 7% of global turnover.
- De-identification is now a shipping product category: PrivaGuard (YOLO11n + OSNet ReID + CUDA
  pixelation + NVENC, 1080p30, fail-closed); reversible frequency-separation de-identification at
  **29.35 ms / 34.7 FPS** with face similarity reduced to 0.25 while detection recall stayed 100%;
  edge-cloud systems that send only irreversible feature vectors (Information Bottleneck), with
  formal reconstruction-resistance arguments.
- **Federated learning** is the standard privacy-preserving *training* answer for VRU detection
  (PANet, PEGNet, Smart-VRU-Fed: federated models reach 90–95% of centralised mAP50 with 17–20%
  unlabeled data and human-in-the-loop pseudo-labelling).

**For Arten:** the current wording ("privacy-preserving by construction", "no faces, no plates") is
correct and now *legally* meaningful. Strengthen it by pointing at the actual boundary: Art. 5
regulates identification, and Arten computes counts and motion, not identity. Do not claim
"GDPR-compliant" (deployment-specific); claim "designed to stay outside biometric identification".

---

## 8. Applications: near-miss analytics is where counting becomes safety

This is the strongest adjacent body of work for Arten's story.

- **Surrogate safety measures (SSMs)** are the standard: **TTC** (time to collision) and **PET**
  (post-encroachment time). Operating thresholds in the literature: **TTC < 3 s** and **PET < 1 s**
  indicate risky interactions. Both require a trajectory plus a ground-plane transform — usually a
  **homography** from four or more reference points on a fixed camera.
- **2026 near-miss framework** (MDPI *Vehicles*): enhanced YOLOv11 with a small-object head +
  BoT-SORT + **bird's-eye-view** transform, producing a **Near-Miss Risk Index** combining spatial
  proximity, TTC and motion intensity. Real CCTV, heterogeneous road users.
- **CrossRisk** (ASCE 2026): panoramic 3D trajectories + gap acceptance + stopping-sight distance,
  classifying interactions safe/risky/critical; 98.49% classification accuracy on 206 real
  interactions.
- **A 2026 methodological warning that matters:** representing a road user by the **bounding-box
  centre** biases PET by **0.69 s on average**, exceeds 1 s in **16.7%** of conflicts, and
  reclassifies up to **3.6%** of events into a different risk category. Arten's zone co-presence is
  effectively a coarse PET on box centres — so this bias applies directly.
- **Graph-based Indian-crosswalk work** (Accident Analysis & Prevention, Apr 2026) explicitly models
  pedestrian–vehicle interaction at **unsignalised Indian crosswalks** for proactive warnings. That
  is Arten's exact use case, one step further along.

**For Arten:** the "zone co-presence" feature is one homography and one subtraction away from a
real, citable SSM. See §11.

### Regional data: the COCO gap is now quantified

- **BMD-45** (CVPR 2026): 45K CCTV images, 480K boxes, **14 vehicle classes**, 3,679 cameras,
  Bengaluru. Cross-dataset result on 3,000 expert-annotated Bengaluru CCTV images:
  best IDD-trained model (**D-FINE-X**) = **0.46 mAP@0.5:0.95**; best TrafficCAM-trained = 0.39;
  **zero-shot GroundingDINO = 0.13**. Also documents that ego-centric → fixed-camera transfer is
  poor (dashcam 1–2 m vs pole 4–8 m changes aspect ratios and occlusion).
- **IDD / IDD-117K**: 117,099 images, 13 detection classes including `autorickshaw`, `rider`,
  `vehicle fallback`.
- **Indian Road Dataset** (ThirdEye Labs, HF): 645,714 annotated frames, 3.67M detections,
  `autorickshaw` in 80.3% of clips — but **machine-generated labels**, ~33% below 0.5 confidence, so
  it is a pseudo-labelled corpus, not ground truth.
- **CrowdLense** (COMSNETS 2026): Raspberry Pi 4/5, YOLO fine-tuned on IDD, privacy-safe metadata
  over MQTT; benchmarks find **YOLOv11-S + NCNN** the best Pi configuration. Directly comparable
  prior art for Arten — and it validates the "metadata only, no video leaves the node" pattern.

**For Arten:** the honest sentence is now stronger than before: even a state-of-the-art detector
trained on Indian data reaches only ~0.46 mAP on Bengaluru CCTV, and zero-shot open-vocabulary
detection collapses to 0.13. Arten's COCO numbers are therefore not an implementation failure; they
are a documented domain gap with a published benchmark behind it (BMD-45).

### Deployments that already exist

- **CrowdLense** — Pi + IDD-finetuned YOLO + MQTT, serverless backend, citizen-deployed nodes.
- **OpenTrack** (Mendrisio, CH) — low-cost open hardware, on-device classification, NB-IoT summaries,
  GDPR by avoiding biometric capture, SensorThings API.
- **PAVE** (ACM SEC 2025) — edge server tracks pedestrians/vehicles, predicts trajectories, warns
  phones; **~0.9 s** warning before a vehicle reaches a pedestrian; per-stage memory/compute tuning
  gave up to **10×** latency reduction.
- **PAVE and the 2026 de-ID products prove the pattern is deployable**, which raises the bar for
  "novel": a demo that only counts is now prior art. A demo that measures honestly and reports a
  surrogate safety metric is not.

---

## 9. Tracking: ByteTrack is still the edge default, with cheap upgrades

2026 edge tracking benchmarks (Jetson AGX Orin 30 W, YOLO26n TensorRT FP16, p95 dense):

| Tracker | p95 ms | Ceiling | When |
|---|---:|---:|---|
| ByteTrack | **16** | 30 FPS | Static camera, well-separated targets |
| OC-SORT | 17 | 30 FPS | Brief occlusions, re-acquisition |
| BoT-SORT (no ReID) | 64 | 15 FPS | Moving camera (GMC) |
| BoT-SORT + ReID | 79 | 10 FPS | Persistent identity across full occlusion |

- **ByteTraX** (2026): single unified matching threshold + stricter track initiation. **>40% fewer
  ID switches**, HOTA +3.6, IDF1 +5.6, and **>10% faster** than ByteTrack. Drop-in-ish.
- **LITE-BoostTrack** (2026): reuses the detector's internal feature maps as appearance
  descriptors, removing the separate ReID network; HOTA 27.31 vs 25.73 on MOT20, **−37% ID switches**.
- **motcpp**: C++ implementations, 10–100× faster than Python (ByteTrack 1100 FPS single-thread on
  i9). Relevant if the Python tracker becomes the pipeline bottleneck.

**For Arten:** the IoU/velocity tracker is the right complexity for a one-day PoC, and this
literature now gives a defensible citation for *why* (ByteTrack-class motion-only tracking is the
30 FPS, zero-ReID point on the spectrum). Two cheap credibility upgrades: adopt ByteTraX's unified
threshold idea, and publish an ID-switch count on the sample clip so "indicative unique counts" has a
number attached.

---

## 10. Novelty verdict: what is genuinely new vs. commodity

| Claim | Nov 2026 status | Verdict |
|---|---|---|
| "Runs a detector on a Raspberry Pi" | Commodity since ~2022; Pi 5 + AI HAT makes it trivial | Not novel |
| "ExecuTorch `.pte` on Pi 5" | Still uncommon in hackathon submissions, but one-command via Ultralytics export since Oct 2025 | Weakly novel, not technically |
| "INT8 quantised small detector" | Standard; PTQ is a documented 3–3.8× win | Commodity |
| "Full-pipeline latency, not just model" | Better than most, but 2026 practice expects p99 + energy + thermals | Table stakes, partially met |
| "Attributes: direction, dwell, speed, colour" | No direct equivalent in the literature; closest is SSM (TTC/PET), which is stronger | **Mildly novel, under-powered** |
| "Privacy by construction / aggregate only" | Now the legal default; counting ≠ biometric ID | Necessary, not differentiating |
| "Honest about COCO vs India" | Now backed by BMD-45's 0.46 mAP result | **Credible and rare** |
| "Zone co-presence" | Crude PET on box centres; known 0.69 s bias | Novel framing, weak method |
| "NMS-free" | Not available on ExecuTorch (falls back to one-to-many) | Do not claim |
| "Real-time" | 7.46 FPS on the reference; ~20–36 FPS dev host | Do not claim without defining FPS + device |

---

## 11. Gap analysis: claims in this repo that need updating

1. **"YOLO11n INT8 is the only candidate that combines sub-5 MB, >7 FPS, published accuracy and a
   prebuilt, target-verified `.pte`."**
   Still true *within prebuilt Pi 5 ExecuTorch INT8 artifacts*. It is no longer true as a general
   statement: YOLO26n ExecuTorch is 9.4 MB at 144.8 ms with ~+9 mAP. Narrow the sentence to the
   artifact class, and add YOLO26 as the accuracy alternative with its own number.

2. **"Why this is different — it is actually ExecuTorch on the edge."**
   Keep, but reframe: the differentiator is **the measurement contract and the decision**, not the
   runtime. A judge from Arm will know Ultralytics exports ExecuTorch in one command.

3. **"Starts from Arm's verified optimization … 2.23× speedup, 1.56× lower peak memory."**
   Accurate and still the strongest vendor anchor. Add that Arm's card is **1 thread**, and that the
   repo's dev-host numbers are not Pi numbers (already stated — keep it).

4. **"No NMS / end-to-end"** — nowhere claimed today; **keep it that way**. YOLO26's end-to-end
   operators do not lower on ExecuTorch, so the pipeline legitimately needs NMS.

5. **"Indicative unique counts."**
   Now quantifiable: report ID switches and fragmentation on the sample clip, and cite ByteTrack as
   the edge-default operating point.

6. **Attributes as "derived, not learned".**
   Correct, but the literature has a better frame: these are **surrogate safety variables**. Move
   from "direction/speed/colour" to "PET/TTC proxy with stated thresholds" and the work joins a
   citable field instead of inventing a private vocabulary.

---

## 12. Three upgrades, ranked by novelty per hour

### Upgrade A — turn zone co-presence into PET/TTC (highest value)

- Add a 4-point homography (`cv2.getPerspectiveTransform`) mapping the ground plane to metres. The
  repo already supports a manual polygon; this reuses the same four points.
- Compute per-track ground-plane position, then:
  - **PET** = time between user A leaving a conflict point and user B arriving.
  - **TTC** = closing distance / closing speed, for approaching pairs.
- Report counts of interactions with **PET < 1 s** and **TTC < 3 s**, plus the thresholds in the
  output.
- State the known bias: box-centre trajectories shift PET by ~0.69 s on average; use a foot-point
  (bottom-centre) or geometry-derived representative point instead, and say which you used.
- Why it wins: it converts a descriptive feature into a **safety metric with published thresholds**,
  and it directly answers "how does counting help?" with something better than a count.

### Upgrade B — content-aware inference instead of fixed stride

- Albireo (2026) wraps any detector: when per-object Kalman uncertainty is low, skip the detector and
  predict boxes from the filter. Reported **−12.1 to −17.6% energy** with AP@50 within **±1.2 pp**
  on YOLO26x, and a **+0.8 pp** AP gain on the primary config. A fixed skip-2 baseline loses
  **8.6 pp** — the point is that *content-blind* skipping is bad, *uncertainty-gated* skipping is not.
- Arten already has `--stride` and an IoU/velocity tracker. The upgrade is to trigger inference when
  tracker predicted-vs-measured IoU drops below a threshold, and to report the frames skipped and
  the latency/energy saved.
- This is a genuine 2026 systems contribution that fits in an afternoon because the tracker exists.

### Upgrade C — meet the 2026 measurement bar

- Run **10+ minutes** (or the full clip repeated) and report **warm p99**, not just p50/p95.
- Log temperature and CPU frequency (`vcgencmd measure_temp`, `vcgencmd get_throttled`) during the
  run; report throttled minutes.
- If a USB-C power meter is available, report **J/inference** and total Wh for the run. If not, say
  "energy not measured" — the 2026 critique is about *conflating* metrics, not about lacking a meter.
- Report **deadline misses** (frames that exceeded 1/7.46 s at the reference point) and **useful
  goodput** (frames detected, tracked and annotated within budget).
- Why it wins: it is the difference between "we measured" and "we measured the thing reviewers now
  ask for", and it costs a script flag, not a redesign.

### Optional, only if there is spare time

- **Auto-rickshaw**: either a YOLOE-26 text prompt (if it exports) or a one-class fine-tune on IDD.
  If neither is feasible, cite BMD-45 (0.13 zero-shot vs 0.46 trained) and present it as a known,
  quantified gap rather than an omission.
- **ByteTraX threshold rule**: one unified association threshold, stricter track initiation; fewer ID
  switches at no extra cost.

---

## 13. Do not chase (for this brief)

| Idea | Why not |
|---|---|
| **AI HAT / Hailo-10H** | Not ExecuTorch (HEF/HailoRT); ~$130; single-stream is decode-bound at ~12 FPS; fails the runtime requirement |
| **SME2 / KleidiAI** | Needs Armv9.2; Pi 5 is Cortex-A76 (Armv8.2). Zero gain |
| **RF-DETR / D-FINE on Pi** | 10–30× the parameters of YOLO26n; no ExecuTorch INT8 path on Arm; would not hit 7 FPS |
| **Edge VLM cascade** (Qwen2-VL / Reka Edge / Moondream) | 1.7–7B models, GB-scale memory, seconds per image; needs an NPU or GPU. Cascade is the right *architecture* claim, not the right *build* |
| **TinyVLM-style MCU zero-shot** | 892 KB flash / 285 KB RAM, 26 FPS on STM32H7 — impressive, but the brief is a Pi and ExecuTorch |
| **Federated learning** | No multi-node fleet, no training budget, no privacy gain over "no data leaves the device" for a single-node PoC |
| **Re-ID / face / plate** | Crosses the AI Act identification line and contradicts the aggregate-only design |

---

## 14. Sources

Primary or vendor-documented:

1. Ultralytics, *Raspberry Pi guide / YOLO26 benchmarks* — <https://docs.ultralytics.com/guides/raspberry-pi>
2. Ultralytics, *ExecuTorch integration* (FP32-only, pinned `executorch<1.5`) — <https://docs.ultralytics.com/integrations/executorch>
3. Ultralytics, *End-to-end NMS-free detection* (ExecuTorch falls back to one-to-many) — <https://docs.ultralytics.com/guides/end2end-detection>
4. Ultralytics, *YOLOE* — <https://docs.ultralytics.com/models/yoloe>
5. Ultralytics, *YOLO26 paper* — <https://arxiv.org/html/2606.03748>
6. Roboflow, *RF-DETR benchmarks / edge deployment* — <https://rfdetr.roboflow.com/> and <https://roboflow.com/blog/rf-detr-for-the-edge>
7. Roboflow, *RF-DETR ICLR 2026 paper* — <https://proceedings.iclr.cc/paper_files/paper/2026/file/5bed8703db85ab27dc32f6a42f8fbdb6-Paper-Conference.pdf>
8. PyTorch, *Accelerating on-device ML with ExecuTorch and Arm SME2* — <https://pytorch.org/blog/accelerating-on-device-ml-inference-with-executorch-and-arm-sme2>
9. ExecuTorch releases (1.5.1 latest) — <https://github.com/pytorch/executorch/releases>
10. Arm, *Bringing ExecuTorch to Cortex-M CPUs* (Oct 2026) — <https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/bringing-executorch-to-arm-cortex-m-cpus>
11. Intel, *ExecuTorch + OpenVINO/NNCF in 2026* — <https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/ExecuTorch-with-OpenVINO-Backend-in-2026-New-Capabilities-and/post/1743166>
12. Raspberry Pi, *AI HAT+ 2 launch* — <https://www.raspberrypi.com/news/introducing-the-raspberry-pi-ai-hat-plus-2-generative-ai-on-raspberry-pi-5>
13. Raspberry Pi, *AI HAT documentation* — <https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html>
14. Sixfab, *Raspberry Pi 5 AI HAT benchmark (DEEPX vs Hailo)* — <https://sixfab.com/blog/raspberry-pi-5-ai-hat-benchmark-deepx-vs-hailo/>
15. Marius Quabeck, *AI HAT+ 2 hands-on* (18.8 FPS end-to-end, 37.76×) — <https://marius.blog/raspberry-pi-ai-hat-plus-2/>
16. *Edge-LLM-Bench* (MobiSys workshop 2026) — <https://dl.acm.org/doi/10.1145/3812836.3814998>

Independent / peer-reviewed:

17. *Object detection on low-compute edge SoCs* — Scientific Reports, Jan 2026 — <https://www.nature.com/articles/s41598-026-36862-y>
18. *A Comprehensive Evaluation of DL Object Detection Models on Heterogeneous Edge Devices* — <https://arxiv.org/abs/2409.16808>
19. *Optimization of Real-Time Object Detection on Edge Devices via Lightweight CNN Architectures and Quantization* — <https://iieta.org/journals/isi/paper/10.18280/isi.310530>
20. *Quantization Noise Analysis: YOLO26 on Hailo-8L* — <https://danieldubinsky.github.io/personal-site/case-studies/quantization_noise_analysis_yolo26/>
21. *Measurement Boundaries … in Edge-AI Energy Benchmarking* — Electronics, Jul 2026 — <https://www.mdpi.com/2079-9292/15/15/3372>
22. *LLM Inference at the Edge: sustained load / thermals* — <https://arxiv.org/html/2603.23640v2>
23. *Albireo: Adaptive, Energy-Efficient Inference for Video Object Detection on the Edge* — <https://arxiv.org/html/2609.29648>
24. *BMD-45: Large-Scale CCTV Vehicle Detection Dataset (Bengaluru)* — CVPR 2026 — <https://arxiv.org/html/2604.24419v1>
25. *Real-Time Detection of Near-Miss Events … BEV Mapping* — Vehicles, Apr 2026 — <https://www.mdpi.com/2673-7590/6/2/80>
26. *A novel method for determining VRU trajectories* (PET bias, 0.69 s) — <https://pubmed.ncbi.nlm.nih.gov/42105696>
27. *Graph-based spatio-temporal framework for safety-critical pedestrian–vehicle interactions at unsignalised crosswalks* — AAP, Apr 2026 — <https://www.sciencedirect.com/science/article/abs/pii/S0001457526000187>
28. *CrowdLense* — COMSNETS 2026 — <https://doi.org/10.1109/comsnets67989.2026.11418200>
29. *OpenTrack: a sensor for monitoring the usage of territory* — ISPRS 2025 — <https://isprs-archives.copernicus.org/articles/XLVIII-4-W13-2025/239/2025/>
30. *PAVE: Real-Time Video Analytics for Urban Safety* — ACM SEC 2025 — <https://doi.org/10.1145/3769102.3770618>
31. *ByteTraX: Enhancing the ByteTrack Architecture* — <https://arxiv.org/html/2609.37801>
32. *LITE-BoostTrack* — Kinetik, May 2026 — <https://kinetik.umm.ac.id/index.php/kinetik/article/view/2478>
33. *TinyVLM: Zero-Shot Object Detection on Microcontrollers* — <https://arxiv.org/html/2603.00136v1>
34. *IDD / IDD-117K* — <https://idd.insaan.iiit.ac.in/dataset/details>
35. *Edge-based AI: hardware/software evolution and future trends* — <https://www.sciencedirect.com/science/article/pii/S0952197626008079>

Regulation / privacy:

36. EU AI Act Art. 5 — <https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-5>
37. *AI cameras at events: what the EU AI Act allows* (counting ≠ identification; Omnibus dates) — <https://www.movetos.com/en/knowledge/ai-act-video-analytics-events/>
38. FPF, *Red Lines under the EU AI Act: real-time RBI* — <https://fpf.org/blog/red-lines-under-the-eu-ai-act-restricting-real-time-remote-biometric-identification-systems-for-law-enforcement-purposes>
39. *Privacy-Native Edge Video Analytics for Regulated Rail Environments* (face-free continuity) — <https://dhi-tech.com/labs/papers/dhi-privacy-whitepaper.pdf>
40. *Frequency-separation reversible de-identification* (29.35 ms / 34.7 FPS) — <https://scholarworks.dongguk.edu/item/21015a55-a0cb-4d64-a358-694996080a5d>
