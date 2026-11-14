# The full opportunity map: what could make Arten stand out (Nov 2026)

**Prepared:** 14 November 2026
**Scope:** deliberately wide. Edge AI detection/recognition across sensing, models, learning,
compression, systems, uncertainty, privacy, hardware, governance, and adjacent application domains.
**Question:** what novelty can be *added* to a small offline pedestrian/vehicle counting PoC so it
stands out to judges, to the literature, and to a municipality?
**How to read the scores:** Novelty = distance from what a typical submission does. Effort = hours
for one competent person with the repo already working. Demo = how visible it is in a 2-minute pitch.
All three are 1–5 (5 = best/easiest).

> Vendor claims are labelled as vendor claims. Benchmark numbers are quoted with their hardware.
> Nothing here has been reproduced on the Pi in this repo.

---

## 0. The one thing to understand first

In 2026 the *technical* bar for "object detection on an edge device" has collapsed. YOLO26 exports to
ExecuTorch in one command; SAM 3 labels your dataset; a 270M model does on-device function calling.
So novelty has moved to five places:

1. **The question you ask of the pixels** (counts → conflicts → interventions).
2. **The guarantee you attach to the answer** (conformal coverage, calibrated uncertainty).
3. **The modality you sense with** (event, radar, audio, multi-camera).
4. **The learning loop you close** (self-supervised, synthetic, federated, test-time adaptation).
5. **The system properties you can prove** (energy, p99, privacy, no-network, governance).

A project that is strong on 2–3 of these is memorable. A project that is strong on all five is a
workshop paper. Most submissions are strong on none.

---

## 1. The 2026 landscape in 14 bullets

1. **NMS-free detectors** (YOLO26) and **real-time DETRs** (RF-DETR at 60.1 COCO AP, D-FINE, DEIM)
   moved the accuracy frontier. Not all of it reaches a Pi CPU, but the ceiling moved.
2. **Open-vocabulary** went real-time (YOLOE/YOLOE-26, YOLO-World v2, OV-DEIM) and **promptable
   segmentation** went concept-level (SAM 3/3.1: 47.0 AP on LVIS zero-shot, but 3.45 GB).
3. **Feed-forward 3D** arrived: VGGT, Depth Anything 3 (22M–1.1B params), Pi3, MapAnything. Single
   pass → camera poses, depth, point maps, 3DGS. Small DA3 variants are 22–110M params.
4. **Event cameras** became a real ITS sensor: 120 dB dynamic range, <10 mW sensing, microsecond
   timestamps; a 2026 surveillance-style pedestrian/cyclist/car event dataset exists.
5. **Neuromorphic + sparse** detection is deployable: 66 mW / 138 ms / 7.2 FPS on Akida; fully sparse
   voxel detection at 83.4 mAP@50 with 0.23% active voxels.
6. **RF sensing** matured: mmWave counting at 98.86% (0–3 people), 802.11bf passive counting at
   98.57% (up to 4), crowd-flow analytics from commodity MIMO radar.
7. **Self-supervised and synthetic** pipelines became the default answer to label scarcity:
   +11.4% small-object AP from synthetic data; anomaly detection from normal-only data at
   94.2% AUROC and 4.0 mJ/inference on an in-sensor IMX500.
8. **Federated learning** became practical at camera scale: +7.04% accuracy and 80% communication
   savings (FedFit), single-round encrypted aggregation with ~70% energy savings (FedHENet).
9. **Continual test-time adaptation** is now a detection topic, not just classification: +3.2 mAP on
   Cityscapes→Cityscapes-C with 20% efficiency gain (AMROD); memory-efficient on-device variants.
10. **Uncertainty got rigorous**: conformal prediction for boxes and class labels, with finite-sample
    coverage, now benchmarked on YOLO11 and RT-DETRv2, and cheap enough for edge use.
11. **On-device agents** became small and real: 270M function calling (FunctionGemma), 2.6B tool-use
    models at 30 tok/s on a phone, on-device RAG at 25% of the storage (MiniRAG).
12. **Hardware diversified**: RISC-V NPUs (Coral 1 TOPS @10 mW, Synaptics Astra SR200 400 GOPS,
    Axelera Europa 629 TOPS @30 W), analog in-memory (TetraMem 12nm, 64M INT4 on-chip, 50–80 TOPS/W
    target), Jetson Thor (128 GB, 130 W, 27B agent at 52 tok/s).
13. **Measurement got audited**: requested power modes not realised, 2.7× thermal latency swings,
    270% memory-contention slowdowns. Warm p99, joules per useful result, and thermal state are the
    new credibility markers.
14. **Governance became a product feature**: AI Act Art. 5 live since Feb 2025, high-risk duties
    deferred to Dec 2027, ISO/IEC 42001 (38 controls across 9 areas) becoming the procurement
    language for AI management systems.

---

## 2. The innovation stack — 14 layers

Each layer lists what is genuinely new, the ideas it unlocks, and how well it fits Arten.

### L1. Sensing — change the input, not the model

| Idea | What is new | Fit | Effort | Novelty |
|---|---|---|---|---|
| **Event-camera crossing sensor** | 120 dB vs ~60 dB for a normal camera; <10 mW sensing; microsecond timestamps. A 2026 ITS dataset captures pedestrian/cyclist/car from a 4 m pole, 103° FOV, at three sensitivity levels | Directly addresses glare, headlights, deep shadow, dusk — the exact failure modes of a school crossing at 6 pm | 5 (hardware) | 5 |
| **mmWave radar counting** | 60 GHz FMCW: 98.86% for 0–3 people; 802.11bf passive sensing: 98.57% up to 4; radar works in darkness and is intrinsically less identifying than video | Privacy-first fallback; "no image ever formed" is a stronger claim than "we don't recognise faces" | 5 | 5 |
| **WiFi CSI occupancy/flow** | 802.11bf standardises sensing; CSI classifiers reach high accuracy indoors but drop 13–54% across unseen rooms | Not outdoor-robust yet; good for a school corridor or gate | 4 | 4 |
| **Two-camera geometry** | Two cheap phones + Depth Anything 3 / classical stereo → metric trajectories and real km/h without a single-camera homography | Turns "35.6 px/s" into "18 km/h", which a ward officer understands | 3 | 4 |
| **Audio cue channel** | Acoustic event detection (horn, screech, brake) as a near-miss corroboration | Adds a modality with no privacy cost; hard to validate in a day | 4 | 3 |
| **Thermal / NIR** | Robust to low light, cheap for a fixed pole, but lower resolution and no colour | Night-time school crossings | 3 | 3 |

**Verdict:** the event camera is the single most *interesting* hardware idea and the least feasible
today. It belongs in the pitch as a named next step with a concrete reason (dynamic range), not as a
demo.

### L2. Perception — models and tasks

| Idea | Why it stands out | Effort | Novelty |
|---|---|---|---|
| **YOLO26n instead of / alongside YOLO11n** | +9 mAP on the same Pi 5 (vendor, FP32 ExecuTorch 144.8 ms vs your 134.1 ms INT8 1-thread). The accuracy gap is bigger than the latency gap | 2 | 2 |
| **YOLOE-26 text prompts** | `auto-rickshaw`, `hand cart`, `cattle`, `wheelchair`, `pram` — the COCO gap becomes a prompt, not a limitation | 3 | 4 |
| **SAM 3 as a labelling teacher, tiny student on the Pi** | Use SAM 3 offline to label 200 frames, distill/fine-tune a 2–5M detector, deploy the student. The *pipeline* is the contribution: "foundation model in the loop, no foundation model on the device" | 4 | 4 |
| **Pose-conditioned counting** | Pose tells you *intent*: head orientation, crossing direction, gait. "42 people crossed" → "31 crossed, 11 waited, 4 stepped back" | 3 | 4 |
| **Depth-based scale** | Monocular depth (DA3-Small, 22M) or stereo gives distance → metric speed, metric zone, metric near-miss | 4 | 4 |
| **Vulnerable-user subclasses** | Segmentation or open-vocab prompts for pram, wheelchair, child, dog — categories COCO lacks and a school-zone audit needs | 3 | 4 |
| **Crop-level small VLM** | SmolVLM-500M / LFM2.5-VL-450M / Moondream on a sampled crop: "bus stopped at kerb", "person pushing cart". 1 crop every 2 s, not every frame | 4 | 4 |
| **Anomaly detection instead of class lists** | Train only on "normal" crossing behaviour (GLASS/PatchCore-style); flag the unusual. No need to enumerate every hazard | 4 | 4 |

**Verdict:** pose + open-vocab is the best accuracy-per-hour combination. The SAM-3-teacher pipeline
is the best *story*.

### L3. Temporal reasoning — from frames to behaviour

| Idea | Why it stands out | Effort | Novelty |
|---|---|---|---|
| **PET / TTC near-miss index** | The standard surrogate safety measures; thresholds TTC < 3 s, PET < 1 s are citable | 2 | 4 |
| **Bias-corrected trajectories** | 2026 work shows box-centre trajectories bias PET by 0.69 s on average (>1 s in 16.7% of conflicts). Use a foot-point or geometry-derived representative point and say so | 2 | 5 |
| **Crossing compliance** | Signal phase + crossing location → aggregate "crossed against signal" rate, no identity | 3 | 3 |
| **Gap acceptance** | At unsignalised crossings, measure the time gap pedestrians accept. A real traffic-engineering output | 3 | 4 |
| **Conflict hotspot map** | Aggregate pairwise proximity into a heatmap over the ground plane; "this 4 m² accounts for 60% of conflicts" | 2 | 4 |
| **Behavioural states** | waiting / crossing / dawdling / running, from track kinematics | 3 | 3 |
| **Digital twin what-if** | Export trajectories to SUMO; simulate "speed hump", "shorter crossing", "signal retiming"; compare delay/conflict proxies | 5 | 5 |
| **World-model prediction** | Predict 2 s ahead, flag predicted conflicts before they happen | 5 | 5 |

**Verdict:** PET/TTC + bias correction is the highest-value addition in the entire document. It is
hours of work and it converts a counter into a safety instrument.

### L4. Learning — closing loops with little data

| Idea | Evidence | Fit | Effort |
|---|---|---|---|
| **Self-supervised pretraining on site footage** | Parking-occupancy study: two-stage SSL (generic → target) then supervised fine-tune gave the best accuracy; 0.23 s/spot on a Pi 5 | 3 | 4 |
| **Synthetic data for small objects** | +11.4% APs from Unity/CAD synthetic data and proportional class balancing | 3 | 4 |
| **Dataset distillation** | OD3 compresses COCO to 1% and beats the prior method by >14% mAP50 at that ratio | 2 | 5 |
| **Federated multi-site learning** | FedFit +7.04% over next-best with 80% comm savings; FedHENet single-round, ~70% energy savings; camera FL 67% uplink reduction | 3 | 5 |
| **Continual test-time adaptation** | AMROD +3.2 mAP and 20% efficiency on CTTA detection; BOSA memory-efficient; AdaCoTTA +4% on minor classes | 4 | 5 |
| **Active learning / HITL** | Smart-VRU-Fed: federated + pseudo-labels reaches 90–95% of centralised mAP; HITL prevents long-term drift | 3 | 4 |
| **Normal-only anomaly learning** | TinyGLASS: 94.2% image AUROC from defect-free data, 20 FPS, 4.0 mJ/inference on IMX500 | 3 | 4 |

**Verdict:** CTTA and federated learning are the two "system grows over time" stories. Both are too
much for a one-day build, both are excellent "phase 2" slides. The self-supervised + synthetic
combination is feasible as an offline experiment if you have unlabeled frames.

### L5. Compression — accuracy per joule

| Idea | Evidence | Effort | Standout? |
|---|---|---|---|
| **Per-layer mixed precision** | 2026 YOLO26-on-Hailo study: uniform INT8 retained only 84.0% on the M variant; the cause was structural (early convs, splitters, attention softmax), and post-quantisation fine-tuning recovered <0.5%. The fix is mixed precision on bottlenecks | 4 | Moderate |
| **QAT for head + stem** | PTQ cost ~3.2 mAP; QAT recovered to within 0.9 of FP32 in a 2026 study | 4 | Moderate |
| **Structured pruning + recovery** | Compression studies now demand matched-teacher controls and native-scale baselines, not raw pruning claims | 3 | Low |
| **Early exit** | Naive early-exit architectures cost 133–145% FLOPs because all exit modules execute; UniExit routes before exit modules and keeps FLOPs near 100% | 5 | Low on Pi CPU |
| **Content-aware inference** | Albireo: ±1.2 pp AP@50 with 12–18% energy reduction; blind skip-2 loses 8.6 pp | 3 | **High** |
| **Distilled site-specific model** | OD3-style 1% dataset → fine-tune in minutes at a new site | 5 | High as a concept |

**Verdict:** content-aware inference is the only compression idea that is both cheap and visible.
"Blind frame skipping loses accuracy; uncertainty-gated skipping doesn't" is a great one-liner.

### L6. Systems — where engineering credibility lives

| Idea | Why it stands out | Effort | Novelty |
|---|---|---|---|
| **C++ ExecuTorch deployment (1.5 SDK in the wheel)** | Most hackathon projects ship Python. A C++ runner with no interpreter drops memory and start-up latency, and is what an actual device vendor would do | 4 | 4 |
| **Warm p99 + deadline miss rate** | The 2026 measurement bar; almost nobody reports it | 1 | 4 |
| **Joules per useful result** | Energy per accepted, on-time, quality-gated inference; requires only a USB-C meter | 2 | 4 |
| **Thermal trace + throttled minutes** | `vcgencmd get_throttled` during a 10-minute run | 1 | 3 |
| **Adaptive scheduling** | Deadline-aware model/exit/batch choice (EdgeServing) — overkill for one model, but the *idea* of a latency budget is cheap | 4 | 3 |
| **Split computing** | 73–99% latency and 40–95% energy reductions vs fully local, but needs a server; contradicts "no network" | 4 | 2 |
| **Multi-stream on one Pi** | Two cameras, staggered inference; throughput vs per-stream latency curve | 3 | 3 |
| **Model rollback / A-B** | Two `.pte` files, checksum, switch on failure — a real deployment property | 2 | 3 |
| **No-network proof** | `strace`/`tcpdump` capture showing zero outbound sockets during a run | 1 | 4 |

**Verdict:** the measurement trio (p99, joules, thermal) plus the C++ runner is the most credible
"we are not a demo" signal available in a day.

### L7. Uncertainty — the most under-used differentiator

| Idea | Evidence | Effort | Novelty |
|---|---|---|---|
| **Conformal box intervals** | Scaled CP: up to 19% higher IoU and 39% lower interval score than unscaled, with coverage preserved; Bonferroni correction for box-level guarantees | 3 | 5 |
| **Conformal class sets** | Inductive CP on YOLO11 and RT-DETRv2 achieves nominal 90/95/99% coverage with empty sets <0.1% | 3 | 5 |
| **Calibrated counts** | Report "17 cars [15, 19] at 90% coverage" instead of a point count | 2 | 5 |
| **OOD / "I don't know"** | Energy score or Mahalanobis distance on features; flag night/rain frames outside calibration | 2 | 4 |
| **Reliability diagram** | Temperature scaling + ECE; cheap and rigorous | 2 | 3 |
| **Grad-CAM on detections** | The wildlife two-stage paper uses it for trust and failure analysis; also great for judges | 2 | 3 |
| **Uncertainty-aware tracking** | Gate track updates by predicted variance; skip inference when confident (Albireo) | 3 | 4 |

**Verdict:** conformal prediction is the single best "nobody else will have this" feature that is
also cheap. It turns "honest about accuracy" into "mathematically guaranteed coverage".

### L8. Privacy & security — from claim to proof

| Idea | Why it matters | Effort | Novelty |
|---|---|---|---|
| **Fail-closed face/plate pixelation on saved frames** | PrivaGuard pattern: detect, store coordinates, then pixelate before writing; fail-closed masking | 2 | 3 |
| **Retention policy in code** | Auto-purge evidence after N days; audit log of writes/deletes (ISO 42001 A.6/A.8 evidence) | 2 | 4 |
| **No-network proof** | See L6 | 1 | 4 |
| **DP aggregates for multi-site** | Add calibrated noise to shared counts; differential privacy, not just aggregation | 3 | 3 |
| **FHE for RF sensing** | mmFHE runs the whole mmWave DSP+ML pipeline under FHE (84.5% gesture accuracy, 1.2–5.8 s latency) | 5 | 5 |
| **CHERI / memory-safe runtime** | Codasip ran a Hailo-10H inference stack under CHERI with ~100 lines changed | 5 | 3 |
| **Model watermark / provenance** | C2PA-style signed outputs; overkill for a PoC | 4 | 2 |

**Verdict:** pixelation + retention + no-network proof is a half-day of work that makes the privacy
claim *demonstrable* rather than rhetorical.

### L9. Output & decision — what the user actually gets

| Idea | Why it stands out | Effort | Novelty |
|---|---|---|---|
| **Counts → conflicts → recommendation** | "This crossing has 3.1 PET<1 s events per 100 pedestrians at 08:15; a crossing guard between 08:00–08:45 would cover 78% of them" | 2 | 5 |
| **On-device natural-language ward report** | 270M function-calling models exist; LFM2.5-2.6B runs at 30 tok/s on a phone. Even a template + slot-filling version is compelling | 3 | 4 |
| **Kannada + English output** | A Bengaluru ward officer reads Kannada. Almost no hackathon project localises the *output* | 2 | 4 |
| **Digital twin what-if** | See L3 | 5 | 5 |
| **JSON schema / open API** | Procurement-ready ingest format; cheap | 1 | 3 |
| **Time-of-day heatmap** | Morning vs afternoon flow; "the 08:15 peak is 3.4× the 14:00 flow" | 1 | 3 |
| **Cost per 1,000 crossings** | ₹/1k observations; the metric a buyer actually compares | 1 | 4 |

**Verdict:** the report generator (even template-based) + Kannada + cost-per-observation is a
disproportionate win because it changes the audience from "CV judges" to "the person who decides".

### L10. Deployment — from demo to installation

| Idea | Why it stands out | Effort |
|---|---|---|
| **10-minute install kit** | Printed 4-point calibration sheet, one-page procedure, auto-generated homography | 3 |
| **Multi-node mesh** | 3 cheap nodes, summaries over LoRaWAN/NB-IoT (OpenTrack pattern) | 5 |
| **Battery/solar duty cycling** | Syntiant runs a streaming wake-word benchmark at 3.3% duty cycle; camera traps last months | 5 |
| **OTA update + rollback** | LAN update with checksum and fallback `.pte` | 3 |
| **Weatherproof enclosure + thermal** | Field credibility; also a p99 story | 3 |
| **Remote health telemetry** | Heartbeat + inference latency + temperature, no imagery | 3 |

**Verdict:** the install kit is the cheapest credibility item: it proves the system is meant to be
used by a non-engineer.

### L11. Hardware — the frontier, and why most of it is not for this brief

| Platform | What it offers | Fit |
|---|---|---|
| **Raspberry Pi AI HAT+ 2 (Hailo-10H)** | 40 TOPS INT4, 8 GB, 18.8 FPS end-to-end in a real field test, 37.76× CPU inference | Not ExecuTorch; ~$130; decode-bound single stream |
| **Coral NPU (RISC-V)** | 1 TOPS at ~10 mW average; open RTL; IREE/MLIR compiler; Synaptics Astra SR200 at 400 GOPS | Future MCU-class sensor node |
| **Axelera Europa** | 629 TOPS at ~30 W, 16 RISC-V vector cores, 200 GB/s | Multi-camera edge server |
| **Jetson AGX Thor** | 128 GB, 130 W, 27B agent at 52 tok/s, VLA at 41.5 tok/s | Out of budget, but the "scale-up" slide |
| **Analog in-memory (TetraMem MLX300)** | 12nm, 64M INT4 on-chip, 50–80 TOPS/W target | 2027+ |
| **Neuromorphic (Akida)** | 66 mW, 138 ms, 7.2 FPS, 83% sparsity | Event-camera future |
| **Pi Zero 2W smart cane** | $88, 330 ms, 2.8 W, F1 0.82 | Proof the low end is viable |

**Verdict:** mention the ladder, deploy on the Pi. The interesting claim is not "we could use an
NPU"; it is "we measured what the CPU actually delivers and here is the headroom".

### L12. Governance — the municipal unlock

| Artifact | What it is | Effort | Novelty |
|---|---|---|---|
| **AI Act classification memo** | Why counting is not Art. 5 biometric identification; why a decision-support tool is likely outside Annex III "safety component" — *but document the reasoning, this is not legal advice* | 2 | 4 |
| **DPIA template** | Purpose, necessity, proportionality, retention, data subject rights, for a street-counting deployment | 2 | 4 |
| **ISO/IEC 42001 mini-pack** | Scope, AI policy, risk assessment, impact assessment, Statement of Applicability against the 38 Annex A controls, model card, data sheet | 3 | 5 |
| **Model card + datasheet** | Already partly done in this repo; extend with intended use, limitations, calibration data | 1 | 3 |
| **Incident reporting channel** | A named contact and a form; required by 42001 A.8 | 1 | 3 |

**Verdict:** a 6-page governance pack is unusual at a hackathon and directly addresses why
municipalities stall on AI pilots. It is also the most "adult" artifact you can produce.

### L13. Interaction — the human loop

| Idea | Why it stands out | Effort |
|---|---|---|
| **"This is wrong" correction capture** | Saves the crop + timestamp to a review queue; feeds active learning | 2 |
| **Community labelling session** | 20 residents label 20 frames each; creates local ground truth and buy-in | 3 |
| **Active learning loop** | Flag low-confidence / OOD frames for labelling; retrain weekly | 4 |
| **Human-in-the-loop review UI** | Local-only HTML page; no server, no cloud | 3 |

### L14. Ecosystem — what outlives the hackathon

| Idea | Why it matters | Effort |
|---|---|---|
| **Release 200–500 annotated Bengaluru crossing frames** | A public dataset is a citation magnet and a durable contribution | 3 |
| **Publish the measurement protocol** | Warm p99, joules, thermal, conformal coverage — a reusable harness | 2 |
| **Leaderboard for Indian crossing detection** | Nobody owns this niche; BMD-45 is CCTV vehicles, not crossings | 5 |
| **Procurement spec sheet** | "₹ per 1,000 crossings, 7.5 FPS, 4.7 MB, no network, 90% coverage guarantee" | 2 |

---

## 3. The idea catalogue, scored

Sorted by (novelty × demo) / effort. 1 = low, 5 = high.

| # | Idea | Layer | Novelty | Demo | Effort | Score |
|---|---|---|---:|---:|---:|---:|
| 1 | PET/TTC near-miss index with thresholds | L3 | 4 | 5 | 2 | **10.0** |
| 2 | Conformal count/box intervals | L7 | 5 | 5 | 3 | **8.3** |
| 3 | Foot-point trajectory bias correction | L3 | 5 | 3 | 2 | **7.5** |
| 4 | On-device ward report, English + Kannada | L9 | 4 | 5 | 3 | **6.7** |
| 5 | OOD / "I don't know" flag | L7 | 4 | 4 | 2 | **8.0** |
| 6 | Pixelation + retention + no-network proof | L8 | 4 | 4 | 2 | **8.0** |
| 7 | Warm p99 + joules + thermal | L6 | 4 | 3 | 1 | **12.0** |
| 8 | Uncertainty-gated adaptive inference | L5 | 5 | 4 | 3 | **6.7** |
| 9 | Cost per 1,000 crossings | L9 | 4 | 4 | 1 | **16.0** |
| 10 | Pose-conditioned intent counting | L2 | 4 | 5 | 3 | **6.7** |
| 11 | YOLOE open-vocab rickshaw prompt | L2 | 4 | 5 | 3 | **6.7** |
| 12 | Two-camera metric speed | L1 | 4 | 4 | 3 | **5.3** |
| 13 | C++ ExecuTorch runner | L6 | 4 | 2 | 4 | **2.0** |
| 14 | SAM 3 teacher → tiny student | L2 | 5 | 4 | 4 | **5.0** |
| 15 | Digital twin what-if (SUMO) | L3 | 5 | 5 | 5 | **5.0** |
| 16 | Event-camera pilot | L1 | 5 | 4 | 5 | **4.0** |
| 17 | mmWave radar fallback | L1 | 5 | 4 | 5 | **4.0** |
| 18 | On-device SLM report generator | L9 | 4 | 5 | 3 | **6.7** |
| 19 | Crossing compliance rate | L3 | 3 | 4 | 3 | **4.0** |
| 20 | Gap acceptance distribution | L3 | 4 | 3 | 3 | **4.0** |
| 21 | Conflict hotspot map | L3 | 4 | 4 | 2 | **8.0** |
| 22 | Normal-only anomaly detection | L4 | 4 | 4 | 4 | **4.0** |
| 23 | CTTA site adaptation | L4 | 4 | 3 | 5 | **2.4** |
| 24 | Federated multi-site learning | L4 | 4 | 3 | 5 | **2.4** |
| 25 | Self-supervised site pretraining | L4 | 3 | 2 | 4 | **1.5** |
| 26 | Synthetic small-object augmentation | L4 | 3 | 3 | 4 | **2.3** |
| 27 | Dataset distillation for site tuning | L5 | 4 | 2 | 5 | **1.6** |
| 28 | Mixed-precision per-layer quantisation | L5 | 3 | 2 | 4 | **1.5** |
| 29 | QAT for head and stem | L5 | 3 | 2 | 4 | **1.5** |
| 30 | 10-minute install kit | L10 | 3 | 4 | 3 | **4.0** |
| 31 | Multi-node mesh | L10 | 3 | 3 | 5 | **1.8** |
| 32 | Battery/solar duty cycling | L10 | 2 | 3 | 5 | **1.2** |
| 33 | OTA + rollback | L10 | 3 | 2 | 3 | **2.0** |
| 34 | AI Act classification memo | L12 | 4 | 3 | 2 | **6.0** |
| 35 | DPIA template | L12 | 3 | 2 | 2 | **3.0** |
| 36 | ISO 42001 mini-pack | L12 | 5 | 2 | 3 | **3.3** |
| 37 | Correction capture + active learning | L13 | 3 | 4 | 3 | **4.0** |
| 38 | Community labelling session | L13 | 3 | 4 | 3 | **4.0** |
| 39 | Public annotated dataset release | L14 | 4 | 2 | 3 | **2.7** |
| 40 | Measurement protocol release | L14 | 4 | 2 | 2 | **4.0** |
| 41 | Leaderboard for Indian crossings | L14 | 5 | 2 | 5 | **2.0** |
| 42 | Small VLM crop captioning | L2 | 4 | 5 | 4 | **5.0** |
| 43 | Audio near-miss cue | L1 | 3 | 3 | 4 | **2.3** |
| 44 | FHE RF sensing | L8 | 5 | 1 | 5 | **1.0** |
| 45 | RISC-V MCU node | L11 | 4 | 2 | 5 | **1.6** |
| 46 | In-memory compute | L11 | 4 | 1 | 5 | **0.8** |
| 47 | Neuromorphic event node | L11 | 5 | 1 | 5 | **1.0** |
| 48 | World-model prediction | L3 | 5 | 4 | 5 | **4.0** |
| 49 | Split computing | L6 | 2 | 3 | 4 | **1.5** |
| 50 | Grad-CAM explanations | L7 | 2 | 3 | 2 | **3.0** |

---

## 4. Top 10 standout bets (ranked by what to do *now*)

### #1 — PET/TTC with foot-point correction (hours)

**Claim:** "We don't just count people; we estimate how close they came to being hit, with the
standard surrogate safety metrics and a correction for a known 0.69 s trajectory bias."

**Build:** 4-point homography → ground-plane tracks → pairwise TTC and PET → counts of PET<1 s and
TTC<3 s per 100 pedestrians → per-hour breakdown. Use the bottom-centre (foot point) or a
geometry-derived representative point, not the box centre. Print the method and thresholds in the
report.

**Why it wins:** it is the only idea here that changes the *question* rather than the *implementation*.
It is citable, it maps to a real decision (crossing guard, speed hump), and it costs an afternoon.

### #2 — Conformal intervals on counts and boxes (hours)

**Claim:** "Every number we report comes with a distribution-free coverage guarantee: 17 cars
[15, 19] at 90% coverage, and these box intervals contain the true box 90% of the time."

**Build:** split your 150 frames into calibration/evaluation; compute per-corner nonconformity
scores with a Bonferroni correction; report empirical coverage and mean interval width. For class
sets, use APS/RAPS. No retraining required.

**Why it wins:** almost no hackathon team can say "guarantee". It also directly answers the judge
question "how do you know it's right?" without inventing an accuracy number.

### #3 — The measurement trio: warm p99, joules, thermal (hours)

**Claim:** "We report the numbers a buyer asks for: p99 after 10 minutes, joules per accepted
detection, and throttled minutes."

**Build:** repeat the clip for 10+ minutes, log `vcgencmd measure_temp` and `get_throttled`, plug a
USB-C meter, compute J per useful frame, report deadline misses at 7.46 FPS.

**Why it wins:** it is the difference between "we benchmarked" and "we benchmarked correctly", and
the 2026 literature is explicit that most edge claims fail this test.

### #4 — OOD flag + "I don't know" (hours)

**Claim:** "When the scene leaves the calibration distribution — night, rain, a new camera angle —
the system says so instead of silently miscounting."

**Build:** energy score or Mahalanobis distance on penultimate features; calibrate on daytime
frames; threshold at 95th percentile; tag each frame and each run.

**Why it wins:** it is the honest-AI story made operational, and it pairs perfectly with conformal.

### #5 — Pixelation + retention + no-network proof (hours)

**Claim:** "The privacy claim is provable: faces and plates are removed before write, evidence is
auto-purged after N days, and a packet capture shows zero outbound connections."

**Build:** detect person/vehicle, pixelate head and plate regions on the *saved copy*, write a
retention manifest, run `tcpdump`/`strace` during a demo run.

**Why it wins:** it converts a paragraph into evidence. A judge can verify it in 30 seconds.

### #6 — Counts → conflicts → recommendation report, English + Kannada (hours)

**Claim:** "The output is a one-page ward report that names the intervention and the hours it
covers."

**Build:** template with slots (peak hour, PET<1 s count, share of pedestrians, recommended action).
Optionally a 270M function-calling model or a 0.5B VLM to fill slots; otherwise pure Python. Add a
Kannada translation column.

**Why it wins:** it changes the audience. It is also the artifact a municipal officer would actually
forward.

### #7 — Cost per 1,000 crossings (minutes)

**Claim:** "₹X per 1,000 observed crossings, on hardware that costs less than a traffic cone
contract."

**Build:** BOM + measured throughput + duty cycle → ₹/1k. Compare with a manual survey and a
commercial camera.

**Why it wins:** it is the metric a procurement officer uses, and nobody presents it.

### #8 — Uncertainty-gated adaptive inference (1 day)

**Claim:** "We skip inference when the tracker is confident, and we show the accuracy cost of doing
it blindly vs. with uncertainty gating."

**Build:** Kalman/IoU predictor; if predicted-vs-measured IoU > threshold, skip the detector and
coast; log skipped frames, saved time, and AP/recall delta. Compare with fixed stride-2.

**Why it wins:** it is a real 2026 systems contribution with a clean ablation, and the repo already
has a tracker.

### #9 — Pose-conditioned intent (1 day)

**Claim:** "42 people crossed; 31 crossed, 11 waited, 4 stepped back — intent, not just presence."

**Build:** use the existing LiteHRNet support or YOLO26-pose; head/hip keypoints + track direction →
waiting/crossing/retreating states.

**Why it wins:** it is visually obvious in the annotated video and immediately understandable.

### #10 — Open-vocabulary rickshaw prompt (1 day, risky)

**Claim:** "COCO has no auto-rickshaw. A text prompt does."

**Build:** YOLOE-26 or YOLO-World v2; prompt `auto-rickshaw`, `hand cart`, `cattle`; compare counts
and latency against COCO YOLO11n. If the ExecuTorch export fails, present it as measured CPU
feasibility + export risk.

**Why it wins:** it turns the project's biggest admitted limitation into a demo moment.

---

## 5. Publishable angles (if you want it to outlive the hackathon)

1. **"From counts to conflicts: bias-corrected surrogate safety metrics on a $80 edge device."**
   Contribution: the pipeline + the box-centre bias correction applied at the edge + a measurement
   protocol. Venue: workshop on edge vision / ITS.
2. **"Conformal counts: distribution-free coverage for edge object counting."**
   Contribution: count intervals and box intervals on a quantised detector, with empirical coverage
   on a small clip. Venue: conformal prediction workshop (COPA-like).
3. **"What 7.5 FPS actually buys you: a measurement study of ExecuTorch INT8 on Raspberry Pi 5."**
   Contribution: warm p99, joules, thermal, and accuracy under quantisation for a real deployment.
   Venue: edge systems workshop.
4. **"A governance pack for street-counting AI."**
   Contribution: AI Act classification + DPIA + ISO 42001 AIMS skeleton, instantiated for one
   deployment. Venue: AI ethics / policy venue, or a municipal white paper.

Any one of these is a stronger artifact than a hackathon demo video, and each is achievable from
what the repo already contains plus 1–2 weeks.

---

## 6. Roadmaps

### If you have 6 hours (hackathon day)

1. PET/TTC + foot-point (2 h).
2. Conformal count intervals (1.5 h).
3. Report generator with Kannada column (1 h).
4. Pixelation + retention manifest (0.5 h).
5. Update README/report claims (1 h).

### If you have 1 week

Add: OOD flag, warm p99/joules/thermal, cost per 1k, uncertainty-gated skipping, pose intent.

### If you have 1 month

Add: two-camera metric speed, SAM 3 teacher → tiny student, CTTA experiment, synthetic augmentation,
public dataset release, governance pack.

### If you have 6 months

Add: event-camera pilot, digital twin what-if, federated multi-site pilot, RISC-V MCU node,
conformal-counts paper.

---

## 7. Anti-patterns — what looks innovative but is not

| Tempting idea | Why it fails |
|---|---|
| **Wrap the output in an LLM** | The model invents numbers; the value is the measurement, not the prose |
| **Dashboard-first** | The brief explicitly deprioritises it; judges have seen 40 dashboards |
| **"Real-time" without a definition** | 7.46 FPS on the reference is not real-time at 30 FPS; be precise |
| **Cloud API fallback** | Contradicts offline/privacy and fails at the venue |
| **Bigger model = better** | RF-DETR/D-FINE are 10–30× the parameters; they will not hit 7 FPS on a Pi 5 CPU |
| **Report mAP on the demo clip** | No ground truth; it is a made-up number and judges know |
| **Aggregate-only privacy claim without proof** | Now table stakes; prove it or drop it |
| **Neuro-symbolic / world-model hand-waving** | Interesting, but cannot be built and validated in a day |
| **Event camera demo without a sensor** | The idea is strong; the demo is not available |

---

## 8. The one-sentence positioning

If the project adopts only the top three bets, the sentence becomes:

> **Arten turns a ₹10,000 Raspberry Pi into a street-safety instrument: it counts who moves, measures
> how close they came to being hit, and reports every number with a stated uncertainty — offline,
> with no identity data, on hardware a school can actually buy.**

That is a different sentence from "edge object detection and counting", and it is much harder to
dismiss.

---

## 9. Sources

### Models, runtimes, architectures
1. Ultralytics, YOLO26 Raspberry Pi benchmarks — <https://docs.ultralytics.com/guides/raspberry-pi>
2. Ultralytics, ExecuTorch integration — <https://docs.ultralytics.com/integrations/executorch>
3. Ultralytics, NMS-free end-to-end detection — <https://docs.ultralytics.com/guides/end2end-detection>
4. Ultralytics, YOLOE — <https://docs.ultralytics.com/models/yoloe>
5. Ultralytics, YOLO26 paper — <https://arxiv.org/html/2606.03748>
6. Roboflow, RF-DETR — <https://rfdetr.roboflow.com/> · <https://roboflow.com/blog/rf-detr-for-the-edge>
7. Meta, SAM 3 / SAM 3.1 — <https://github.com/facebookresearch/sam3> · <https://ai.meta.com/research/sam3>
8. Ultralytics, SAM 3 — <https://docs.ultralytics.com/models/sam-3>
9. Depth Anything 3 (ICLR 2026) — <https://depth-anything-3.github.io/> · <https://arxiv.org/html/2511.10647>
10. VGGT / feed-forward 3D survey — <https://www.digitalsense.ai/blog/best-of-cvpr-2026>

### Sensing and hardware
11. Prophesee event sensors — <https://www.prophesee.ai/event-based-sensors>
12. Event-based camera infrastructure node dataset (IEEE DataPort) — <https://ieee-dataport.org/documents/event-based-camera-infrastructure-node-dataset>
13. SparseVoxelDet — <https://arxiv.org/html/2603.21638v1>
14. Neuromorphic edge system (66 mW) — <https://repository.kaust.edu.sa/server/api/core/bitstreams/f688abca-c964-4da2-91f8-e86a64fe0a8b/content>
15. mmWave counting (98.86%) — <https://www.mdpi.com/1424-8220/26/4/1289>
16. 802.11bf passive counting (98.57%) — <https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=957538>
17. mmFlux crowd flow — <https://www.nature.com/articles/s44459-025-00007-w>
18. Raspberry Pi AI HAT+ 2 — <https://www.raspberrypi.com/news/introducing-the-raspberry-pi-ai-hat-plus-2-generative-ai-on-raspberry-pi-5>
19. AI HAT benchmark (Sixfab) — <https://sixfab.com/blog/raspberry-pi-5-ai-hat-benchmark-deepx-vs-hailo/>
20. AI HAT+ 2 hands-on — <https://marius.blog/raspberry-pi-ai-hat-plus-2/>
21. Coral NPU — <https://developers.googleblog.com/unifying-edge-ai-how-coral-npus-open-standard-brings-portable-energy-viable-intelligence-from-model-to-silicon>
22. Axelera Europa — <https://axelera.ai/axelera-products/edge>
23. Andes AnDLA I370 v2.0 — <https://www.jonpeddie.com/news/andes-teaches-its-risc-v-edge-platform-to-speak-transformer>
24. TetraMem MLX300 — <https://www.edn.com/analog-in-memory-computing-advances-to-12-nm/>
25. Jetson Thor — <https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai>
26. MLPerf Tiny v1.4 — <https://mlcommons.org/2026/07/mlperf-tiny-v1-4-results>

### Learning, compression, uncertainty
27. Self-supervised parking occupancy (Pi 5) — <https://arxiv.org/pdf/2606.20886>
28. TinyGLASS in-sensor anomaly detection — <https://arxiv.org/html/2603.16451v3>
29. Synthetic data small objects (+11.4% APs) — <https://arxiv.org/html/2401.12729v2>
30. OD3 dataset distillation — <https://arxiv.org/html/2506.01942v3>
31. FedFit — <https://mbzuai.ac.ae/news/a-principled-approach-to-cross-device-federated-learning>
32. FedHENet — <https://arxiv.org/html/2602.13024v1>
33. Communication-efficient camera FL — <https://europepmc.org/article/PPR/PPR1341041>
34. AMROD CTTA detection — <https://arxiv.org/html/2406.16439v6>
35. BOSA on-device TTA — <https://www.ijcai.org/proceedings/2026/101>
36. Conformal object detection — <https://arxiv.org/html/2605.07549v2>
37. Class-label conformal detection — <https://pmc.ncbi.nlm.nih.gov/articles/PMC13514218>
38. Quantisation on Hailo-8L (YOLO26) — <https://danieldubinsky.github.io/personal-site/case-studies/quantization_noise_analysis_yolo26/>
39. PTQ/QAT comparison — <https://iieta.org/journals/isi/paper/10.18280/isi.310530>
40. Albireo adaptive inference — <https://arxiv.org/html/2609.29648>
41. UniExit early exit — <https://arxiv.org/html/2611.09159v1>
42. Split computing survey — <https://arxiv.org/html/2103.04505v4>

### Applications and cross-domain
43. Near-miss PET/TTC framework — <https://www.mdpi.com/2673-7590/6/2/80>
44. PET bias from box-centre trajectories — <https://pubmed.ncbi.nlm.nih.gov/42105696>
45. CrowdLense — <https://doi.org/10.1109/comsnets67989.2026.11418200>
46. OpenTrack — <https://isprs-archives.copernicus.org/articles/XLVIII-4-W13-2025/239/2025/>
47. PAVE — <https://doi.org/10.1145/3769102.3770618>
48. BMD-45 — <https://arxiv.org/html/2604.24419v1>
49. Assistive smart cane ($88, Pi Zero 2W) — <https://arxiv.org/html/2609.22277v1>
50. Mobilio — <https://www.nature.com/articles/s41551-026-01772-x>
51. Wildlife two-stage edge classifier — <https://www.mdpi.com/1424-8220/26/4/1366>
52. Scout open-world edge recognition — <https://arxiv.org/html/2609.22897v1>
53. MegaDetectorV6 — <https://microsoft.github.io/MegaDetector>
54. Pose-guided PPE compliance — <https://www.nature.com/articles/s41598-026-66580-4>
55. PPE persistence threshold deployment — <https://www.nexcom.com/news/Detail/ai-powered-ppe-monitoring-for-safer-steel-coil-packaging-operations>

### Systems, measurement, governance
56. Edge-AI energy benchmarking audit — <https://www.mdpi.com/2079-9292/15/15/3372>
57. Thermal throttling on-device — <https://xckevin.github.io/en/blog/android-on-device-ai-benchmark>
58. Useful goodput / warm p99 — <https://mubibai.com/sustained-on-device-inference-needs-a-warm-device-p99-test>
59. "Name the silicon" — <https://talkingiot.io/edge-ai-benchmarks-are-not-numbers-until-you-name-the-silicon>
60. SMART-TWIN digital twin — <https://transweb.sjsu.edu/sites/default/files/2537-Hovannes-Modeling-Real-Time-Transportation-Digital-Twins.pdf>
61. NVIDIA Metropolis calibration (homography) — <https://docs.nvidia.com/vss/vision-agent/agent-workflows/industry-specific-examples/smart-city-blueprint/development-workflow/calibration>
62. Mobile-GS (ICLR 2026) — <https://proceedings.iclr.cc/paper_files/paper/2026/file/298281b9e89197195eb461e68ad20136-Paper-Conference.pdf>
63. ISO/IEC 42001 explainer — <https://www.iso.org/insights/iso-42001-explained>
64. ISO 42001 control structure — <https://www.kertos.io/en/blog/iso-42001>
65. EU AI Act Art. 5 — <https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-5>
66. AI Act Omnibus dates / counting vs identification — <https://www.movetos.com/en/knowledge/ai-act-video-analytics-events/>
67. On-device agents (LFM2.5-2.6B) — <https://huggingface.co/blog/LiquidAI/lfm2-5-2-6b>
68. FunctionGemma 270M — <https://developers.googleblog.com/on-device-function-calling-in-google-ai-edge-gallery>
69. MiniRAG — <https://aclanthology.org/2026.acl-long.1721.pdf>
