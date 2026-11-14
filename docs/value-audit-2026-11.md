# Value audit: which additions actually earn their place

**Prepared:** 14 November 2026
**Question:** of the ideas in the innovation map, which add *real* value — to a judge in two minutes,
to an engineer reading the repo, and to the person who would actually deploy this?
**Method:** score each idea on demoability, defensibility, cost and risk, then apply three consumer
lenses. The result is deliberately harsh: several ideas that looked good on paper do not survive.

---

## 1. How "real value" is judged

Three people will look at this project, and they want different things:

| Consumer | Attention span | Rewards | Punishes |
|---|---|---|---|
| **Hackathon judge** | 2 minutes + 1 question | a visible "wow", a clear decision, numbers that survive a challenge | jargon, unverifiable claims, a feature that produces zero on the demo clip |
| **Engineer / Arm reviewer** | 10 minutes in the repo | correct methodology, reproducibility, honest limits, small artifacts | dev-host numbers presented as Pi numbers, missing thread counts, hand-waving |
| **Buyer (school / ward)** | 1 page | a decision, a cost, a report they can forward, legal comfort | a dashboard, a research prototype, a claim they cannot repeat |

An idea has real value only if it moves at least one of these **and** does not add failure risk at the
venue. Four criteria:

- **Demo** — can a judge see it in the 2-minute window?
- **Defence** — can it survive "how do you know?" and "what does it mean?"
- **Cost** — hours, given the repo already works.
- **Risk** — can it fail on the day (hardware, export, data, calibration)?

---

## 2. Verdict matrix

Scores 1–5; "Verdict" is the decision, not the score.

| Idea | Demo | Defence | Cost | Risk | Verdict |
|---|---:|---:|---:|---:|---|
| **Measurement protocol (Pi 5, ExecuTorch INT8)** | 3 | **5** | 3 | 1 | **Tier 1 — the backbone** |
| **Ward report, English, grounded** | **5** | 4 | 2 | 1 | **Tier 1 — the translation layer** |
| **Natural-language investigation over detections** | **5** | 4 | 4 | 3 | **Tier 1 — the investigation layer** |
| **Conformal count/box intervals** | 2 | **5** | 2 | 2 | **Tier 1 — the trust layer** |
| Cost per 1,000 crossings | 4 | 4 | 1 | 1 | Tier 2 — do it, it is minutes |
| No-network + retention proof | 3 | 4 | 1 | 1 | Tier 2 — cheap credibility |
| Kinematic intent (crossed/waited/retreated) | **5** | 3 | 2 | 2 | Tier 2 — cheap version of "pose intent" |
| Conflict exposure heatmap | 4 | 3 | 2 | 3 | Tier 2 — the salvageable part of near-miss |
| OOD / "I don't know" flag | 3 | 4 | 2 | 3 | Tier 3 — only if a night image exists |
| Uncertainty-gated adaptive inference | 2 | 3 | 3 | 3 | Tier 3 — needs a sustained run to mean anything |
| Pose-conditioned intent | 4 | 3 | 4 | 4 | Tier 3 — model cost and export risk |
| YOLOE rickshaw prompt | 4 | 3 | 3 | **5** | Tier 3 — documented experiment, not a demo |
| **PET/TTC near-miss count** | **1** | 2 | 2 | **5** | **Downgrade — see §3** |
| VLM describes every frame | 4 | 1 | 5 | 4 | **Drop — see the NL doc** |
| Digital twin / SUMO | 5 | 3 | 5 | 4 | Drop for now |
| Event camera / radar / neuromorphic | 3 | 4 | 5 | 5 | Drop — phase 2 slide only |
| Federated / CTTA | 1 | 3 | 5 | 4 | Drop — no data, no time |

---

## 3. The near-miss problem (you were right)

### Why it fails as a demo

1. **It will probably be zero.** Arten's sample clip has 18 tracks over 15 seconds. A discrete
   `PET < 1 s` event requires two road users to pass within one second of the same conflict point. On
   a sparse pedestrian clip, that is often **0 events**. A demo slide that says "0 near misses" is
   worse than not having the feature: it invites "so nothing happened".
2. **It needs calibration you cannot validate.** A homography from four ground points is only valid
   for a flat road surface and a fixed camera. A phone clip may not show four usable points; lens
   distortion breaks it; the error propagates into every distance.
3. **The known bias is large.** Box-centre trajectories shift PET by **0.69 s on average**, exceed
   1 s in **16.7%** of conflicts, and reclassify up to **3.6%** of events. If you do not correct it,
   the number is wrong; if you correct it with a foot-point heuristic, it fails under occlusion.
4. **No ground truth.** You cannot say "this is a near miss" with any authority from one clip. A
   traffic engineer will ask for validation against observed conflicts, and there is none.
5. **The claim is causal-adjacent.** "3 near misses" sounds like an enforcement or safety finding. It
   is a surrogate metric with assumptions, and a careful judge will push on that.

### What survives instead: conflict exposure

Replace the **event count** with a **continuous exposure index** that is always defined:

```
exposure = mean over pedestrian-seconds of  1 / max(TTC, τ)
```

or, more simply and more explainably:

```
exposure = share of pedestrian-seconds with TTC < 2 s
```

Properties:
- **Always non-zero** on any clip with people and vehicles, so the demo works.
- **Statistically stable**: it aggregates over time instead of counting rare events.
- **Still citable**: TTC < 3 s and PET < 1 s are the literature thresholds; you are reporting a
  proximity-weighted exposure, which is a standard surrogate-safety construct.
- **Visually demoable**: render a ground-plane heatmap of exposure; the hot spot is the finding.
- **Honest**: label it "exposure index, unvalidated, assumes flat ground and a fixed camera".

Keep the `PET < 1 s` count as a **secondary** number if it occurs. Do not headline it.

**Verdict:** near-miss as a count is a **Tier 4** demo. Conflict exposure is a **Tier 2** supporting
visual. Neither should be the centrepiece.

---

## 4. The four that earn their place

### 4.1 Measurement protocol — the backbone

**Why it is Tier 1:** it is the only addition that makes *every other number* trustworthy, and it is
the one an Arm judge will recognise as real engineering. It is also the only artifact here that is
reusable after the hackathon.

**What it changes in practice:**
- The headline FPS will probably go **down** (sustained, thermally soaked) versus the cold burst.
  That is the feature: "we report the number the device actually sustains, not the first 50 frames".
- It answers the questions judges ask: p99, energy, thermal, thread count, n.
- It produces a table no other submission will have.

**Cost:** 3–4 hours to add `--protocol`, 90 minutes to run it. **Risk:** low. **Upside:** high.

**The one number to remember:** Arm's own team saw ExecuTorch + XNNPACK latency *increase* over
successive runs on an uncooled Pi 5 due to heat. A benchmark that warms up and measures 50 frames is
measuring the device at its coolest and fastest. The protocol fixes that.

Full detail: [measurement-protocol-pi5.md](measurement-protocol-pi5.md).

### 4.2 Ward report — the translation layer

**Why it is Tier 1:** it changes the audience from "CV judge" to "the person who decides". It is the
artifact a school principal or ward officer would forward. It is also the cheapest way to make the
project feel like a product rather than a script.

**What makes it good vs. bad:**

| Good | Bad |
|---|---|
| Every number traceable to `summary.json` | An LLM writing prose from the numbers |
| States method, time, site, weather, limitations | "AI-powered insights" |
| Gives one recommended action with a coverage estimate | Claims a causal effect ("will reduce crashes") |
| Includes cost per 1,000 observations | A dashboard screenshot |
| One page, English, plain language | Ten pages of technical detail |

**Suggested structure (one page):**

```
Street observation report — [site], [date], [time window]
1. What was observed: X pedestrians, Y vehicles, peak hour, direction split
2. How close they came: exposure index, share of time within 2 s TTC, PET<1 s events (if any)
3. How confident: conformal interval on the count; OOD flag; calibration split
4. What it means: e.g. "78% of pedestrian crossings occur in the 08:00–08:45 window;
   a crossing guard in that window would cover 78% of observed crossings"
5. Limits: COCO classes, one camera, 15 s sample, no ground truth, unvalidated exposure
6. Cost: ₹X per 1,000 observations; hardware ₹Y; power Z W
7. Method: model, runtime, precision, latency, checksum
```

**Cost:** 2–3 hours. **Risk:** low, provided the numbers are templated, not generated.

### 4.3 Natural-language investigation — the investigation layer

**Why it is Tier 1:** it is the only idea here that changes *how a person uses the output*. It turns
`detections.jsonl` into an interface. It is also highly demoable: type a question, get a contact
sheet. And it is defensible if — and only if — the language model **never counts**.

**The rule:**

> **YOLO produces the facts. The small language model compiles the question into a filter. The VLM,
> if used at all, describes a crop. The VLM never counts.**

Why: 2026 benchmarks show VLM counting is poor — **no model exceeds 27.6% exact match** on counts
5–50 (CountVideoBench), counting is **20–30 points below human** (Encord), and counting is the
lowest-accuracy VLM task at **64.0–74.7%** (GroundCount). Grounding the VLM with YOLO boxes improves
counting by **6.2–7.5 pp** and cuts inference time **22–23%**, but does not make it a counter.

**What works with a filter compiler:**
- "How many people crossed between 8 and 9?"
- "Show frames with more than 5 vehicles."
- "Which track moved left fastest?"
- "Was a bus and a person in the zone at the same time?"
- "Top 3 highest-exposure moments."
- "Compare 08:00–09:00 with 14:00–15:00."
- "Which frames were out of distribution?"

**What needs a VLM (optional):**
- "Person carrying a red bag."
- "Someone running."
- "Is the bus stopped or moving?"
- "What is the person doing?"

**What needs a different detector (not a VLM):**
- "Auto-rickshaw", "hand cart", "cattle" — these are open-vocabulary detection prompts (YOLOE), or a
  fine-tune, not descriptions.

Full design: [natural-language-investigation.md](natural-language-investigation.md).

### 4.4 Conformal intervals — the trust layer

**Why it is Tier 1:** it is the cheapest way to say something no other team can say — "17 cars
[15, 19] at 90% coverage" — and it directly answers "how do you know?". It also pairs perfectly with
the ward report: the report quotes intervals, not point estimates.

**What to do:**
- Split by **time blocks**, not random frames (adjacent frames are correlated).
- Calibrate on the first 70%, evaluate on the last 30%.
- For boxes: per-corner nonconformity with a Bonferroni correction.
- For counts: interval on the per-frame count, and on the unique-track count if you can bootstrap
  tracks.
- Report empirical coverage and mean interval width.
- State the exchangeability caveat and the small-sample limitation.

**Cost:** 2 hours. **Risk:** low. If coverage comes out at 88% for a 90% nominal on 45 frames, say so
— that is within noise and saying it is more impressive than hiding it.

---

## 5. Tier 2: cheap, do if the clock allows

| Idea | Time | Why it earns its place |
|---|---:|---|
| **Cost per 1,000 crossings** | 15 min | The metric a buyer compares; nobody presents it |
| **No-network + retention proof** | 1 h | Turns the privacy paragraph into evidence |
| **Kinematic intent** | 2 h | "31 crossed, 11 waited, 4 stepped back" from track direction + dwell, no second model |
| **Conflict exposure heatmap** | 2 h | The salvageable, demoable part of near-miss |

**Kinematic intent** deserves a note: you do not need pose to get most of the demo. A track that
enters the zone, reverses direction and leaves is a "stepped back"; a track that dwells > 3 s before
crossing is a "waited". That is a 2-hour feature with a strong visual. Pose adds head orientation and
is a 1-day, higher-risk feature; only do it if the kinematic version is already working.

---

## 6. Tier 3: only under conditions

| Idea | Condition to attempt | Why it is not Tier 1 |
|---|---|---|
| **OOD flag** | You have a night/rain image to test on | Without an out-of-distribution example, the demo is a claim, not a result |
| **Adaptive inference** | You have the sustained run from the protocol | On a 15 s clip the energy saving is invisible; the ablation is the point |
| **Pose intent** | The pose `.pte` already runs at ≥ 5 FPS on the Pi | Otherwise it halves the pipeline FPS for a marginal gain |
| **YOLOE rickshaw** | The ExecuTorch export succeeds in the first hour | High narrative value, high export risk; do it as a documented experiment |

---

## 7. Drop, and why

| Idea | Why it is dropped |
|---|---|
| **VLM describes every frame** | Seconds per frame on a Pi CPU; hallucinated counts; new privacy surface (stored captions); the detector already produces better facts |
| **Digital twin / SUMO** | 5/5 effort; a great month-two project, not a day-one feature |
| **Event camera / radar / neuromorphic** | No hardware, no time; the *argument* (120 dB dynamic range, no image formed) belongs on a future-work slide |
| **Federated / CTTA** | No fleet, no unlabeled corpus, no training budget; one node cannot demonstrate either |
| **Full near-miss count** | Zero-event demo, unvalidated, calibration-fragile; use exposure instead |
| **Dashboard** | Explicitly deprioritised by the brief; judges have seen many |

---

## 8. The minimum standout set

If only four things can be added, add these:

1. **Measurement protocol** — credibility backbone (3–4 h + 90 min run).
2. **Ward report** — translation layer (2–3 h).
3. **NL investigation over detections** — investigation layer (3–4 h).
4. **Conformal intervals** — trust layer (2 h).

Together they change the claim from:

> "We ran YOLO on a Raspberry Pi and counted people."

to:

> "We measured what the Pi actually sustains, we report every count with a stated uncertainty, and a
> non-engineer can ask the evidence a question in plain English — offline."

That is a different project, and it is defensible in all three rooms.

**Supporting, in order:** cost per 1k (15 min), no-network proof (1 h), kinematic intent (2 h),
exposure heatmap (2 h).

---

## 9. Claims ladder: what to say, in what order

For the 2-minute pitch, lead with the claim that survives the most scrutiny:

1. **The measurement** (10 s): "7.x FPS sustained on a Pi 5 after 20 minutes, p99 Y ms, Z J per frame,
   throttled=0x0 — here is the table."
2. **The uncertainty** (10 s): "Every count has a coverage interval; 17 cars means 15 to 19 at 90%."
3. **The decision** (30 s): "78% of crossings happen in a 45-minute window; that is where a guard
   goes; here is the one-page report."
4. **The investigation** (30 s): "Ask it 'show me frames where a bus and a person were in the zone
   within 2 seconds' — it compiles the question to a filter, shows the filter, returns the frames."
5. **The honesty** (20 s): "COCO is not Bengaluru; here is the documented gap; here is what we did
   not measure."
6. **The demo** (20 s): annotated clip + contact sheet.

Do not lead with the model. The model is a commodity; the measurement, the report and the query
interface are not.

---

## 10. Sources

1. Arm/PyTorch, ExecuTorch + XNNPACK thermal behaviour on Pi 5 —
   <https://pytorch.org/blog/efficient-edge-ai-on-arm-cpus-and-npus>
2. PET bias from box-centre trajectories — <https://pubmed.ncbi.nlm.nih.gov/42105696>
3. Near-miss / SSM framework — <https://www.mdpi.com/2673-7590/6/2/80>
4. CountVideoBench (VLM counting, 27.6% exact match) — <https://arxiv.org/html/2610.28356v2>
5. Encord, egocentric VQA (counting −20 to −30 points) —
   <https://encord.com/data-labs/vlms-egocentric-understanding-keeping-score/>
6. GroundCount (YOLO grounding improves VLM counting 6.2–7.5 pp) — <https://arxiv.org/html/2603.10978>
7. Conformal object detection — <https://arxiv.org/html/2605.07549v2>
8. Class-label conformal detection — <https://pmc.ncbi.nlm.nih.gov/articles/PMC13514218>
9. VQS, natural-language traffic query over structured index (95% exact match) —
   <https://www.ijcaonline.org/archives/volume187/number81/kathait-2026-ijca-926403.pdf>
10. Text2SQirreL (0.6B local text-to-SQL, 74% judge accuracy) — <https://www.distillabs.ai/blog/distil-text2sql>
11. Arm, Raspberry Pi 5 energy per token (thread count, governor) —
    <https://learn.arm.com/learning-paths/cross-platform/rpi5-energy-per-token>
