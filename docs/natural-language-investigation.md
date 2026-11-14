# Natural-language investigation over detections (YOLO + small LM, optional VLM)

**Prepared:** 14 November 2026
**Question:** does adding a VLM to YOLO add real value for investigations and searches in natural
language?
**Short answer:** yes — but only if the language model **compiles questions into filters over
detector output**. It must not be the thing that looks at pixels and counts. The evidence for that
distinction is strong and recent.

> **Implementation note (14 Nov 2026):** the first version is built in
> `src/arten/query/` with `arten index` and `arten ask`. Measured on Qwen2.5-0.5B,
> free-form JSON output was unreliable; the shipped design uses a closed-label
> classifier plus deterministic slot extraction, and still emits a validated
> `QueryPlan`. See [plan-vlm-yolo.md](plan-vlm-yolo.md) §17.

---

## 1. The rule

> **YOLO produces the facts. A small language model compiles the question into a structured filter.
> A VLM, if used at all, describes a single crop. The VLM never counts.**

Three reasons:

1. **VLMs cannot count.** CountVideoBench (2026, 2,600 videos, 16 models): **no model exceeds 27.6%
   exact match** on counts 5–50. Encord's egocentric benchmark: counting is **20–30 points below
   human**. GroundCount: counting is the **lowest-accuracy** VLM task at **64.0–74.7%**, below object
   recognition (70.3–89.6%) and attributes (83.2–86.5%). CAPTURe: VLMs are far worse than humans on
   occluded counting.
2. **Grounding helps, but does not fix it.** GroundCount feeds YOLO boxes into the VLM prompt and
   gains **6.2–7.5 pp** (up to 81.3%), and cuts inference time **22–23%** by avoiding
   hallucination-driven reasoning loops. That is a real improvement — and still not a counter.
3. **Cost.** A 0.5B VLM on a phone has a warm TTFT of ~6 s (VisionPsy-Flash, Pixel 9). On a Pi 5 CPU,
   per-frame captioning is seconds to tens of seconds. You cannot caption 150 frames in a demo, let
   alone a 20-minute survey.

So the useful architecture is not "VLM watches video". It is "**language model queries a database
that YOLO filled**".

---

## 2. Three architectures, scored

### A. VLM captions every frame → vector search

```
frames → VLM caption → embedding → vector DB → semantic search
```

- **Pros:** open-ended; finds things no detector was trained for.
- **Cons:** seconds per frame; hallucinated counts; captions are a new text log of scene content
  (privacy surface); retrieval is fuzzy and hard to audit; the answer is not reproducible.
- **Verdict:** no for this project. This is the Verkada/Brivo/NVIDIA VSS pattern, and it assumes a
  GPU and a cloud or edge server.

### B. YOLO → structured index → small LM compiles a filter

```
frames → YOLO → detections.jsonl → SQLite → [NL query → JSON filter → SQL] → frames/rows
```

- **Pros:** exact; auditable (you can show the filter and the SQL); fast; offline; no hallucinated
  numbers; the model only has to do a small, closed task.
- **Cons:** limited to the attributes in the schema; needs a schema and a few examples.
- **Verdict:** **yes — this is the core.**

### C. Hybrid: B + VLM on demand for a single crop

```
query → filter → candidate frames → top-K crops → VLM describes each crop → answer
```

- **Pros:** handles "red bag", "running", "wheelchair" without captioning everything; bounded cost
  (K ≤ 3); human can review.
- **Cons:** extra model, extra latency, still fallible.
- **Verdict:** optional add-on, after B works.

---

## 3. What the data already contains

The repo already writes enough to support a real query language:

| File | Fields | Supports |
|---|---|---|
| `detections.jsonl` | frame, class, confidence, box, track_id, direction, speed, dwell, colour | class/time/confidence/geometry queries |
| `counts.csv` | per-frame counts, stage timings | aggregates, peaks, throughput |
| `attribute_tracks.csv` | track_id, class, direction, dwell_s, speed_px_s, colour | track-level queries |
| `attribute_summary.json` | distributions | summaries |
| `zone_counts.csv` | zone co-presence | interaction queries |
| `summary.json` | environment, latency, model | provenance queries |

A query compiler only needs a **closed schema** over these fields.

---

## 4. Query schema

Keep it small enough that a 0.5B model can emit it reliably, and rich enough to be useful.

```json
{
  "select": "frames | count | tracks | aggregate",
  "filters": {
    "class": ["person", "car"],
    "frame": {"min": 0, "max": 150},
    "time": {"from": "08:00", "to": "09:00"},
    "conf_min": 0.3,
    "track": {
      "min_dwell_s": 2.0,
      "direction": "left | right | up | down | stationary",
      "min_speed_px_s": 10,
      "colour": ["white", "red"]
    },
    "zone": "selected_area",
    "co_presence": {"classes": ["person", "car"], "within_s": 2.0}
  },
  "group_by": ["minute", "class"],
  "order_by": {"field": "speed_px_s", "dir": "desc"},
  "limit": 12,
  "return": ["frame", "class", "track_id", "confidence", "box"]
}
```

This is deliberately **not** free-form SQL. A closed JSON schema is easier to validate, easier to
explain, and safer to execute.

---

## 5. Query catalogue

### Works with the filter compiler (no VLM)

| Natural language | Compiled filter |
|---|---|
| "How many people crossed between 8 and 9?" | `select=count, class=person, time=08:00–09:00` |
| "Show frames with more than five vehicles." | `select=frames, group_by=frame, having count(vehicle) > 5` |
| "Which track moved left fastest?" | `select=tracks, direction=left, order_by=speed desc, limit=1` |
| "Was a bus and a person in the zone at the same time?" | `co_presence.classes=[bus,person], within_s=2` |
| "Top three highest-exposure moments." | `select=frames, order_by=exposure desc, limit=3` |
| "Compare morning and afternoon." | `select=aggregate, group_by=hour` |
| "Which frames were out of distribution?" | `select=frames, ood=true` |
| "Show me the annotated frames where a car was red." | `select=frames, class=car, colour=red` |
| "How long did people wait before crossing?" | `select=aggregate, class=person, metric=dwell` |
| "Which minute had the most two-wheelers?" | `select=aggregate, class=motorcycle, group_by=minute, order_by=count desc` |

### Needs a VLM on a crop (bounded)

| Query | Why the schema cannot answer |
|---|---|
| "Person carrying a red bag" | bag/colour is not a detector class |
| "Someone running" | gait is not in the schema |
| "Is the bus stopped or moving?" | requires scene context |
| "What is the person doing?" | open-ended |

### Needs a different detector, not a VLM

| Query | Better tool |
|---|---|
| "Auto-rickshaw" | YOLOE text prompt or an IDD fine-tune |
| "Hand cart", "cattle" | open-vocabulary detector |
| "Wheelchair", "pram" | open-vocabulary detector or a small classifier |

---

## 6. Model choice and Pi 5 feasibility

| Model | Size | Pi 5 CPU feasibility | Notes |
|---|---:|---|---|
| **Qwen2.5-0.5B-Instruct Q4** | ~0.4 GB | **~32 tok/s** (Arm's measured Pi 5, 1 thread, 2.4 GHz) | A 60-token JSON filter ≈ 2 s; best starting point |
| Qwen3-0.6B Q4 | ~0.5 GB | Similar class | Stronger instruction following, similar cost |
| Qwen2.5-1.5B Q4 | ~1 GB | ~10–15 tok/s (estimated) | Better reasoning; 4–6 s per query; still usable |
| **Text2SQirreL 0.6B** | ~0.4 GB | n/a on Pi, but reference | Tuned text-to-SQL: 74% judge accuracy; proves the 0.6B class is viable |
| FunctionGemma 270M | ~0.3 GB | Not in llama.cpp today | Function calling; watch for a GGUF port |
| VLM (SmolVLM-500M, LFM2.5-VL-450M) | ~0.5 GB | Seconds per image | Only for the optional crop-describe path |

**Constrained decoding is the key trick.** llama.cpp supports GBNF grammars; give it the JSON schema
as a grammar so the model *cannot* emit invalid JSON. This converts a probabilistic task into a
constrained one and removes most of the failure modes.

**Fallback:** a rule-based parser with 10 intents and synonyms. If the model is too slow or
misbehaves at the venue, the demo still works. Build the rule-based path first, then add the model.

---

## 7. Execution and "show your work"

The trust mechanism is transparency:

1. Parse the question into the JSON filter.
2. **Show the filter** to the user.
3. Compile it to SQL and **show the SQL**.
4. Execute against SQLite.
5. Return rows + a contact sheet of the matching frames.
6. Let the user edit the filter and re-run.

This makes the system auditable and turns a wrong answer into a visible, correctable filter rather
than a hallucination.

**SQLite schema sketch:**

```sql
CREATE TABLE detections (
  frame INTEGER, ts REAL, class TEXT, conf REAL,
  x1 REAL, y1 REAL, x2 REAL, y2 REAL,
  track_id INTEGER, direction TEXT, speed_px_s REAL,
  dwell_s REAL, colour TEXT, zone TEXT
);
CREATE INDEX idx_class_frame ON detections(class, frame);
CREATE INDEX idx_track ON detections(track_id);
```

Queries are fast (150 frames × ~20 detections = 3,000 rows; even a 20-minute run is ~100k rows).

---

## 8. Evaluation (make it a mini-benchmark)

Write 20 questions with expected filters and expected result sets. Measure:

| Metric | What it means | Target |
|---|---|---|
| **Parse validity** | valid JSON against the schema | ≥ 18/20 |
| **Exact filter match** | same filters as the reference | ≥ 14/20 |
| **Execution accuracy** | same result set as the reference | ≥ 16/20 |
| **Latency** | seconds per query on the Pi | ≤ 3 s |
| **Repair rate** | queries needing a second attempt | ≤ 20% |

Report the table. This is a real, small evaluation and it is far more convincing than "it works".

---

## 9. Privacy

- Queries and filters stay on the device; no query log by default.
- No captions are stored in the core path; only structured detections.
- If the optional VLM crop path is enabled, process the crop in memory, store only the requested
  description, and apply the same retention policy.
- The query interface should never expose raw frames to a network; it reads local files.

---

## 10. What it adds, and what it does not

**Adds:**
- A usable interface over the evidence bundle for a non-engineer.
- Auditable investigation: question → filter → SQL → rows.
- A demo moment nobody else will have: "ask the crossing".
- A reason for the CSV/JSONL artifacts to exist.

**Does not add:**
- Better detection or counting. It inherits whatever YOLO produced.
- Open-ended visual understanding (that needs the optional VLM path or open-vocab detection).
- Real-time interaction; 1–3 s per query is fine for investigation, not for live alerts.

**Do not claim:** "the AI understands the scene". Claim: "the AI compiles your question into an
auditable query over detector output".

---

## 11. Implementation plan (4 hours)

| Time | Step | Output |
|---|---|---|
| 0:00–0:30 | Export `detections.jsonl` + `attribute_tracks.csv` to SQLite | `evidence.db` |
| 0:30–1:00 | Write the JSON schema and 10 reference queries | `queries.json` |
| 1:00–2:00 | Rule-based parser (10 intents + synonyms) | working fallback |
| 2:00–3:00 | llama.cpp + Qwen2.5-0.5B + GBNF grammar; prompt with 8 examples | NL → filter |
| 3:00–3:30 | Execute, render contact sheet, show filter + SQL | CLI/HTML |
| 3:30–4:00 | Evaluate 20 queries; record the table | `nl_eval.md` |

If llama.cpp on the Pi is too slow to set up, run the model on the dev host and keep the *interface*
on the Pi; the demo can show the query flow over a local socket, still offline. Do not quietly switch
to a cloud API.

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Model emits invalid JSON | GBNF grammar + schema validation |
| Model picks the wrong filter | Show the filter; let the user edit; keep a rule-based fallback |
| Model too slow on Pi | 0.5B Q4, short prompts, 8 examples max, cache the system prompt |
| Query asks for an attribute not in the schema | Return "not in the index" and offer the VLM crop path or an open-vocab detector |
| Venue Wi-Fi absent | Everything local; model file committed or on USB |
| Judge asks "is this just ChatGPT?" | Show the local model, the grammar, the SQL, and the absence of network calls |

---

## 13. The one-slide version

```
YOLO  →  evidence.db  →  [question] → JSON filter → SQL → frames
                              ↑
                   0.5B local model + grammar
                   (never sees a pixel, never counts)
```

The language model is a **query compiler**, not a perceiver. That is the defensible, demoable,
offline design.

---

## 14. Sources

1. CountVideoBench (VLM counting, 27.6% exact match) — <https://arxiv.org/html/2610.28356v2>
2. Encord, egocentric VQA benchmark (counting −20 to −30 points) —
   <https://encord.com/data-labs/vlms-egocentric-understanding-keeping-score/>
3. GroundCount (YOLO grounding improves VLM counting 6.2–7.5 pp, −22% inference time) —
   <https://arxiv.org/html/2603.10978>
4. CAPTURe (occluded counting; VLMs far below humans) — <https://arxiv.org/html/2504.15485v2>
5. VQS, natural-language traffic query over a structured violation index (95% exact match) —
   <https://www.ijcaonline.org/archives/volume187/number81/kathait-2026-ijca-926403.pdf>
6. Text2SQirreL (0.6B text-to-SQL, 74% judge accuracy) — <https://www.distillabs.ai/blog/distil-text2sql>
7. Struct-SQL (distilling structured reasoning for small text-to-SQL models) —
   <https://proceedings.mlr.press/v318/thaker26a.html>
8. NVIDIA VSS search workflow (embed / attribute / fusion / image search) —
   <https://docs.nvidia.com/vss/latest/agent-workflow-search.html>
9. ForeSea (forensic search with multimodal queries) — <https://arxiv.org/html/2603.22872v2>
10. Axxon VLM meta-search — <https://www.axxonsoft.com/products/video-management-software/key-features/vlm-metadata-tool>
11. Arm, Pi 5 energy per token (Qwen2.5-0.5B Q4 at ~32 tok/s, 1 thread) —
    <https://learn.arm.com/learning-paths/cross-platform/rpi5-energy-per-token>
12. Google AI Edge, on-device function calling (FunctionGemma 270M) —
    <https://developers.googleblog.com/on-device-function-calling-in-google-ai-edge-gallery>
13. MiniRAG (lightweight RAG with SLMs, 25% storage) — <https://aclanthology.org/2026.acl-long.1721.pdf>
