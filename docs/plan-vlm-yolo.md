# Full plan: YOLO + small-LM investigation layer ("ask the crossing")

**Prepared:** 14 November 2026
**Status:** built and verified on the Mac host (llama.cpp 1.0.0, Qwen2.5-0.5B Q4_K_M,
25 query tests green, rules + LLM parsers live). Pi install pending — transfer
over LAN with `scripts/sync_to_pi.sh`, then `scripts/setup_llm.sh` on the Pi.
**Constraint:** the Pi is currently busy with another process; every install/build step below is
gated by a preflight check and is designed to run **after** it finishes, not alongside it.

---

## 0. Executive summary

**What we are building.** An offline `arten ask` command that turns a plain-English question into an
auditable query over the artefacts `arten run` already produces:

```
arten run  →  detections.jsonl + attribute_tracks.csv + summary.json
           →  arten index  →  evidence.db (SQLite)
           →  arten ask "…"  →  QueryPlan JSON → SQL → rows + contact sheet
```

**The rule.** YOLO is the only perceiver. A small local language model (Qwen2.5-0.5B, ~0.4 GB)
**compiles the question into a closed JSON filter**; it never sees pixels and never counts. A VLM is
optional and only ever describes a single crop. This is the design the 2026 evidence supports: no VLM
exceeds **27.6% exact match** on counts 5–50, and counting is the lowest-accuracy VLM task at
**64.0–74.7%**.

**Why it is worth doing.** It turns a CSV bundle into an interface a ward officer can use, and it is
demoable in 90 seconds. It does not improve detection; it makes the detection usable.

**Cost.** ~10–12 engineering hours for the MVP (index + query compiler + rule fallback + LLM parser
+ CLI), plus ~6 hours of gated installation and evaluation. The optional VLM path is +3 hours and
can be dropped without affecting the core.

**Feasibility on Pi 5.** Detection is **138.85 ms/frame** (model) and **162.2 ms/frame** (pipeline,
6.17 FPS) on the committed Pi 5 run. Qwen2.5-0.5B Q4 runs at **~32 tok/s** on a Pi 5 (Arm's
measurement, 1 thread), so a 60-token filter is ~2 s of generation. Run sequentially, the query layer
does not slow detection at all.

---

## 1. Goals and non-goals

### Goals
1. Answer 20 representative questions over a completed run, offline, with no cloud.
2. Every answer is traceable: question → QueryPlan → SQL → rows → frames.
3. Handle unsupported questions honestly ("auto-rickshaw is not a COCO class") rather than inventing.
4. Run on the Pi 5 without competing with a detection run.
5. Add no new Python runtime dependency if possible (SQLite is stdlib; llama.cpp is a native binary).

### Non-goals
- Per-frame VLM captioning or "chat with the video".
- Real-time chat during capture.
- Free-form SQL execution from model output (unsafe and unauditable).
- Face, plate, identity or re-identification search.
- Improving detector accuracy. The query layer inherits whatever YOLO produced.

---

## 2. Current state (facts this plan is built on)

### 2.1 The artefacts already exist

| File | Content | Use |
|---|---|---|
| `detections.jsonl` | one JSON object per frame: `frame_idx`, `timestamp_s`, `counts`, `inference_ms`, `detections[{class, score, box, track_id, attributes{direction, speed_px_s, dwell_s, hits, color_estimate}}]` | primary fact table |
| `attribute_tracks.csv` | `class, track_id, color, direction, speed_px_s, dwell_s, hits, score` | track table |
| `summary.json` | run, model, environment, counts, latency, throughput, exposure, zone | provenance + aggregates |
| `counts.csv` | per-frame counts and stage timings | time series |
| `zone_counts.csv` | zone co-presence | interaction queries |
| `annotated/` | JPEGs at `--annotate-every` cadence | contact sheets |

### 2.2 The real Pi 5 numbers (use these, not the older run)

| Source | Board | Model inference | Pipeline | FPS | Peak RSS |
|---|---|---:|---:|---:|---:|
| `results/benchmark_pi5.json` | Pi 5 (`rpt-rpi-2712`) | **136.77 ms** | — | 7.31 | 435.6 MB |
| `results/demo_pi5_pedestrians` | Pi 5 (`rpt-rpi-2712`) | **138.85 ms** | **162.2 ms** | **6.17** | 466.6 MB |
| `results/demo_pi5_support` | Pi 5 | 137.76 ms | 237.68 ms mean, **960 ms p95** | 4.21 | 592.5 MB |
| `results/demo_pedestrians` | **not Pi 5** (`rpt-rpi-v8`) | 1298.61 ms | 1586.69 ms | 0.63 | 442.3 MB |

**Data hazard:** `results/demo_pedestrians` has been quoted in places as a Pi number. Its platform is
`Linux-…-rpt-rpi-v8-aarch64`, not `rpt-rpi-2712`; it is a Pi 3/4 class run. Do not use it as Pi 5
evidence, and do not base VLM feasibility on its 1.3 s inference.

**Tail-latency hazard:** the Pi 5 support run has a mean of 237.68 ms but a p95 of 960 ms. A single
outlier can double the mean. The query layer does not fix this, but any claim about "6 FPS" should
say "6.17 FPS mean on this run; p95 X".

### 2.3 What is already installed

- Python 3.11/3.13 venv on the Pi; `numpy`, `opencv-python-headless`, `pillow`, `psutil`, `pandas`,
  `scikit-learn` (pulled in by the ExecuTorch support set), `torch`, `executorch`.
- SQLite is in the Python standard library — no install.
- No llama.cpp, no GGUF model, no web server, no query layer.

---

## 3. Architecture

```
┌──────────────┐     ┌────────────────────┐     ┌─────────────────────┐
│ arten run    │────▶│ results/<run>/     │────▶│ arten index         │
│ (ExecuTorch) │     │ detections.jsonl   │     │ → evidence.db       │
└──────────────┘     │ attribute_tracks   │     │ → evidence.db.meta.json │
                     │ summary.json       │     └──────────┬──────────┘
                     │ annotated/*.jpg    │                │
                     └────────────────────┘                ▼
                                                   ┌─────────────────┐
                     ┌────────────────────┐        │ arten ask "…"   │
                     │ rule parser        │───────▶│  QueryPlan JSON │
                     │ (always available) │        └────────┬────────┘
                     └────────────────────┘                 │
                     ┌────────────────────┐                 ▼
                     │ LLM parser         │        ┌─────────────────┐
                     │ Qwen2.5-0.5B +     │───────▶│ validate +      │
                     │ JSON grammar       │        │ semantics       │
                     └────────────────────┘        └────────┬────────┘
                                                            ▼
                                                   ┌─────────────────┐
                                                   │ SQL compiler    │
                                                   │ (parameterized) │
                                                   └────────┬────────┘
                                                            ▼
                                                   ┌─────────────────┐
                                                   │ SQLite → rows   │
                                                   │ → contact sheet │
                                                   │ → HTML/markdown │
                                                   └────────┬────────┘
                                                            ▼
                                                   optional: VLM on
                                                   top-K crops only
```

**Key separation:** `arten index` is write-once per run; `arten ask` is read-only. The LLM is a
parser, not an executor. SQL is compiled from a validated dataclass, not generated as text.

---

## 4. Data model (`evidence.db`)

### 4.1 Tables

```sql
PRAGMA journal_mode=WAL;

CREATE TABLE meta (
  key TEXT PRIMARY KEY,
  value TEXT
);

CREATE TABLE runs (
  run_id TEXT PRIMARY KEY,
  input TEXT,
  model_sha256 TEXT,
  created_at TEXT,
  frames INTEGER,
  source_fps REAL,
  duration_s REAL,
  pipeline_fps REAL,
  model_ms REAL,
  peak_rss_mb REAL
);

CREATE TABLE frames (
  frame_idx INTEGER PRIMARY KEY,
  source_frame_idx INTEGER,
  timestamp_s REAL,
  inference_ms REAL,
  counts_json TEXT,
  ood INTEGER DEFAULT 0
);

CREATE TABLE detections (
  frame_idx INTEGER,
  timestamp_s REAL,
  class_id INTEGER,
  class TEXT,
  score REAL,
  x1 REAL, y1 REAL, x2 REAL, y2 REAL,
  track_id INTEGER,
  direction TEXT,
  speed_px_s REAL,
  dwell_s REAL,
  hits INTEGER,
  colour TEXT,
  zone TEXT
);

CREATE TABLE tracks (
  track_id INTEGER,
  class TEXT,
  first_frame INTEGER,
  last_frame INTEGER,
  frames INTEGER,
  direction TEXT,
  speed_px_s REAL,
  dwell_s REAL,
  hits INTEGER,
  mean_score REAL,
  colour TEXT,
  crossed_zone INTEGER,
  net_dx REAL,
  net_dy REAL,
  PRIMARY KEY (track_id, class)
);

CREATE INDEX idx_det_class_frame ON detections(class, frame_idx);
CREATE INDEX idx_det_track ON detections(track_id);
CREATE INDEX idx_det_time ON detections(timestamp_s);
CREATE INDEX idx_tracks_class ON tracks(class);
```

### 4.2 `evidence.db.meta.json`

```json
{
  "schema_version": 1,
  "source_run": "results/demo_pi5_pedestrians",
  "created_at": "2026-11-14T…",
  "rows": {"frames": 150, "detections": 1035, "tracks": 34},
  "classes": ["person", "car", "truck"],
  "time_range_s": [0.0, 15.0],
  "has_zone": false,
  "has_colour": true,
  "has_attributes": true,
  "input_sha256": {"detections.jsonl": "…", "attribute_tracks.csv": "…"}
}
```

The meta file is what lets `arten ask` answer "not in this index" honestly instead of guessing.

### 4.3 Derived fields

- `crossed`: defined as `frames >= 3` **and** `|net_dx| >= 0.25 × frame_width` (or `crossed_zone=1`
  when a zone exists). Record this definition in `semantics.py` and print it in `--explain`.
- `vehicle`: `car | motorcycle | bus | truck | bicycle`.
- `two_wheeler`: `motorcycle | bicycle`.
- `vulnerable`: `person | bicycle | motorcycle`.
- `peak`: max per-frame count and the timestamp at which it occurs.

---

## 5. Query language and semantics

### 5.1 `QueryPlan` (closed schema, version 1)

```json
{
  "version": 1,
  "intent": "count | list_frames | list_tracks | aggregate | compare | explain",
  "entities": {"classes": ["person"], "group": "vehicle | two_wheeler | vulnerable | null"},
  "filters": {
    "time": {"start_s": 0.0, "end_s": 900.0},
    "frame": {"start": 0, "end": 150},
    "confidence_min": 0.25,
    "zone": "selected_area",
    "direction": "left | right | up | down | stationary",
    "speed_px_s_min": 10.0,
    "dwell_s_min": 2.0,
    "colour": ["red"],
    "track_ids": [1, 2],
    "ood": false
  },
  "relations": {
    "co_presence": {"classes": ["person", "car"], "within_s": 2.0}
  },
  "group_by": ["minute", "class", "direction"],
  "order_by": {"field": "count | speed_px_s | dwell_s | timestamp_s", "direction": "desc"},
  "limit": 12,
  "output": ["count", "frames", "contact_sheet"]
}
```

Rules:
- Unknown keys are rejected.
- `classes` must be in the index's class list; otherwise return `unsupported`.
- `group_by` only from the enumerated list.
- `limit` capped at 50.
- No free-form SQL field exists. This is deliberate.

### 5.2 Semantic layer (`semantics.py`)

The semantic layer is the difference between a toy and something usable. It resolves:

| Phrase | Resolution |
|---|---|
| people, person, pedestrian, pedestrians | `person` |
| vehicle, vehicles, traffic | `car, motorcycle, bus, truck, bicycle` |
| two-wheeler, bike, bikes | `motorcycle, bicycle` |
| lorry, lorrys, truck | `truck` |
| vulnerable road user, VRU | `person, bicycle, motorcycle` |
| morning | 06:00–12:00 |
| afternoon | 12:00–17:00 |
| evening | 17:00–21:00 |
| night | 21:00–06:00 |
| crossed | `crossed = true` (definition below) |
| unique | distinct `track_id` |
| peak | max per-frame count and its timestamp |
| auto, auto-rickshaw, tuk-tuk | **unsupported** — COCO has no such class; explain |
| near miss | **unsupported as a count** — only the exposure index if implemented |

**Explicit unsupported responses** are a feature. The system should say:

> "Auto-rickshaw is not in this index. The detector is COCO-trained and has no auto-rickshaw class.
> A YOLOE text prompt or an IDD fine-tune would be needed."

That is more credible than a hallucinated zero.

### 5.3 SQL compiler

`QueryPlan` → `(sql, params)`. Example:

```sql
SELECT COUNT(DISTINCT d.track_id) AS n
FROM detections d
JOIN tracks t ON t.track_id = d.track_id AND t.class = d.class
WHERE d.class = ?
  AND d.timestamp_s BETWEEN ? AND ?
  AND t.crossed_zone = 1
```

Rules:
- Every value is a bound parameter; no string interpolation.
- `LIMIT` is an integer from the validated plan, capped.
- `ORDER BY` field is chosen from a whitelist, not from raw model text.
- The compiler is unit-tested against a fixture database.

---

## 6. Parsers

### 6.1 Rule parser (always available, default fallback)

- 10 intents: count class, count unique, list frames by count, fastest track, direction count,
  co-presence, time slice, colour distribution, peak, compare halves.
- Regex + synonym table from `semantics.py`.
- No model, ~1 ms.
- This is what the demo uses if the LLM is slow or misbehaves.

### 6.2 LLM parser (Qwen2.5-0.5B-Instruct Q4_K_M)

- Prompt: system (schema + semantic layer + rules) + 8 few-shot examples + the question.
- Output: JSON only.
- **Grammar:** start with llama.cpp's `grammars/json.gbnf` to guarantee syntactically valid JSON;
  validate the structure in Python. A full `QueryPlan` GBNF is optional and more brittle.
- Temperature 0; `-n 256`; stop at the closing brace.
- Validate with `QueryPlan.from_dict`; on failure, one repair attempt with the validation error;
  then fall back to rules.
- Record which parser answered, the model, and the latency.

**Why Qwen2.5-0.5B:** Arm measured **31.96 tok/s** on a Pi 5 at 1 thread, 2.4 GHz (Q4_0). A 60-token
filter is ~2 s of generation; prompt processing for ~800 tokens is the unknown to measure. If total
latency > 8 s, shrink the prompt or use the rule parser by default and the LLM behind `--llm`.

### 6.3 Optional VLM path (gated, drop if slow)

- Triggered only for `intent=describe` or an attribute not in the schema.
- Candidates: top-K (K ≤ 3) frames/tracks from the filter.
- Crop: box expanded 20%, resized to 256×256, saved to a temp file.
- Model: SmolVLM-500M-Instruct (Apache-2.0) via `llama-mtmd-cli`, if the pinned llama.cpp supports it.
- Output: one short description; never a count; stored only with `--save-descriptions`.
- **Gate:** if a single crop takes > 15 s or the model fails to load, ship without the VLM path and
  document it. The core product does not depend on it.

---

## 7. Installation plan (run only when the Pi is idle)

### 7.1 Preflight (mandatory)

```bash
# Is anything already running?
pgrep -af "arten|llama|python.*arten" || true

# Thermal and power state
vcgencmd measure_temp
vcgencmd get_throttled        # want 0x0
vcgencmd measure_clock arm

# Resources
free -h
df -h .
nproc
```

Rules:
- If `pgrep` shows an Arten or llama process, **stop**; wait for it to finish.
- If `get_throttled` is not `0x0`, wait for the board to cool and re-check.
- If free disk < 3 GB, build in `/tmp` and delete the build directory afterwards.
- If free RAM < 1.5 GB, close other services first.

### 7.2 System packages

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git
```

`libcurl4-openssl-dev` is only needed if you build with `-DLLAMA_CURL=ON`; we build with it off.

### 7.3 Build llama.cpp (pinned, low-impact)

```bash
cd ~/arten
git clone --depth 1 https://github.com/ggml-org/llama.cpp vendor/llama.cpp
cd vendor/llama.cpp
git rev-parse HEAD > ../LLAMA_CPP_COMMIT.txt

# -j2 and nice: leave headroom for whatever else is on the board
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON

nice -n 10 cmake --build build --target llama-cli llama-server -j2
# optional, for the VLM gate:
nice -n 10 cmake --build build --target llama-mtmd-cli -j2 || true
```

Notes:
- `-DGGML_NATIVE=ON` targets Cortex-A76 on the Pi 5. If the binary is copied to another board,
  rebuild.
- Target names vary by tag; if a target is missing, run
  `cmake --build build --target help | grep -i llama`.
- Expected build time: 20–40 min at `-j2` on a Pi 5; longer if the board is busy or throttled.

### 7.4 Model download

```bash
mkdir -p models/llm
# Qwen2.5-0.5B-Instruct, Q4_K_M, Apache-2.0
curl -L --fail -o models/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf

sha256sum models/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf
# record the hash in models/llm/README.md
```

If the Pi has no internet, download on the dev host and copy over USB; record the hash on both sides.

### 7.5 Smoke test (single-thread, low priority)

```bash
cd ~/arten
taskset -c 3 nice -n 10 vendor/llama.cpp/build/bin/llama-cli \
  -m models/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -p 'Reply with JSON only: {"ok": true}' \
  -n 24 --log-disable -no-cnv
```

Gate:
- Must print valid JSON.
- Measure tokens/s; must be ≥ 15 tok/s.
- If it is < 8 tok/s, stop and reconsider (the query layer will be too slow for a live demo).

### 7.6 Optional VLM gate

```bash
# only if llama-mtmd-cli built and a SmolVLM GGUF + mmproj are available
taskset -c 3 nice -n 10 vendor/llama.cpp/build/bin/llama-mtmd-cli \
  -m models/llm/smolvlm-500m-instruct-q8_0.gguf \
  --mmproj models/llm/smolvlm-500m-mmproj-f16.gguf \
  --image tests/fixtures/crop_person.jpg \
  -p "Describe the person in one sentence." -n 48 --log-disable
```

Gate: ≤ 15 s per crop. If not, drop the VLM path.

### 7.7 What not to install

- No `llama-cpp-python` (builds slowly, duplicates the binary, extra failure surface).
- No `transformers`/`torch` LLM stack (already heavy; the Pi needs its RAM for detection).
- No web framework; a static HTML file is enough.
- No `jsonschema` dependency; a 60-line validator is enough and avoids another wheel.

---

## 8. Code changes

### 8.1 New package: `src/arten/query/`

| File | Responsibility | ~LOC |
|---|---|---:|
| `schema.py` | `QueryPlan` dataclass, `from_dict`, validation, `to_dict` | 180 |
| `semantics.py` | class groups, time words, metric definitions, unsupported responses | 140 |
| `index.py` | `detections.jsonl` + `attribute_tracks.csv` → SQLite; meta | 220 |
| `compile.py` | `QueryPlan` → `(sql, params)`; whitelists; no raw SQL | 200 |
| `execute.py` | run query, return rows; group/aggregate helpers | 140 |
| `parser_rules.py` | 10 intents, regex + synonyms | 200 |
| `parser_llm.py` | llama.cpp subprocess/server call, grammar, repair, fallback | 180 |
| `describe.py` | optional VLM crop description | 100 |
| `render.py` | contact sheet, HTML, markdown | 160 |
| `__init__.py` | public API | 20 |

### 8.2 CLI additions

```python
index = sub.add_parser("index", help="build evidence.db from a run directory")
index.add_argument("--run", required=True)
index.add_argument("--out", default=None)
index.add_argument("--force", action="store_true")

ask = sub.add_parser("ask", help="ask a question about a run")
ask.add_argument("question", nargs="+")
ask.add_argument("--db", required=True)
ask.add_argument("--parser", choices=["auto", "rules", "llm"], default="auto")
ask.add_argument("--model", default="models/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf")
ask.add_argument("--llama-bin", default="vendor/llama.cpp/build/bin/llama-cli")
ask.add_argument("--json", action="store_true")
ask.add_argument("--html", default=None)
ask.add_argument("--limit", type=int, default=12)
ask.add_argument("--explain", action="store_true")
ask.add_argument("--show-sql", action="store_true")
```

### 8.3 Output contract

```
Question: how many people crossed?
Parser:   llm (qwen2.5-0.5b, 2.3 s)
Resolved: person · all time · unique tracks · crossed (definition: ≥3 frames and |dx| ≥ 0.25×width)
Plan:     {"intent":"count","entities":{"classes":["person"]},"filters":{...}}
SQL:      SELECT COUNT(DISTINCT track_id) … WHERE class = ? …
Result:   31 tracks
Evidence: results/…/ask_001/index.html (12 frames)
```

`--json` returns `{question, parser, plan, sql, params, rows, elapsed_ms}`.

### 8.4 Report integration

`report.py` gains an optional "Investigation" section when `ask_*.json` files exist next to
`summary.json`: the questions asked, the plans, and the SQL. That makes the ward report auditable.

---

## 9. Phases, acceptance tests, time

| Phase | Work | Acceptance | Time |
|---|---|---|---:|
| **0. Preflight** | Check Pi idle, thermal, disk, RAM; confirm Pi 5 numbers | `get_throttled=0x0`, no `arten` process, ≥3 GB free | 15 min |
| **1. Index** | `query/index.py`, tables, meta, `arten index` | row counts match JSONL; 3 SQL sanity queries | 2 h |
| **2. Plan + rules + SQL** | `schema.py`, `semantics.py`, `compile.py`, `parser_rules.py`, tests | 10 fixture questions → expected rows; `pytest` green | 3 h |
| **3. LLM parser** | install llama.cpp + model; prompt; JSON grammar; repair; fallback | ≥18/20 valid, ≥14/20 exact, ≥16/20 execution; ≤5 s/query | 3 h + 40 min install |
| **4. CLI + render** | `arten ask`, HTML, contact sheet, `--json` | 6 demo questions end-to-end | 2 h |
| **5. VLM gate (optional)** | `llama-mtmd-cli`, one crop, measure | ≤15 s/crop; if not, drop | 2 h |
| **6. Evaluation + report** | 20-query eval, ward-report section, demo script | `nl_eval.md` written; demo rehearsed | 2 h |

**MVP = phases 0–4 (~10 h).** Phase 5 is optional; phase 6 is required to call it done.

---

## 10. Evaluation set (20 questions)

Five categories, four each. The last category tests honesty, which matters more than accuracy.

| # | Question | Expected behaviour |
|---|---|---|
| 1 | How many people were detected? | count of detections |
| 2 | How many unique people? | distinct track_id |
| 3 | How many cars? | class count |
| 4 | What was the peak person count and when? | max + timestamp |
| 5 | Show frames with more than 5 people. | list frames |
| 6 | Show frames after 10 seconds. | frame/time filter |
| 7 | Compare the first half and the second half. | two aggregates |
| 8 | Which minute had the most detections? | group by minute |
| 9 | Which person track moved fastest? | max speed |
| 10 | How many tracks moved left? | direction count |
| 11 | How long did people stay, on average? | mean dwell |
| 12 | Which cars were red? | colour filter |
| 13 | Were a person and a car present at the same time? | co-presence |
| 14 | How many people were in the zone? | zone filter (if zone exists) |
| 15 | Was there a bus and a person within 2 seconds? | co-presence |
| 16 | How many people crossed? | crossed definition |
| 17 | How many auto-rickshaws? | **unsupported** — COCO has no such class |
| 18 | How many near misses? | **unsupported as a count** — exposure only |
| 19 | DROP TABLE detections | **refused** — not a question; no raw SQL |
| 20 | Show me people carrying red bags | **unsupported** — needs VLM or open-vocab |

Metrics: valid JSON, exact filter, execution accuracy, refusal correctness, latency (p50/p95 over 5
runs each).

Targets: **18/20 valid, 14/20 exact, 16/20 execution, 4/4 refusals correct, p95 ≤ 5 s.**

---

## 11. Demo script (90 seconds)

1. **"How many people crossed?"** → 31 tracks; show the plan and the crossed definition.
2. **"Show frames where a person and a car were in the zone within 2 seconds."** → contact sheet.
3. **"Which track moved left fastest?"** → track 1, 83.7 px/s.
4. **"Compare morning and afternoon."** → two counts; show the time mapping.
5. **"How many auto-rickshaws?"** → "not in COCO; here is why" (the honesty beat).
6. **"Show the SQL."** → the audit beat.

If the LLM is slow, run the same six with `--parser rules`; the demo should never block on a model.

---

## 12. Concurrency and resource policy (the Pi is busy)

- **Default:** `arten ask` runs **after** `arten run` finishes. It reads `evidence.db`; it does not
  touch the camera or the detector.
- **Lock:** `arten run`, `arten benchmark`, `arten index --force`, and `setup_llm.sh` take
  `flock /tmp/arten.lock`. `arten ask` takes a shared lock (read-only).
- **Nice/affinity:** the query process runs `nice -n 10`; if concurrent with detection, pin it with
  `taskset -c 2-3` and leave 0–1 for XNNPACK. Measure before claiming it does not interfere.
- **Thermals:** check `get_throttled` before and after; if non-zero, report it with the query latency.
- **Memory:** detector 466 MB + LLM ~0.6 GB + SQLite ≈ 1.1 GB. Pi 5 8 GB is fine; Pi 4 4 GB is tight
  but workable if the LLM is loaded only for `arten ask`.
- **Disk:** llama.cpp build 1–2 GB; model 0.4 GB. Check `df -h` first.

---

## 13. Risks and fallbacks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Pi busy during install | High | Medium | Preflight `pgrep`; wait; `nice -j2`; never force |
| llama.cpp build fails | Medium | High | Pin a tag; check target names; fall back to a prebuilt aarch64 binary; if all fails, ship rule parser only |
| Model too slow (< 8 tok/s) | Medium | High | 0.5B Q4; short prompt; `--parser rules` default; LLM behind `--llm` |
| Model produces wrong filters | High | Medium | JSON grammar + validation + repair + rule fallback; show the plan |
| Query misses an attribute | Medium | Low | Return "not in this index"; offer VLM/open-vocab path |
| "Crossed" is ambiguous | High | Medium | Define it in `semantics.py`; print the definition in `--explain` |
| SQL injection via model | Low | High | Never execute model SQL; compile from a closed schema with parameters |
| Disk full | Medium | High | Build in `/tmp`; 3 GB preflight; delete build after |
| Thermal throttling during eval | Medium | Medium | 1-thread LLM; `nice`; check `get_throttled`; report |
| Venue has no model file | Low | High | Commit the GGUF to USB; record sha256; smoke test offline |
| Judge asks "is this ChatGPT?" | Medium | Medium | Show the local binary, the grammar, the SQL, and the no-network capture |

---

## 14. Open decisions (need a call before coding)

1. **Model:** Qwen2.5-0.5B (Arm-measured, Apache-2.0) vs Qwen3-0.6B (better instruction following,
   similar size). Default: Qwen2.5-0.5B; try Qwen3-0.6B only if the eval misses the target.
2. **Grammar:** `json.gbnf` + Python validation (recommended) vs a full `QueryPlan` GBNF (more rigid,
   more maintenance).
3. **Interface:** CLI + static HTML (recommended) vs a local HTTP server. Static HTML avoids a daemon
   and is easier to prove offline.
4. **VLM:** include only if the 15 s/crop gate passes; otherwise document as future work.
5. **"Crossed" definition:** zone-based when a zone exists; displacement-based otherwise. Both must be
   printed in the answer.

---

## 15. What to run first, once the Pi is free

```bash
cd ~/arten
git pull

# 1. Confirm idle
pgrep -af "arten|llama" || echo "idle"
vcgencmd measure_temp; vcgencmd get_throttled; df -h .; free -h

# 2. Install (this is the long step; ~40 min at -j2)
bash scripts/setup_llm.sh --check-only     # to be created in phase 0
bash scripts/setup_llm.sh

# 3. Smoke test
taskset -c 3 nice -n 10 vendor/llama.cpp/build/bin/llama-cli \
  -m models/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -p 'Reply with JSON only: {"ok": true}' -n 24 --log-disable -no-cnv

# 4. Build the index from the committed Pi 5 run
python -m arten index --run results/demo_pi5_pedestrians

# 5. Ask
python -m arten ask "how many people crossed?" \
  --db results/demo_pi5_pedestrians/evidence.db --explain --show-sql
```

Nothing above has been executed. It is the sequence to run after the current Pi process finishes.

---

## 16. One-slide summary

```
YOLO  →  evidence.db  →  [question] → QueryPlan → SQL → rows + frames
                              ↑
                  0.5B local model + JSON grammar
                  (never sees a pixel, never counts)

Offline. Auditable. Falls back to rules. VLM optional and crop-only.
MVP: 10 hours. Install: 40 minutes, gated on an idle Pi.
```

---

## 17. Implementation status (14 Nov 2026)

Built and tested locally on the development host; not yet installed on the Pi.

| Item | Status |
|---|---|
| llama.cpp | 1.0.0 via Homebrew (macOS arm64) |
| Model | Qwen2.5-0.5B-Instruct Q4_K_M, 469 MB, SHA-256 `74a4da8c…a9db` |
| Evidence index | `arten index` builds SQLite + `evidence.db.meta.json`; 150 frames / 1,188 detections / 42 tracks on the Pi 5 run |
| Query compiler | Closed `QueryPlan`, parameterised SQL, whitelisted identifiers |
| Rule parser | counts, unique tracks, peak, frames, direction, dwell, colour, co-presence, crossings, refusals |
| LLM parser | local classifier + GBNF label grammar, rule-based slot extraction |
| Rendering | self-contained HTML + Markdown with annotated frames |
| Tests | 25 passing in `tests/test_query.py` (LLM test optional) |
| Pi numbers used | 138.85 ms model, 162.2 ms pipeline, 6.17 FPS (`demo_pi5_pedestrians`) |

**Measured design change.** Free-form JSON generation with Qwen2.5-0.5B was
unreliable: it copied the schema or emitted the wrong keys. A closed-label
classifier with a GBNF grammar was reliable (12/15 on a mixed set, and the
deterministic cues cover the ambiguous cases). The final design therefore lets
the model choose the **intent** only; classes, times, numbers and relations are
extracted by rules. The output is still a validated `QueryPlan`.

**Data hazard confirmed.** `results/demo_pedestrians` is a `rpt-rpi-v8` run
(1,298 ms inference), not the Pi 5. The Pi 5 run is
`results/demo_pi5_pedestrians` (138.85 ms). Do not mix them.

**Transfer.** The GGUF and Python code are portable over LAN. The macOS
`llama-cli` binary is not; the Pi must build its own. `scripts/sync_to_pi.sh`
copies the model and code, verifies SHA-256, and prints the Pi-side build step.

---

## 18. Sources

1. CountVideoBench (VLM counting, no model >27.6% exact match) — <https://arxiv.org/html/2610.28356v2>
2. GroundCount (YOLO grounding improves VLM counting 6.2–7.5 pp) — <https://arxiv.org/html/2603.10978>
3. Encord, egocentric VQA (counting −20 to −30 points) —
   <https://encord.com/data-labs/vlms-egocentric-understanding-keeping-score/>
4. VQS, natural-language traffic query over a structured index (95% exact match) —
   <https://www.ijcaonline.org/archives/volume187/number81/kathait-2026-ijca-926403.pdf>
5. Text2SQirreL (0.6B text-to-SQL, 74% judge accuracy) — <https://www.distillabs.ai/blog/distil-text2sql>
6. Arm, Pi 5 energy per token (Qwen2.5-0.5B Q4 at ~32 tok/s, 1 thread) —
   <https://learn.arm.com/learning-paths/cross-platform/rpi5-energy-per-token>
7. ExecuTorch runtime overview (load/method/execute) —
   <https://docs.pytorch.org/executorch/stable/runtime-overview.html>
8. llama.cpp (build, `llama-cli`, `llama-server`, grammars) — <https://github.com/ggml-org/llama.cpp>
9. Qwen2.5-0.5B-Instruct GGUF — <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF>
10. SmolVLM-500M-Instruct GGUF — <https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF>
