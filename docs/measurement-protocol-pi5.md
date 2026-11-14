# Measurement protocol: ExecuTorch INT8 on Raspberry Pi 5

**Prepared:** 14 November 2026
**Purpose:** a defensible, reproducible protocol for the numbers Arten reports. Written after auditing
what this repo measures today against what the 2026 literature, MLPerf rules and Arm's own guidance
require. It is deliberately stricter than a hackathon needs — because the *gap* between this and a
typical demo is the differentiator.

> **The one finding that should change the benchmark:** Arm's own PyTorch team observed that on a
> Raspberry Pi 5 without active cooling, ExecuTorch + XNNPACK latency **increases over successive
> runs** as the SoC heats up. A "warm up 10, measure 50" benchmark therefore measures the coolest,
> fastest state the device will ever be in. The honest number is the sustained one.
> Source: <https://pytorch.org/blog/efficient-edge-ai-on-arm-cpus-and-npus>

---

## 1. What the protocol has to answer

A reader should be able to answer these from the output alone, without asking a question:

1. How long does **one inference** take, on this exact board, at this exact clock, with this exact
   thread count, after the device has been running for 20 minutes?
2. How long does the **whole pipeline** take, and which stage dominates?
3. What **throughput** does that imply, and how many frames miss a stated deadline?
4. How much **energy** does one accepted frame cost?
5. How much **memory** does the process actually hold?
6. How **stable** is all of this — what is the p99, the max, the drift?
7. What **accuracy** does the quantised artifact have on data that was not used to tune it?
8. What **cannot** be concluded from this run?

If the output cannot answer #1–#8, it is a demo, not a measurement.

---

## 2. Metric definitions (write these into the report)

Use these names consistently. Ambiguity between "latency" and "throughput" is the most common way
edge numbers become misleading.

| Metric | Definition | Boundary | Unit |
|---|---|---|---|
| `t_load` | Deserialise `.pte`, create `Program`, `load_method` | from process start after imports to method ready | ms |
| `t_first` | First `execute` after load | includes lazy allocation, page faults, delegate init | ms |
| `t_model` | `Method.execute` only, warm | input tensor already resident; output not copied | ms |
| `t_pre` | Letterbox + normalise + tensor write | source frame in memory → input tensor ready | ms |
| `t_post` | Decode + confidence filter + NMS + inverse letterbox | output tensor → detection list | ms |
| `t_track` | Tracker update | detection list → tracks | ms |
| `t_annot` | Draw boxes/labels/HUD | detection list → annotated frame | ms |
| `t_write` | CSV/JSONL/JPEG write | in memory → flushed | ms |
| `t_pipeline` | `t_pre + t_model + t_post + t_track + t_annot + t_write` | excludes video encode and final mux | ms |
| `t_decode` | Video frame decode | container → BGR frame | ms |
| `fps_model` | `1000 / mean(t_model)` | **model only** | FPS |
| `fps_pipeline` | `1000 / mean(t_pipeline)` | **end to end, excluding encode** | FPS |
| `p50/p90/p95/p99/max` | Percentiles over `n` warm samples | state `n` and the window | ms |
| `drift` | `median(last 100) / median(first 100) - 1` | sustained run | % |
| `miss_rate` | frames with `t_pipeline > budget` / total | budget must be stated (e.g. 134 ms) | % |
| `goodput` | accepted frames / second | accepted = detected ∧ tracked ∧ written ∧ on time ∧ not OOD | fps |
| `E_inf` | joules per warm inference | whole-board input, idle subtracted | J |
| `E_frame` | joules per accepted frame | includes pipeline overhead | J |
| `P_idle` | idle power, 5-min mean, screen/SSH/WiFi state stated | whole-board input | W |
| `peak_rss` | `VmHWM` | process, includes Python/torch | MB |
| `model_mb` | `.pte` file size | on disk | MB |
| `runtime_mb` | ExecuTorch + delegate shared objects | on disk / RSS | MB |
| `mAP@0.5:0.95` | on a labelled held-out set | **not** the demo clip | — |
| `coverage` | empirical conformal coverage at nominal 1−α | block-wise split | % |
| `interval_width` | mean conformal box interval width | pixels | px |

**Never** write "real-time" without `fps_pipeline`, the device, and the resolution.

---

## 3. Reference numbers, and why they are not comparable

| Source | Model | Runtime | Precision | Threads | Latency | Notes |
|---|---|---|---|---:|---:|---|
| Arm model card | YOLO11n | ExecuTorch + XNNPACK | INT8 | **1** | **134.10 ms p50** | Pi 5, 7.46 FPS, 59.17 MB peak |
| Ultralytics docs | YOLO26n | ExecuTorch | FP32 | *not stated* | **144.83 ms** | Pi 5, 9.4 MB, 47.7 mAP |
| Ultralytics docs | YOLO26n | NCNN | FP32 | *not stated* | **67.03 ms** | same board, different runtime |
| This repo (dev host) | YOLO11n | ExecuTorch 1.5.1 | INT8 | *not stated* | 20.34 ms mean | Apple Silicon, **not a Pi** |
| This repo (Pi stack, dev host) | YOLO11n | ExecuTorch 1.1.0 | INT8 | *not stated* | 23.83 ms mean | **still not a Pi** |

The 134 ms and 145 ms numbers differ by 8% in latency but by ~9 mAP in accuracy, and they were not
measured under the same thread count or precision. They must not be placed in one comparison table
without those columns. The protocol exists to make such comparisons legitimate.

---

## 4. Reference environment record (fill this in for every run)

Write this into `summary.json` and into the report. If a field is unknown, write `"unknown"` — do not
omit it.

### Hardware
- Board: Pi 5 model/revision, RAM (4/8/16 GB)
- Cooling: official Active Cooler / third-party fan / passive heatsink / none
- Enclosure: open bench / case / sealed box
- Ambient temperature at start and end (°C)
- Power supply: official 27 W USB-C PD / other; measured voltage under load
- Storage: microSD / NVMe; model and output location
- Camera/USB devices attached

### Software
- OS image and kernel (`uname -a`), page size (`getconf PAGESIZE`; Pi 5 defaults to 16 KB)
- Python version, `torch`, `executorch`, `flatbuffers`, `numpy`, `opencv-python`
- ExecuTorch delegate: XNNPACK; delegate library version
- Model: filename, SHA-256, size, input shape, precision, calibration dataset
- CLI command line, config file, seed

### CPU state
- Governor: `cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor`
- `scaling_max_freq`, `scaling_cur_freq` before/after
- Overclock / `force_turbo` / `temp_limit`
- Thread count used by XNNPACK; CPU affinity (`taskset`); `nice`
- Background services stopped; WiFi/Bluetooth state
- Swap state (`swapoff -a` preferred)

### Measurement
- Warmup count; measured count `n`; run duration
- Whether the model was reloaded between runs
- Whether outputs were written to disk or `/dev/null`
- Whether annotation/video encode were included (state explicitly)

---

## 5. Controls (the part most benchmarks skip)

### 5.1 CPU frequency

The Pi 5 ships with `ondemand` and a single frequency policy for all four cores
(`/sys/devices/system/cpu/cpufreq/policy0`). Frequencies step 1.5–2.4 GHz. For measurement:

```bash
# Record
cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq

# Pin to performance for the run (restore afterwards)
echo performance | sudo tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
echo 2400000  | sudo tee /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq

# Restore
echo ondemand | sudo tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
```

Report which governor was used. A run under `ondemand` and a run under `performance` are different
experiments.

### 5.2 Thermal

The Pi 5 begins throttling at **80 °C** and throttles harder at **85 °C**; a sustained full load with
no cooling can throttle within **9 seconds** and drop to 1.5 GHz. With the official cooler, throttling
can still begin after ~4 minutes and settle around 83 °C. Consequences:

- Cool the board to a stable idle baseline before the first run (15–30 min).
- Log `vcgencmd measure_temp`, `measure_clock arm`, `get_throttled` at 1 Hz for the whole run.
- Report: start temp, max temp, mean clock, min clock, `throttled` before/after, and throttled minutes.
- Report both **cold** (first 60 s) and **sustained** (last 5 min) statistics. Arm observed latency
  *rising* over runs on an uncooled Pi 5 — that is the effect this control exists to expose.

Bitmask (`get_throttled`) — record the hex value and decode:
`0x1` under-voltage, `0x2` arm frequency capped, `0x4` currently throttled, `0x8` soft temp limit,
`0x10000` under-voltage occurred, `0x20000` arm frequency capped occurred, `0x40000` throttled occurred,
`0x80000` soft temp limit occurred.

### 5.3 Threads

Test **1, 2 and 4 threads**; do not assume more is faster. Evidence:

- Arm's Pi 5 LLM sweep: 4 threads was the worst setting at every clock cap; at 2.4 GHz it used **61%
  more energy per token while running 25% slower** (memory-bandwidth-bound).
- XNNPACK's own docs report multi-thread gains on phones, but the correct count is workload-dependent.
- There is a long history of Python bindings silently ignoring `num_threads` for FP32 TFLite; verify
  the actual thread count with `top -H -p <pid>` or `ps -L`, not by trusting the API.
- In ExecuTorch C++, the threadpool is set explicitly:
  `executorch::extension::threadpool::get_threadpool()->_unsafe_reset_threadpool(n);`
  and must be called before loading the model.

### 5.4 Affinity and scheduling

```bash
taskset -c 3 python -m arten ...        # single core
taskset -c 0-3 python -m arten ...      # all cores
nice -n -20 ...                         # if permitted
```

Record the affinity. A single-core run and a four-core run are different experiments.

### 5.5 Background noise

- Stop desktop services, package updates, indexing, Bluetooth, Avahi.
- Prefer Ethernet over WiFi, or block WiFi (`rfkill block wifi`) if not needed; Arm measured the WiFi
  rail moving idle power between 1.45 and 1.76 W across runs, enough to reorder close settings.
- Disable swap or ensure zero swapping.
- Write outputs to NVMe or `/tmp` if possible; microSD write latency can dominate `t_write`.

### 5.6 Power supply

Use the official 27 W supply. Under-voltage throttling shows as `0x1`/`0x10000` in `get_throttled`;
it invalidates latency and energy numbers silently.

---

## 6. Run design

### 6.1 Three phases, not one

| Phase | What it measures | Procedure |
|---|---|---|
| **Cold** | `t_load`, `t_first` | Fresh process, fresh page cache if possible, no warmup; 5 repetitions, report each |
| **Warm** | `t_model`, `t_pipeline`, percentiles | 10–20 warmups discarded, then `n ≥ 1000` samples |
| **Sustained** | drift, thermals, p99, goodput | 20–30 min at the target rate; log at 1 Hz; report first 100 vs last 100 |

The headline number should be the **sustained** one, with the warm one beside it and the cold one in
an appendix. That is the opposite of common practice, and that is the point.

### 6.2 Sample size

- `n ≥ 1000` for a usable p99; `n ≥ 3000` to quote p99 with a narrower interval.
- MLPerf's own table: a 99th-percentile estimate with 99% confidence and 0.05% margin needs **262,742**
  inferences; 90th percentile with 0.5% margin needs **23,886**. A hackathon does not need that, but
  it must not quote a p99 from 100 samples as if it were stable.
- Always report `n` and a bootstrap 95% CI for the median and p99 (10,000 resamples is fine).

### 6.3 Order

Thermal state persists between runs (the 2026 Jetson audit found the realised clock regime of a "fast"
run depended on the *previous* run's state). Therefore:

- Randomise or alternate the order of settings (e.g. 1-thread, 4-thread, 2-thread, then reverse).
- Repeat each setting 3–5 times; report mean ± SD and coefficient of variation.
- Do not compare a cold run of setting A with a hot run of setting B.

### 6.4 Repetitions

For a hackathon: 3 repetitions of each configuration. For a report: 5. For a paper: 10 with
randomised blocks. If the CV of a metric is > 5%, the difference you are claiming is probably noise
unless you have repeats.

---

## 7. Instrumentation

### 7.1 Latency

```python
import time
t0 = time.perf_counter_ns()
out = method.execute(inputs)
t1 = time.perf_counter_ns()
```

Rules:
- Use `perf_counter_ns`, not `time.time()`.
- Preallocate the input tensor and reuse it; creating a tensor inside the timed region measures the
  allocator, not the model.
- Time `execute` only for `t_model`. Time each pipeline stage separately with the same clock.
- Keep the timer in-process; do not add logging inside the loop.
- Flush and close files outside the measured region; measure `t_write` separately.

### 7.2 Operator attribution (ETDump)

ExecuTorch's Developer Tools give operator-level timings and delegate attribution:

- Build with `-DEXECUTORCH_BUILD_DEVTOOLS=ON -DEXECUTORCH_ENABLE_EVENT_TRACER=ON`.
- Generate an `ETRecord` at export (optional, for source mapping) and an `ETDump` at runtime.
- Inspect with the Python `Inspector` API (`print_data_tabular()`, `find_total_for_module()`).
- Delegate (XNNPACK) events are reported in **cycles**, not milliseconds; use them for relative
  attribution, not headline latency.
- Event tracing adds overhead. Use it to answer "which ops are not delegated?" and "where does time
  go?", then re-run without tracing for the numbers you publish.

The most useful question ETDump answers for this repo: **what fraction of the graph is actually
delegated to XNNPACK?** Any `aten.*` op left on portable kernels is a candidate bottleneck. The
ExecuTorch issue tracker has multiple cases where a single non-delegated `aten.mm` dominated runtime.

### 7.3 Thermals and clocks

```bash
while true; do
  echo "$(date +%s),$(vcgencmd measure_temp),$(vcgencmd measure_clock arm),$(vcgencmd get_throttled)"
  sleep 1
done | tee thermal.csv
```

### 7.4 Energy

Two options, in order of preference:

**A. USB-C power meter / inline INA260 (preferred).** Put it between the supply and the Pi. Log at
≥ 1 Hz. Run 5 min idle, then 5 min at the target rate. `E = ∫P dt`, subtract idle, divide by accepted
frames. Report `P_idle`, `P_run`, `E_inf`, `E_frame`.

**B. Pi 5 PMIC rails (secondary).** `vcgencmd pmic_read_adc` returns current and voltage per rail.
Sum `V×I` over rails that have both. Caveats:
- It is **not** wall power: `EXT5V_V` has voltage but no current, and USB/peripheral/fan/regulator
  losses are excluded.
- A full read takes ~25 ms (~40 Hz); a single named channel is faster.
- It undercounts; one user calibration found `real ≈ pmic × 1.1451 + 0.5879`, but that is board-
  specific and must not be presented as a general correction.

If the inference is very short (< 1 ms), a single event is below the sampler's resolution. Use the
batch-slope method: run batches of B = 100/500/1000/2000, fit `E(B) = E_fixed + B·E_marginal`, and
report the slope. For Arten's ~134 ms inference this is not necessary; a 1 Hz meter resolves it.

### 7.5 Memory

- `VmHWM` (peak RSS) and `VmRSS` (current) from `/proc/self/status`.
- `resource.getrusage(RUSAGE_SELF).ru_maxrss` (KB on Linux) as a cross-check.
- Model file size and runtime/delegate `.so` sizes separately.
- Note that Python + torch dominate RSS; the model's 59 MB peak (Arm's card) is not process RSS.

### 7.6 Accuracy

- Use a labelled held-out set, not the demo clip. If none exists, label 50–100 frames once (or use
  SAM 3 / a VLM as a teacher and review).
- Report mAP@0.5:0.95 and mAP@0.5 for INT8 vs FP32 if both artifacts exist.
- If no labels: report **count agreement** on 20 frames (MAE, bias) and call it that, never "mAP".
- For conformal: split by **time blocks**, not random frames, because adjacent frames are correlated.
  Calibrate on the first 70%, evaluate on the last 30%; report empirical coverage and mean interval
  width. State the exchangeability caveat explicitly.

### 7.7 No-network proof

```bash
sudo tcpdump -i any -n -c 1000 > /tmp/net.log &
strace -f -e trace=network -o /tmp/net.strace python -m arten run ...
```

Report: zero outbound connections during the run (excluding loopback), or list them.

---

## 8. Statistics

Report this table for every metric:

| Statistic | Why |
|---|---|
| `n` | without it, percentiles are meaningless |
| mean, SD | comparability with vendor numbers |
| median | robust central value |
| p90, p95, p99 | tail behaviour; the number a deadline cares about |
| max | worst case; reveals stalls |
| CV = SD/mean | decides whether a difference is real |
| bootstrap 95% CI for median and p99 | uncertainty without distributional assumptions |
| drift (last/first − 1) | thermal or memory degradation |

For A/B comparisons:
- Use a paired test (Wilcoxon signed-rank) if runs are paired; otherwise a Mann–Whitney U.
- Report effect size and CI, not just p.
- If the CIs overlap and the difference is < 5%, call it "within noise".

---

## 9. Reporting template

### 9.1 JSON (write to `summary.json`)

```json
{
  "protocol_version": "1.0",
  "run_id": "2026-11-14T10:00:00+05:30",
  "hardware": {
    "board": "Raspberry Pi 5 8GB",
    "cooling": "official_active_cooler",
    "ambient_c_start": 24.0,
    "ambient_c_end": 25.5,
    "psu": "official_27w",
    "storage": "nvme"
  },
  "software": {
    "os": "Raspberry Pi OS 64-bit",
    "kernel": "6.6.x",
    "page_size": 16384,
    "python": "3.11.x",
    "torch": "2.10.0",
    "executorch": "1.1.0",
    "delegate": "xnnpack",
    "model_sha256": "154c54b5...",
    "model_mb": 4.73,
    "input_shape": [1, 3, 640, 640],
    "precision": "int8"
  },
  "cpu": {
    "governor": "performance",
    "scaling_max_freq_khz": 2400000,
    "threads": 1,
    "affinity": "3",
    "throttled_before": "0x0",
    "throttled_after": "0x0"
  },
  "phases": {
    "cold": {"t_load_ms": 0, "t_first_ms": 0, "n": 5},
    "warm": {"n": 1000, "t_model_ms": {"mean": 0, "sd": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0, "ci95_p99": [0, 0]}},
    "sustained": {"duration_s": 1200, "n": 8900, "drift_pct": 0, "temp_max_c": 0, "clock_min_mhz": 0, "throttled_minutes": 0}
  },
  "pipeline": {
    "t_pre_ms": 0, "t_model_ms": 0, "t_post_ms": 0, "t_track_ms": 0,
    "t_annot_ms": 0, "t_write_ms": 0, "t_decode_ms": 0, "t_pipeline_ms": 0
  },
  "throughput": {"fps_model": 0, "fps_pipeline": 0, "goodput": 0, "miss_rate_pct": 0},
  "memory": {"peak_rss_mb": 0, "runtime_mb": 0},
  "energy": {"p_idle_w": 0, "p_run_w": 0, "j_per_inference": 0, "j_per_accepted_frame": 0, "method": "usb_c_meter"},
  "accuracy": {"map_50_95": null, "map_50": null, "labelled_frames": 0, "conformal_coverage": null, "interval_width_px": null},
  "notes": []
}
```

### 9.2 Markdown table (for the report)

```
| Metric | Value | Conditions |
|---|---:|---|
| Model | YOLO11n INT8, 4.73 MB, SHA-256 154c54b5… | XNNPACK, 640×640 |
| Cold load / first inference | X ms / X ms | fresh process |
| Warm model-only | X ms p50 / X ms p99 (n=1000) | 1 thread, 2.4 GHz, 24 °C |
| Pipeline | X ms p50 / X ms p99 | pre+infer+post+track+annot+write |
| Throughput | X FPS model / X FPS pipeline | stated budget 134 ms |
| Deadline misses | X % | 134 ms |
| Sustained drift | X % over 20 min | active cooler, 24→X °C |
| Energy | X J/inference / X J/frame | USB-C meter, idle subtracted |
| Peak RSS | X MB | process, includes Python |
| Accuracy | X mAP@0.5:0.95 | held-out X frames |
| Conformal coverage | X % at 90% nominal | block-wise split |
```

---

## 10. Pitfalls, with evidence

1. **Uncooled Pi 5 gets slower over successive runs.** Arm observed increasing ExecuTorch + XNNPACK
   latency on a Pi 5 as it heated. Report cold vs sustained separately; do not report the cold burst
   as the sustained rate.
2. **`ondemand` governor.** The default. A short run may never reach 2.4 GHz. Pin `performance` or
   pre-heat, and record the governor.
3. **Thread count is not monotonic.** Arm's LLM sweep found 4 threads worst at every clock cap
   (61% more energy/token, 25% slower at 2.4 GHz). Test 1/2/4 for the detector.
4. **Python may not apply the thread count you think.** Verify with `top -H`/`ps -L`.
5. **Model load and first inference are not inference.** ExecuTorch's `Module` lazy-loads on first
   `forward`; `to_executorch` memory planning does not guarantee allocation-free execution. Separate
   `t_load`, `t_first`, `t_warm`.
6. **Non-delegated ops can dominate.** Check ETDump/Inspector for `aten.*` nodes outside the XNNPACK
   partition. One `aten.mm` can turn a 25 ms model into a 3.5 s model (real ExecuTorch issue).
7. **Page size.** Pi 5 defaults to 16 KB pages; a 4 KB kernel is available. 16 KB gives ~7% on random
   memory access and ~1.3% on Geekbench single-core. Record `getconf PAGESIZE`; a cross-board
   comparison with different page sizes is not apples-to-apples.
8. **PMIC ≠ wall power.** `vcgencmd pmic_read_adc` has no input current; it excludes USB, fan and
   regulator losses; a full read is ~25 ms. Use a USB-C meter for absolute energy, PMIC for relative.
9. **Idle power drifts on WiFi.** Arm measured 1.45–1.76 W idle variation on the WiFi rail; enough to
   reorder close settings. Block WiFi or report idle.
10. **SD card I/O can dominate.** Model load and JPEG/CSV writes may be slower than inference. Use
    NVMe or `/tmp`; measure `t_decode` and `t_write` separately.
11. **Annotation can dominate the pipeline.** Drawing boxes and text in PIL is slow; measure it.
12. **NMS cost scales with detections.** Dense scenes change the model/pipeline ratio; report both.
13. **Mean hides the tail.** Report p99 and max; a 7.46 FPS mean can still miss deadlines.
14. **`ru_maxrss` is peak, not current.** Report `VmHWM` and `VmRSS`; note Python overhead.
15. **Vendor numbers differ in threads and precision.** Arm: 1 thread INT8. Ultralytics: unspecified
    threads FP32. Never compare without those two columns.
16. **Quantisation accuracy is dataset-specific.** COCO mAP is not your clip's mAP. Report both,
    clearly separated.
17. **Thermal state carries across runs.** Randomise/alternate order; the Jetson audit found a run's
    realised clock regime depended on the previous run's state.
18. **Video decode is often the real bottleneck.** On a Pi 5 single stream, decode can cap end-to-end
    FPS well below the NPU/CPU inference rate. Measure it.
19. **A p99 from 100 samples is not a p99.** Report `n` and a CI; 1000+ for a rough p99.
20. **"Real-time" is not a number.** State FPS, resolution, device, thread count and budget.

---

## 11. What this repo does today, and the gap

**Already good:**
- Model-only timer around `Method.execute`, separate from pipeline timer.
- p50/p95, warmups, throughput both ways, `ru_maxrss`, checksum, environment metadata.
- Explicit labelling that dev-host numbers are not Pi numbers.
- A benchmark harness that can be re-run.

**Missing (in priority order):**
1. `t_load` / `t_first` separation.
2. `n` and bootstrap CI; p99.
3. Sustained 20–30 min run with thermal log and drift.
4. Energy per inference / per accepted frame.
5. Deadline miss rate and useful goodput.
6. Thread count, governor, clock, page size, cooling, ambient in the report.
7. Per-stage breakdown including decode, annotation and write.
8. Delegate coverage from ETDump (what fraction is actually XNNPACK).
9. Accuracy on a labelled held-out set; conformal coverage with block split.
10. No-network proof and retention manifest.

**Suggested CLI:**
```bash
arten benchmark --protocol \
  --input data/samples/pedestrians_15s.mp4 \
  --phase all \            # cold,warm,sustained
  --warmup 20 --samples 1000 --sustained 1200 \
  --threads 1 --affinity 3 --governor performance \
  --energy usb-c-meter --thermal-log thermal.csv \
  --out results/protocol
```
Outputs: `protocol.json`, `PROTOCOL.md`, `thermal.csv`, `latency.csv`, `energy.csv`, `env.json`.

---

## 12. A 90-minute execution plan

| Time | Step | Output |
|---|---|---|
| 0–10 | Freeze environment: stop services, block WiFi, set governor, record `doctor --json`, cool to baseline | `env.json` |
| 10–20 | Cold: 5 fresh-process runs, record `t_load`/`t_first` | cold table |
| 20–35 | Warm: 1000 inferences at 1, 2, 4 threads; randomise order | warm table |
| 35–55 | Sustained: 20 min at the chosen thread count, 1 Hz thermal log | drift, p99, thermal |
| 55–65 | Energy: 5 min idle + 5 min run with USB-C meter | J/inference |
| 65–75 | Pipeline breakdown: 100 frames, per-stage timings | stage table |
| 75–85 | Accuracy: 50–100 labelled frames + conformal block split | mAP, coverage |
| 85–90 | Write `PROTOCOL.md`, check every claim maps to a row | report |

If there is no USB-C meter, skip energy and write "energy not measured"; do not substitute PMIC rails
without the caveat.

---

## 13. Threats to validity (state these)

- One board, one clip, one ambient temperature; this is not a fleet study.
- The model is COCO-trained; accuracy on Bengaluru footage is a separate, harder question.
- Sustained 20 min is not a 24/7 deployment; thermal equilibrium may not be reached.
- The 16 KB page size and kernel version are part of the result; a different image may differ.
- Conformal coverage assumes exchangeability; video frames are correlated, hence the block split.
- Vendor comparisons are not controlled experiments; they are context, not evidence.
- Energy measured at the USB-C input includes the board and peripherals, not just the SoC.

---

## 14. What not to claim

| Do not claim | Because | Say instead |
|---|---|---|
| "7.46 FPS real-time" | 7.46 FPS is the model-only reference, 1 thread | "7.46 model-only FPS on the reference Pi 5, 1 thread; X pipeline FPS measured here" |
| "mAP 38.9 on our clip" | No ground truth on the clip | "COCO val mAP 38.9 from the model card; qualitative on our clip" |
| "Energy 5 W" | PMIC rails are not wall power | "USB-C input X W; PMIC rail sum Y W (SoC only)" |
| "p99 = X ms" from 100 runs | Too few samples | "p99 = X ms (n=100, wide CI); sustained run n=8900 gives Y ms" |
| "Same as Arm's 134 ms" | Threads/precision/runtime may differ | "Within X% of Arm's 1-thread INT8 reference under matched conditions" |
| "No thermal throttling" | Only true if `get_throttled` shows 0 and temp < 80 °C | "throttled=0x0 throughout, max temp X °C" |
| "Runs in real time" | Undefined | "X FPS end-to-end at 640×640 on Pi 5, 1 thread, 24 °C" |

---

## 15. Sources

1. Arm/PyTorch, *Efficient Edge AI on Arm CPUs and NPUs* (ExecuTorch + XNNPACK on Pi 5; thermal
   latency increase without cooling) — <https://pytorch.org/blog/efficient-edge-ai-on-arm-cpus-and-npus>
2. ExecuTorch, *Runtime overview* (load/method/execute, memory planning, lazy `Module`) —
   <https://docs.pytorch.org/executorch/stable/runtime-overview.html>
3. Arm, *ExecuTorch Insights* (record shapes, precision, thread counts, thermal conditions; separate
   startup, repeated inference, model size, runtime size, peak memory) —
   <https://learn.arm.com/insights/executorch>
4. ExecuTorch, *Developer Tools overview* — <https://docs.pytorch.org/executorch/stable/devtools-overview.html>
5. ExecuTorch, *Profiling models* — <https://docs.pytorch.org/executorch/stable/runtime-profiling.html>
6. ExecuTorch, *ETDump* (CMake flags, tracing) — <http://docs.pytorch.org/executorch/stable/etdump.html>
7. ExecuTorch, *Inspector APIs* (delegate events in cycles) — <https://docs.pytorch.org/executorch/stable/model-inspector>
8. ExecuTorch issue #10297 (non-delegated `aten.mm` dominating runtime) —
   <https://github.com/pytorch/executorch/issues/10297>
9. ExecuTorch issue #7880 (release build, threadpool reset) —
   <https://github.com/pytorch/executorch/issues/7880>
10. Arm, *Measure llama.cpp energy per token on Raspberry Pi 5* (governor, clock caps, threads,
    PMIC, idle variation) — <https://learn.arm.com/learning-paths/cross-platform/rpi5-energy-per-token>
11. Arm, *Read power from the Raspberry Pi 5 PMIC* (rail sum, 25 ms read, no input current) —
    <https://learn.arm.com/learning-paths/cross-platform/rpi5-energy-per-token/read-the-pmic>
12. Raspberry Pi, *Heating and cooling Raspberry Pi 5* (80/85 °C thresholds, stress results) —
    <https://www.raspberrypi.com/news/heating-and-cooling-raspberry-pi-5>
13. Raspberry Pi, *Benchmarking Raspberry Pi 5* (16 KB vs 4 KB page size; ±6 run variance) —
    <https://www.raspberrypi.com/news/benchmarking-raspberry-Pi-5>
14. Raspberry Pi, *Use-case-specific thermal performance* (USB meter vs inline analyser vs bench PSU) —
    <https://pip-assets.raspberrypi.com/categories/685-app-notes-guides-whitepapers/documents/RP-010139-WP-1-Use-case-specific%20thermal%20performance%20of%20Raspberry%20Pi%20SBCs.pdf>
15. TUM thesis, *Towards Edge Power Measurement* (batch-slope method; 0.55 mJ/inference ResNet-8 on
    Pi 5; meter vs reference within 3%) — <https://wgeng.site/teaching/theses/tum-msc-chang-thesis.pdf>
16. MLCommons, *MLPerf Inference rules* (single-stream p90, sample counts, early stopping) —
    <https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc>
17. MLCommons, *MLPerf Tiny v1.4* (energy per inference, operating-point trade-offs) —
    <https://mlcommons.org/2026/07/mlperf-tiny-v1-4-results>
18. *Measurement Boundaries and Sequence-Dependent Hardware-State Realization in Edge-AI Energy
    Benchmarking* (558 Jetson runs; requested modes not realised) —
    <https://www.mdpi.com/2079-9292/15/15/3372>
19. *LLM Inference at the Edge* (thermal degradation curves) — <https://arxiv.org/html/2603.23640v2>
20. *Sustained on-device inference needs a warm-device P99 test* (useful goodput) —
    <https://mubibai.com/sustained-on-device-inference-needs-a-warm-device-p99-test>
21. *Edge AI benchmarks are not numbers until you name the silicon* —
    <https://talkingiot.io/edge-ai-benchmarks-are-not-numbers-until-you-name-the-silicon>
22. XNNPACK README (single vs multi-thread Raspberry Pi numbers) —
    <https://github.com/google/XNNPACK>
23. Ultralytics, *Raspberry Pi guide / YOLO26 benchmarks* (144.83 ms ExecuTorch, 67.03 ms NCNN) —
    <https://docs.ultralytics.com/guides/raspberry-pi>
24. Arm model card, YOLO11n INT8 XNNPACK ExecuTorch (134.10 ms, 7.46 FPS, 59.17 MB, 1 thread) —
    <https://huggingface.co/Arm/yolo11n-int8-xnnpack-executorch-raspberrypi5>
