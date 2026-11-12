# Model combinations on ARMv8.0

Measured on a Raspberry Pi 3B+ (Cortex-A53, 1 GB), 64-bit Bookworm,
torch 2.9.0, ExecuTorch 1.1.0a0+17adba1 (ARMv8.0 build).

The board was firmware-capped at **600 MHz** during all runs
(`throttled=0x50005`, under-voltage). Absolute numbers are pessimistic;
the ranking between combinations still holds. A proper 5 V / 2.5 A supply
should reduce every latency.

Detection counts are from `data/samples/bengaluru_traffic.jpg`, one frame.
No ground-truth labels exist for this image, so counts are indicative.

## 1. Threshold sweep

YOLO11n’s INT8 confidence distribution is shifted low. The default 0.35
misses most riders; 0.15 recovers many, but two-wheeler recall stays weak.

| Model | conf | person | car | motorcycle | bus | truck | p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | 0.35 | 2 | 19 | 0 | 1 | 1 | 1300 ms |
| YOLO11n | 0.20 | 9 | 25 | 4 | 2 | 1 | 1300 ms |
| YOLO11n | 0.15 | 16 | 26 | 5 | 2 | 2 | 1296 ms |
| YOLOv8s | 0.35 | 10 | 21 | 4 | 1 | 2 | 3200 ms |
| YOLOv8s | 0.20 | 18 | 25 | 13 | 1 | 3 | 3199 ms |
| YOLOv9s | 0.35 | 12 | 19 | 8 | 1 | 3 | 5777 ms |
| YOLOv9s | 0.20 | 18 | 24 | 12 | 1 | 4 | 5796 ms |

Key result: at comparable thresholds YOLO11n finds 16 people but only 5
motorcycles, while YOLOv8s finds 18 people and 13 motorcycles. For a
Bengaluru street scene where two-wheelers are 44.5% of road deaths, that
gap matters. YOLOv9s matches YOLOv8s at 1.8x the latency, so it is dominated.

## 2. Every-frame combinations

All models loaded together, run back-to-back on each frame. Model-only
latency; preprocessing, decode, tracking and annotation are extra.

| Combo | p50 / frame | FPS | Peak RSS |
|---|---:|---:|---:|
| YOLO11n | 1297 ms | 0.77 | 417 MB |
| YOLO11n + pose | 2643 ms | 0.38 | 449 MB |
| YOLO11n + DDRNet | 3230 ms | 0.31 | 494 MB |
| YOLO11n + pose + DDRNet | 4610 ms | 0.22 | 534 MB |
| YOLOv8s | 3204 ms | 0.31 | 441 MB |
| YOLOv8s + pose | 4547 ms | 0.22 | 481 MB |
| YOLOv8s + DDRNet | 5127 ms | 0.20 | 507 MB |
| YOLOv8s + pose + DDRNet | 6451 ms | 0.16 | 547 MB |

All eight fit inside 1 GB. Pose adds roughly 1.35 s; DDRNet at 720x960 adds
roughly 1.88 s. Running everything every frame buys little: pose and zones do
not change that fast.

## 3. Layered cadence (recommended)

Run detection every frame, pose every 10th frame, segmentation every 30th
frame. The values below are derived from the measured per-model costs.

| Combo | Est. p50 / frame | Est. FPS |
|---|---:|---:|
| YOLO11n + pose/10 + DDRNet/30 | ~1495 ms | ~0.67 |
| YOLOv8s + pose/10 + DDRNet/30 | ~3401 ms | ~0.29 |

That is almost detection-only cost while still producing posture samples and
static road/sidewalk masks.

## 4. Decision

**Best counting accuracy:** YOLOv8s at `--conf 0.2`, plus pose every 10th
frame and DDRNet every 30th. On the throttled Pi 3B+ that is ~3.4 s/frame;
on a Pi 4 expect roughly 1.1–1.4 s, on a Pi 5 with the prebuilt runtime
~0.49 s. This is the best fit for the project’s premise: people **and**
two-wheelers are counted, and zones/posture come almost for free.

**Best live throughput:** YOLO11n at `--conf 0.15`, plus the same sampled
pose/zones. ~1.5 s/frame on the throttled Pi 3B+, ~0.5–0.6 s on a Pi 4,
~0.16 s on a Pi 5. Weaker two-wheeler recall (5 vs 13) is the price.

**Pi 3B+:** still images or very short clips only. At 0.3–0.8 FPS, motion
attributes are not meaningful. Fix the power supply before trusting the
absolute numbers.

**Pi 4 / Pi 5:** the layered YOLOv8s combination is the most complete
solution. Use YOLO11n only if the demo must run continuously in real time.

## 5. Ground-truth check (manual count)

Counted from four non-overlapping 2x crops of the original 1040x780 image.
Approximate: dense occlusion, tiny distant vehicles and auto-rickshaws make an
exact count impossible. Ranges are given where the count is uncertain.

| Class | Approx. actual | YOLO11n @0.15 | YOLOv8s @0.2 | YOLOv9s @0.2 |
|---|---:|---:|---:|---:|
| person | 30–40 | 16 | 18 | 18 |
| motorcycle | 25–30 | 5 | 13 | 12 |
| car | 45–55 | 26 | 25 | 24 |
| bus | 1 | 2 | 1 | 1 |
| truck | 3–4 | 2 | 3 | 4 |
| auto-rickshaw (not COCO) | 3 | — | — | — |

Recall at the midpoint of each range:

| Model | person | motorcycle | car | mean |
|---|---:|---:|---:|---:|
| YOLO11n @0.15 | 46% | **18%** | 52% | 39% |
| **YOLOv8s @0.2** | **51%** | **46%** | 50% | **49%** |
| YOLOv9s @0.2 | 51% | 43% | 48% | 47% |

YOLOv8s and YOLOv9s are effectively tied on recall; YOLOv8s reaches that
recall in 3.2 s instead of 5.8 s. YOLO11n’s motorcycle recall is the outlier:
it sees roughly one in five two-wheelers.

## 6. Trade-offs

- **Recall vs latency.** YOLOv8s costs 2.5x YOLO11n but recovers 2.6x more
  motorcycles. For a two-wheeler-heavy Indian street, that is the right trade.
- **Resolution is the ceiling.** All models sit near 50% person recall at
  640x640. A 960 export or 2x tiling would raise recall, but costs ~2.25x or
  ~4x per frame and needs a new `.pte`; the prebuilt models do not support it.
- **Class mismatch.** Auto-rickshaws are not in COCO. They are missed or
  counted as car/truck, so “car” counts are not exact for Indian scenes.
- **Threshold.** Lowering to 0.1 would raise recall but add false positives and
  duplicates. 0.2 on YOLOv8s was the best tested operating point.
- **Distant objects.** The 640 input cannot resolve the far traffic; manual
  counts include those vehicles, so model recall on the full frame is
  understated relative to a foreground-only count.
- **Board cap.** All latencies are at 600 MHz. Fix the supply before quoting
  absolute numbers.

## 7. Caveats

- Pose and segmentation need new decoders; the current `arten` CLI decodes
  detection boxes only. The latency numbers above are runtime-only.
- Counts at low thresholds may include false positives; a small labelled
  sample would settle the optimal threshold.
- Pi 3B+ timings are at 600 MHz. Re-measure after fixing the supply.
