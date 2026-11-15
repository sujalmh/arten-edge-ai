# YOLO26 check: Mac results and Pi 5 plan

**Date:** 14 November 2026
**Question:** does YOLO26 run on the Pi 5, and does it have an auto-rickshaw class?
**Short answer:** yes to ExecuTorch export and local inference; **no auto-rickshaw class** in the pretrained model. Pi 5 has a published ExecuTorch number but was busy, so it is not yet measured on our board.

---

## 1. What YOLO26 actually is

- Ultralytics YOLO26n/s/m/l/x are pretrained on **COCO, 80 classes**.
- The class list is the standard COCO list: person, bicycle, car, motorcycle, bus, truck, ... toothbrush.
- **There is no `auto-rickshaw` class.** An auto-rickshaw is detected as `truck` or `car`, or missed.
- Ultralytics' ExecuTorch export is **FP32 by default** and runs through XNNPACK. It also supports static INT8 export with a calibration dataset.
- On ExecuTorch, YOLO26 falls back to the one-to-many path, so the app still needs NMS. The "NMS-free" benefit is not available on this runtime.

---

## 2. Mac results (verified locally)

Environment: Apple M4, macOS, Python 3.12.12, Ultralytics 8.4.212, torch 2.15.0, ExecuTorch 1.6.0. **These are Mac numbers, not Pi numbers.**

| Run | Model | Size | Result |
|---|---|---:|---|
| Export, FP32 | YOLO26n | 9.4 MB | export OK, 68 s |
| Export, INT8 | YOLO26n, 11 calibration images | **3.1 MB** | export OK, 31 s |
| Video, FP32 | YOLO26n ExecuTorch | 9.4 MB | 239 frames (stride 5), model p50 **18.7 ms**, e2e p50 19.7 ms |
| Video, INT8 | YOLO26n ExecuTorch | 3.1 MB | 239 frames (stride 5), model p50 **20.2 ms**, e2e p50 21.3 ms |
| App integration | `arten run` with the INT8 `.pte` | 3.1 MB | worked end-to-end: counts, tracks, attributes, CSV/JSON, annotated video |

The test video is 854×480, 25 fps, 47.64 s (1,191 frames). Sampling every 5th frame gives 239 frames.

**Video counts, `arten` run (INT8, conf 0.20, IoU 0.50):**

| Class | Peak / frame | Mean / frame | Unique tracks |
|---|---:|---:|---:|
| car | 14 | 7.65 | 54 |
| bus | 4 | 1.41 | 9 |
| truck | 2 | 0.27 | 9 |
| person | 2 | 0.50 | 11 |
| motorcycle | 1 | 0.03 | 0 |

Full output: `results/yolo26-check/REPORT.md`, `contact_sheet.jpg`, `annotated.mp4`, `counts.csv`, `summary.json`.

The raw Ultralytics runs on the same video are in `video_fp32_summary.json` and `video_int8_summary.json`. They show the same picture: ~8 cars per frame, ~1.4 buses per frame, 3 people peak.

**INT8 caveat:** the export used only 11 calibration images. Ultralytics recommends 300+. The counts stayed close to FP32 on this clip, but this is not a validated INT8 model. Calibrate on the varied set before trusting it.

---

## 3. Auto-rickshaw test (the important negative result)

Test image: `data/samples/bengaluru_street.jpg`, with one clear yellow/black auto-rickshaw.

| Model / prompt | Auto-rickshaw result |
|---|---|
| YOLO26n, COCO | No class. The auto is boxed as **`truck 0.27`**. |
| YOLOE-26n, prompt `auto-rickshaw` | 1 box, confidence 0.69, but the box is loose and shifted onto the foreground car; not a clean detection. |
| YOLOE-26n, prompt `auto-rickshaw` on the dense traffic image | **0 detections.** |
| YOLOE-26n, prompt `tuk tuk` on the dense traffic image | 3 boxes, all on blue cars; **false positives**. |

Evidence:
- `results/yolo26-check/street_auto_labelled_truck.jpg`
- `results/yolo26-check/street_yoloe_auto_prompt.jpg`
- `results/yolo26-check/traffic_yoloe_tuk_tuk_false_positives.jpg`

**Conclusion:** do not claim auto-rickshaw detection. YOLO26 does not have the class, and YOLOE-26 text prompting is not reliable enough on these images. A real auto-rickshaw class needs a fine-tune on an Indian street dataset (IDD or similar), not a prompt.

---

## 4. Pi 5: what is known, what is not

**Published Ultralytics Pi 5 benchmark (FP32, 640, inference only):**

| Model | Format | Size | mAP@0.5:0.95 | ms/image |
|---|---|---:|---:|---:|
| YOLO26n | ExecuTorch | 9.4 MB | 47.7 | **144.83** |
| YOLO26s | ExecuTorch | 36.5 MB | 56.7 | 376.48 |
| YOLO11n (current, Arm INT8) | ExecuTorch | 4.73 MB | 38.90 | 134.10 |

Source: Ultralytics Raspberry Pi guide and ExecuTorch integration page.

**What this means:** YOLO26n should run on the Pi 5 at roughly 145 ms per image, about 6.9 FPS model-only. That is ~11 ms slower than the current INT8 YOLO11n, with a much higher COCO mAP (47.7 vs 38.9) and a 2× larger file. The numbers are not perfectly comparable: different precision, thread count and post-processing.

**Not yet verified:** the exported `.pte` loading and running on our Pi 5. The Pi was busy. The exported file is at `results/yolo26-check/yolo26n_int8.pte`.

**Main blocker: runtime version.** The export used ExecuTorch **1.6.0**. The project's Pi stack pins **1.1.0**. A newer `.pte` may fail to load on the older runtime. Test this before promising YOLO26 on the Pi.

---

## 5. Pi 5 test commands for when the board is free

First try the existing app environment, which is the real deployment target:

```bash
# on the Mac
scp results/yolo26-check/yolo26n_int8.pte pi@<pi-host>:~/arten/models/yolo26n_int8_calib11.pte
scp <test-video>.mp4 pi@<pi-host>:~/arten/data/samples/yolo26_test.mp4

# on the Pi
cd ~/arten
source .venv/bin/activate
arten run -i data/samples/yolo26_test.mp4 \
  --model models/yolo26n_int8_calib11.pte \
  --conf 0.2 --iou 0.5 --stride 5 --max-frames 250 \
  --save-video -o results/pi5_yolo26
```

If it fails with a schema, operator or delegate error, the runtime is too old. In that case, use a **separate** venv so the working demo is not touched:

```bash
python3 -m venv ~/yolo26-venv
~/yolo26-venv/bin/pip install --upgrade pip
~/yolo26-venv/bin/pip install ultralytics
~/yolo26-venv/bin/yolo predict model=~/arten/models/yolo26n_int8_calib11.pte \
  source=~/arten/data/samples/yolo26_test.mp4 imgsz=640 conf=0.2
```

Then measure, with the board idle and cooled:

```bash
# p50/p95 over repeated frames; record peak RSS and platform
/usr/bin/time -v ~/yolo26-venv/bin/yolo predict ... 2>&1 | tail -30
```

Record: `uname -a`, `cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor`, `vcgencmd measure_temp`, `vcgencmd get_throttled`, and the exact ExecuTorch version.

---

## 6. Recommendation

1. **Keep YOLO11n INT8 as the demo default until the Pi 5 run is done.** It is measured on the board, 4.73 MB, 136.73 ms p50, 7.31 FPS.
2. **Treat YOLO26n as a tested alternative, not a drop-in.** It exports and runs locally, and the app's decoder handles it. The Pi needs a runtime-compatibility check first.
3. **Do not claim auto-rickshaw detection.** The class does not exist in COCO; YOLOE prompting was unreliable. If the demo needs it, fine-tune on an Indian dataset with an explicit auto-rickshaw class.
4. **Fix the reference line before any YOLO26 screenshot.** `arten run` prints the Arm YOLO11n card (134.10 ms) even when the loaded model is YOLO26. That line is wrong for this model.

---

## 7. Reproduction

```bash
# isolated environment
uv venv --python 3.12 /tmp/yolo26-venv
uv pip install --python /tmp/yolo26-venv/bin/python ultralytics

# FP32 export
python -c 'from ultralytics import YOLO; YOLO("yolo26n.pt").export(format="executorch", imgsz=640)'

# INT8 export needs a calibration YAML; 300+ images recommended
python -c 'from ultralytics import YOLO; YOLO("yolo26n.pt").export(format="executorch", imgsz=640, quantize=8, data="calib.yaml")'

# run through the app
PYTHONPATH=src python -m arten run -i <video> --model <model.pte> --stride 5 --max-frames 250 -o results/yolo26_check
```

Artifacts: `results/yolo26-check/`.
