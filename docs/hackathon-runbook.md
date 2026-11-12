# Hackathon runbook

Everything here is verified on a Raspberry Pi 3B+ unless marked as a reference
or an estimate. The Pi 3B+ used for verification was firmware-capped at
600 MHz by under-voltage, so its absolute numbers are pessimistic.

## 1. Hardware and OS

- Raspberry Pi 4 or 5, 8 GB, 64-bit Raspberry Pi OS Bookworm.
- Official power supply (5 V / 3 A for Pi 4, 5 V / 5 A USB-C PD for Pi 5).
- Active cooling for a Pi 5; a heatsink or fan for a Pi 4.
- 32 GB+ A2 microSD. Ethernet if the venue Wi-Fi is unreliable.

## 2. Install

Pi 5 (prebuilt ARMv8.2 runtime):

```bash
git clone https://github.com/sujalmh/arten-edge-ai arten
cd arten
bash scripts/setup_pi.sh
source .venv/bin/activate
```

Pi 4 / Pi 3 (ARMv8.0 runtime, built from source and committed):

```bash
git clone -b raspberry-pi-3-4-armv8 https://github.com/sujalmh/arten-edge-ai arten
cd arten
bash scripts/setup_pi.sh
source .venv/bin/activate
```

The script detects the board, installs the matching torch/ExecuTorch set, runs
`arten doctor` and a one-frame smoke test.

## 3. Choose the operating point

Full-frame 640 is the fast path. Add `--tile 2` for dense or high-resolution
frames; it runs four native-resolution crops and merges them, at about 4x cost.

| Board | Model / mode | Expected latency | Expected throughput | Source |
|---|---|---:|---:|---|
| Pi 5 | YOLO11n, full | 134 ms | 7.46 FPS | Arm card |
| Pi 5 | YOLOv8s, full | 457 ms | 2.19 FPS | Arm card |
| Pi 5 | YOLO11n, `--tile 2` | ~0.55 s | ~1.9 FPS | estimate (4x) |
| Pi 5 | YOLOv8s, `--tile 2` | ~1.8 s | ~0.55 FPS | estimate (4x) |
| Pi 4 | YOLO11n, full | ~0.35–0.45 s | ~2.2–2.9 FPS | estimate |
| Pi 4 | YOLOv8s, full | ~1.1–1.4 s | ~0.7–0.9 FPS | estimate |
| Pi 4 | YOLO11n, `--tile 2` | ~1.4–1.8 s | ~0.6–0.7 FPS | estimate |
| Pi 3B+ | YOLO11n, full | 1.30 s | 0.77 FPS | measured |
| Pi 3B+ | YOLOv8s, full | 3.20 s | 0.31 FPS | measured |
| Pi 3B+ | YOLO11n, `--tile 2` | 5.2 s | 0.19 FPS | measured |
| Pi 3B+ | YOLOv8s, `--tile 2` | 12.8 s | 0.08 FPS | measured |

Recommended:

- **Best accuracy:** YOLOv8s `--conf 0.2`, with `--tile 2` for stills. Add pose
  and DDRNet only on sampled frames (not every frame).
- **Best live throughput:** YOLO11n `--conf 0.15`, full-frame; `--tile 2` only
  when the source is 1080p and the demo is offline.
- **Pi 3B+:** still images or very short clips; do not promise live video.

The CLI default is 0.2. YOLO11n at 0.35 under-counts badly on dense scenes;
use 0.15 there. The INT8 confidence distribution is shifted low.

## 4. Commands

Single image, highest accuracy:

```bash
arten run -i data/samples/bengaluru_traffic.jpg \
  --conf 0.2 --tile 2 --annotate-every 1 \
  -o results/demo_still
```

Clip, balanced:

```bash
arten run -i clip.mp4 --conf 0.2 --tile 2 --stride 2 \
  --save-video --annotate-every 10 -o results/demo_clip
```

Fast live demo:

```bash
arten run -i clip.mp4 --conf 0.15 --stride 1 \
  --save-video --annotate-every 10 -o results/demo_live
```

Evidence to show: `summary.txt`, `contact_sheet.jpg`, `attribute_tracks.csv`,
`annotated/`, and `annotated.mp4`.

## 5. Offline preparation

The first install needs the Python wheels. At the venue this may not be
available, so build a wheelhouse in advance on the same board:

```bash
python -m pip download -r requirements-pi.txt -d ~/wheelhouse     # Pi 5
python -m pip download -r requirements-pi-armv8.txt -d ~/wheelhouse  # Pi 3/4
python -m pip download executorch==1.1.0 -d ~/wheelhouse          # Pi 5 runtime
```

Run the download on the same architecture and Python version as the target
board; the wheels are platform-specific.

Verified on the Pi 3B+: **22 wheels, 156 MB**, a clean `.venv` installed with
**zero downloads**, and `arten doctor` plus the one-frame smoke test passed.
`setup_pi.sh` detects a `wheelhouse/` directory next to the repo and adds
`--no-index`, so the same one-command install works offline.

Copy the repo, `models/`, `vendor/executorch-armv8/*.whl` and the wheelhouse
to a USB stick. Offline install:

```bash
python -m venv .venv && source .venv/bin/activate
pip install --no-index --find-links ~/wheelhouse -r requirements-pi-armv8.txt
pip install --no-deps vendor/executorch-armv8/*.whl
pip install -e . --no-deps
```

Optional extra models (not committed, download before the event):

```bash
# YOLOv8s, YOLOv9s, YOLO11n-pose, DDRNet23-slim
# https://huggingface.co/Arm/yolov8s-int8-xnnpack-executorch
# https://huggingface.co/Arm/yolov9s-int8-xnnpack-executorch
# https://huggingface.co/Arm/yolo11n-pose-int8-xnnpack-executorch-raspberrypi5
# https://huggingface.co/Arm/ddrnet23-slim-int8-xnnpack-executorch-raspberrypi5
```

Pass them with `--model path/to/model.pte`; the CLI resolves the model card
from the checksum and labels the summary and annotation footer correctly.

## 6. Demo flow (5 minutes)

1. One still image, tiled, YOLOv8s: show the counts and annotated frame.
2. One 15-second clip, balanced mode: show the summary and annotated video.
3. Mention the limits: COCO has no auto-rickshaw class, two-wheelers are the
   hardest class, and tiling costs 4x.
4. Fallback if the board or network fails: open the committed evidence under
   `results/demo_pedestrians/`.

## 7. Checklist

- [ ] Board boots, SSH works, `arten doctor` passes.
- [ ] Power supply is adequate (`vcgencmd get_throttled` returns `0x0`).
- [ ] Sample image and clip open.
- [ ] One tiled still and one clip processed end to end.
- [ ] Wheelhouse and models on USB.
- [ ] Precomputed results copied for the fallback.
- [ ] HDMI cable, keyboard, Ethernet cable, card reader.

## 8. Known limits

- Auto-rickshaws are not a COCO class; they are missed or counted as car/truck.
- Tiled inference raises recall substantially (people 46→74%, motorcycles
  18→47% for YOLO11n) but costs about 4x.
- Motion attributes need temporal resolution. Below ~1 FPS, direction and
  speed are indicative only.
- The Pi 3B+ verification board was capped at 600 MHz; re-measure on the
  event board before quoting numbers.
