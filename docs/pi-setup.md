# Raspberry Pi runbook

## Supported boards

| Board | CPU | ISA | Runtime |
|---|---|---|---|
| Pi 5 | Cortex-A76 | ARMv8.2 (dot-product) | prebuilt ExecuTorch 1.1.0 + torch 2.10 |
| Pi 4 / CM4 / Pi 400 | Cortex-A72 | ARMv8.0 | source-built ExecuTorch + torch 2.9.0 |
| Pi 3B / 3B+ | Cortex-A53 | ARMv8.0 | same as Pi 4, much slower |

The upstream ExecuTorch and PyTorch 2.10 aarch64 wheels contain ARMv8.2
`sdot` instructions. On a Pi 3 or Pi 4 they trap with `Illegal instruction`
(SIGILL). `setup_pi.sh` reads `/proc/device-tree/model` and installs the right
set automatically.

Recommended: 64-bit Raspberry Pi OS Bookworm, 8 GB for Pi 4/5, 32 GB+ A2
microSD, official power supply, active cooling. A Pi 3B+ has 1 GB RAM and is a
compatibility target, not a performance target.

## 1. Flash and connect

1. Flash 64-bit Raspberry Pi OS with Raspberry Pi Imager.
2. In the imager's settings: set hostname, username, Wi-Fi/Ethernet and enable
   SSH.
3. Boot the Pi, then find its address:
   ```bash
   hostname -I
   ```
4. From the laptop:
   ```bash
   ssh <user>@<pi-ip>
   ```

If `raspberrypi.local` resolves, use that instead of the IP.

## 2. Install Arten

```bash
sudo apt update
sudo apt install -y git python3-venv python3-pip libgomp1
git clone https://github.com/sujalmh/arten-edge-ai arten
cd arten
bash scripts/setup_pi.sh
source .venv/bin/activate
```

`setup_pi.sh` creates the virtual environment, installs the pinned ExecuTorch
runtime, installs the package, verifies the model checksum and runs a one-frame
smoke test.

On a Pi 3 or Pi 4 it installs `requirements-pi-armv8.txt` (torch 2.9.0,
NumPy 1.26, OpenCV 4.10) and the wheel in `vendor/executorch-armv8/`, which is
built with `-march=armv8-a`, KleidiAI off and the XNNPACK dot-product/i8mm
kernels disabled. On a Pi 5 it uses the upstream wheels.

To rebuild that wheel from source on an arm64 Linux host:

```bash
bash scripts/build_executorch_armv8.sh
mkdir -p vendor/executorch-armv8
cp wheels/executorch-*.whl vendor/executorch-armv8/
```

Expected first-run time: several minutes on a Pi 5, longer on a Pi 4, and
substantially longer on a Pi 3B+. The download is dominated by the PyTorch and
ExecuTorch wheels.

## 3. Verify

```bash
arten doctor
arten info
arten run -i data/samples/bengaluru_traffic.jpg \
  -o results/pi_smoke --annotate-every 1 --estimate-vehicle-colors
```

The run should report the model checksum as OK and produce an annotated image.

## 4. Measure the real device

```bash
arten benchmark \
  --input data/samples/bengaluru_traffic.jpg \
  --runs 50 --warmup 10 \
  --output results/benchmark_pi5.json

arten run \
  -i data/samples/pedestrians_15s.mp4 \
  -o results/pi_clip \
  --save-video --annotate-every 10 --estimate-vehicle-colors
```

Or run the hardware proof script, which captures board/kernel/OS details,
doctor output, a benchmark and a clip:

```bash
bash scripts/run_pi5_hardware_proof.sh
```

The proof script exits if the board is not a Pi 5. On a Pi 4, use the commands
above and label the results as Pi 4.

## 5. What to look for

- `summary.txt` has counts, p50/p95 and model-only/pipeline FPS.
- `attribute_tracks.csv` has direction, dwell, relative speed and optional
  colour.
- `contact_sheet.jpg` and `annotated.mp4` are the visual evidence.
- `benchmark_pi5.json` is the standalone model-only latency.

Arm's Pi 5 reference for the prebuilt model is **134.10 ms p50 / 7.46 FPS** with
**59.17 MB** peak memory. Treat that as the reference until the device run
replaces it. The attribute layer and video encoding add time beyond the model
benchmark.

## 6. Troubleshooting

**`python3-venv` is missing**
```bash
sudo apt install -y python3-venv
```

**`libgomp.so.1` missing**
```bash
sudo apt install -y libgomp1
```

**ExecuTorch install tries to pull `coremltools`**
The setup script installs ExecuTorch with `--no-deps` and installs the runtime
support packages explicitly. Do not replace that step with a normal
`pip install executorch`.

**`torchao` prints a compatibility warning**
Arm's runtime pins `torchao` against the Pi's PyTorch build. The warning is
expected; the model still runs. Do not upgrade `torch` or `torchao` on the Pi.

**Low FPS or throttling**
Use the official power supply, fit an active cooler, keep the Pi off soft
surfaces, and use `--stride 2` or fewer frames for a quick survey. Report the
measured p95, not a single best run.

**Video will not open**
Convert it first:
```bash
ffmpeg -i input.mov -c:v libx264 -pix_fmt yuv420p -an clip_h264.mp4
```

**No detections**
Lower `--conf` to `0.25`, check the requested categories, and inspect a frame.
The object may be too small or occluded.

**Model checksum mismatch**
```bash
bash scripts/download_model.sh
```

## 7. Offline note

The first install needs internet for the Python wheels. After the virtual
environment is built, the model and all inference run locally. For an offline
event, pack the repo, the `.venv` and the model on a USB stick after a
successful setup.
