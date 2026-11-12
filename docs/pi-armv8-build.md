# ARMv8.0 build and Pi 3B+ evidence

This branch adds a Raspberry Pi 3 / Pi 4 (ARMv8.0) runtime. It exists because
the prebuilt Arm ExecuTorch wheel and PyTorch 2.10 contain ARMv8.2
dot-product instructions (`sdot`). Cortex-A53 (Pi 3) and Cortex-A72 (Pi 4)
trap with `Illegal instruction` (SIGILL) when those instructions execute.

## What was built

- ExecuTorch `v1.1.0` from source, `-march=armv8-a`
- XNNPACK dot-product and i8mm kernels disabled
- KleidiAI disabled
- LLM kernels disabled (not used by detection)
- PyTorch **2.9.0**, the newest release that runs on ARMv8.0
- torch headers ordered ahead of ExecuTorch's header-only `c10` copy
- the runtime's `portable_lib.py` no longer imports `executorch.exir`
  (the export stack requires torch 2.10; inference does not)

Build command:

```bash
bash scripts/build_executorch_armv8.sh
cp wheels/executorch-*.whl vendor/executorch-armv8/
```

## Pi 3B+ result

| Item | Value |
|---|---|
| Board | Raspberry Pi 3 Model B+ Rev 1.3 |
| CPU | Cortex-A53, 4 cores, ARMv8.0 |
| OS | Raspberry Pi OS Lite 64-bit Bookworm |
| Kernel | 6.12.25+rpt-rpi-v8, `aarch64` |
| Python | 3.11.2 |
| PyTorch | 2.9.0+cpu |
| ExecuTorch | 1.1.0a0+17adba1 (source build) |
| Model | `yolo11n_int8_xnnpack.pte`, 4.73 MB, checksum OK |

One-frame run, `data/samples/bengaluru_traffic.jpg`:

```text
model infer    1293.67 ms   (0.77 FPS)
preprocess       88.71 ms
decode + NMS     56.80 ms
pipeline total 2173.86 ms   (0.46 FPS)

person 2 · car 19 · bus 1 · truck 1
```

Five-run benchmark (`results/benchmark_pi3b_armv8.json`, warmup 1):

| Metric | Pi 3B+ |
|---|---:|
| model load | 126.28 ms |
| mean | 1298.78 ms |
| p50 | 1300.08 ms |
| p95 | 1300.43 ms |
| throughput | 0.77 FPS |
| peak RSS | 443.8 MB |

**Measurement caveat.** These timings were taken while the board was
firmware-capped at **600 MHz** (`vcgencmd get_throttled` = `0x50005`,
under-voltage; nominal clock 1.4 GHz). A proper 5 V / 2.5 A supply should
reduce the absolute latencies. Model-to-model ratios remain indicative.

## Model comparison on the Pi 3B+

All runs on the same board, runtime and image, at the default confidence 0.35
unless noted. Same 600 MHz cap applies.

| Model | p50 | FPS | Peak RSS | person | car | motorcycle | bus | truck |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | 1300 ms | 0.77 | 444 MB | 2 | 19 | 0 | 1 | 1 |
| YOLO11n, `--conf 0.2` | ~1300 ms | 0.77 | 444 MB | **9** | 25 | 4 | 2 | 1 |
| YOLOv8s | 3196 ms | 0.31 | 470 MB | 10 | 21 | 4 | 1 | 2 |
| YOLOv9s | 5777 ms | 0.17 | 484 MB | 12 | 19 | 8 | 1 | 3 |
| YOLO11n-pose | 1357 ms | 0.74 | 436 MB | (pose) | — | — | — | — |
| DDRNet23-slim, 720×960 | 1879 ms | 0.53 | — | (segmentation) | — | — | — | — |

The YOLO11n INT8 confidence distribution is shifted low: the default 0.35
misses most two-wheeler riders in this dense scene, while 0.2 recovers them
at no latency cost. YOLOv8s at 0.35 reaches a similar count with 2.5× the
latency, and YOLOv9s finds the most at 4.4× the latency.

Practical reading:

- Tune the threshold before changing models. `--conf 0.2` is the cheaper win.
- YOLOv8s is the accuracy upgrade when stills or offline clips are acceptable.
- YOLOv9s is still-image only on ARMv8.0.
- YOLO11n-pose costs almost nothing over YOLO11n and can add posture attributes.
- DDRNet23-slim can supply road/sidewalk masks for zone audits at ~0.5 FPS.

Instruction check on the installed binding:

```text
sdot  count: 0
ldadd count: 1   (GCC outline atomics; runtime LSE check, A53 takes ldxr/stxr)
```

The Pi 3B+ is a compatibility target. A Pi 4 (Cortex-A72, 8 GB) is the
practical ARMv8.0 demo board; a Pi 5 (Cortex-A76, ARMv8.2) remains the
performance target and uses the prebuilt wheels.

## Run it

```bash
git clone -b raspberry-pi-3-4-armv8 https://github.com/sujalmh/arten-edge-ai arten
cd arten
bash scripts/setup_pi.sh
source .venv/bin/activate
arten doctor
arten run -i data/samples/bengaluru_traffic.jpg -o results/pi_armv8 --annotate-every 1
```

`setup_pi.sh` detects the board: Pi 5 gets the upstream ARMv8.2 wheels, Pi 3/4
get `requirements-pi-armv8.txt` plus the wheel in `vendor/executorch-armv8/`.
