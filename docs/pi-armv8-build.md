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
