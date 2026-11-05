#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT_DIR="${OUT_DIR:-results/raspberry_pi5_hardware_proof}"
MODEL="$(tr -d '\000' < /proc/device-tree/model 2>/dev/null || true)"
mkdir -p "$OUT_DIR"

if [[ "$MODEL" != *"Raspberry Pi 5"* ]]; then
  {
    printf 'Hardware proof requires Raspberry Pi 5; detected: %s\n' "${MODEL:-unknown}"
    uname -a
  } | tee "$OUT_DIR/hardware.txt"
  exit 2
fi

{
  printf 'Captured: %s\n' "$(date -Is)"
  printf 'Board: %s\n\n' "$MODEL"
  printf '%s\n' '=== Kernel ==='
  uname -a
  printf '\n%s\n' '=== OS ==='
  cat /etc/os-release
  printf '\n%s\n' '=== CPU ==='
  cat /proc/cpuinfo
  printf '\n%s\n' '=== Runtime ==='
  python3 -c 'import cv2, executorch, torch; print("Python runtime imports: OK"); print("PyTorch:", torch.__version__); print("OpenCV:", cv2.__version__); print("ExecuTorch module:", executorch.__file__)'
  printf '\n%s\n' '=== Model checksum ==='
  sha256sum models/yolo11n_int8_xnnpack.pte
} | tee "$OUT_DIR/hardware.txt"

namma-drishti doctor 2>&1 | tee "$OUT_DIR/doctor.log"
namma-drishti benchmark \
  --input data/samples/bengaluru_traffic.jpg \
  --runs 50 --warmup 10 \
  --output "$OUT_DIR/benchmark.json" \
  2>&1 | tee "$OUT_DIR/benchmark.log"
namma-drishti run \
  --input data/samples/pedestrians_15s.mp4 \
  --output "$OUT_DIR/clip" \
  --classes person,bicycle,car,motorcycle,bus,truck \
  --stride 10 --annotate-every 10 --save-video \
  2>&1 | tee "$OUT_DIR/clip.log"

printf 'Hardware proof completed on %s\n' "$MODEL" | tee "$OUT_DIR/COMPLETE.txt"
