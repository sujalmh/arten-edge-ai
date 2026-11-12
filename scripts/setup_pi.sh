#!/usr/bin/env bash
# One-command setup for a 64-bit Raspberry Pi 3, 4, CM4 or 5.
# Idempotent: safe to run again after a failed download.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"

echo "== Arten · Raspberry Pi setup =="
echo "repo:   $ROOT"
echo "arch:   $(uname -m)"
echo "python: $($PYTHON --version 2>&1)"

if [[ "$(uname -m)" != "aarch64" && "$(uname -m)" != "arm64" ]]; then
  echo "WARNING: expected aarch64/arm64. The pipeline also runs on x86_64,"
  echo "         but the Pi reference numbers will not apply."
fi

BOARD="$(tr -d '\000' < /proc/device-tree/model 2>/dev/null || echo unknown)"
echo "board:  ${BOARD:-unknown}"

if ! $PYTHON -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "ERROR: Python 3.10+ is required. On Raspberry Pi OS:"
  echo "  sudo apt update && sudo apt install -y python3-venv python3-pip"
  exit 1
fi

if [[ ! -d "$VENV" ]]; then
  echo "\n[1/5] Creating virtual environment at $VENV"
  $PYTHON -m venv "$VENV" || {
    echo "Could not create a venv. Install the venv package:"
    echo "  sudo apt update && sudo apt install -y python3-venv"
    exit 1
  }
else
  echo "\n[1/5] Reusing existing virtual environment at $VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "\n[2/5] Installing Python dependencies (this can take a few minutes)"
PIP_OFFLINE=()
if [[ -d "$ROOT/wheelhouse" ]]; then
  PIP_OFFLINE=(--no-index --find-links "$ROOT/wheelhouse")
  echo "  offline wheelhouse: $ROOT/wheelhouse"
fi
if [[ ${#PIP_OFFLINE[@]} -eq 0 ]]; then
  python -m pip install --upgrade pip wheel
fi

if [[ "$BOARD" == *"Raspberry Pi 5"* ]]; then
  # Pi 5 (Cortex-A76, ARMv8.2): use the prebuilt Arm runtime.
  python -m pip install "${PIP_OFFLINE[@]}" -r requirements-pi.txt
  # coremltools has no Linux aarch64 wheel, so install ExecuTorch without its
  # declared dependency set. The runtime packages above mirror Arm's working
  # runtime project, which also excludes coremltools.
  python -m pip install "${PIP_OFFLINE[@]}" --no-deps executorch==1.1.0
elif [[ "$(uname -m)" == "aarch64" || "$(uname -m)" == "arm64" ]]; then
  # Pi 3 / Pi 4 / CM4 (Cortex-A53/A72, ARMv8.0): the prebuilt wheels use
  # ARMv8.2 dot-product instructions and crash with SIGILL here. Use torch
  # 2.9.0 plus the runtime built by scripts/build_executorch_armv8.sh.
  ET_WHEEL="$(ls "$ROOT"/vendor/executorch-armv8/*.whl 2>/dev/null | head -1 || true)"
  if [[ -z "$ET_WHEEL" ]]; then
    echo "ERROR: no ARMv8.0 ExecuTorch wheel found in vendor/executorch-armv8/."
    echo "Build it once on an arm64 Linux host (Docker works):"
    echo "  bash scripts/build_executorch_armv8.sh"
    echo "Then copy wheels/executorch-*.whl into vendor/executorch-armv8/."
    exit 1
  fi
  python -m pip install "${PIP_OFFLINE[@]}" -r requirements-pi-armv8.txt
  python -m pip install --no-deps "$ET_WHEEL"
else
  echo "ERROR: this setup script targets 64-bit Raspberry Pi OS (aarch64)."
  exit 1
fi

echo "\n[3/5] Installing the arten package"
python -m pip install -e . --no-deps

echo "\n[4/5] Checking the model and runtime"
if [[ ! -f models/yolo11n_int8_xnnpack.pte ]]; then
  bash scripts/download_model.sh
fi

arten doctor || {
  echo "\nDoctor found an issue. Check the messages above."
  exit 1
}

echo "\n[5/5] Running a one-frame smoke test (loads the .pte and runs inference)"
SMOKE_DIR="$(mktemp -d)"
arten run -i data/samples/bengaluru_traffic.jpg -o "$SMOKE_DIR" \
  --annotate-every 0 --quiet || {
  echo "\nSmoke test failed. Check the messages above."
  rm -rf "$SMOKE_DIR"
  exit 1
}
rm -rf "$SMOKE_DIR"

echo "\nSetup complete. Try:"
echo "  source $VENV/bin/activate"
echo "  arten run -i data/samples/pedestrians_15s.mp4 --save-video"
echo "  arten benchmark --runs 50"
