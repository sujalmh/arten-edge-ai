#!/usr/bin/env bash
# One-command setup for a 64-bit Raspberry Pi (Pi 4 or Pi 5 recommended).
# Idempotent: safe to run again after a failed download.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"

echo "== Namma Drishti · Raspberry Pi setup =="
echo "repo:   $ROOT"
echo "arch:   $(uname -m)"
echo "python: $($PYTHON --version 2>&1)"

if [[ "$(uname -m)" != "aarch64" && "$(uname -m)" != "arm64" ]]; then
  echo "WARNING: expected aarch64/arm64. The pipeline also runs on x86_64,"
  echo "         but the Pi reference numbers will not apply."
fi

if ! $PYTHON -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "ERROR: Python 3.10+ is required. On Raspberry Pi OS:"
  echo "  sudo apt update && sudo apt install -y python3-venv python3-pip"
  exit 1
fi

if [[ ! -d "$VENV" ]]; then
  echo "\n[1/4] Creating virtual environment at $VENV"
  $PYTHON -m venv "$VENV" || {
    echo "Could not create a venv. Install the venv package:"
    echo "  sudo apt update && sudo apt install -y python3-venv"
    exit 1
  }
else
  echo "\n[1/4] Reusing existing virtual environment at $VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "\n[2/4] Installing Python dependencies (this can take a few minutes)"
python -m pip install --upgrade pip wheel
python -m pip install -r requirements-pi.txt
# coremltools has no Linux aarch64 wheel, so install ExecuTorch without its
# declared dependency set. The runtime packages above mirror Arm's working
# runtime project, which also excludes coremltools.
python -m pip install --no-deps executorch==1.1.0

echo "\n[3/4] Installing the namma-drishti package"
python -m pip install -e . --no-deps

echo "\n[4/4] Checking the model and runtime"
if [[ ! -f models/yolo11n_int8_xnnpack.pte ]]; then
  bash scripts/download_model.sh
fi

namma-drishti doctor || {
  echo "\nDoctor found an issue. Check the messages above."
  exit 1
}

echo "\nSetup complete. Try:"
echo "  source $VENV/bin/activate"
echo "  namma-drishti run -i data/samples/pedestrians_15s.mp4 --save-video"
echo "  namma-drishti benchmark --runs 50"
