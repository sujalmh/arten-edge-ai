#!/usr/bin/env bash
# Download Arm's pre-optimized YOLO11n INT8 ExecuTorch model and verify it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/models/yolo11n_int8_xnnpack.pte}"
EXPECTED="154c54b52389219ebd982e1003227c1abbc3f29a99bdad241147b684d3041b96"
URL="https://huggingface.co/Arm/yolo11n-int8-xnnpack-executorch-raspberrypi5/resolve/main/yolo11n_raspberry_executorch_optimized.pte"

mkdir -p "$(dirname "$OUT")"
echo "Downloading Arm-optimized YOLO11n INT8 (.pte, ~4.7 MB)"
echo "  $URL"
curl -L --fail --retry 3 --progress-bar -o "$OUT" "$URL"

if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL="$(sha256sum "$OUT" | awk '{print $1}')"
else
  ACTUAL="$(shasum -a 256 "$OUT" | awk '{print $1}')"
fi

if [[ "$ACTUAL" != "$EXPECTED" ]]; then
  echo "ERROR: checksum mismatch"
  echo "  expected $EXPECTED"
  echo "  actual   $ACTUAL"
  exit 1
fi

echo "Checksum OK: $ACTUAL"
echo "Model ready at: $OUT"
