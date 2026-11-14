#!/usr/bin/env bash
# One-command hackathon demo: pick the best available model and write evidence.
#
# Usage:
#   bash scripts/demo_hackathon.sh [input] [output_dir]
#   MODEL=/path/to/model.pte CONF=0.2 TILE=2 bash scripts/demo_hackathon.sh clip.mp4
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INPUT="${1:-data/samples/bengaluru_traffic.jpg}"
OUT="${2:-results/hackathon_demo}"
TILE="${TILE:-2}"

if [[ -x "$ROOT/.venv/bin/arten" ]]; then
  ARTEN="$ROOT/.venv/bin/arten"
else
  ARTEN="arten"
fi

if [[ -z "${MODEL:-}" ]]; then
  if [[ -f "$ROOT/models/yolov5s_int8_xnnpack.pte" ]]; then
    MODEL="$ROOT/models/yolov5s_int8_xnnpack.pte"
  elif [[ -f "$ROOT/models/yolov8s_int8_xnnpack.pte" ]]; then
    MODEL="$ROOT/models/yolov8s_int8_xnnpack.pte"
  else
    MODEL="$ROOT/models/yolo11n_int8_xnnpack.pte"
  fi
fi

if [[ -z "${CONF:-}" ]]; then
  if [[ "$MODEL" == *yolo11n* ]]; then
    CONF=0.15
  else
    CONF=0.2
  fi
fi

SUPPORT_ARGS=()
if [[ "${SUPPORT:-0}" == "1" ]]; then
  SUPPORT_ARGS=(
    --pose-model "$ROOT/models/lite_hrnet18_int8_xnnpack.pte" --pose-every 10 --pose-max-persons 3
    --seg-model "$ROOT/models/ddrnet23_slim_int8_xnnpack.pte" --seg-every 30
  )
fi

echo "model: $MODEL"
echo "input: $INPUT"
echo "conf:  $CONF"
echo "tile:  ${TILE}x${TILE} (0/1 disables)"
echo

case "$INPUT" in
  *.mp4|*.avi|*.mov|*.mkv|*.webm|*.m4v)
    "$ARTEN" run -i "$INPUT" -o "$OUT" --model "$MODEL" \
      --conf "$CONF" --tile "$TILE" --save-video --annotate-every 10 \
      "${SUPPORT_ARGS[@]}"
    ;;
  *)
    "$ARTEN" run -i "$INPUT" -o "$OUT" --model "$MODEL" \
      --conf "$CONF" --tile "$TILE" --annotate-every 1 \
      "${SUPPORT_ARGS[@]}"
    ;;
esac

echo
echo "Evidence: $OUT/summary.txt"
echo "Open:     $OUT/contact_sheet.jpg"
