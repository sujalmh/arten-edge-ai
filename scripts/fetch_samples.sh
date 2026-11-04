#!/usr/bin/env bash
# Re-fetch the public sample media (already committed, for provenance).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/data/samples"
mkdir -p "$OUT"

echo "Fetching the OpenCV pedestrian clip (public sample data)..."
curl -L --fail --retry 3 -o /tmp/vtest.avi \
  "https://cdn.jsdelivr.net/gh/opencv/opencv@4.x/samples/data/vtest.avi"

if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -y -v error -i /tmp/vtest.avi -t 15 -an \
    -vf "scale=640:-2" -c:v libx264 -preset medium -crf 26 -pix_fmt yuv420p \
    "$OUT/pedestrians_15s.mp4"
else
  echo "ffmpeg not found; keeping the full clip instead of trimming."
  cp /tmp/vtest.avi "$OUT/pedestrians_15s.avi"
fi
rm -f /tmp/vtest.avi

echo "Fetching CC0 / public-domain Bengaluru street images..."
curl -L --fail -o "$OUT/bengaluru_traffic.jpg" \
  "https://upload.wikimedia.org/wikipedia/commons/2/2d/TRAFFIC_VIEW_IN_BANGALORE.jpg"
curl -L --fail -o "$OUT/bengaluru_street.jpg" \
  "https://upload.wikimedia.org/wikipedia/commons/8/8c/Traffic_in_Bangalore.JPG"

echo "Done. See data/README.md for attribution and licences."
