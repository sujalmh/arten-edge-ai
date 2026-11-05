#!/usr/bin/env bash
# Render the single project report to a styled, self-contained HTML, then print
# it to PDF with headless Chrome.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/docs"

if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc is required to build the HTML. Install it, or open report.md in a viewer."
  exit 1
fi

pandoc report.md \
  --from markdown \
  --to html5 \
  --standalone \
  --embed-resources \
  --css report.css \
  --metadata title="Arten — Edge Object Detection, Counting & Attributes" \
  --metadata author="Sujal M H" \
  --output report.html

echo "Wrote docs/report.html"

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [[ ! -x "$CHROME" ]]; then
  echo "Google Chrome not found. Open docs/report.html in a browser and print to PDF."
  exit 0
fi

"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$ROOT/docs/report.pdf" \
  "file://$ROOT/docs/report.html"

echo "Wrote docs/report.pdf"
