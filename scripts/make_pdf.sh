#!/usr/bin/env bash
# Render the proposal and technical report to styled, self-contained HTML,
# then print both to PDF with headless Chrome.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/docs"

if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc is required to build the HTML. Install it, or open the Markdown files in a viewer."
  exit 1
fi

render() {
  local source="$1"
  local title="$2"
  local stem="${source%.md}"
  pandoc "$source" \
    --from markdown \
    --to html5 \
    --standalone \
    --embed-resources \
    --css proposal.css \
    --metadata title="$title" \
    --output "${stem}.html"
  echo "Wrote docs/${stem}.html"
}

render proposal.md "Namma Drishti — Proposal"
render report.md "Namma Drishti — Technical Report"
render judge-brief.md "Namma Drishti — Judge Brief"

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [[ ! -x "$CHROME" ]]; then
  echo "Google Chrome not found. Open the HTML files in a browser and print to PDF."
  exit 0
fi

for stem in proposal report judge-brief; do
  "$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="$ROOT/docs/${stem}.pdf" \
    "file://$ROOT/docs/${stem}.html"
  echo "Wrote docs/${stem}.pdf"
done
