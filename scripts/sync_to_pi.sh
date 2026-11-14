#!/usr/bin/env bash
# Copy the local LLM model and the query-layer code to a Raspberry Pi over LAN.
#
# The macOS llama.cpp binary is NOT copied: it cannot run on Linux aarch64.
# The Pi builds its own with scripts/setup_llm.sh. The GGUF model and all
# Python code are portable.
#
# Usage:
#   PI_HOST=pi@raspberrypi.local bash scripts/sync_to_pi.sh
#   PI_HOST=pi@192.168.1.42 PI_DIR=/home/pi/arten bash scripts/sync_to_pi.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PI_HOST="${PI_HOST:-}"
PI_DIR="${PI_DIR:-arten}"          # relative to the Pi user's home if not absolute
MODEL_NAME="qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL="$ROOT/models/llm/$MODEL_NAME"

say() { printf '%s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ -n "$PI_HOST" ]] || fail "set PI_HOST, e.g. PI_HOST=pi@raspberrypi.local bash scripts/sync_to_pi.sh"
[[ -f "$MODEL" ]] || fail "model missing: $MODEL (run scripts/setup_llm.sh first)"
command -v rsync >/dev/null 2>&1 || fail "rsync is required"

say "== Arten · sync to Pi =="
say "host:  $PI_HOST"
say "dir:   $PI_DIR"
say "model: $MODEL_NAME ($(du -h "$MODEL" | awk '{print $1}'))"

say ""
say "[1/4] Checking SSH"
ssh -o BatchMode=yes -o ConnectTimeout=8 "$PI_HOST" "echo connected: \$(hostname)" \
  || fail "cannot SSH to $PI_HOST (set up a key or try ssh first)"

ssh "$PI_HOST" "mkdir -p '$PI_DIR/models/llm' '$PI_DIR/grammars' '$PI_DIR/src' '$PI_DIR/tests/fixtures' '$PI_DIR/scripts'"

say ""
say "[2/4] Copying the model (resumable)"
rsync -av --partial --progress -e ssh "$MODEL" "$PI_HOST:$PI_DIR/models/llm/"

say ""
say "[3/4] Copying code, grammar, tests and scripts"
rsync -av --progress -e ssh \
  --exclude '__pycache__' --exclude '*.pyc' \
  src/ "$PI_HOST:$PI_DIR/src/"
rsync -av --progress -e ssh grammars/ "$PI_HOST:$PI_DIR/grammars/"
rsync -av --progress -e ssh \
  pyproject.toml README.md models/llm/README.md "$PI_HOST:$PI_DIR/" 2>/dev/null || true
rsync -av --progress -e ssh \
  scripts/setup_llm.sh scripts/sync_to_pi.sh "$PI_HOST:$PI_DIR/scripts/"
rsync -av --progress -e ssh \
  tests/test_query.py "$PI_HOST:$PI_DIR/tests/" 2>/dev/null || true
rsync -av --progress -e ssh \
  tests/fixtures/run_mini/ "$PI_HOST:$PI_DIR/tests/fixtures/run_mini/"

say ""
say "[4/4] Verifying the model hash on both sides"
LOCAL_SHA="$(shasum -a 256 "$MODEL" | awk '{print $1}')"
REMOTE_SHA="$(ssh "$PI_HOST" "sha256sum '$PI_DIR/models/llm/$MODEL_NAME'" | awk '{print $1}')"
if [[ "$LOCAL_SHA" != "$REMOTE_SHA" ]]; then
  fail "hash mismatch: local $LOCAL_SHA, remote $REMOTE_SHA"
fi
say "sha256: $LOCAL_SHA (match)"

say ""
say "Done. On the Pi:"
say "  ssh $PI_HOST"
say "  cd $PI_DIR"
say "  bash scripts/setup_llm.sh --check-only"
say "  bash scripts/setup_llm.sh          # builds Linux aarch64 llama.cpp; ~40 min"
say "  source .venv/bin/activate"
say "  arten index --run results/demo_pi5_pedestrians --force"
say "  arten ask \"how many people?\" --db results/demo_pi5_pedestrians/evidence.db --parser llm"
