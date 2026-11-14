#!/usr/bin/env bash
# Install the optional local LLM used by `arten ask`.
#
# Safe defaults:
#   - refuses to run while an Arten or llama process is active;
#   - builds with -j2 and nice;
#   - verifies the GGUF SHA-256 before use;
#   - never installs anything when --check-only is passed.
#
# Usage:
#   bash scripts/setup_llm.sh --check-only
#   bash scripts/setup_llm.sh
#   bash scripts/setup_llm.sh --install-deps      # Linux only, uses apt
#   bash scripts/setup_llm.sh --force             # ignore a busy-board warning
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_DIR="$ROOT/models/llm"
MODEL_NAME="qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL="$MODEL_DIR/$MODEL_NAME"
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/$MODEL_NAME"
MODEL_SHA="74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"
LLAMA_DIR="$ROOT/vendor/llama.cpp"
LLAMA_BIN_LOCAL="$LLAMA_DIR/build/bin/llama-cli"
GRAMMAR="$ROOT/grammars/query_labels.gbnf"

CHECK_ONLY=0
FORCE=0
NO_DOWNLOAD=0
INSTALL_DEPS=0

usage() {
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
}

for arg in "$@"; do
  case "$arg" in
    --check-only) CHECK_ONLY=1 ;;
    --force) FORCE=1 ;;
    --no-download) NO_DOWNLOAD=1 ;;
    --install-deps) INSTALL_DEPS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $arg" >&2; usage; exit 2 ;;
  esac
done

say() { printf '%s\n' "$*"; }
ok() { printf '  %-14s %s\n' "$1" "$2"; }
warn() { printf '  %-14s %s\n' "WARN" "$*"; }
fail() { printf '  %-14s %s\n' "ERROR" "$*" >&2; exit 1; }

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    return 1
  fi
}

find_llama() {
  if command -v llama-cli >/dev/null 2>&1; then
    command -v llama-cli
  elif [[ -x "$LLAMA_BIN_LOCAL" ]]; then
    printf '%s\n' "$LLAMA_BIN_LOCAL"
  else
    return 1
  fi
}

board_name() {
  if [[ -r /proc/device-tree/model ]]; then
    tr -d '\000' < /proc/device-tree/model
  else
    uname -m
  fi
}

disk_free_kb() {
  df -kP "$ROOT" | awk 'NR==2 {print $4}'
}

busy_processes() {
  ps -axo pid,command 2>/dev/null \
    | grep -E 'arten (run|benchmark)|llama-cli|llama-server' \
    | grep -v 'grep' || true
}

echo "== Arten · local LLM setup =="
say "repo:   $ROOT"
say "board:  $(board_name)"
say "python: $(python3 --version 2>&1)"

# ---------------------------------------------------------------------- #
# Preflight
# ---------------------------------------------------------------------- #
BUSY="$(busy_processes)"
if [[ -n "$BUSY" ]]; then
  say ""
  say "A process is already using the board:"
  printf '%s\n' "$BUSY" | sed 's/^/  /'
  if [[ "$FORCE" -eq 0 ]]; then
    say ""
    say "Wait for it to finish, then re-run. Use --force only if you accept contention."
    exit 3
  fi
  warn "--force given; continuing with a busy board"
fi

if command -v vcgencmd >/dev/null 2>&1; then
  THROTTLED="$(vcgencmd get_throttled 2>/dev/null || true)"
  TEMP="$(vcgencmd measure_temp 2>/dev/null || true)"
  ok "thermal" "${TEMP:-unknown} · ${THROTTLED:-unknown}"
  if [[ "$THROTTLED" != *"0x0"* && "$FORCE" -eq 0 ]]; then
    fail "the board reports throttling ($THROTTLED); let it cool, then re-run"
  fi
fi

FREE_KB="$(disk_free_kb)"
if [[ -n "$FREE_KB" && "$FREE_KB" -lt 3145728 && "$FORCE" -eq 0 ]]; then
  fail "less than 3 GB free ($((FREE_KB / 1024)) MB); free space or pass --force"
fi
ok "disk free" "$((FREE_KB / 1024)) MB"

LLAMA_BIN="$(find_llama || true)"
MODEL_OK=0
if [[ -f "$MODEL" ]]; then
  ACTUAL_SHA="$(sha256_of "$MODEL" || true)"
  if [[ "$ACTUAL_SHA" == "$MODEL_SHA" ]]; then
    MODEL_OK=1
  fi
fi

# ---------------------------------------------------------------------- #
# Check-only report
# ---------------------------------------------------------------------- #
if [[ "$CHECK_ONLY" -eq 1 ]]; then
  say ""
  say "Status"
  ok "llama-cli" "${LLAMA_BIN:-missing}"
  if [[ -n "$LLAMA_BIN" ]]; then
    ok "llama ver" "$("$LLAMA_BIN" --version 2>&1 | head -1)"
  fi
  if [[ "$MODEL_OK" -eq 1 ]]; then
    ok "model" "$MODEL_NAME · $(du -h "$MODEL" | awk '{print $1}') · sha OK"
  elif [[ -f "$MODEL" ]]; then
    ok "model" "$MODEL_NAME · SHA MISMATCH"
  else
    ok "model" "missing: $MODEL"
  fi
  ok "grammar" "$([[ -f "$GRAMMAR" ]] && echo "$GRAMMAR" || echo missing)"
  if [[ -n "$LLAMA_BIN" && "$MODEL_OK" -eq 1 && -f "$GRAMMAR" ]]; then
    say ""
    say "Ready. Try:"
    say "  arten ask \"how many people?\" --db results/demo_pi5_pedestrians/evidence.db --parser llm"
    exit 0
  fi
  say ""
  say "Run without --check-only to install the missing pieces."
  exit 1
fi

# ---------------------------------------------------------------------- #
# Install llama.cpp
# ---------------------------------------------------------------------- #
if [[ -z "$LLAMA_BIN" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    if ! command -v brew >/dev/null 2>&1; then
      fail "Homebrew is required on macOS (https://brew.sh)"
    fi
    say ""
    say "[1/3] Installing llama.cpp with Homebrew"
    brew install llama.cpp
    LLAMA_BIN="$(find_llama || true)"
  elif [[ "$(uname -s)" == "Linux" ]]; then
    if ! command -v cmake >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
      if [[ "$INSTALL_DEPS" -eq 1 ]] && command -v apt-get >/dev/null 2>&1; then
        say ""
        say "[1/3] Installing build dependencies (apt)"
        sudo apt-get update
        sudo apt-get install -y build-essential cmake git
      else
        fail "cmake/git missing; install them or re-run with --install-deps"
      fi
    fi

    if [[ ! -d "$LLAMA_DIR/.git" ]]; then
      say ""
      say "[1/3] Cloning llama.cpp"
      git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
    else
      say ""
      say "[1/3] Reusing $LLAMA_DIR"
    fi
    ( cd "$LLAMA_DIR" && git rev-parse HEAD > "$ROOT/vendor/LLAMA_CPP_COMMIT.txt" )

    say "      building llama-cli and llama-server with -j2 (this takes a while)"
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=ON \
      -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=ON >/dev/null
    nice -n 10 cmake --build "$LLAMA_DIR/build" --target llama-cli llama-server -j2

    # Optional multimodal CLI; absence is fine.
    nice -n 10 cmake --build "$LLAMA_DIR/build" --target llama-mtmd-cli -j2 >/dev/null 2>&1 || true
    LLAMA_BIN="$(find_llama || true)"
  else
    fail "unsupported platform: $(uname -s)"
  fi
fi
[[ -n "$LLAMA_BIN" ]] || fail "llama-cli was not found after installation"
ok "llama-cli" "$LLAMA_BIN"

# ---------------------------------------------------------------------- #
# Download and verify the model
# ---------------------------------------------------------------------- #
mkdir -p "$MODEL_DIR"
if [[ "$MODEL_OK" -eq 0 ]]; then
  if [[ "$NO_DOWNLOAD" -eq 1 ]]; then
    fail "model missing or SHA mismatch and --no-download was given"
  fi
  say ""
  say "[2/3] Downloading $MODEL_NAME (469 MB)"
  curl -L --fail --progress-bar -o "$MODEL.part" "$MODEL_URL"
  ACTUAL_SHA="$(sha256_of "$MODEL.part" || true)"
  if [[ "$ACTUAL_SHA" != "$MODEL_SHA" ]]; then
    rm -f "$MODEL.part"
    fail "SHA-256 mismatch: expected $MODEL_SHA, got ${ACTUAL_SHA:-unknown}"
  fi
  mv "$MODEL.part" "$MODEL"
  MODEL_OK=1
fi
ok "model" "$MODEL_NAME · sha OK"

[[ -f "$GRAMMAR" ]] || fail "grammar missing: $GRAMMAR"

# ---------------------------------------------------------------------- #
# Smoke test
# ---------------------------------------------------------------------- #
say ""
say "[3/3] Smoke test (one label, one thread)"

PROMPT="Label the question with exactly one label from: count, unique_count, list_frames, fastest, direction_count, dwell, colour, co_presence, compare, peak, unsupported.
Examples:
how many people? -> count
which track moved fastest -> fastest
Question: how many people?
Label:"

BASE=("$LLAMA_BIN" -m "$MODEL" -p "$PROMPT" -n 8 --log-disable -st --grammar-file "$GRAMMAR" --temp 0)
if command -v taskset >/dev/null 2>&1; then
  BASE=(taskset -c 3 nice -n 10 "${BASE[@]}")
elif command -v nice >/dev/null 2>&1; then
  BASE=(nice -n 10 "${BASE[@]}")
fi

smoke_label() {
  local output="$1"
  printf '%s\n' "$output" \
    | grep -E '^(count|unique_count|list_frames|fastest|direction_count|dwell|colour|co_presence|compare|peak|unsupported)$' \
    | head -1 || true
}

OUTPUT="$("${BASE[@]}" --no-jinja 2>&1 || true)"
LABEL="$(smoke_label "$OUTPUT")"
if [[ -z "$LABEL" ]]; then
  # Older llama.cpp builds may not have --no-jinja.
  OUTPUT="$("${BASE[@]}" 2>&1 || true)"
  LABEL="$(smoke_label "$OUTPUT")"
fi
if [[ -z "$LABEL" ]]; then
  printf '%s\n' "$OUTPUT" | tail -20 >&2
  fail "the smoke test did not return a label"
fi
ok "label" "$LABEL"

cat > "$MODEL_DIR/install.json" <<EOF
{
  "model": "$MODEL_NAME",
  "sha256": "$MODEL_SHA",
  "llama_cli": "$LLAMA_BIN",
  "llama_version": "$("$LLAMA_BIN" --version 2>&1 | head -1)",
  "installed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

say ""
say "Ready. Try:"
say "  arten index --run results/demo_pi5_pedestrians --force"
say "  arten ask \"how many people?\" --db results/demo_pi5_pedestrians/evidence.db --parser llm"
say ""
say "To copy the model and code to the Pi over LAN:"
say "  PI_HOST=pi@raspberrypi.local bash scripts/sync_to_pi.sh"
