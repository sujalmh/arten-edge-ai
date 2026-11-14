# Local LLM for `arten ask`

This directory holds the optional GGUF model used by the natural-language query
compiler. **The model is not committed** (469 MB); fetch it with
`bash scripts/setup_llm.sh` or copy it over LAN.

## Default model

| Field | Value |
|---|---|
| Model | Qwen2.5-0.5B-Instruct |
| Quantisation | Q4_K_M |
| File | `qwen2.5-0.5b-instruct-q4_k_m.gguf` |
| Size | 491,400,032 bytes (469 MB) |
| SHA-256 | `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db` |
| Licence | Apache-2.0 |
| Source | <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF> |

Why this one: it is small enough to leave the detector's memory alone, and Arm
measured Qwen2.5-0.5B Q4 at **~32 tok/s** on a Raspberry Pi 5 at one thread. The
query compiler only needs to emit a short label, so latency is dominated by
prompt processing, not generation.

## Fetch

```bash
curl -L --fail -o models/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
shasum -a 256 models/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

On the Pi, `sha256sum` instead of `shasum`.

## Runtime

`llama.cpp` is installed by `scripts/setup_llm.sh`:

- macOS (development): `brew install llama.cpp`
- Raspberry Pi / Linux aarch64: built from source under `vendor/llama.cpp`,
  `-DGGML_NATIVE=ON -DLLAMA_CURL=OFF`, `-j2`, `nice`.

The macOS binary is **not** transferable to the Pi. The GGUF and the Python code
are; the Pi builds its own `llama-cli`. Use `scripts/sync_to_pi.sh` to copy the
model and code over LAN.

## Alternatives

| Model | Size | Notes |
|---|---:|---|
| Qwen3-0.6B Q4_K_M | ~0.5 GB | Stronger instruction following; same class |
| Qwen2.5-1.5B-Instruct Q4_K_M | ~1 GB | Better reasoning; slower on Pi |

Change `--model` on `arten ask` to try one. Keep the model file out of git.
