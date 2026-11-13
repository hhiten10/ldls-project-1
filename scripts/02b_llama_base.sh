#!/usr/bin/env bash
# Base (non-Instruct) Llama-3.2-1B, converted with the pinned llama.cpp exactly like the Mamba models:
# unsloth/Llama-3.2-1B is an ungated mirror of meta-llama/Llama-3.2-1B (bf16 safetensors).
# --outtype auto = bf16 intermediate (lossless for a bf16 checkpoint), then Q4_K_M; sources deleted after.
set -euo pipefail
source "$(dirname "$0")/env.sh"
PY="$VENV/bin/python"; HF="$VENV/bin/hf"
OUT="$MODELS/gguf/Llama-3.2-1B-Q4_K_M.gguf"
if [ ! -s "$OUT" ]; then
  SRC="$MODELS/src/llama-3.2-1b"
  "$HF" download unsloth/Llama-3.2-1B --local-dir "$SRC" > /dev/null
  "$PY" "$LLAMA/convert_hf_to_gguf.py" "$SRC" --outtype auto --outfile "$MODELS/gguf/llama-3.2-1b-intermediate.gguf"
  "$BIN/llama-quantize" "$MODELS/gguf/llama-3.2-1b-intermediate.gguf" "$OUT" Q4_K_M
  rm -rf "$SRC" "$MODELS/gguf/llama-3.2-1b-intermediate.gguf"
fi
ls -la "$OUT"; sha256sum "$OUT" | tee "$RESULTS/hw/llama_base_sha256.txt"
