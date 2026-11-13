#!/usr/bin/env bash
# Download official checkpoints, convert to GGUF with the pinned llama.cpp, quantize.
# Quantization rules (spec): Q4_K_M for every model; Q8_0 only for <=1.5B params; no F16 runs.
# Disk discipline (the WSL disk lives on C:, which is nearly full on this laptop):
#   * the intermediate GGUF uses --outtype auto = the checkpoint's own 16-bit type (lossless, half of F32)
#   * each raw checkpoint and intermediate is deleted as soon as its quantized GGUFs exist
#   * abort if the Windows C: drive has < MIN_FREE_GB free
set -euo pipefail
source "$(dirname "$0")/env.sh"
PY="$VENV/bin/python"; HF="$VENV/bin/hf"
MIN_FREE_GB=${MIN_FREE_GB:-3}
mkdir -p "$MODELS/src" "$MODELS/gguf"

guard() {
  local free; free=$(df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9)
  if [ "$free" -lt "$MIN_FREE_GB" ]; then echo "ABORT: C: has only ${free} GB free (< ${MIN_FREE_GB})"; exit 3; fi
}
dl() { guard; local repo="$1" dest="$2"; shift 2; "$HF" download "$repo" "$@" --local-dir "$dest" >/dev/null; }

convert_quant() { # name srcdir quant1 [quant2]
  local name="$1" src="$2"; shift 2
  local need=0; for q in "$@"; do [ -s "$MODELS/gguf/$name-$q.gguf" ] || need=1; done
  if [ $need = 1 ]; then
    local mid="$MODELS/gguf/$name-intermediate.gguf"
    guard
    "$PY" "$LLAMA/convert_hf_to_gguf.py" "$src" --outtype auto --outfile "$mid"
    for q in "$@"; do guard; "$BIN/llama-quantize" "$mid" "$MODELS/gguf/$name-$q.gguf" "$q"; done
    rm -f "$mid"
  else
    echo "$name: already converted"
  fi
  [ "${KEEP_SRC:-0}" = 1 ] || rm -rf "$src"
}

"$PY" -m pip install -q -e "$LLAMA/gguf-py"
rm -f "$MODELS/gguf/"*-F32.gguf "$MODELS/gguf/"*-intermediate.gguf      # partial leftovers

# state-spaces/mamba2-* ship no tokenizer: llama.cpp's Mamba2 converter then uses its built-in
# GPT-NeoX vocab (the tokenizer these checkpoints were trained with) - the intended path.
for spec in "mamba2-1.3b:state-spaces/mamba2-1.3b:Q4_K_M Q8_0" \
            "mamba2-2.7b:state-spaces/mamba2-2.7b:Q4_K_M" \
            "mamba-1.4b:state-spaces/mamba-1.4b-hf:Q4_K_M Q8_0"; do
  IFS=: read -r name repo quants <<< "$spec"
  src="$MODELS/src/${repo#state-spaces/}"
  done_all=1; for q in $quants; do [ -s "$MODELS/gguf/$name-$q.gguf" ] || done_all=0; done
  [ $done_all = 1 ] || dl "$repo" "$src"
  [ -d "$src" ] || mkdir -p "$src"
  convert_quant "$name" "$src" $quants      # 1.3B/1.4B <= 1.5B -> Q8_0 allowed; 2.7B -> Q4_K_M only
done

# Shared reference model (Llama-3.2-1B Q4_K_M, prebuilt) + Mamba-3 checkpoints for the bonus (kept)
[ -s "$MODELS/gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf" ] || dl bartowski/Llama-3.2-1B-Instruct-GGUF "$MODELS/gguf" Llama-3.2-1B-Instruct-Q4_K_M.gguf
for m in mamba3-siso-1.5b mamba3-mimo-1.5b; do dl "state-spaces/$m" "$MODELS/src/$m"; done
dl unsloth/Llama-3.2-1B "$MODELS/src/llama3-tok" tokenizer.json tokenizer_config.json special_tokens_map.json

ls -la "$MODELS/gguf"; du -sh "$MODELS/src"/* ; df -h /mnt/c | tail -1
