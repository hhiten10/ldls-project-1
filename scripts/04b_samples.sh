#!/usr/bin/env bash
# Greedy sample outputs only (re-run after fixing the context: llama-completion's default -c 0 means
# "model's trained context" = 131072 for Llama-3.2 -> a 4.3 GB KV cache, violating the 16k / 4.5 GB rules).
# -c 2048 for every model (prompts are < 10 tokens, 48 generated). Peak RSS logged per run.
set -euo pipefail
source "$(dirname "$0")/env.sh"
OUT="$RESULTS/correctness"
echo "model,prompt,peak_rss_kb" > "$OUT/samples_rss.csv"
for m in mamba2-1.3b-Q4_K_M mamba2-1.3b-Q8_0 mamba2-2.7b-Q4_K_M mamba-1.4b-Q4_K_M mamba-1.4b-Q8_0 Llama-3.2-1B-Q4_K_M; do
  f="$MODELS/gguf/$m.gguf"; : > "$OUT/samples_$m.txt"
  for prompt in "The capital of France is" "Water boils at a temperature of" "def fibonacci(n):"; do
    echo "### $prompt" >> "$OUT/samples_$m.txt"
    OMP_NUM_THREADS=8 /usr/bin/time -f "%M" -o "$OUT/.rss" "$BIN/llama-completion" -m "$f" -c 2048 -p "$prompt" -n 48 \
      --temp 0 -t 8 -no-cnv 2>/dev/null >> "$OUT/samples_$m.txt" || true
    echo "$m,\"$prompt\",$(cat "$OUT/.rss")" >> "$OUT/samples_rss.csv"; echo >> "$OUT/samples_$m.txt"
  done
done
rm -f "$OUT/.rss"; cat "$OUT/samples_rss.csv"
