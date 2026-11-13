#!/usr/bin/env bash
# Correctness (spec step 3): perplexity on a FIXED small text sample + a few greedy sample outputs.
# Sample: wikitext-2-raw test set, first 40 chunks of 512 tokens (same file & settings for every model).
# PPL is only comparable between quantizations of the SAME model (tokenizers differ across families).
set -euo pipefail
source "$(dirname "$0")/env.sh"
OUT="$RESULTS/correctness"; mkdir -p "$OUT" "$WORK/data"
cd "$WORK/data"
if [ ! -f wikitext-2-raw/wiki.test.raw ]; then
  curl -sSfL -o wt2.zip https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
  "$VENV/bin/python" -c "import zipfile;zipfile.ZipFile('wt2.zip').extractall('.')"; rm -f wt2.zip
fi
sha256sum wikitext-2-raw/wiki.test.raw | tee "$OUT/text_sample.sha256"
PLACES="{0},{2},{4},{6},{8},{10},{12},{14}"
# main models (gguf/) + the own-experiment variants (exp/), so every perplexity run logs peak RSS
models=(gguf/mamba2-1.3b-Q4_K_M gguf/mamba2-1.3b-Q8_0 gguf/mamba2-2.7b-Q4_K_M gguf/mamba-1.4b-Q4_K_M gguf/mamba-1.4b-Q8_0
        gguf/Llama-3.2-1B-Q4_K_M)
for v in q4_0 q4_K q5_K q6_K q8_0; do models+=("exp/m2-1.3b-ssm_in-$v"); done
echo "model,ppl,ppl_err,peak_rss_kb" > "$OUT/perplexity.csv"
echo "model,prompt,peak_rss_kb" > "$OUT/samples_rss.csv"
for mp in "${models[@]}"; do
  m=$(basename "$mp"); f="$MODELS/$mp.gguf"; [ -s "$f" ] || { echo "missing $f"; continue; }
  OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES="$PLACES" /usr/bin/time -f "%M" -o "$OUT/.rss" \
    "$BIN/llama-perplexity" -m "$f" -f wikitext-2-raw/wiki.test.raw -c 512 --chunks 40 -t 8 -b 512 \
    > "$OUT/ppl_$m.log" 2>&1
  est=$(grep -o "Final estimate: PPL = [0-9.]* +/- [0-9.]*" "$OUT/ppl_$m.log" | awk '{print $5","$7}')
  echo "$m,$est,$(cat "$OUT/.rss")" | tee -a "$OUT/perplexity.csv"
  [[ $mp == exp/* ]] && continue          # sample outputs only for the main models
  # Greedy sample outputs (read them: a broken model produces garbage even if it "runs")
  : > "$OUT/samples_$m.txt"
  for prompt in "The capital of France is" "Water boils at a temperature of" "def fibonacci(n):"; do
    echo "### $prompt" >> "$OUT/samples_$m.txt"
    OMP_NUM_THREADS=8 /usr/bin/time -f "%M" -o "$OUT/.rss" "$BIN/llama-completion" -m "$f" -c 2048 -p "$prompt" -n 48 --temp 0 -t 8 -no-cnv \
      2>/dev/null >> "$OUT/samples_$m.txt" || true
    echo "$m,\"$prompt\",$(cat "$OUT/.rss")" >> "$OUT/samples_rss.csv"
    echo >> "$OUT/samples_$m.txt"
  done
done
rm -f "$OUT/.rss"
cat "$OUT/perplexity.csv"
