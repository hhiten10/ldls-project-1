#!/usr/bin/env bash
# Own experiment (spec step 8) - see docs/experiment_preregistration.md. One variable: ssm_in quant type.
set -euo pipefail
source "$(dirname "$0")/env.sh"
OUT="$RESULTS/experiment"; mkdir -p "$OUT" "$MODELS/exp"
SRC="$MODELS/gguf/mamba2-1.3b-Q8_0.gguf"
export OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES="{0},{2},{4},{6},{8},{10},{12},{14}"
echo "ssm_in_type,file_MB,test,mean_tps,sd_tps,peak_rss_kb" > "$OUT/speed.csv"
echo "ssm_in_type,ppl,ppl_err" > "$OUT/ppl.csv"
for T in q4_0 q4_K q5_K q6_K q8_0; do
  f="$MODELS/exp/m2-1.3b-ssm_in-$T.gguf"
  [ -s "$f" ] || "$BIN/llama-quantize" --allow-requantize --tensor-type "ssm_in=$T" "$SRC" "$f" Q4_K_M > "$OUT/quant_$T.log" 2>&1
  mb=$(( $(stat -c %s "$f") / 1000000 ))
  for args in "-p 0 -n 128" "-p 512 -n 0"; do
    /usr/bin/time -f "%M" -o "$OUT/time.tmp" "$BIN/llama-bench" -m "$f" -t 8 -r 3 $args -o jsonl > "$OUT/bench_$T.tmp" 2>/dev/null
    "$VENV/bin/python" -c "
import json,sys
j=json.loads([l for l in open('$OUT/bench_$T.tmp') if l.startswith('{')][0])
test=('tg%d'%j['n_gen']) if j['n_gen'] else ('pp%d'%j['n_prompt'])
print('$T,$mb,%s,%.3f,%.3f,%s'%(test,j['avg_ts'],j['stddev_ts'],open('$OUT/time.tmp').read().strip()))" | tee -a "$OUT/speed.csv"
  done
  "$BIN/llama-perplexity" -m "$f" -f "$WORK/data/wikitext-2-raw/wiki.test.raw" -c 512 --chunks 40 -t 8 -b 512 > "$OUT/ppl_$T.log" 2>&1
  echo "$T,$(grep -o 'Final estimate: PPL = [0-9.]* +/- [0-9.]*' "$OUT/ppl_$T.log" | awk '{print $5","$7}')" | tee -a "$OUT/ppl.csv"
done
# byte accounting per variant for the prediction
ROOFLINE_OUT="$OUT/roofline_variants.csv" "$VENV/bin/python" "$REPO/scripts/06_roofline.py" "$RESULTS" "$MODELS"/exp/*.gguf > "$OUT/roofline_variants.txt"
rm -f "$OUT"/*.tmp
