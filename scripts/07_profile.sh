#!/usr/bin/env bash
# Profiling (spec step 6). Three independent views per model, decode and prefill separately:
#  (1) per-op wall-clock breakdown from the GGML_OP_PROFILE build (exact, includes barrier waits)
#  (2) perf record (cycles:u sampling) on the normal benchmark build -> hottest functions
#  (3) perf stat hardware counters: IPC, cache misses, LLC (memory-boundness evidence)
# usage: bash scripts/07_profile.sh [model-tag ...]
set -euo pipefail
source "$(dirname "$0")/env.sh"
PERF=/usr/lib/linux-tools-6.8.0-107/perf
PROF_BIN="$LLAMA/build-prof/bin"
OUT="$RESULTS/profile"; mkdir -p "$OUT"
declare -A GGUF=(
  [m2-1.3b-q4]="mamba2-1.3b-Q4_K_M.gguf" [m2-2.7b-q4]="mamba2-2.7b-Q4_K_M.gguf"
  [m1-1.4b-q4]="mamba-1.4b-Q4_K_M.gguf"  [llama-1b-q4]="Llama-3.2-1B-Q4_K_M.gguf"
  [m2-1.3b-q8]="mamba2-1.3b-Q8_0.gguf"   [m1-1.4b-q8]="mamba-1.4b-Q8_0.gguf"
)
tags=("$@"); [ ${#tags[@]} -gt 0 ] || tags=(m2-1.3b-q4 m2-2.7b-q4 m1-1.4b-q4 llama-1b-q4)
export OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES="{0},{2},{4},{6},{8},{10},{12},{14}"
declare -A PHASE=( [decode]="-p 0 -n 256" [prefill512]="-p 512 -n 0" [prefill4096]="-p 4096 -n 0 -ub 512"
                   [prefill8192]="-p 8192 -n 0 -ub 512" [decode_d8192]="-p 0 -n 256 -d 8192 -ub 512" )
EVENTS=cycles:u,instructions:u,cache-references:u,cache-misses:u,L1-dcache-loads:u,L1-dcache-load-misses:u
TMPD="$WORK/tmp_profile"; mkdir -p "$TMPD"
echo "tag,phase,run,peak_rss_kb" > "$OUT/profile_rss.csv"

for tag in "${tags[@]}"; do
  m="$MODELS/gguf/${GGUF[$tag]}"
  phases="decode prefill512 prefill4096"
  # Llama only: decode with 8k context (attention-dominated). llama-bench -d runs the 8k prefill in the same
  # process, so we also profile that prefill alone; analysis subtracts it (decode_d8192 - prefill8192).
  [[ $tag == llama* ]] && phases="$phases prefill8192 decode_d8192"
  for ph in $phases; do
    args=(${PHASE[$ph]})
    rss() { echo "$tag,$ph,$1,$(cat "$TMPD/rss")" >> "$OUT/profile_rss.csv"; }
    # (1) op breakdown, 3 repetitions (separate processes) -> mean/spread computed in analysis
    for rep in 1 2 3; do
      GGML_OP_PROFILE="$OUT/ops_${tag}_${ph}_r$rep.csv" /usr/bin/time -f "%M" -o "$TMPD/rss" \
        "$PROF_BIN/llama-bench" -m "$m" -t 8 -r 1 --no-warmup "${args[@]}" -o jsonl \
        > "$OUT/ops_${tag}_${ph}_r$rep.jsonl" 2>/dev/null
      rss "opprofile_r$rep"
    done
    # (2) perf sampling on the clean build (sample file on the WSL ext4 disk: /mnt/c is slow)
    /usr/bin/time -f "%M" -o "$TMPD/rss" "$PERF" record -q -e cycles:u -F 2999 -o "$TMPD/perf.data" -- \
      "$BIN/llama-bench" -m "$m" -t 8 -r 1 "${args[@]}" > /dev/null 2>&1
    rss "perf_record(incl. perf)"
    # write the full report, then truncate (a `| head` would SIGPIPE perf and kill the script under pipefail)
    "$PERF" report -q -i "$TMPD/perf.data" --no-children --sort sym --stdio 2>/dev/null > "$TMPD/full.txt" || true
    grep -v '^$' "$TMPD/full.txt" | awk 'NR<=40' > "$OUT/perf_${tag}_${ph}_top.txt" || true
    "$PERF" report -q -i "$TMPD/perf.data" --no-children --sort dso --stdio 2>/dev/null | grep -v '^$' \
      | awk 'NR<=12' > "$OUT/perf_${tag}_${ph}_dso.txt" || true
    rm -f "$TMPD/perf.data" "$TMPD/full.txt"
    # (3) counters
    /usr/bin/time -f "%M" -o "$TMPD/rss" "$PERF" stat -x, -e "$EVENTS" -o "$OUT/stat_${tag}_${ph}.csv" -- \
      "$BIN/llama-bench" -m "$m" -t 8 -r 1 "${args[@]}" > /dev/null 2>&1
    rss "perf_stat"
    echo "profiled $tag $ph"
  done
done
