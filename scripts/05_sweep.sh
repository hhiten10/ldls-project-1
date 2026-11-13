#!/usr/bin/env bash
# Scripted benchmark sweeps (spec step 5). One llama-bench process per configuration so that
# peak RSS (/usr/bin/time %M) and swap activity (/proc/vmstat pswpin/pswpout deltas) are per run.
# Every configuration: -r 3 (llama-bench reports mean and stddev over the 3 repetitions).
# Threads are pinned one-per-physical-core via OpenMP places (logical CPUs 0,2,4,...).
#
# Usage: bash scripts/05_sweep.sh <sweep> [model-tag ...]
#   sweeps: prefill | decode | threads | all | cross (threads x context; CROSS_PREFILL="tag ..." adds threads x pp4096)
set -euo pipefail
source "$(dirname "$0")/env.sh"
RAW="$RESULTS/raw"; mkdir -p "$RAW"
REPS=${REPS:-3}
declare -A GGUF=(
  [m2-1.3b-q4]="mamba2-1.3b-Q4_K_M.gguf"  [m2-1.3b-q8]="mamba2-1.3b-Q8_0.gguf"
  [m2-2.7b-q4]="mamba2-2.7b-Q4_K_M.gguf"
  [m1-1.4b-q4]="mamba-1.4b-Q4_K_M.gguf"   [m1-1.4b-q8]="mamba-1.4b-Q8_0.gguf"
  [llama-1b-q4]="Llama-3.2-1B-Q4_K_M.gguf"
)
# Largest allowed context per spec: pure recurrent 32k, Llama-3.2-1B 16k.
maxctx() { case "$1" in llama-1b*) echo 16384;; *) echo 32768;; esac; }

PCPUS=(0 2 4 6 8 10 12 14)
places() { local n=$1 s=""; for ((i=0;i<n;i++)); do s+="{${PCPUS[$i]}},"; done; echo "${s%,}"; }
# swap-ins = some process waited on disk during the run (invalidates timing); swap-outs alone are the
# kernel evicting idle pages to grow page cache (swappiness=60) and cost a running process nothing.
swpin()  { awk '/^pswpin /{print $2}'  /proc/vmstat; }
swpout() { awk '/^pswpout /{print $2}' /proc/vmstat; }

run() { # tag id threads -- llama-bench args...
  local tag=$1 id=$2 t=$3; shift 3
  local out="$RAW/${tag}__${id}"
  if [ -s "$out.jsonl" ] && [ "${FORCE:-0}" != 1 ]; then echo "skip $tag $id"; return; fi
  local avail_mb; avail_mb=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  local i0 o0; i0=$(swpin); o0=$(swpout)
  OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES="$(places "$t")" \
  /usr/bin/time -f "%M %e %F" -o "$out.time" \
    "$BIN/llama-bench" -m "$MODELS/gguf/${GGUF[$tag]}" -t "$t" -r "$REPS" -o jsonl "$@" \
    > "$out.jsonl" 2> "$out.err" || { echo "FAILED $tag $id (see $out.err)"; return 0; }
  local din=$(( $(swpin) - i0 )) dout=$(( $(swpout) - o0 ))
  read -r rss_kb wall majflt < "$out.time"
  echo "$tag,$id,$t,$rss_kb,$wall,$avail_mb,$din,$dout,$majflt" >> "$RAW/runs_meta.csv"
  printf "%-12s %-16s t=%-2s peakRSS=%6.2f GB wall=%6ss swap_in=%s swap_out=%s majflt=%s\n" \
    "$tag" "$id" "$t" "$(awk -v k="$rss_kb" 'BEGIN{print k/1048576}')" "$wall" "$din" "$dout" "$majflt"
  # Invalid if > SWAPIN_MAX pages (default 256 = 1 MB) were swapped IN during the run. A handful of pages
  # (an idle daemon paged back in) costs milliseconds on a minutes-long run; the model itself swapping
  # shows up as thousands of pages. Raw counts are kept in runs_meta.csv for every run.
  if [ "$din" -gt "${SWAPIN_MAX:-256}" ]; then
    echo "  !! SWAP-IN ($din pages) during $tag $id -> timing invalid"
    mv "$out.jsonl" "$out.swapped.jsonl"
    if [ "${RETRY:-1}" = 1 ]; then RETRY=0 run "$tag" "$id" "$t" "$@"; fi
  fi
  return 0
}
[ -s "$RAW/runs_meta.csv" ] || echo "tag,id,threads,peak_rss_kb,wall_s,mem_avail_mb_before,swap_in_pages,swap_out_pages,major_faults" > "$RAW/runs_meta.csv"

sweep=$1; shift
tags=("$@"); [ ${#tags[@]} -gt 0 ] || tags=(m2-1.3b-q4 m2-1.3b-q8 m2-2.7b-q4 m1-1.4b-q4 m1-1.4b-q8 llama-1b-q4)
T=8

for tag in "${tags[@]}"; do
  mc=$(maxctx "$tag")
  if [[ $sweep == prefill || $sweep == all ]]; then
    for p in 512 4096 "$mc"; do run "$tag" "pp$p" $T -p "$p" -n 0 -b 2048 -ub 512; done
  fi
  if [[ $sweep == decode || $sweep == all ]]; then
    # Context lengths per spec: 1k, 8k, max. Plus d=0 as the empty-context baseline.
    # -d prefills the context first (not timed), then times 128 generated tokens.
    for d in 0 1024 8192 "$mc"; do run "$tag" "tg128_d$d" $T -p 0 -n 128 -d "$d" -b 2048 -ub 512; done
  fi
  if [[ $sweep == threads || $sweep == all ]]; then
    for t in 1 2 3 4 5 6 7 8; do
      run "$tag" "pp512_t$t" "$t" -p 512 -n 0
      run "$tag" "tg128_t$t" "$t" -p 0 -n 128
    done
  fi
  if [[ $sweep == cross ]]; then
    # threads x context (decode) - the 8-thread column is the decode sweep above (tg128_d1024 / _d8192)
    for t in 1 4; do
      for d in 1024 8192; do run "$tag" "tg128_t${t}_d$d" "$t" -p 0 -n 128 -d "$d" -b 2048 -ub 512; done
    done
    # threads x prompt length (prefill), for the models named in CROSS_PREFILL (8 threads = pp4096 above)
    if [[ " ${CROSS_PREFILL:-} " == *" $tag "* ]]; then
      for t in 1 4; do run "$tag" "pp4096_t$t" "$t" -p 4096 -n 0 -b 2048 -ub 512; done
    fi
  fi
done
