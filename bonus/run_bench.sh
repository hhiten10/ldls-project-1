#!/usr/bin/env bash
# Bonus timing: Mamba-3 SISO vs MIMO(R=4) vs Mamba-2 decode step, same state (64x64x128 fp32 = 2 MB/layer).
# Two cache regimes: layers=1 (state stays in cache: best case) and layers=24 (48 MB >> 8 MB L3: DRAM-cold,
# what a real 24-layer model sees). Threads 1 and 8 (pinned to physical cores). 3 reps inside each run.
# Plus perf stat counters (IPC, cache misses) for the AVX2 kernels in both regimes.
set -euo pipefail
source "$(dirname "$0")/../scripts/env.sh"
OUT="$RESULTS/bonus"; mkdir -p "$OUT"
K="$REPO/bonus/kernel/m3bench"; G="$REPO/bonus/ggml/ggml_driver"; TR="$WORK/traces"
PERF=/usr/lib/linux-tools-6.8.0-107/perf
places() { local s=""; for c in 0 2 4 6 8 10 12 14; do s+="{$c},"; done; echo "${s%,}"; }
export OMP_PROC_BIND=close OMP_PLACES="$(places)"
H="variant,impl,R,threads,layers,us_per_step,sd_us,gflops,state_GBs,flop_per_byte"
echo "$H" > "$OUT/kernel_timing.csv"
for th in 1 8; do
  for L in 1 24; do
    steps=$(( L == 1 ? 4000 : 2400 ))
    vars="m3avx2 m2avx2"; [ $th = 1 ] && vars="m3scalar m3avx2 m2scalar m2avx2"
    OMP_NUM_THREADS=$th "$K" bench "$TR/siso_L12" $th $L $steps $vars | tee -a "$OUT/kernel_timing.csv"
    OMP_NUM_THREADS=$th "$K" bench "$TR/mimo_L12" $th $L $steps m3avx2 $([ $th = 1 ] && echo m3scalar) | tee -a "$OUT/kernel_timing.csv"
    OMP_NUM_THREADS=$th "$G" "$TR/siso_L12" $th $L $steps | grep -v '^check' | tee -a "$OUT/kernel_timing.csv"
    OMP_NUM_THREADS=$th "$G" "$TR/mimo_L12" $th $L $steps | grep -v '^check' | grep m3_ | tee -a "$OUT/kernel_timing.csv"
  done
done
# hardware counters, single thread, hot vs cold
for L in 1 24; do
  for tr in siso mimo; do
    OMP_NUM_THREADS=1 "$PERF" stat -x, -e cycles:u,instructions:u,cache-references:u,cache-misses:u,L1-dcache-load-misses:u \
      -o "$OUT/stat_${tr}_L$L.csv" -- "$K" bench "$TR/${tr}_L12" 1 $L 3000 m3avx2 > /dev/null
  done
  OMP_NUM_THREADS=1 "$PERF" stat -x, -e cycles:u,instructions:u,cache-references:u,cache-misses:u,L1-dcache-load-misses:u \
    -o "$OUT/stat_m2_L$L.csv" -- "$K" bench "$TR/siso_L12" 1 $L 3000 m2avx2 > /dev/null
done
# correctness logs alongside the timings
"$K" check "$TR/siso_L12" > "$OUT/check_siso.txt"; cp "$TR/siso_L12/check_errors.csv" "$OUT/check_errors_siso.csv"
"$K" check "$TR/mimo_L12" > "$OUT/check_mimo.txt"; cp "$TR/mimo_L12/check_errors.csv" "$OUT/check_errors_mimo.csv"
OMP_NUM_THREADS=8 "$G" "$TR/siso_L12" 8 2 10 | grep '^check' > "$OUT/check_ggml.txt"
OMP_NUM_THREADS=8 "$G" "$TR/mimo_L12" 8 2 10 | grep '^check' >> "$OUT/check_ggml.txt"
echo BONUS_BENCH_DONE
