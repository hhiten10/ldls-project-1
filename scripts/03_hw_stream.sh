#!/usr/bin/env bash
# Hardware record + STREAM memory bandwidth (McCalpin, official stream.c).
# Arrays: 80M doubles x 3 = 1.9 GB  (>> 4x the 8 MB L3, as STREAM's rules require).
# Run with 1..8 threads pinned one-per-physical-core; 3 repetitions each (spec: repeat 3x).
set -euo pipefail
source "$(dirname "$0")/env.sh"
OUT="$RESULTS/hw"; mkdir -p "$OUT" "$WORK/stream"
cd "$WORK/stream"
[ -s stream.c ] || curl -sSfL -o stream.c https://www.cs.virginia.edu/stream/FTP/Code/stream.c
gcc -O3 -march=native -fopenmp -mcmodel=medium -DSTREAM_ARRAY_SIZE=80000000 -DNTIMES=20 stream.c -o stream

{
  echo "## date";          date -Is
  echo "## uname";         uname -a
  echo "## lscpu";         lscpu
  echo "## caches";        lscpu -C 2>/dev/null || true
  echo "## topology (cpu,core,socket,node)"; lscpu -p=CPU,CORE,SOCKET,NODE,CACHE
  echo "## simd flags";    grep -o -w -E 'sse4_2|avx|avx2|fma|f16c|avx512[a-z_]*|avx_vnni' /proc/cpuinfo | sort | uniq -c
  echo "## memory";        free -h; grep -E 'MemTotal|SwapTotal' /proc/meminfo
  echo "## gcc";           gcc --version | head -1
  echo "## llama.cpp";     cat "$LLAMA_COMMIT_FILE" 2>/dev/null || true
} > "$OUT/hw_linux.txt" 2>&1

# One thread per physical core: pick the first logical CPU of each core id.
PCORES=$(lscpu -p=CPU,CORE | grep -v '^#' | sort -t, -k2,2n -k1,1n | awk -F, '!seen[$2]++{print $1}' | paste -sd,)
echo "physical-core cpus: $PCORES" | tee -a "$OUT/hw_linux.txt"

gcc -O3 -march=native -fopenmp "$REPO/scripts/tools/readbw.c" -o readbw

echo "threads,rep,copy_MBs,scale_MBs,add_MBs,triad_MBs,readonly_MBs" > "$OUT/stream.csv"
for t in 1 2 3 4 5 6 7 8; do
  cpus=$(echo "$PCORES" | cut -d, -f1-"$t")
  places="{$(echo "$cpus" | sed 's/,/},{/g')}"
  for rep in 1 2 3; do
    res=$(OMP_NUM_THREADS=$t OMP_PROC_BIND=true OMP_PLACES="$places" ./stream)
    line=$(echo "$res" | awk '/^Copy:/{c=$2}/^Scale:/{s=$2}/^Add:/{a=$2}/^Triad:/{tr=$2}END{print c","s","a","tr}')
    ro=$(OMP_NUM_THREADS=$t OMP_PROC_BIND=true OMP_PLACES="$places" ./readbw | awk '{print $2}')
    echo "$t,$rep,$line,$ro" | tee -a "$OUT/stream.csv"
  done
done
# Also all 16 logical CPUs (SMT on), for the record
for rep in 1 2 3; do
  line=$(OMP_NUM_THREADS=16 ./stream | awk '/^Copy:/{c=$2}/^Scale:/{s=$2}/^Add:/{a=$2}/^Triad:/{tr=$2}END{print c","s","a","tr}')
  ro=$(OMP_NUM_THREADS=16 ./readbw | awk '{print $2}')
  echo "16smt,$rep,$line,$ro" | tee -a "$OUT/stream.csv"
done
