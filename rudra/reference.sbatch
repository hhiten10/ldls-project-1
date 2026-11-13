#!/bin/bash
# Shared reference run (spec): Llama-3.2-1B Q4_K_M, 512-token prompt, 128-token decode, PARAM Rudra CPU node.
# This account's QoS (aicte_qip): max 4 CPUs/user, 1 node, 5 min wall, 2 jobs -> 4 cores, shared node.
# (An --exclusive 48-core request pends forever with QOSMaxCpuPerUserLimit.)
#SBATCH --job-name=p1-ref
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=/home/bapic_iiitd/2_group/aditya/p1_rudra/ref_%j.out
D=/home/bapic_iiitd/2_group/aditya/p1_rudra
cd "$D"
module load compiler/gcc/12.3 2>/dev/null || true
echo "node: $(hostname)  allocated cpus: $(taskset -cp $$ | awk -F: '{print $2}')"
lscpu | grep -E "Model name|^CPU\(s\)|Socket|NUMA node\(s\)|L3|MHz"
B=llama.cpp/build/bin/llama-bench; M=${MODEL:-models/Llama-3.2-1B-Q4_K_M.gguf}   # base model (sbatch --export=ALL,MODEL=...)
echo "model: $M  sha256: $(sha256sum "$M" | cut -d' ' -f1)"
export OMP_PROC_BIND=close OMP_PLACES=cores
# The fixed reference test at 4 threads (= all allocated cores), 3 repetitions; then 1 and 2 threads.
# compute nodes have no /usr/bin/time: peak RSS = kernel high-water mark VmHWM, polled until exit
peak_run() {
  "$@" & local pid=$! hwm=0 v
  while kill -0 $pid 2>/dev/null; do
    v=$(awk '/VmHWM/{print $2}' /proc/$pid/status 2>/dev/null); [ -n "$v" ] && hwm=$v; sleep 0.1
  done
  wait $pid; echo "peak_rss_kb=$hwm"
}
for t in 4 2 1; do
  OMP_NUM_THREADS=$t peak_run "$B" -m "$M" -p 512 -n 128 -t $t -r 3 -o jsonl
done
OMP_NUM_THREADS=4 "$B" -m "$M" -p 512 -n 128 -t 4 -r 3 -o md
