#!/usr/bin/env bash
# Third pass: closes the requirement gaps found in the audit (all timed, strictly sequential).
#  1. base Llama-3.2-1B (replaces the Instruct variant; Instruct results archived in results/llama_instruct/)
#  2. threads x context cross sweep (decode t=1,4 x d=1k,8k for all six models; prefill pp4096 t=1,4 for two)
#  3. peak RSS for every perplexity / sample / profiling run (04 and 07 now log it)
source "$(dirname "$0")/env.sh"
cd "$REPO"
stage() { echo "[$(date -Is)] START $1"; bash scripts/run_logged.sh "${@:2}" > /dev/null 2>&1; echo "[$(date -Is)] END   $1 rc=$?"; }
stage sweep-llama-base 05_sweep.sh all llama-1b-q4
export CROSS_PREFILL="m2-1.3b-q4 llama-1b-q4"
for m in m2-1.3b-q4 m2-1.3b-q8 m2-2.7b-q4 m1-1.4b-q4 m1-1.4b-q8 llama-1b-q4; do
  stage "cross-$m" 05_sweep.sh cross "$m"
done
stage correctness 04_correctness.sh
rm -f "$RESULTS"/profile/ops_* "$RESULTS"/profile/perf_* "$RESULTS"/profile/stat_*
stage profile 07_profile.sh
"$VENV/bin/python" scripts/06_roofline.py "$RESULTS" "$MODELS"/gguf/*.gguf > "$RESULTS/roofline_bytes.txt" 2>&1
"$VENV/bin/python" scripts/08_analyze.py "$REPO" > "$WORK/logs/analyze.log" 2>&1
"$VENV/bin/python" scripts/10_gap.py "$REPO" > "$WORK/logs/gap.log" 2>&1
"$VENV/bin/python" bonus/analyze_bonus.py "$REPO" > "$WORK/logs/analyze_bonus.log" 2>&1
echo "[$(date -Is)] GAPS DONE"
