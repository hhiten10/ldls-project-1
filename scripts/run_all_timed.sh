#!/usr/bin/env bash
# All timed work, strictly sequential (nothing else may run on the CPU meanwhile).
# Stages log to $WORK/logs/<stage>.log; a failing stage is reported and the pipeline continues.
source "$(dirname "$0")/env.sh"
cd "$REPO"
stage() { echo "[$(date -Is)] START $1"; bash scripts/run_logged.sh "${@:2}" > /dev/null 2>&1; echo "[$(date -Is)] END   $1 rc=$?"; }
while pgrep -x llama-perplexity > /dev/null; do sleep 20; done      # let correctness finish first

stage stream        03_hw_stream.sh
"$VENV/bin/python" scripts/06_roofline.py "$RESULTS" "$MODELS"/gguf/*.gguf > "$RESULTS/roofline_bytes.txt" 2>&1
echo "[$(date -Is)] roofline done"
echo "[$(date -Is)] START bonus"; bash bonus/run_bench.sh > "$WORK/logs/bonus_bench.log" 2>&1; echo "[$(date -Is)] END bonus rc=$?"
for m in m2-1.3b-q4 m1-1.4b-q4 m2-2.7b-q4 llama-1b-q4 m2-1.3b-q8 m1-1.4b-q8; do
  stage "sweep-$m" 05_sweep.sh all "$m"
done
stage profile       07_profile.sh
stage experiment    09_experiment.sh
"$VENV/bin/python" scripts/08_analyze.py "$REPO" > "$WORK/logs/analyze.log" 2>&1
echo "[$(date -Is)] ALL DONE"
