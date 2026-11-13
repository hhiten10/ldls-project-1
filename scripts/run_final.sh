#!/usr/bin/env bash
# Fourth pass, after run_gaps.sh: (a) sample outputs re-run with -c 2048 (memory/context fix);
# (b) Llama tg128_d0 re-run - its first pass had one transient repetition (22.7 vs 31.5-34.6 tok/s); the first
#     pass is kept as llama-1b-q4__tg128_d0.firstpass.jsonl and both are reported; (c) analysis regenerated.
source "$(dirname "$0")/env.sh"
cd "$REPO"
while pgrep -f "scripts/run_gaps.sh" > /dev/null; do sleep 20; done
bash scripts/run_logged.sh 04b_samples.sh > /dev/null 2>&1; echo "[$(date -Is)] samples rc=$?"
R="$RESULTS/raw"
mv "$R/llama-1b-q4__tg128_d0.jsonl" "$R/llama-1b-q4__tg128_d0.firstpass.jsonl"
sed -i 's/^llama-1b-q4,tg128_d0,/llama-1b-q4,tg128_d0_firstpass,/' "$R/runs_meta.csv"
bash scripts/run_logged.sh 05_sweep.sh decode llama-1b-q4 > /dev/null 2>&1; echo "[$(date -Is)] llama d0 rerun rc=$?"
"$VENV/bin/python" scripts/06_roofline.py "$RESULTS" "$MODELS"/gguf/*.gguf > "$RESULTS/roofline_bytes.txt" 2>&1
"$VENV/bin/python" scripts/08_analyze.py "$REPO" > "$WORK/logs/analyze.log" 2>&1
"$VENV/bin/python" scripts/10_gap.py "$REPO" > "$WORK/logs/gap.log" 2>&1
"$VENV/bin/python" bonus/analyze_bonus.py "$REPO" > "$WORK/logs/analyze_bonus.log" 2>&1
echo "[$(date -Is)] FINAL DONE"
