#!/usr/bin/env bash
# Second pass after run_all_timed.sh: stages that failed there (fixed since), then the analysis.
#  - Mamba-1 1.4B Q8_0 sweep: aborted at pp4096 by a retry-logic bug (swap guard); completed runs are skipped
#  - profiling: aborted by SIGPIPE from `perf report | head`
source "$(dirname "$0")/env.sh"
cd "$REPO"
while pgrep -f "scripts/run_all_timed.sh" > /dev/null; do sleep 20; done
stage() { echo "[$(date -Is)] START $1"; bash scripts/run_logged.sh "${@:2}" > /dev/null 2>&1; echo "[$(date -Is)] END   $1 rc=$?"; }
stage sweep-m1-1.4b-q8 05_sweep.sh all m1-1.4b-q8
rm -f "$RESULTS"/profile/*
stage profile 07_profile.sh
"$VENV/bin/python" scripts/08_analyze.py "$REPO" > "$WORK/logs/analyze.log" 2>&1
"$VENV/bin/python" bonus/analyze_bonus.py "$REPO" > "$WORK/logs/analyze_bonus.log" 2>&1
echo "[$(date -Is)] REST DONE"
