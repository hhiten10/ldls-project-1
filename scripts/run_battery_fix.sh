#!/usr/bin/env bash
# Re-run the two configurations that overlapped short off-AC windows in results/hw/cpu_freq_log_pass3.csv
# (13:51-13:55 and 14:16:57 IST). Originals kept as *.battery.jsonl (excluded from analysis); rows renamed.
source "$(dirname "$0")/env.sh"
cd "$REPO"; R="$RESULTS/raw"
for x in llama-1b-q4__pp16384 m2-1.3b-q4__tg128_t1_d8192; do
  [ -e "$R/$x.jsonl" ] && mv "$R/$x.jsonl" "$R/$x.battery.jsonl"
  tag=${x%%__*}; id=${x#*__}
  sed -i "s/^$tag,$id,/$tag,${id}_battery,/" "$R/runs_meta.csv"
done
bash scripts/run_logged.sh 05_sweep.sh prefill llama-1b-q4 > /dev/null 2>&1; echo "llama pp16384 rerun rc=$?"
bash scripts/run_logged.sh 05_sweep.sh cross m2-1.3b-q4 > /dev/null 2>&1; echo "m2 t1 d8192 rerun rc=$?"
"$VENV/bin/python" scripts/08_analyze.py "$REPO" > "$WORK/logs/analyze.log" 2>&1
for x in llama-1b-q4__pp16384 m2-1.3b-q4__tg128_t1_d8192; do
  for s in battery.jsonl jsonl; do printf "%-40s " "$x.$s"; grep -o '"avg_ts": [0-9.]*, "stddev_ts": [0-9.]*' "$R/$x.$s"; done
done
echo BATTERY_FIX_DONE
