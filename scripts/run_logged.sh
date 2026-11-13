#!/usr/bin/env bash
# Usage (from repo root): bash scripts/run_logged.sh <script.sh> [args...]
# Runs a script, logs to $WORK/logs/<name>.log, prints the tail.
source "$(dirname "$0")/env.sh"
mkdir -p "$WORK/logs"
name="$(basename "$1" .sh)"; log="$WORK/logs/$name.log"
bash "$REPO/scripts/$1" "${@:2}" > "$log" 2>&1; rc=$?
echo "EXIT=$rc" >> "$log"
tail -n 30 "$log"
exit $rc
