#!/bin/bash
# VisRetain demo.
#
#   ./demo.sh           start the LIVE app: drop in any image, watch the model run
#   ./demo.sh static    open the offline pages (no model needed): stages + full pipeline
#
# Fully local: the model runs on this machine. Ctrl+C in this window stops the live app.

set -e
cd "$(dirname "$0")"                       # the repo root, so `python -m visretain...` resolves
JOBS="visretain\.(app|run_timeline|build_stages)|tests\.test_(pipeline|stage1|stage2)"

if [ "$1" = "static" ]; then
  python3 -m visretain.build_demo
  [ -f demo/stages.html ] && open demo/stages.html        # built by: python -m visretain.build_stages
  open demo/visretain_demo.html
  exit 0
fi

if pgrep -f "$JOBS" >/dev/null; then
  if curl -s localhost:8765/api/status >/dev/null 2>&1; then
    echo "The live app is already running. Opening it."
    open http://localhost:8765; exit 0
  fi
  echo "Another VisRetain job is using the GPU. Wait for it to finish:"
  pgrep -fl "$JOBS" | cut -c1-120
  exit 1
fi

echo "Starting the live app (loads the model once, ~20-60 seconds)..."
( until curl -s localhost:8765/api/status >/dev/null 2>&1; do sleep 2; done
  open http://localhost:8765 ) &
TRANSFORMERS_VERBOSITY=error python3 -u -m visretain.app 2>&1 \
  | grep --line-buffered -vE "^\[ERROR\]|Loading weights|Fetching|Warning:|mpsgraph|out of space|Error creating"
