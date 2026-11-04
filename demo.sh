#!/bin/bash
# VisRetain demo.
#
#   ./demo.sh           start the LIVE app: drop in any image, watch the model run
#   ./demo.sh static    open the offline pages (no model needed): stages + full pipeline
#
# Fully local: the model runs on this Mac. Ctrl+C in this window stops the live app.

set -e
cd "$(dirname "$0")"

if [ "$1" = "static" ]; then
  python3 build_demo.py --runs results_timeline --out demo/visretain_demo.html
  [ -f demo/stages.html ] && open demo/stages.html        # stage 1, precomputed (build_stages.py)
  open demo/visretain_demo.html
  exit 0
fi

if pgrep -f "(run_timeline|test_pipeline)\.py|[ /]app\.py" >/dev/null; then
  if curl -s localhost:8765/api/status >/dev/null 2>&1; then
    echo "The live app is already running. Opening it."
    open http://localhost:8765; exit 0
  fi
  echo "Another VisRetain job is using the GPU. Wait for it to finish:"
  pgrep -fl "(run_timeline|test_pipeline)\.py|[ /]app\.py" | cut -c1-120
  exit 1
fi

echo "Starting the live app (loads the model once, ~20-60 seconds)..."
( until curl -s localhost:8765/api/status >/dev/null 2>&1; do sleep 2; done
  open http://localhost:8765 ) &
TRANSFORMERS_VERBOSITY=error python3 -u app.py 2>&1 \
  | grep --line-buffered -vE "^\[ERROR\]|Loading weights|Fetching|Warning:|mpsgraph|out of space|Error creating"
