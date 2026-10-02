#!/bin/sh
# Strata for Linux: pick one of the installed models (run-<model>-vision|novision.sh) and start it.
cd "$(dirname "$0")" || exit 1
PY=.venv-linux/bin/python
[ -x "$PY" ] || PY=python3
exec "$PY" MoreSimpleStart.py "$@"
