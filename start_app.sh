#!/usr/bin/env bash
set -euo pipefail
IPO_APP_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$IPO_APP_ROOT"
if [[ ! -x "$IPO_APP_ROOT/.venv/bin/python" ]]; then
  echo "Python environment not found or not executable in this WSL distribution."
  echo "See submission/START_HERE.md to rebuild .venv with Python 3.14."
  exit 1
fi
exec "$IPO_APP_ROOT/.venv/bin/python" -m uvicorn app.server:app --host 127.0.0.1 --port 8501
