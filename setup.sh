#!/usr/bin/env bash
# One-time environment setup. Creates .venv and installs dependencies.
# Works on Mac (M1/Intel) and Linux. Uses `uv` if available (fast), else venv+pip.
set -e
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if command -v uv >/dev/null 2>&1; then
  echo "[setup] using uv"
  uv venv --python "$PYTHON_BIN" .venv || uv venv .venv
  uv pip install -r requirements.txt --python .venv/bin/python
else
  echo "[setup] uv not found; using python venv + pip"
  "$PYTHON_BIN" -m venv .venv
  ./.venv/bin/python -m pip install --upgrade pip
  ./.venv/bin/python -m pip install -r requirements.txt
fi

echo
echo "[setup] done. Activate with:  source .venv/bin/activate"
echo "[setup] verify device with:   python common/device.py"
