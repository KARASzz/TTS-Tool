#!/bin/zsh
set -e
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "未找到 Python 3，请先安装 Python 3.11 或更高版本。"
  exit 1
fi
if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
fi
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py
