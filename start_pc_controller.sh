#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "[错误] 未找到 Python，请先安装 Python 3.10–3.12。"
  exit 1
fi

echo "正在启动手势操控电脑客户端（订阅 ws://127.0.0.1:8000/ws/events）..."
exec "${PYTHON_BIN}" scripts/pc_controller.py "$@"
