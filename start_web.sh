#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PORT="${PORT:-8000}"
URL="http://localhost:${PORT}"
HEALTH_URL="http://127.0.0.1:${PORT}/api/status"

open_browser() {
  if command -v open >/dev/null 2>&1; then
    open "$1" >/dev/null 2>&1 || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$1" >/dev/null 2>&1 || true
  fi
}

echo ""
echo "============================================================"
echo "  统一手势图卷积网络 - 一键启动 Web Studio (macOS / Linux)"
echo "  页面地址: ${URL}"
echo "  按 Ctrl+C 可随时停止服务"
echo "============================================================"
echo ""

# 如果服务已经在运行，直接打开浏览器即可，避免端口冲突
if command -v curl >/dev/null 2>&1 && curl -fsS --max-time 2 "${HEALTH_URL}" >/dev/null 2>&1; then
  echo "[提示] 检测到服务已在 ${URL} 运行，正在直接打开页面..."
  open_browser "${URL}"
  exit 0
fi

# 自动选择可用 Python：优先项目虚拟环境 -> python3 -> python
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

# 首次运行如缺少本地 MediaPipe 手部/防人脸误检静态资源，自动下载
if [ ! -f "src/gesture_gcn/web/static/mediapipe/hands/hands.js" ] || [ ! -f "src/gesture_gcn/web/static/mediapipe/tasks/blaze_face_short_range.tflite" ]; then
  echo "[0/2] 检测到本地 MediaPipe 静态资源不完整，正在自动补全下载..."
  "${PYTHON_BIN}" scripts/download_assets.py
fi

echo "[1/2] 正在启动后端服务，就绪后会自动打开浏览器..."
(
  for _ in $(seq 1 180); do
    if command -v curl >/dev/null 2>&1 && curl -fsS --max-time 1 "${HEALTH_URL}" >/dev/null 2>&1; then
      open_browser "${URL}"
      break
    fi
    sleep 0.5
  done
) &

echo "[2/2] 使用 ${PYTHON_BIN} 启动服务..."
echo ""
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" scripts/run_web.py --host 127.0.0.1 --port "${PORT}" "$@"
