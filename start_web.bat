@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PORT=8000"
set "URL=http://localhost:%PORT%"
set "HEALTH_URL=http://127.0.0.1:%PORT%/api/status"

echo.
echo ============================================================
echo   统一手势图卷积网络 - 一键启动 Web Studio
echo   页面地址: %URL%
echo   按 Ctrl+C 可随时停止服务
echo ============================================================
echo.

rem 如果服务已经在运行，直接打开浏览器即可，避免端口冲突
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri '%HEALTH_URL%' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 goto :ALREADY_RUNNING

rem 后台轮询等待服务就绪后自动打开浏览器（最长等待 90 秒）
echo [1/2] 正在启动后端服务，就绪后会自动打开浏览器...
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for ($i=0; $i -lt 180; $i++) { try { $r = Invoke-WebRequest -UseBasicParsing -Uri '%HEALTH_URL%' -TimeoutSec 1; if ($r.StatusCode -eq 200) { Start-Process '%URL%'; break } } catch {} Start-Sleep -Milliseconds 500 }"

rem 自动选择可用运行环境：优先 WSL 虚拟环境 -> Windows 虚拟环境 -> 系统 Python
if exist ".venv\bin\python" goto :RUN_WSL
if exist ".venv\Scripts\python.exe" goto :RUN_WIN_VENV
goto :RUN_SYS_PY

:ALREADY_RUNNING
echo [提示] 检测到服务已在 %URL% 运行，正在直接打开页面...
start "" "%URL%"
exit /b 0

:RUN_WSL
echo [2/2] 检测到 WSL 虚拟环境 [.venv/bin/python]，正在启动...
echo.
wsl .venv/bin/python scripts/run_web.py --host 0.0.0.0 --port %PORT% %*
goto :END

:RUN_WIN_VENV
echo [2/2] 检测到 Windows 虚拟环境 [.venv\Scripts\python.exe]，正在启动...
echo.
set "PYTHONPATH=src"
".venv\Scripts\python.exe" scripts\run_web.py --host 127.0.0.1 --port %PORT% %*
goto :END

:RUN_SYS_PY
echo [2/2] 使用系统 Python 启动...
echo.
set "PYTHONPATH=src"
python scripts\run_web.py --host 127.0.0.1 --port %PORT% %*
goto :END

:END
echo.
echo [服务已停止] 如异常退出，请查看上方报错信息。
pause
