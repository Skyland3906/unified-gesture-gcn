@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在启动手势操控电脑客户端（订阅 ws://127.0.0.1:8000/ws/events）...
python scripts\pc_controller.py %*
pause
