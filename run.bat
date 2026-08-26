@echo off
rem PixelAnimIDE 启动器：固定使用 .venv 里的 Python（避免系统 python 缺依赖）
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py %*
) else (
    echo 未找到 .venv，请先创建虚拟环境并安装依赖：
    echo   python -m venv .venv
    echo   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
)
