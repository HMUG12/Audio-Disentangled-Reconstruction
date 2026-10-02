@echo off
rem ============================================================
rem  ADR WebUI 一键启动
rem  自动探测 Python 环境, 兼容不同机器:
rem    1. 环境变量 ADR_PYTHON (用户指定解释器)
rem    2. 项目内 .venv (python -m venv .venv 创建的虚拟环境)
rem    3. D:\pyhon\python.exe (本机开发环境)
rem    4. 系统 PATH 中的 python
rem ============================================================
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"

rem ---- 1. 探测 Python ----
set "PY="
if defined ADR_PYTHON (
    if exist "%ADR_PYTHON%" set "PY=%ADR_PYTHON%"
)
if not defined PY if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY if exist "D:\pyhon\python.exe" set "PY=D:\pyhon\python.exe"
if not defined PY (
    where python >nul 2>nul
    if !errorlevel! == 0 set "PY=python"
)
if not defined PY (
    echo [错误] 找不到 Python 解释器。
    echo        请安装 Python 3.10+, 或设置环境变量 ADR_PYTHON 指向 python.exe
    pause
    exit /b 1
)
echo [环境] Python: %PY%

rem ---- 2. 环境变量 (跨机器兼容) ----
set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH=%CD%"
rem HuggingFace 国内镜像 (BigVGAN 等预训练模型下载用)
if not defined HF_ENDPOINT set "HF_ENDPOINT=https://hf-mirror.com"

rem ---- 3. 依赖检查 ----
"%PY%" -c "import torch" >nul 2>nul
if errorlevel 1 (
    echo [错误] 未安装 PyTorch。请先运行: %PY% -m pip install torch
    pause
    exit /b 1
)
"%PY%" -c "import gradio" >nul 2>nul
if errorlevel 1 (
    echo [错误] 未安装 Gradio。请先运行: %PY% -m pip install "gradio>=6"
    pause
    exit /b 1
)
"%PY%" -c "import bitsandbytes" >nul 2>nul
if errorlevel 1 (
    echo [提示] 未安装 bitsandbytes, QLoRA 功能不可用 (LoRA 不受影响)
)

rem ---- 4. 启动 ----
echo [启动] ADR WebUI  http://127.0.0.1:7860
"%PY%" -m adr.cli webui %*
if errorlevel 1 (
    echo.
    echo [错误] WebUI 异常退出, 错误码 !errorlevel!
    pause
)
endlocal
