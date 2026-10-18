@echo off
rem ADR Studio headless server launcher (installer.iss ships this to {app})
chcp 65001 >nul
setlocal EnableExtensions
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
rem ---- data dir (mirrors desktop shell & container: adr.core.config.adr_data_dir)
rem ---- ADR_DATA_DIR env > F:\ADR_data (exists -> keep legacy) > %LOCALAPPDATA%\ADR\data
if defined ADR_DATA_DIR goto :have_data_dir
if exist "F:\ADR_data" (
    set "ADR_DATA_DIR=F:\ADR_data"
    goto :have_data_dir
)
set "ADR_DATA_DIR=%LOCALAPPDATA%\ADR\data"
:have_data_dir
if not exist "%ADR_DATA_DIR%" mkdir "%ADR_DATA_DIR%" 2>nul
rem ---- mirror desktop shell sidecar launch: cwd=runtime + PYTHONPATH=.
if exist "%~dp0runtime\python\python.exe" goto :have_python
echo [ADR] runtime\python\python.exe not found. Run this script from the install root.
exit /b 1
:have_python
cd /d "%~dp0runtime"
set "PYTHONPATH=."
"%~dp0runtime\python\python.exe" -m adr.server %*
endlocal
