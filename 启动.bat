@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (set "PYTHON=py") else (set "PYTHON=python")
if not exist .venv %PYTHON% -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python run.py
pause
