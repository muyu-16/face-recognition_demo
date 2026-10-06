@echo off
rem ===========================================================================
rem  Face Recognition Demo - ONE CLICK start (Windows)
rem
rem  Just double-click this file. It will:
rem    1. look for a usable Python / uv toolchain
rem    2. create the virtual environment .venv when it is missing
rem    3. install everything listed in requirements.txt (mirror first)
rem    4. pre-download the pretrained model weights (about 150 MB, first run)
rem    5. start the web UI on http://localhost:8501
rem
rem  NOTE: this file is intentionally ASCII-only. cmd.exe parses .bat files with
rem        the system OEM code page, so UTF-8 Chinese text here would break it.
rem        Chinese documentation: README.md and the file "????.md".
rem
rem  Usage:
rem      start.bat             one click: setup (only if needed) + launch
rem      start.bat --check     also run self_test.py before launching
rem      start.bat --setup     force a dependency check / reinstall
rem      start.bat --dry-run   report what would be done, change nothing
rem ===========================================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "DO_CHECK=0"
set "FORCE_SETUP=0"
set "DRY_RUN=0"

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--check" set "DO_CHECK=1"
if /i "%~1"=="--setup" set "FORCE_SETUP=1"
if /i "%~1"=="--dry-run" set "DRY_RUN=1"
shift
goto :parse_args
:args_done

echo ============================================================
echo   Face Recognition Demo - one click start
echo ============================================================
echo.

if not exist "app.py" goto :no_app
if "%DRY_RUN%"=="1" goto :dry_run
if "%FORCE_SETUP%"=="1" goto :need_setup

rem ---------------------------------------------------------------------------
rem 1. Is the environment already usable?  (venv exists AND packages import)
rem ---------------------------------------------------------------------------
if not exist "%VENV_PY%" goto :need_setup

echo [1/3] Checking the existing virtual environment ...
"%VENV_PY%" -c "import streamlit, torch, torchvision, facenet_pytorch, numpy, PIL, pandas" >nul 2>nul
if errorlevel 1 goto :deps_missing
echo       OK - .venv is ready.
goto :setup_done

:deps_missing
echo       Some packages are missing in .venv, they will be installed now.
goto :need_setup

rem ---------------------------------------------------------------------------
rem 2. First run (or repair): let run.bat build the environment
rem ---------------------------------------------------------------------------
:need_setup
echo [1/3] Preparing the environment (first run or repair) ...
echo.
echo       Steps: create .venv -^> install requirements.txt -^> download weights.
echo       This needs a network connection and may take several minutes.
echo       A Tsinghua PyPI mirror is tried first (much faster in mainland China).
echo.
call "%~dp0run.bat" --setup-only
if errorlevel 1 goto :setup_failed
echo.

:setup_done
echo [1/3] Environment is ready.
echo.

if "%DO_CHECK%"=="1" goto :run_check
goto :launch

:run_check
echo [2/3] Running the project self test ...
echo.
"%VENV_PY%" self_test.py
echo.
goto :launch

rem ---------------------------------------------------------------------------
rem 3. Launch the web UI
rem ---------------------------------------------------------------------------
:launch
echo [3/3] Starting the web UI ...
echo ------------------------------------------------------------
echo   URL : http://localhost:8501
echo         (the browser should open automatically)
echo   Stop: press Ctrl+C here, or simply close this window
echo ------------------------------------------------------------
echo.
echo   Reminder: this app must run through Streamlit.
echo             Never start it with "python app.py".
echo.
"%VENV_PY%" -m streamlit run app.py
echo.
echo The app has stopped.
pause
exit /b 0

rem ---------------------------------------------------------------------------
rem Error handling
rem ---------------------------------------------------------------------------
:no_app
echo [ERROR] app.py was not found next to this script.
echo         Put start.bat inside the project folder and run it there.
echo.
pause
exit /b 1

:setup_failed
echo.
echo [ERROR] Automatic setup failed. Please check the messages above, then:
echo         1. read the troubleshooting section in README.md
echo         2. or run the steps manually:
echo              uv venv .venv --python 3.11
echo              uv pip install --python .venv\Scripts\python.exe -r requirements.txt --default-index https://pypi.tuna.tsinghua.edu.cn/simple
echo              .venv\Scripts\python.exe -m streamlit run app.py
echo.
pause
exit /b 1

rem ---------------------------------------------------------------------------
rem Dry run: detection only, nothing is created, installed or launched
rem ---------------------------------------------------------------------------
:dry_run
echo [dry-run] project folder    : %~dp0
echo [dry-run] venv interpreter  : "%VENV_PY%"
if exist "%VENV_PY%" echo [dry-run] venv status       : exists, would be checked with "import streamlit, torch, ..."
if not exist "%VENV_PY%" echo [dry-run] venv status       : missing, would run "run.bat --setup-only"
echo [dry-run] self test         : %DO_CHECK% (1 = would run self_test.py)
echo [dry-run] launch command    : "%VENV_PY%" -m streamlit run app.py
echo [dry-run] URL               : http://localhost:8501
echo [dry-run] Nothing was changed.
exit /b 0
