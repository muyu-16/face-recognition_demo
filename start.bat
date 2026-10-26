@echo off
rem ===========================================================================
rem  Face Recognition Demo - ONE CLICK launch (Windows)
rem
rem  This is THE ONLY file a beginner needs. Double-click it.
rem
rem  What it does, automatically:
rem    1. finds a usable toolchain (uv, or Python 3.10~3.12)
rem    2. creates the virtual environment .venv if missing / broken
rem    3. installs requirements.txt if anything is missing (mirror first)
rem    4. pre-downloads the pretrained model weights (~150 MB, first run only)
rem    5. runs the self test (skipped automatically once the env is healthy)
rem    6. starts the web UI and opens http://localhost:8501 in your browser
rem
rem  Advanced flags (optional, for power users):
rem      start.bat --check       always run self_test.py before launching
rem      start.bat --setup       force a dependency re-check / reinstall
rem      start.bat --no-browser  do not auto-open the browser
rem      start.bat --dry-run     report what would be done, change nothing
rem
rem  NOTE: this file is intentionally ASCII-only. cmd.exe parses .bat files
rem        with the system OEM code page, so UTF-8 Chinese text here would be
rem        mis-decoded and break the script. Chinese docs: README.md and the
rem        startup guide (Chinese file name).
rem ===========================================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "DO_CHECK=0"
set "FORCE_SETUP=0"
set "NO_BROWSER=0"
set "DRY_RUN=0"

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--check"        set "DO_CHECK=1"
if /i "%~1"=="--setup"        set "FORCE_SETUP=1"
if /i "%~1"=="--no-browser"   set "NO_BROWSER=1"
if /i "%~1"=="--dry-run"      set "DRY_RUN=1"
shift
goto :parse_args
:args_done

echo ============================================================
echo   Face Recognition Demo - one click start
echo ============================================================
echo.

if not exist "app.py" goto :no_app
if "%DRY_RUN%"=="1" goto :dry_run

rem ---------------------------------------------------------------------------
rem 1. Detect the toolchain (uv preferred, then py launcher, then python)
rem ---------------------------------------------------------------------------
set "UV_CMD="
where uv >nul 2>nul
if not errorlevel 1 set "UV_CMD=uv"

set "PY_CMD="
if not defined PY_CMD py -3.12 -c "import sys" >nul 2>nul & if not errorlevel 1 set "PY_CMD=py -3.12"
if not defined PY_CMD py -3.11 -c "import sys" >nul 2>nul & if not errorlevel 1 set "PY_CMD=py -3.11"
if not defined PY_CMD py -3.10 -c "import sys" >nul 2>nul & if not errorlevel 1 set "PY_CMD=py -3.10"
if not defined PY_CMD py -c "import sys" >nul 2>nul & if not errorlevel 1 set "PY_CMD=py"
if not defined PY_CMD python -c "import sys" >nul 2>nul & if not errorlevel 1 set "PY_CMD=python"

if defined UV_CMD goto :have_tool
if defined PY_CMD goto :have_tool
echo [ERROR] Neither "uv" nor a usable Python was found.
echo.
echo   Install uv (recommended, one line, no admin rights needed):
echo       powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
echo.
echo   Or install Python 3.10 - 3.12 from python.org and tick
echo   "Add Python to PATH" during setup.
echo.
pause
exit /b 1

:have_tool
echo [1/6] Toolchain detected:
if defined UV_CMD uv --version
if defined PY_CMD %PY_CMD% -c "import sys;print('      Python:', sys.version.split()[0])"

rem ---------------------------------------------------------------------------
rem 2. Create / verify the virtual environment
rem    An existing .venv is health-checked: a moved/renamed venv keeps absolute
rem    paths in its launchers and silently misbehaves, so it is rebuilt.
rem ---------------------------------------------------------------------------
if "%FORCE_SETUP%"=="1" goto :deps_check
if exist "%VENV_PY%" goto :venv_check
goto :create_venv

:venv_check
"%VENV_PY%" -c "import sys" >nul 2>nul
if errorlevel 1 goto :create_venv
if not defined UV_CMD goto :venv_check_pip
goto :venv_ok

:venv_check_pip
"%VENV_PY%" -m pip --version >nul 2>nul
if not errorlevel 1 goto :venv_ok
echo [WARN] Existing .venv has no pip, rebuilding ...
goto :create_venv

:venv_ok
echo [2/6] Virtual environment .venv is usable.
goto :deps_check

:create_venv
echo [2/6] Creating virtual environment .venv ...
if exist "%VENV_PY%" (
    echo        Removing a broken .venv first ...
    rmdir /s /q .venv 2>nul
)
if exist "%VENV_PY%" goto :venv_failed
if defined UV_CMD (
    uv venv .venv --python 3.11 >nul 2>nul
    if not exist "%VENV_PY%" uv venv .venv >nul 2>nul
) else (
    %PY_CMD% -m venv .venv
)
if not exist "%VENV_PY%" goto :venv_failed
if not defined UV_CMD (
    "%VENV_PY%" -m pip --version >nul 2>nul
    if errorlevel 1 (
        echo [WARN] This Python cannot bootstrap pip into a venv. Please install uv:
        echo         powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
        rmdir /s /q .venv 2>nul
        goto :venv_failed
    )
)
echo        Virtual environment ready.

rem ---------------------------------------------------------------------------
rem 3. Install dependencies (single source of truth = requirements.txt)
rem    Tsinghua mirror tried first (faster in mainland China), then PyPI.
rem ---------------------------------------------------------------------------
:deps_check
echo [3/6] Checking dependencies ...
"%VENV_PY%" -c "import streamlit, torch, torchvision, facenet_pytorch, numpy, PIL, pandas, sklearn, cv2" >nul 2>nul
if not errorlevel 1 goto :deps_ok
echo        Some packages are missing, installing requirements.txt ...
if defined UV_CMD (
    uv pip install --python "%VENV_PY%" -r requirements.txt --default-index https://pypi.tuna.tsinghua.edu.cn/simple >nul
    if errorlevel 1 uv pip install --python "%VENV_PY%" -r requirements.txt
) else (
    "%VENV_PY%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 60
    if errorlevel 1 "%VENV_PY%" -m pip install -r requirements.txt --timeout 60
)
"%VENV_PY%" -c "import streamlit, torch, torchvision, facenet_pytorch, numpy, PIL, pandas, sklearn, cv2" >nul 2>nul
if errorlevel 1 goto :deps_failed
echo        Dependencies ready.

:deps_ok
echo.
echo [4/6] Preparing pretrained model weights (first run downloads ~150 MB) ...
"%VENV_PY%" prefetch_models.py
if errorlevel 1 echo [WARN] Weight download had failures - the app will retry when needed.

rem ---------------------------------------------------------------------------
rem 5. Self test (skipped by default on a healthy env; --check forces it)
rem ---------------------------------------------------------------------------
if "%DO_CHECK%"=="1" goto :run_check
goto :launch

:run_check
echo.
echo [5/6] Running the project self test ...
"%VENV_PY%" self_test.py
if errorlevel 1 echo [WARN] Self test reported failures - see above, but still launching.

:launch
echo.
echo [6/6] Starting the web UI ...
echo ------------------------------------------------------------
echo   URL : http://localhost:8501
echo   Do NOT use "python app.py" - this app runs under Streamlit.
echo   Close this window (or press Ctrl+C) to stop the app.
echo ------------------------------------------------------------
echo.
if "%NO_BROWSER%"=="1" goto :launch_headless
start "" http://localhost:8501

:launch_headless
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
pause
exit /b 1

:venv_failed
echo.
echo [ERROR] Could not create a usable virtual environment .venv
echo         Please install uv once, then run this script again:
echo             powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
pause
exit /b 1

:deps_failed
echo.
echo [ERROR] Dependency installation failed. Check the network and run again.
echo         Alternative mirrors:
echo             https://mirrors.aliyun.com/pypi/simple
echo             https://mirrors.ustc.edu.cn/pypi/simple
pause
exit /b 1

:dry_run
echo [dry-run] project folder    : %~dp0
echo [dry-run] venv interpreter  : "%VENV_PY%"
if exist "%VENV_PY%" echo [dry-run] venv status       : exists, would be checked with "import streamlit, torch, ..."
if not exist "%VENV_PY%" echo [dry-run] venv status       : missing, would be created
echo [dry-run] self test         : %DO_CHECK% (1 = would run self_test.py)
echo [dry-run] launch command    : "%VENV_PY%" -m streamlit run app.py
echo [dry-run] URL               : http://localhost:8501
echo [dry-run] Nothing was changed.
exit /b 0
