@echo off
rem ===========================================================================
rem  Face Recognition Demo - ONE CLICK setup + launch (Windows)
rem
rem  Double-click this file. It will:
rem    [1/6] detect a usable Python / uv toolchain
rem    [2/6] create the virtual environment .venv (skipped if already usable)
rem    [3/6] install everything in requirements.txt (mirror tried first)
rem    [4/6] pre-download the pretrained model weights (about 150 MB, first run)
rem    [5/6] run the project self test
rem    [6/6] start the web UI on http://localhost:8501
rem
rem  NOTE 1: this file is intentionally ASCII-only. cmd.exe parses .bat files
rem          with the system OEM code page, so UTF-8 Chinese text here would be
rem          mis-decoded and break the script. Chinese docs live in README.md
rem          and in the startup guide file (Chinese file name).
rem  NOTE 2: "uv" is used when available, because on machines whose Python is
rem          managed by uv (or conda), "python -m venv" produces a venv with no
rem          pip at all, and plain "pip install" is refused by PEP 668.
rem
rem  Usage:
rem      run.bat                 one click: setup + self test + start the UI
rem      run.bat --setup-only    setup only (used by start.bat), then exit
rem      run.bat --skip-test     setup + start, without the self test
rem      run.bat --dry-run       detect interpreters only, change nothing
rem ===========================================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "SETUP_ONLY=0"
set "SKIP_TEST=0"
set "DRY_RUN=0"

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--setup-only" set "SETUP_ONLY=1"
if /i "%~1"=="--skip-test" set "SKIP_TEST=1"
if /i "%~1"=="--dry-run" set "DRY_RUN=1"
shift
goto :parse_args
:args_done

echo ============================================================
echo   Face Recognition Demo - setup and launch
echo   (the first run downloads dependencies and model weights)
echo ============================================================
echo.

rem ---------------------------------------------------------------------------
rem 1. Detect uv and a usable Python (3.10 ~ 3.12 preferred)
rem ---------------------------------------------------------------------------
set "UV_CMD="
if not defined UV_CMD uv --version >nul 2>nul
if not defined UV_CMD if not errorlevel 1 set "UV_CMD=uv"

set "PY_CMD="
if not defined PY_CMD py -3.12 -c "import sys" >nul 2>nul
if not defined PY_CMD if not errorlevel 1 set "PY_CMD=py -3.12"
if not defined PY_CMD py -3.11 -c "import sys" >nul 2>nul
if not defined PY_CMD if not errorlevel 1 set "PY_CMD=py -3.11"
if not defined PY_CMD py -3.10 -c "import sys" >nul 2>nul
if not defined PY_CMD if not errorlevel 1 set "PY_CMD=py -3.10"
if not defined PY_CMD py -c "import sys" >nul 2>nul
if not defined PY_CMD if not errorlevel 1 set "PY_CMD=py"
if not defined PY_CMD python -c "import sys" >nul 2>nul
if not defined PY_CMD if not errorlevel 1 set "PY_CMD=python"

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
if defined PY_CMD %PY_CMD% -c "import sys;print('      Python:', sys.version.split()[0]);print('      Exe   :', sys.executable)"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"

if "%DRY_RUN%"=="1" goto :dry_run

rem ---------------------------------------------------------------------------
rem 2. Create the virtual environment (skipped when it already exists)
rem    An existing .venv is first health-checked: a venv that was moved/renamed
rem    keeps absolute paths inside its launchers (pip.exe, streamlit.exe), so it
rem    silently misbehaves. If the check fails, it is rebuilt from scratch.
rem ---------------------------------------------------------------------------
if not exist "%VENV_PY%" goto :create_venv

echo [2/6] Found an existing .venv, checking it ...
"%VENV_PY%" -c "import sys" >nul 2>nul
if errorlevel 1 goto :rebuild_venv
if not defined UV_CMD goto :venv_check_pip
goto :venv_ok

:venv_check_pip
rem Without uv we rely on the venv's own pip, so a pip-less venv is useless here.
"%VENV_PY%" -m pip --version >nul 2>nul
if not errorlevel 1 goto :venv_ok
echo [WARN] The existing .venv contains no pip ...
goto :rebuild_venv

:rebuild_venv
echo [WARN] The existing .venv is not usable (moved/renamed, or has no pip).
echo        Removing it and building a fresh one ...
rmdir /s /q .venv
if exist "%VENV_PY%" goto :venv_failed
goto :create_venv

:venv_ok
echo [2/6] Virtual environment .venv is usable, skipping creation.
goto :venv_done

:create_venv
echo [2/6] Creating virtual environment .venv ...

if not defined UV_CMD goto :venv_by_python
uv venv .venv --python 3.11
if exist "%VENV_PY%" goto :venv_done
echo [WARN] "uv venv --python 3.11" failed, retrying with the default Python ...
uv venv .venv
if exist "%VENV_PY%" goto :venv_done
echo [WARN] uv could not create the environment, falling back to "python -m venv" ...

:venv_by_python
if not defined PY_CMD goto :venv_failed
%PY_CMD% -m venv .venv
if not exist "%VENV_PY%" goto :venv_failed
"%VENV_PY%" -m pip --version >nul 2>nul
if not errorlevel 1 goto :venv_done
echo.
echo [WARN] The new environment contains no pip. This happens when Python is
echo        managed by uv or conda ("python -m venv" cannot bootstrap pip).
echo        Removing the unusable .venv ...
rmdir /s /q .venv
echo.
echo        Please install uv once and run this script again:
echo            powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
goto :venv_failed

:venv_failed
echo.
echo [ERROR] Could not create a usable virtual environment .venv
pause
exit /b 1

rem ---------------------------------------------------------------------------
rem 3. Install dependencies
rem    The Tsinghua mirror is tried FIRST: connecting straight to pypi.org from
rem    mainland China often stalls with a spinning cursor and no progress.
rem ---------------------------------------------------------------------------
:venv_done
rem Fast path: when every required package already imports there is nothing to
rem do. This keeps daily startup quick and survives a temporarily broken mirror.
echo [3/6] Checking dependencies ...
"%VENV_PY%" -c "import streamlit, torch, torchvision, facenet_pytorch, numpy, PIL, pandas, sklearn, cv2" >nul 2>nul
if not errorlevel 1 goto :deps_present
echo       Some packages are missing, installing requirements.txt now ...
echo       Using the Tsinghua mirror first (faster in mainland China) ...

if not defined UV_CMD goto :deps_by_pip
uv pip install --python "%VENV_PY%" -r requirements.txt --default-index https://pypi.tuna.tsinghua.edu.cn/simple
if not errorlevel 1 goto :deps_ok
echo [WARN] Mirror failed, retrying against the default PyPI index ...
uv pip install --python "%VENV_PY%" -r requirements.txt
if not errorlevel 1 goto :deps_ok
goto :deps_failed

:deps_by_pip
"%VENV_PY%" -m pip install --upgrade pip -q -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 60
"%VENV_PY%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 60
if not errorlevel 1 goto :deps_ok
echo [WARN] Mirror failed, retrying against the default PyPI index ...
"%VENV_PY%" -m pip install -r requirements.txt --timeout 60
if not errorlevel 1 goto :deps_verify

:deps_failed
echo.
echo [ERROR] Dependency installation failed. Check the network and run again.
echo         Manual command with uv + mirror:
echo             uv pip install --python .venv\Scripts\python.exe -r requirements.txt --default-index https://pypi.tuna.tsinghua.edu.cn/simple
echo         Alternative mirrors:
echo             https://mirrors.aliyun.com/pypi/simple
echo             https://mirrors.ustc.edu.cn/pypi/simple
echo.
pause
exit /b 1

rem ---------------------------------------------------------------------------
rem 4. Pre-download the pretrained weights (non fatal: the app retries on demand)
rem ---------------------------------------------------------------------------
:deps_verify
"%VENV_PY%" -c "import streamlit, torch, torchvision, facenet_pytorch, numpy, PIL, pandas, sklearn, cv2" >nul 2>nul
if errorlevel 1 goto :deps_failed

:deps_present
if not exist "%VENV_PY%" goto :deps_failed
echo [3/6] Dependencies are already satisfied.

:deps_ok
echo.
echo [4/6] Preparing the pretrained model weights (about 150 MB on first run) ...
"%VENV_PY%" prefetch_models.py
if errorlevel 1 echo [WARN] Weight download had failures - the app will retry when needed.

if "%SETUP_ONLY%"=="1" goto :setup_only_done

rem ---------------------------------------------------------------------------
rem 5. Self test  6. Start Streamlit
rem ---------------------------------------------------------------------------
if "%SKIP_TEST%"=="1" goto :launch

echo.
echo [5/6] Running the project self test ...
"%VENV_PY%" self_test.py

:launch
echo.
echo [6/6] Starting the web UI ...
echo ------------------------------------------------------------
echo  http://localhost:8501 will open in your browser shortly.
echo  Do NOT use "python app.py" - this app must run under Streamlit.
echo  Close this window to stop the application.
echo ------------------------------------------------------------
echo.
"%VENV_PY%" -m streamlit run app.py

echo.
echo The app has stopped.
pause
exit /b 0

:setup_only_done
echo.
echo [OK] Environment is ready (.venv + dependencies + weights).
if "%SETUP_ONLY%"=="1" exit /b 0

rem ---------------------------------------------------------------------------
rem Dry run: detection only, nothing is created or installed
rem ---------------------------------------------------------------------------
:dry_run
echo.
echo [dry-run] Nothing was changed.
echo [dry-run] Virtual env path : %VENV_PY%
if defined UV_CMD echo [dry-run] Create env       : uv venv .venv --python 3.11
if not defined UV_CMD echo [dry-run] Create env       : %PY_CMD% -m venv .venv
if defined UV_CMD echo [dry-run] Install deps     : uv pip install --python "%VENV_PY%" -r requirements.txt --default-index https://pypi.tuna.tsinghua.edu.cn/simple
if not defined UV_CMD echo [dry-run] Install deps     : "%VENV_PY%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo [dry-run] Fetch weights    : "%VENV_PY%" prefetch_models.py
echo [dry-run] Self test        : "%VENV_PY%" self_test.py
echo [dry-run] Launch UI        : "%VENV_PY%" -m streamlit run app.py
exit /b 0
