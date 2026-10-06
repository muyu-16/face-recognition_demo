@echo off
rem ===========================================================================
rem  Push this project to Gitee:  https://gitee.com/muyu1213/face_demo.git
rem
rem  Double-click it and follow the prompts:
rem    Username: your Gitee account name
rem    Password: your Gitee password, or better a personal access token
rem              (Gitee: ?? -> ????, give it "projects" scope)
rem
rem  NOTE: this file is intentionally ASCII-only (cmd.exe + OEM code page).
rem ===========================================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "REMOTE=https://gitee.com/muyu1213/face_demo.git"

echo ============================================================
echo   Push this project to Gitee
echo   %REMOTE%
echo ============================================================
echo.

where git >nul 2>nul
if errorlevel 1 goto :no_git

if not exist ".git" goto :init_repo
goto :have_repo

:init_repo
echo [1/4] Initializing a local repository ...
git init -q
if errorlevel 1 goto :failed

:have_repo
echo [1/4] Local repository is ready.

echo [2/4] Configuring the remote "origin" ...
git remote get-url origin >nul 2>nul
if errorlevel 1 goto :add_remote
git remote set-url origin "%REMOTE%"
goto :remote_ok

:add_remote
git remote add origin "%REMOTE%"

:remote_ok
git remote -v

echo.
echo [3/4] Committing local changes ...
git add -A
git diff --cached --quiet
if not errorlevel 1 goto :nothing_to_commit
git -c core.safecrlf=false commit -m "update: %DATE% %TIME%"
if errorlevel 1 goto :failed
goto :commit_ok

:nothing_to_commit
echo       Nothing new to commit.

:commit_ok
echo.
echo [4/4] Pushing to Gitee (enter your account when asked) ...
echo.
git push -u origin master
if errorlevel 1 goto :push_failed

echo.
echo ============================================================
echo   Done. Open: https://gitee.com/muyu1213/face_demo
echo ============================================================
pause
exit /b 0

:no_git
echo [ERROR] git was not found. Install Git for Windows first:
echo         https://git-scm.com/download/win
pause
exit /b 1

:push_failed
echo.
echo [ERROR] Push failed. Common reasons:
echo   1. wrong account / password  -^> use a Gitee personal access token
echo   2. the remote already has commits -^> run:
echo          git pull --rebase origin master
echo      then run this script again
echo   3. TLS error on old Windows   -^> run:
echo          git config --global http.sslBackend openssl
echo.
pause
exit /b 1

:failed
echo.
echo [ERROR] Something went wrong, see the messages above.
pause
exit /b 1
