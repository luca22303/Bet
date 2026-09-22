@echo off
rem Start RueBet on Windows: find Python, set up a virtual environment, launch.
rem
rem Double-click this file, or run it from a terminal with extra arguments:
rem   start.bat --port 9000
rem
rem It does not assume `pip` is on PATH. A Windows Python install frequently
rem leaves pip unexported even when python itself works, so everything here
rem goes through `python -m pip`, which works wherever Python does.

setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PYTHON="

rem The py launcher ships with the official installer and is the most reliable
rem way to find a suitable interpreter, so it is tried first.
where py >nul 2>&1
if %errorlevel%==0 (
    py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
    if !errorlevel!==0 set "PYTHON=py -3"
)

if not defined PYTHON (
    where python >nul 2>&1
    if !errorlevel!==0 (
        python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
        if !errorlevel!==0 set "PYTHON=python"
    )
)

if not defined PYTHON (
    echo.
    echo No Python 3.10 or newer was found.
    echo.
    echo   Install it from https://www.python.org/downloads/
    echo   During setup, tick "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating a virtual environment in .venv ...
    %PYTHON% -m venv .venv
    if !errorlevel! neq 0 (
        echo.
        echo Could not create a virtual environment.
        pause
        exit /b 1
    )
)

set "VPY=.venv\Scripts\python.exe"

rem Install only when something is missing, so the usual start is immediate.
"%VPY%" -c "import bet" >nul 2>&1
if !errorlevel! neq 0 (
    echo Installing RueBet and its dependencies ^(first run takes a minute^) ...
    "%VPY%" -m pip install --quiet --upgrade pip >nul 2>&1
    "%VPY%" -m pip install --quiet -e .
    if !errorlevel! neq 0 (
        echo.
        echo Install failed. Run this to see why:
        echo   .venv\Scripts\python.exe -m pip install -e .
        pause
        exit /b 1
    )
)

echo.
"%VPY%" -m bet.cli serve %*

rem Keep the window open if it was double-clicked, so an error is readable.
if !errorlevel! neq 0 pause
