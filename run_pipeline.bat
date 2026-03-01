@echo off
setlocal enabledelayedexpansion
title DTM Drainage AI - Pipeline Runner

:: ==============================================================
::  EDIT THIS LINE - set path to your .las or .laz file
:: ==============================================================
set INPUT_FILE=data\input\DEVDI_511671.las

:: Output folder (auto-created if missing)
set OUTPUT_DIR=data\output

:: Log folder
set LOG_DIR=logs

:: Extra flags: add --no-ml to skip ML (faster), or --help to see all options
set EXTRA_FLAGS=

:: ==============================================================

echo.
echo ==============================================================
echo   DTM Drainage AI - MoPR Hackathon Pipeline Runner
echo ==============================================================
echo.

:: Check venv exists
if not exist "dtm-env\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo         Please run install.bat first.
    pause
    exit /b 1
)

:: Activate
call dtm-env\Scripts\activate.bat
echo [OK] Environment activated

:: Check input file exists
if not exist "%INPUT_FILE%" (
    echo.
    echo [ERROR] Input file not found: %INPUT_FILE%
    echo.
    echo   Please either:
    echo     1. Copy your .las or .laz file into data\input\
    echo     2. Edit the INPUT_FILE line at the top of this file
    echo.
    pause
    exit /b 1
)

echo [OK] Input : %INPUT_FILE%
echo [OK] Output: %OUTPUT_DIR%
echo [OK] Logs  : %LOG_DIR%
echo.

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "%LOG_DIR%"    mkdir "%LOG_DIR%"

echo Starting pipeline at %time%...
echo Logs are being written to %LOG_DIR%\
echo.

python run_pipeline.py --input "%INPUT_FILE%" --output "%OUTPUT_DIR%" --log-dir "%LOG_DIR%" %EXTRA_FLAGS%

echo.
if errorlevel 1 (
    echo [FAILED] Pipeline finished with errors.
    echo          Check %LOG_DIR%\ for details.
) else (
    echo [SUCCESS] Pipeline complete!
    echo           Results are in: %OUTPUT_DIR%\
)
echo.
echo Finished at %time%
pause
