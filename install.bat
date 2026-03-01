@echo off
setlocal enabledelayedexpansion
title DTM Drainage AI - Installer

echo.
echo ==============================================================
echo   DTM Drainage AI - Windows Setup Script
echo   MoPR Geospatial Intelligence Hackathon
echo ==============================================================
echo.

:: ── Step 1: Check Python ──────────────────────────────────────────────
echo [STEP 1/7] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo         Install Python 3.10 from https://python.org
    echo         Check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER% found
echo.

:: ── Step 2: Create virtual environment ────────────────────────────────
echo [STEP 2/7] Creating virtual environment (dtm-env)...
if exist "dtm-env\" (
    echo [SKIP] dtm-env already exists.
) else (
    python -m venv dtm-env
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)
call dtm-env\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Could not activate virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

:: ── Step 3: Upgrade pip ───────────────────────────────────────────────
echo [STEP 3/7] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel --quiet
echo [OK] pip upgraded
echo.

:: ── Step 4: PyTorch (CPU) ─────────────────────────────────────────────
echo [STEP 4/7] Installing PyTorch (CPU build)...
echo           This downloads ~200MB, may take a few minutes...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
if errorlevel 1 (
    echo [WARN] PyTorch failed. PointNet classifier will be unavailable.
) else (
    echo [OK] PyTorch installed
)
echo.

:: ── Step 5: All other packages via requirements-win.txt ───────────────
echo [STEP 5/7] Installing geospatial and ML packages...
echo           This may take 5-10 minutes...
pip install -r requirements-win.txt --quiet
if errorlevel 1 (
    echo [WARN] Batch install had issues. Retrying individually...
    pip install laspy --quiet
    pip install pyproj --quiet
    pip install shapely --quiet
    pip install rasterio --quiet
    pip install fiona --quiet
    pip install geopandas --quiet
    pip install "rio-cogeo" --quiet
    pip install pysheds --quiet
    pip install numpy --quiet
    pip install scipy --quiet
    pip install scikit-learn --quiet
    pip install xgboost --quiet
    pip install pandas --quiet
    pip install networkx --quiet
    pip install matplotlib --quiet
    pip install pyyaml --quiet
    pip install tqdm --quiet
    pip install loguru --quiet
    pip install click --quiet
    pip install rich --quiet
    pip install joblib --quiet
)
echo [OK] Packages installed
echo.

:: ── Step 6: PDAL (optional) ───────────────────────────────────────────
echo [STEP 6/7] Trying PDAL (optional, for SMRF ground classification)...
pip install pdal --quiet >nul 2>&1
if errorlevel 1 (
    echo [SKIP] PDAL not installed - pipeline uses fallback classifier.
    echo        To install manually: conda install -c conda-forge pdal python-pdal
) else (
    echo [OK] PDAL installed
)
echo.

:: ── Step 7: Create project folders ────────────────────────────────────
echo [STEP 7/7] Creating project folders...
if not exist "data\input"  mkdir data\input
if not exist "data\output" mkdir data\output
if not exist "data\cache"  mkdir data\cache
if not exist "logs"        mkdir logs
if not exist "models"      mkdir models
echo [OK] Folders ready
echo.

:: ── Verify ────────────────────────────────────────────────────────────
echo Verifying key packages...
echo.
python -c "import laspy;     print('  [OK] laspy')"        2>nul || echo   [FAIL] laspy
python -c "import rasterio;  print('  [OK] rasterio')"     2>nul || echo   [FAIL] rasterio
python -c "import geopandas; print('  [OK] geopandas')"    2>nul || echo   [FAIL] geopandas
python -c "import xgboost;   print('  [OK] xgboost')"      2>nul || echo   [FAIL] xgboost
python -c "import pysheds;   print('  [OK] pysheds')"      2>nul || echo   [FAIL] pysheds
python -c "import sklearn;   print('  [OK] scikit-learn')" 2>nul || echo   [FAIL] scikit-learn
python -c "import torch;     print('  [OK] pytorch')"      2>nul || echo   [SKIP] pytorch (optional)
python -c "import pdal;      print('  [OK] pdal')"         2>nul || echo   [SKIP] pdal (optional)
echo.
echo ==============================================================
echo   SETUP COMPLETE
echo ==============================================================
echo.
echo Next steps:
echo   1. Copy your .las or .laz file into:  data\input\
echo   2. Edit INPUT_FILE at the top of:     run_pipeline.bat
echo   3. Double-click:                       run_pipeline.bat
echo.
pause
