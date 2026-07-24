@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ========================================
echo  Setup - Khat Classification Web
echo ========================================
echo.

set "PYTHON="
where python >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=python"
) else (
    where py >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON=py -3"
    )
)

if not defined PYTHON (
    echo [ERROR] Python tidak ditemukan.
    echo Install Python 3.10 atau 3.11 dari https://www.python.org/downloads/
    echo Centang opsi "Add Python to PATH" saat instalasi.
    pause
    exit /b 1
)

echo Menggunakan: %PYTHON%
echo.

if not exist "venv\Scripts\python.exe" (
    echo Membuat virtual environment...
    %PYTHON% -m venv venv
    if errorlevel 1 (
        echo [ERROR] Gagal membuat virtual environment.
        pause
        exit /b 1
    )
)

set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo [ERROR] venv\Scripts\python.exe tidak ditemukan setelah pembuatan venv.
    pause
    exit /b 1
)

echo Mengupgrade pip...
"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Gagal mengupgrade pip.
    pause
    exit /b 1
)

echo Menginstall dependencies dari requirements.txt...
"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Gagal install dependencies.
    echo Coba jalankan manual:
    echo   venv\Scripts\activate
    echo   pip install tensorflow==2.15.0
    pause
    exit /b 1
)

echo.
echo Setup selesai. Jalankan aplikasi dengan double-click run.bat
echo.
if /i not "%~1"=="from-run" pause
