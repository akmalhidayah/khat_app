@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Arabic Khat AI - Installer

set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

echo.
echo ============================================================
echo   Arabic Khat AI - Windows Installer
echo ============================================================
echo   Project: %CD%
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    where py >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python belum terpasang.
        echo Silakan install Python 3.10/3.11 dan centang "Add Python to PATH".
        echo Download: https://www.python.org/downloads/
        goto :END
    )
    set "PY=py -3"
) else (
    set "PY=python"
)

echo [1/8] Checking Python version...
for /f "delims=" %%V in ('%PY% -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set "PYVER=%%V"
echo        Python %PYVER%
%PY% -c "import sys; raise SystemExit(0 if sys.version_info >= (3,9) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.10 atau 3.11 direkomendasikan.
    goto :END
)

echo [2/8] Creating virtual environment .venv ...
if exist ".venv\Scripts\python.exe" (
    echo        .venv already exists, skipping creation.
) else (
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Gagal membuat virtual environment.
        goto :END
    )
)

echo [3/8] Activating virtual environment...
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Gagal mengaktifkan .venv
    goto :END
)

echo [4/8] Upgrading pip...
python -m pip install --upgrade pip >nul

echo [5/8] Installing dependencies...
if exist "requirements.txt" (
    pip install -r requirements.txt
) else if exist "installer\requirements_windows.txt" (
    pip install -r installer\requirements_windows.txt
) else (
    echo [ERROR] requirements.txt not found.
    goto :END
)
if errorlevel 1 (
    echo [ERROR] Gagal install dependencies. Coba jalankan repair_windows.bat.
    echo.
    echo Catatan: TensorFlow di Windows kadang gagal jika Python 3.12+ atau MSVC runtime belum ada.
    echo Gunakan Python 3.10/3.11 dan install "Microsoft Visual C++ Redistributable".
    goto :END
)

echo [6/8] Running setup_app.py...
python installer\setup_app.py
if errorlevel 1 (
    echo [WARNING] Setup selesai dengan peringatan. Periksa MySQL/XAMPP dan file .env
)

echo [7/8] Running system check...
python installer\check_system.py
set "CHECK_CODE=%ERRORLEVEL%"

echo [8/8] Installation summary
echo.
if not "%CHECK_CODE%"=="0" (
    echo ============================================================
    echo   Instalasi selesai DENGAN PERINGATAN / ERROR.
    echo   Baca output check_system di atas.
    echo   Perbaiki MySQL XAMPP, file model, atau dependencies.
    echo   Lalu jalankan: installer\repair_windows.bat
    echo ============================================================
    goto :END
)
echo ============================================================
echo   Instalasi selesai.
echo   Jalankan: installer\run_windows.bat
echo   Browser : http://127.0.0.1:5002
echo   Login   : admin / admin123
echo   PENTING : Segera ganti password admin setelah login.
echo ============================================================
echo.
echo Opsional: buat shortcut desktop dengan:
echo   powershell -ExecutionPolicy Bypass -File installer\create_shortcut.ps1
echo.

:END
pause
endlocal
