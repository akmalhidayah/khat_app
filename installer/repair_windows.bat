@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Arabic Khat AI - Repair

set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

echo.
echo ============================================================
echo   Arabic Khat AI - Repair Tool
echo ============================================================
echo   Tidak akan menghapus dataset atau file model.
echo ============================================================
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] .venv tidak ditemukan. Menjalankan install_windows.bat ...
    call "%~dp0install_windows.bat"
    exit /b %ERRORLEVEL%
)

call ".venv\Scripts\activate.bat"

echo [1/6] Reinstalling Python dependencies...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if errorlevel 1 (
    echo [WARNING] Beberapa paket gagal diinstall. Lanjut perbaikan folder...
)

echo [2/6] Recreating missing folders and .env ...
python installer\setup_app.py

echo [3/6] Cleaning temporary files (temp/, dataset/tmp/) ...
python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('installer'))); from installer_lib import clean_temp_files, log_message, project_path; cleaned=clean_temp_files(); log_message('Repair cleaned: '+', '.join(cleaned) if cleaned else 'none', project_path('logs','repair.log')); print('[OK] Cleaned', len(cleaned), 'temp item(s).')"

echo [4/6] Validating model files ...
python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('installer'))); from installer_lib import validate_model_files; [print(f'  [{s}] {p} - {d}') for p,s,d in validate_model_files()]"

echo [5/6] Running full system check ...
python installer\check_system.py
set "CHECK_CODE=%ERRORLEVEL%"

echo [6/6] Repair log saved to logs\repair.log
echo.
if "%CHECK_CODE%"=="0" (
    echo [OK] Perbaikan selesai. Jalankan installer\run_windows.bat
) else (
    echo [WARNING] Masih ada masalah. Baca installer\README_INSTALL_WINDOWS.md
)

pause
endlocal
exit /b %CHECK_CODE%
