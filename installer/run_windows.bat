@echo off
setlocal EnableExtensions
title Arabic Khat AI

set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment belum ada.
    echo Jalankan installer\install_windows.bat terlebih dahulu.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        if /I "%%A"=="APP_PORT" set "APP_PORT=%%B"
        if /I "%%A"=="FLASK_PORT" if not defined APP_PORT set "APP_PORT=%%B"
        if /I "%%A"=="APP_HOST" set "APP_HOST=%%B"
        if /I "%%A"=="FLASK_ENV" set "FLASK_ENV=%%B"
    )
)

if not defined APP_PORT set "APP_PORT=5002"
if not defined APP_HOST set "APP_HOST=127.0.0.1"
if not defined FLASK_ENV set "FLASK_ENV=development"

set "OPEN_BROWSER=1"
set "FLASK_APP=app.py"
set "FLASK_PORT=%APP_PORT%"

echo.
echo [INFO] Pre-flight check...
python installer\check_system.py
if errorlevel 1 (
    echo.
    echo [WARNING] System check reported errors. App may still start if only warnings exist.
    echo Jalankan installer\repair_windows.bat jika aplikasi gagal.
    echo.
)

echo.
echo ============================================================
echo   Arabic Khat AI
echo   Starting server at http://%APP_HOST%:%APP_PORT%
echo   Tekan Ctrl+C untuk menghentikan.
echo ============================================================
echo.

python app.py
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Terjadi kesalahan saat menjalankan aplikasi.
    echo Jalankan installer\repair_windows.bat atau baca installer\README_INSTALL_WINDOWS.md
)

pause
endlocal
exit /b %EXIT_CODE%
