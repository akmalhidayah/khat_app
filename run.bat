@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if /i "%~1"=="INTERNAL" goto MAIN

REM Saat double-click dari Explorer, buka jendela CMD yang tetap terbuka.
if "%~1"=="" (
    start "Khat Classification Web" cmd /k call "%~f0" INTERNAL
    exit /b 0
)

:MAIN
if not defined FLASK_PORT set "FLASK_PORT=5001"
set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo Virtual environment belum ada. Menjalankan setup otomatis...
    echo.
    call "%~dp0setup.bat" from-run
    if errorlevel 1 goto END_ERROR
    if not exist "%PYTHON_EXE%" (
        echo [ERROR] Setup selesai tetapi venv\Scripts\python.exe tidak ditemukan.
        goto END_ERROR
    )
)

echo ========================================
echo  Memeriksa persiapan...
echo ========================================
"%PYTHON_EXE%" "%~dp0scripts\preflight_check.py"
if errorlevel 1 goto END_ERROR

echo.
echo ========================================
echo  Arabic Khat Classification Web
echo ========================================
echo  URL   : http://127.0.0.1:%FLASK_PORT%
echo  Login : admin / admin123
echo.
echo  Browser akan terbuka otomatis setelah server siap.
echo  Pastikan MySQL di XAMPP Control Panel sudah Start.
echo  Tekan Ctrl+C untuk menghentikan server.
echo ========================================
echo.

set "OPEN_BROWSER=1"
set "FLASK_PORT=%FLASK_PORT%"
"%PYTHON_EXE%" "%~dp0app.py"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Aplikasi berhenti dengan kode %EXIT_CODE%.
    echo Jika port %FLASK_PORT% sudah dipakai, tutup jendela ini lalu jalankan:
    echo   set FLASK_PORT=5002 ^&^& run.bat
    echo.
)
goto END

:END_ERROR
set "EXIT_CODE=1"

:END
echo.
pause
exit /b %EXIT_CODE%
