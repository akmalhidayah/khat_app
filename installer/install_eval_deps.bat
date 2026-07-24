@echo off
setlocal EnableExtensions
title Arabic Khat AI - Install Evaluation Dependencies

set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment .venv tidak ditemukan.
    echo Jalankan installer\install_windows.bat terlebih dahulu.
    goto :END
)

echo.
echo ============================================================
echo   Memasang dependensi evaluasi Teachable Machine
echo   tensorflow + tensorflowjs + h5py
echo ============================================================
echo.

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install "tensorflow==2.20.0" "h5py>=3.11.0" "tf-keras>=2.13.0"
if errorlevel 1 (
    echo [ERROR] Gagal memasang tensorflow. Pastikan koneksi internet stabil.
    goto :END
)
python -m pip install "tensorflowjs==4.22.0" --no-deps
if errorlevel 1 (
    echo [ERROR] Gagal memasang tensorflowjs.
    goto :END
)

echo.
python -c "from services.tfjs_import_bootstrap import load_keras_model; import tensorflow as tf; print('OK - TensorFlow', tf.__version__)"
if errorlevel 1 (
    echo [WARNING] Import gagal. Coba restart terminal lalu jalankan evaluasi lagi.
) else (
    echo.
    echo ============================================================
    echo   Dependensi evaluasi siap. Jalankan evaluasi dari menu
    echo   Evaluasi Model -^> Jalankan Evaluasi
    echo ============================================================
)

:END
pause
endlocal
