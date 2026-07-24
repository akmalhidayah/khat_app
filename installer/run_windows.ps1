# Arabic Khat AI — run server (PowerShell / VS Code terminal)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "[ERROR] Virtual environment (.venv) belum ada."
    Write-Host "Jalankan dulu:"
    Write-Host "  installer\install_windows.bat"
    Write-Host "  atau: powershell -ExecutionPolicy Bypass -File installer\install_windows.ps1"
    exit 1
}

# Load APP_PORT from .env if present
$EnvFile = Join-Path $ProjectRoot ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*APP_PORT\s*=\s*(.+)\s*$') { $env:APP_PORT = $Matches[1].Trim() }
        if ($_ -match '^\s*FLASK_PORT\s*=\s*(.+)\s*$' -and -not $env:APP_PORT) { $env:FLASK_PORT = $Matches[1].Trim() }
        if ($_ -match '^\s*APP_HOST\s*=\s*(.+)\s*$') { $env:APP_HOST = $Matches[1].Trim() }
    }
}
if (-not $env:APP_PORT) { $env:APP_PORT = "5002" }
if (-not $env:APP_HOST) { $env:APP_HOST = "127.0.0.1" }

$env:FLASK_APP = "app.py"
$env:FLASK_ENV = "development"
$env:OPEN_BROWSER = "0"

Write-Host ""
Write-Host "[INFO] Pre-flight check ..."
& $VenvPython installer\check_system.py
Write-Host ""
Write-Host "============================================================"
Write-Host "  Arabic Khat AI"
Write-Host "  http://$($env:APP_HOST):$($env:APP_PORT)"
Write-Host "  Tekan Ctrl+C untuk menghentikan."
Write-Host "============================================================"
Write-Host ""

& $VenvPython app.py
exit $LASTEXITCODE
