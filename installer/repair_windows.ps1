# Arabic Khat AI — repair (PowerShell / VS Code terminal)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "[INFO] .venv tidak ditemukan. Menjalankan install ..."
    & "$PSScriptRoot\install_windows.ps1"
    exit $LASTEXITCODE
}

Write-Host "[1/4] Reinstalling dependencies ..."
& $VenvPython -m pip install --upgrade pip | Out-Null
& $VenvPython -m pip install -r requirements.txt

Write-Host "[2/4] setup_app.py ..."
& $VenvPython installer\setup_app.py

Write-Host "[3/4] System check ..."
& $VenvPython installer\check_system.py
$CheckCode = $LASTEXITCODE

Write-Host "[4/4] Repair selesai."
if ($CheckCode -eq 0) {
    Write-Host "Jalankan: installer\run_windows.ps1"
} else {
    Write-Host "Masih ada peringatan — pastikan MySQL XAMPP running dan file model ada di static/model/"
}
exit $CheckCode
