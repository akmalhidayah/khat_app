# Arabic Khat AI — Windows installer (PowerShell / VS Code terminal)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host ""
Write-Host "============================================================"
Write-Host "  Arabic Khat AI - Windows Installer (PowerShell)"
Write-Host "============================================================"
Write-Host "  Project: $ProjectRoot"
Write-Host "============================================================"
Write-Host ""

function Get-PythonCmd {
    if (Get-Command python -ErrorAction SilentlyContinue) { return "python" }
    if (Get-Command py -ErrorAction SilentlyContinue) { return "py -3" }
    throw "Python belum terpasang. Install Python 3.10/3.11 dan centang 'Add Python to PATH'."
}

$Py = Get-PythonCmd
Write-Host "[1/7] Python: $(& $Py -c 'import sys; print(sys.version.split()[0])')"

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "[2/7] Creating virtual environment .venv ..."
    & $Py -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Gagal membuat .venv" }
} else {
    Write-Host "[2/7] .venv already exists."
}

Write-Host "[3/7] Upgrading pip ..."
& $VenvPython -m pip install --upgrade pip | Out-Null

Write-Host "[4/7] Installing dependencies (may take several minutes) ..."
& $VenvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "Gagal install dependencies. Gunakan Python 3.10/3.11 dan jalankan repair_windows.ps1"
}

Write-Host "[5/7] Running setup_app.py ..."
& $VenvPython installer\setup_app.py

Write-Host "[6/7] System check ..."
& $VenvPython installer\check_system.py
$CheckCode = $LASTEXITCODE

Write-Host "[7/7] Done."
Write-Host ""
if ($CheckCode -eq 0) {
    Write-Host "Instalasi selesai. Jalankan:"
    Write-Host "  installer\run_windows.ps1"
    Write-Host "  atau: .\.venv\Scripts\python.exe app.py"
    Write-Host "Browser: http://127.0.0.1:5002  |  Login: admin / admin123"
} else {
    Write-Host "Instalasi selesai dengan peringatan. Periksa MySQL XAMPP dan file model."
    Write-Host "Jalankan: installer\repair_windows.ps1"
}
Write-Host ""
