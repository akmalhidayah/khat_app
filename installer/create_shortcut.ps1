# Creates a desktop shortcut for Arabic Khat AI (Windows)
$ErrorActionPreference = "Stop"

$InstallerDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $InstallerDir "..")
$RunScript = Join-Path $InstallerDir "run_windows.bat"
$Desktop = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $Desktop "Arabic Khat AI.lnk"

if (-not (Test-Path $RunScript)) {
    Write-Error "run_windows.bat not found at $RunScript"
}

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $RunScript
$Shortcut.WorkingDirectory = $ProjectRoot
$Shortcut.WindowStyle = 1
$Shortcut.Description = "Arabic Khat AI - Klasifikasi Khat Arab"
$Shortcut.Save()

Write-Host "Shortcut created: $ShortcutPath"
