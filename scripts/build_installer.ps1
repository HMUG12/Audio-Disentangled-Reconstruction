# Batch 44: build the ADR Studio installer (Inno Setup).
# NSIS (tauri bundle) hit the hard ~2GB data limit on our 3.89GB runtime,
# so the installer is produced by Inno Setup from the no-bundle output.
#
# Usage (from repo root):  powershell -File scripts\build_installer.ps1
# Output:  desktop\src-tauri\target\release\bundle\inno\ADR-Studio-<ver>-x64-setup.exe
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$iscc = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if (-not (Test-Path $iscc)) { throw "Inno Setup 6 not found at: $iscc" }

Push-Location (Join-Path $root "desktop\src-tauri")
try {
  $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
  npx -y @tauri-apps/cli@2 build --no-bundle
  if ($LASTEXITCODE -ne 0) { throw "tauri build failed" }
}
finally { Pop-Location }

& $iscc (Join-Path $PSScriptRoot "installer.iss")
if ($LASTEXITCODE -ne 0) { throw "ISCC failed" }

$out = Join-Path $root "desktop\src-tauri\target\release\bundle\inno"
Get-ChildItem $out | Select-Object Name, @{n = 'MB'; e = { [math]::Round($_.Length / 1MB, 1) } }
Write-Output "Installer build OK"
