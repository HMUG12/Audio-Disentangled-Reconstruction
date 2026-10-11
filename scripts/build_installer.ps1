# Batch 44: build the ADR Studio installer (Inno Setup).
# NSIS (tauri bundle) hit the hard ~2GB data limit on our 3.89GB runtime,
# so the installer is produced by Inno Setup from the no-bundle output.
#
# Batch 52 双包 (docs/m10-plan.md G 节): CPU 包 (默认) + GPU 包 (-Gpu,
# 打包 build_runtime.ps1 -AllowCuda 产出的 cu130 runtime, 文件名带 -gpu)。
#
# Usage (from repo root):
#   powershell -File scripts\build_installer.ps1           # CPU 包
#   powershell -File scripts\build_installer.ps1 -Gpu      # GPU 包
# Output:  desktop\src-tauri\target\release\bundle\inno\ADR-Studio-<ver>-x64[-gpu]-setup.exe
param(
    # 打 GPU 包: 文件名加 -gpu 后缀 + 体检按 --allow-cuda 放行 nvidia_* 库
    [switch]$Gpu
)
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$iscc = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if (-not (Test-Path $iscc)) { throw "Inno Setup 6 not found at: $iscc" }

# 打包前 runtime 体检 (批次 49b 固化规则, 防垃圾权重/个人数据面/底模混入)
$rtDirName = if ($Gpu) { "runtime-gpu" } else { "runtime-cpu" }
$rtPy = Join-Path $root "build\$rtDirName\python\python.exe"
if (Test-Path $rtPy) { $py = $rtPy } else { $py = "python" }
$checkArgs = @((Join-Path $PSScriptRoot "check_runtime.py"),
               "--runtime", (Join-Path $root "build\$rtDirName"))
if ($Gpu) { $checkArgs += "--allow-cuda" }
Write-Host "[build_installer] runtime check: $py $checkArgs" -ForegroundColor Cyan
& $py @checkArgs
if ($LASTEXITCODE -ne 0) { throw "check_runtime failed - 修复 runtime 后再打包" }

Push-Location (Join-Path $root "desktop\src-tauri")
try {
  $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
  npx -y @tauri-apps/cli@2 build --no-bundle
  if ($LASTEXITCODE -ne 0) { throw "tauri build failed" }
}
finally { Pop-Location }

$isccArgs = @()
if ($Gpu) {
  $isccArgs += "/DGPU_BUILD", "/DRUNTIME_DIR_NAME=runtime-gpu"
}
$isccArgs += (Join-Path $PSScriptRoot "installer\installer.iss")
& $iscc @isccArgs
if ($LASTEXITCODE -ne 0) { throw "ISCC failed" }

$out = Join-Path $root "desktop\src-tauri\target\release\bundle\inno"
Get-ChildItem $out | Select-Object Name, @{n = 'MB'; e = { [math]::Round($_.Length / 1MB, 1) } }
Write-Output "Installer build OK"
