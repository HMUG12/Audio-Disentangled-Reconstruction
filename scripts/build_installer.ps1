# Batch 44: build the ADR Studio installer (Inno Setup).
# NSIS (tauri bundle) hit the hard ~2GB data limit on our 3.89GB runtime,
# so the installer is produced by Inno Setup from the no-bundle output.
#
# 单一交付路径: Tauri 只负责产出 adr-desktop.exe (tauri.conf bundle.active=false,
# 本脚本再显式传 --no-bundle); 安装包唯一来源 = Inno Setup (installer.iss)。
#
# Batch 52 双包 (docs/m10-plan.md G 节): CPU 包 (默认) + GPU 包 (-Gpu,
# 打包 build_runtime.ps1 -AllowCuda 产出的 cu130 runtime, 文件名带 -gpu)。
#
# Usage (from repo root):
#   powershell -File scripts\build_installer.ps1           # CPU 包
#   powershell -File scripts\build_installer.ps1 -Gpu      # GPU 包
#   powershell -File scripts\build_installer.ps1 -DryRun   # 只跑门禁 + 前置预检 (批次70)
# Output:  desktop\src-tauri\target\release\bundle\inno\ADR-Studio-<ver>-x64[-gpu]-setup.exe
param(
    # 打 GPU 包: 文件名加 -gpu 后缀 + 体检按 --allow-cuda 放行 nvidia_* 库
    [switch]$Gpu,
    # 批次70 (Q2): 干跑 — 只验版本/合规门禁与前置 (ISCC / tauri CLI / runtime),
    # 不编译不签名; 门禁必须过 (硬失败), 前置缺失仅 WARN。成功结尾输出 DRY RUN OK。
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$iscc = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if (-not (Test-Path $iscc)) {
  if ($DryRun) { Write-Warning "DryRun: Inno Setup 6 not found at $iscc - 真打包会失败" }
  else { throw "Inno Setup 6 not found at: $iscc" }
}

# 打包前 runtime 体检 (批次 49b 固化规则, 防垃圾权重/个人数据面/底模混入)
$rtDirName = if ($Gpu) { "runtime-gpu" } else { "runtime-cpu" }
$rtPy = Join-Path $root "build\$rtDirName\python\python.exe"
if (Test-Path $rtPy) { $py = $rtPy } else { $py = "python" }
if ((-not (Test-Path $rtPy)) -and $DryRun) {
  Write-Warning "DryRun: $rtDirName runtime python not found - 门禁用系统 python ($py)"
}

# 批次69 (Q2): 版本面单源 + 合规门禁前置 — 漂移/缺口在出包前失败, 不产出坏包
foreach ($gate in @("check_versions.py", "check_compliance.py")) {
  Write-Host "[build_installer] gate: $gate" -ForegroundColor Cyan
  & $py (Join-Path $PSScriptRoot $gate)
  if ($LASTEXITCODE -ne 0) { throw "$gate failed - 修复后再打包" }
}

$checkArgs = @((Join-Path $PSScriptRoot "check_runtime.py"),
               "--runtime", (Join-Path $root "build\$rtDirName"))
if ($Gpu) { $checkArgs += "--allow-cuda" }
Write-Host "[build_installer] runtime check: $py $checkArgs" -ForegroundColor Cyan
& $py @checkArgs
if ($LASTEXITCODE -ne 0) {
  if ($DryRun) { Write-Warning "DryRun: check_runtime 未通过 - 真打包会失败" }
  else { throw "check_runtime failed - 修复 runtime 后再打包" }
}

# 批次69 (Q1): 固定版本 Tauri CLI (desktop/package.json 钉死), 不再 npx @2 动态解析
$tauriCli = Join-Path $root "desktop\node_modules\.bin\tauri.cmd"
if (-not (Test-Path $tauriCli)) {
  if ($DryRun) { Write-Warning "DryRun: tauri CLI missing - 真打包前需 cd desktop; npm ci" }
  else { throw "tauri CLI missing - 先执行: cd desktop && npm ci (版本见 package.json)" }
}

if ($DryRun) {
  Write-Output "DRY RUN OK"
  exit 0
}

Push-Location (Join-Path $root "desktop\src-tauri")
try {
  $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
  & $tauriCli build --no-bundle
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
