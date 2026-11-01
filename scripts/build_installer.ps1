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

# MAX_PATH 预检: ISCC 用源文件的完整路径去 open, >260 就报 "系统找不到指定的路径"
# 这种报错完全看不出是路径问题 (v1.0.1 踩过), 所以提前用可读中文拦下。
# 阈值 250 留 10 字符余量给 ISCC 内部临时后缀。
$rtDir = Join-Path $root "build\$rtDirName"
$tooLong = @()
try {
  $tooLong = @(Get-ChildItem -LiteralPath $rtDir -Recurse -File -Force -ErrorAction Stop |
               Where-Object { $_.FullName.Length -gt 250 })
} catch {
  throw "runtime 路径扫描失败 (目录内可能已有超长路径): $($_.Exception.Message)"
}
if ($tooLong.Count -gt 0) {
  $mx = ($tooLong | ForEach-Object { $_.FullName.Length } | Measure-Object -Maximum).Maximum
  throw ("runtime 有 {0} 个文件全路径超过 250 字符 (最长 {1}), ISCC 压缩阶段会报 " -f $tooLong.Count, $mx) +
        "'系统找不到指定的路径' 而中止。请把构建根目录换到更短的路径 (如 C:\adr) 后重试。`n" +
        ("最长示例: {0}" -f $tooLong[0].FullName)
}

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
# 传归一化绝对路径给 installer.iss 的 RUNTIME_DIR_ABS: 相对路径 "..\..\build\..."
# 不会被 ISCC 折叠, 字面多 24 字符会把 MAX_PATH 顶穿 (见 installer.iss 处注释)。
$rtAbs = (Resolve-Path -LiteralPath (Join-Path $root "build\$rtDirName")).Path
Write-Host "[build_installer] runtime payload: $rtAbs" -ForegroundColor Cyan
$isccArgs += "/DRUNTIME_DIR_ABS=$rtAbs"
$isccArgs += (Join-Path $PSScriptRoot "installer\installer.iss")
& $iscc @isccArgs
if ($LASTEXITCODE -ne 0) { throw "ISCC failed" }

$out = Join-Path $root "desktop\src-tauri\target\release\bundle\inno"
Get-ChildItem $out | Select-Object Name, @{n = 'MB'; e = { [math]::Round($_.Length / 1MB, 1) } }
Write-Output "Installer build OK"
