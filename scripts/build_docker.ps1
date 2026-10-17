# ADR Docker 镜像构建脚本 (批次56 / v1.2 多形态分发)
#
# 用法示例:
#   powershell -File scripts/build_docker.ps1                                    # CPU 变体
#   powershell -File scripts/build_docker.ps1 -Cuda                              # CUDA 变体
#   powershell -File scripts/build_docker.ps1 -Tag myreg/adr:1.2.0 -Cuda         # 自定义 tag
#   powershell -File scripts/build_docker.ps1 -ProxyPrefix "https://ghproxy.net/"  # GSV clone 加速
#
# 等价手写命令 (CPU):
#   docker build -f docker/Dockerfile -t adr-server:1.2.0-cpu --build-arg TORCH_VARIANT=cpu .

param(
    [switch]$Cuda,
    [string]$Tag = "",
    [string]$ProxyPrefix = "",
    # deb.debian.org 走 HTTP/80, 部分代理不覆盖 HTTP → apt update 失败;
    # 传 mirrors.tuna.tsinghua.edu.cn 等 HTTPS 镜像可绕开
    [string]$AptMirror = ""
)

$ErrorActionPreference = "Stop"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Variant = if ($Cuda) { "cuda" } else { "cpu" }
if (-not $Tag) { $Tag = "adr-server:1.2.0-$Variant" }

Write-Host "=== ADR Docker 构建 ($Variant) ===" -ForegroundColor Cyan
Write-Host "  镜像 tag    : $Tag"
Write-Host "  构建上下文  : $RepoRoot"
if ($ProxyPrefix) { Write-Host "  GSV 代理前缀: $ProxyPrefix" }

$DockerArgs = @(
    "build",
    "-f", (Join-Path $RepoRoot "docker\Dockerfile"),
    "-t", $Tag,
    "--build-arg", "TORCH_VARIANT=$Variant"
)
if ($ProxyPrefix) {
    $DockerArgs += @("--build-arg", "GSV_PROXY_PREFIX=$ProxyPrefix")
}
if ($AptMirror) {
    $DockerArgs += @("--build-arg", "APT_MIRROR=$AptMirror")
}
$DockerArgs += $RepoRoot

& docker @DockerArgs
if ($LASTEXITCODE -ne 0) {
    throw "docker build 失败 (exit $LASTEXITCODE)"
}

Write-Host ""
Write-Host "构建完成: $Tag" -ForegroundColor Green
Write-Host "启动服务:"
Write-Host "  docker run -d --name adr-server -p 9881:9881 -v adr_data:/data $Tag"
Write-Host "或使用 compose (推荐, 含权重持久化):"
Write-Host "  cd docker; docker compose up -d"
