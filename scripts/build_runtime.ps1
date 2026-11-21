#Requires -Version 5.1
# =====================================================================
# ADR portable runtime builder (batch 44; batch 52 dual-variant)
#
# Output: <repo>/build/runtime
#   python\               CPython 3.13 (python-build-standalone, win64)
#                         + torch stack + pinned deps
#   adr\                  adr package source (PYTHONPATH-friendly flat copy)
#   third_party\gpt_sovits\  GSV source tree
#   scripts\              训练脚本随包 (批次71, 复盘§一: console.py 以
#                         子进程调 REPO/scripts/*.py, v1.0.1/v1.3.1 双包漏拷
#                         → 用户点训练即 500「缺少训练脚本」)
#   bin\                  ffmpeg.exe + ffprobe.exe (gyan 8.1.2, 批次71 依赖
#                         自包含; 启动时 settings.ensure_runtime_bins_on_path
#                         把本目录注入 PATH — GSV 训练切片裸调 "ffmpeg")
#   redist\               VC++ 2015-2022 x64 (批次71: PBS Python 的 CRT 依赖,
#                         installer.iss [Run] 静默安装)
#
# Variants (双包拍板, 内部批次计划 G 节):
#   CPU (default): torch 2.10.0 from PyPI (Windows PyPI wheels are CPU builds)
#   GPU (-AllowCuda): torch 2.10.0+cu130 from download.pytorch.org/whl/cu130
#                     (+ nvidia_* CUDA 库 ~2.5GB); smoke 断言 cuda 可用
#
# NOT included (by design):
#   - pretrained_models / G2PWModel  -> downloaded on first start (rt4 bootstrap)
#   - logs / output / *_weights_* / TEMP / __pycache__ / .git
#   - tools/asr/models (GSV funasr ASR models, unused by adr)
# =====================================================================
[CmdletBinding()]
param(
    # GPU 包构建: 安装 CUDA torch, 供 GPU 用户开箱即用 (CPU 包为默认线)
    [switch]$AllowCuda
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
# 变体隔离目录: CPU/GPU runtime 共存, 重建互不覆盖 (GPU 包装机验证时 CPU runtime 不受影响)
$OutDir   = if ($AllowCuda) { Join-Path $RepoRoot "build\runtime-gpu" } else { Join-Path $RepoRoot "build\runtime-cpu" }
$DlDir    = Join-Path $RepoRoot "build\_dl"
$ReqFile  = Join-Path $RepoRoot "build\_requirements.txt"
$PipCache = Join-Path $RepoRoot "build\_pipcache"

function Log([string]$msg) {
    Write-Host ("[build_runtime {0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg) -ForegroundColor Cyan
}
function Step([string]$msg) {
    Write-Host ""
    Write-Host ("================ {0} ================" -f $msg) -ForegroundColor Yellow
}
# PS 5.1: $ErrorActionPreference=Stop 会把原生命令往 stderr 写的任何一行
# (pip 的 "WARNING: Retrying ..."、nltk.downloader 启动的 runpy RuntimeWarning)
# 升级成终止性 NativeCommandError, 构建当场断掉 —— 这才是 2026-11 起多次构建
# 中断的真因, 而非依赖取不到。进程成败只该由 $LASTEXITCODE 表达, 所以第 4 步
# (全是 pip / nltk 原生命令) 整段降级为 Continue, 结束再还原。
# 注意: 必须在本脚本顶层赋值 —— 在函数里改 $ErrorActionPreference 只影响函数
# 自身作用域, 对调用处无效 (踩过一次)。
function Enter-NativeStep {
    $script:EapSaved = $ErrorActionPreference
    $script:ErrorActionPreference = "Continue"
}
function Exit-NativeStep {
    $script:ErrorActionPreference = $script:EapSaved
}

Step "0/7 prepare directories"
New-Item -ItemType Directory -Force -Path $OutDir, $DlDir, $PipCache | Out-Null
# 批次73: 包型标记 — 桌面壳据此判断"CPU 包 / GPU 包", 与显卡存在性互校后
# 在启动器顶部提示换包。安装器 Source: {#RuntimeDir}\* → {app}\runtime,
# 故该文件最终落在 <安装目录>\runtime\BUILD_VARIANT.txt。
$variantTag = if ($AllowCuda) { "gpu" } else { "cpu" }
Set-Content -Path (Join-Path $OutDir "BUILD_VARIANT.txt") -Value $variantTag -Encoding ASCII
Log ("build variant: {0}" -f $variantTag)

Step "1/7 fetch python-build-standalone (CPython 3.13 win64)"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
# 批次69 (Q1): 钉死发布标签 + 资产名 + SHA256。原先查 releases/latest 每次
# 解析结果都可能不同 (构建不可复现), 且 GitHub 匿名 API 限流 60/h 会让构建
# 中断。缓存命中同样校验哈希, 损坏/换了文件即重下。
$PbsTag    = "20261009"
$PbsAsset  = "cpython-3.13.16+20261009-x86_64-pc-windows-msvc-install_only.tar.gz"
$PbsSha256 = "286285F159BC711E4F0D89BD3D01993D3A321B84BD6593BD66DC816C23609388"
$PbsUrl    = "https://github.com/astral-sh/python-build-standalone/releases/download/$PbsTag/$PbsAsset"
$tarball   = Join-Path $DlDir $PbsAsset
$needDownload = $true
if (Test-Path $tarball) {
    $hash = (Get-FileHash $tarball -Algorithm SHA256).Hash
    if ($hash -eq $PbsSha256) {
        Log ("PBS archive cached+verified: {0} — skip download" -f $PbsAsset)
        $needDownload = $false
    } else {
        Log ("PBS cache SHA256 mismatch ({0}) — re-download" -f $hash)
        Remove-Item -Force $tarball
    }
}
if ($needDownload) {
    Log ("downloading {0} (progress bar below) ..." -f $PbsAsset)
    Invoke-WebRequest -Uri $PbsUrl -OutFile $tarball
    $hash = (Get-FileHash $tarball -Algorithm SHA256).Hash
    if ($hash -ne $PbsSha256) {
        throw "PBS archive SHA256 mismatch: $hash (expected $PbsSha256)"
    }
    Log "download verified (SHA256 ok)"
}
$pyDir = Join-Path $OutDir "python"
if (Test-Path $pyDir) { Remove-Item -Recurse -Force $pyDir }
Log "extracting python ..."
& tar -xzf $tarball -C $OutDir
if ($LASTEXITCODE -ne 0) { throw "tar extraction failed" }
if (-not (Test-Path (Join-Path $pyDir "python.exe"))) { throw "python.exe not found after extraction" }

$py = Join-Path $pyDir "python.exe"
$env:PYTHONUTF8 = "1"
# 冒烟/pip check 只看 runtime 自身: 隔离开发机用户级 site-packages
# (Roaming\Python\Python313 曾让 pip check 误报 mouseinfo 缺 pyperclip)
$env:PYTHONNOUSERSITE = "1"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$env:PIP_NO_WARN_SCRIPT_LOCATION = "1"

Step "2/7 bootstrap pip"
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $py -m pip --version 2>$null | Out-Null
$hasPip = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $prevEap
if (-not $hasPip) {
    Log "pip missing, running ensurepip ..."
    & $py -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) { throw "ensurepip failed" }
} else {
    Log "pip present"
}

Step "3/7 write pinned requirements"
$requirements = @"
numpy==2.4.3
PyYAML==6.0.3
click==8.4.1
rich==14.3.3
tqdm==4.67.3
soundfile==0.13.1
librosa==0.11.0
transformers==4.57.3
tokenizers==0.22.2
safetensors==0.7.0
huggingface_hub==0.36.2
# batch75: GSV ASR scripts (tools/asr/fasterwhisper_asr.py, funasr_asr.py) do a
# TOP-LEVEL "from modelscope import snapshot_download" -- without the package the
# training pipeline step 2/7 (ASR) dies with ModuleNotFoundError.
# Light deps only (filelock/packaging/requests/setuptools/tqdm/urllib3); the
# heavy "datasets" extra is NOT pulled in.
modelscope==1.35.3
fastapi==0.129.0
uvicorn[standard]==0.41.0
python-multipart==0.0.24
faster-whisper==1.2.1
ctranslate2==4.8.1
onnxruntime==1.24.4
pypinyin==0.55.0
jieba==0.42.1
cn2an==0.5.24
g2p-en==2.1.0
nltk==3.10.3
split-lang==2.1.1
wordsegment==1.3.1
fast-langdetect==1.0.1
fasttext-predict==0.9.2.4
jaconv==0.5.0
opencc-python-reimplemented==0.1.7
# batch53 multilingual G2P: ja/ko/yue synthesis import chain (full ko also
# needs g2pk2+eunjeon, unavailable on cp313 -- v1.1 claim: zh/en/ja/yue)
# NOTE: keep this here-string pure ASCII -- PS 5.1 reads BOM-less files as
# ANSI(GBK) and CJK comment bytes here silently drop the 4 lines below.
pyopenjtalk==0.4.1
jamo==0.4.1
ToJyutping==3.2.0
ko_pron==1.3
peft==0.19.1
scipy==1.17.1
einops==0.8.2
psutil==7.2.2
requests==2.34.2
audioop-lts==0.2.2
standard-aifc==3.13.0
pydub==0.25.1
faiss-cpu==1.14.3
ffmpeg-python==0.2.0
pytorch-lightning==2.6.5
matplotlib==3.11.1
tensorboard==2.21.0
tensorboardX==2.6.5
x-transformers==2.25.5
sentencepiece==0.2.2
chardet==7.7.0
"@
# ASCII on purpose: PowerShell 5.1 reads BOM-less files as ANSI.
# torch/torchaudio 不在此清单: 按变体单独安装 (见下一步), 避免解析器
# 从 PyPI 拉回 CPU 轮子覆盖 cu130 本地版本。
Remove-Item -Force $ReqFile -ErrorAction SilentlyContinue
Set-Content -Path $ReqFile -Value $requirements -Encoding ASCII
# Write-back check: v1.3.1 build once ran pip against a stale file missing the G2P pins
$written = Get-Content -Raw $ReqFile
foreach ($pin in "pyopenjtalk==", "jamo==", "ToJyutping==", "ko_pron==", "nltk==", "modelscope==") {
    if ($written -notmatch [regex]::Escape($pin)) { throw "requirements file missing $pin (stale write?)" }
}
Log ("requirements written: {0}" -f $ReqFile)

Step "4/7 pip install (this downloads ~1 GB, progress bars below)"
# 4a. torch 栈先行: 后续 requirements 中的 pytorch-lightning 等看到已装即满足
if ($AllowCuda) {
    Log "GPU variant: torch/torchaudio 2.10.0+cu130 from download.pytorch.org (~2.5 GB)"
    Enter-NativeStep
    & $py -m pip install --prefer-binary --cache-dir $PipCache --no-warn-script-location `
        --index-url https://download.pytorch.org/whl/cu130 `
        torch==2.10.0+cu130 torchaudio==2.10.0+cu130
    $rc = $LASTEXITCODE
    Exit-NativeStep
    if ($rc -ne 0) { throw "pip install torch cu130 failed" }
} else {
    Log "CPU variant: torch/torchaudio 2.10.0 from PyPI (Windows PyPI wheels are CPU builds)"
    Enter-NativeStep
    & $py -m pip install --prefer-binary --cache-dir $PipCache --no-warn-script-location `
        torch==2.10.0 torchaudio==2.10.0
    $rc = $LASTEXITCODE
    Exit-NativeStep
    if ($rc -ne 0) { throw "pip install torch failed" }
}
# 4b. 其余钉版依赖 (torch 已满足, 不会重复解析)
Enter-NativeStep
& $py -m pip install --prefer-binary --cache-dir $PipCache --no-warn-script-location -r $ReqFile
$rc = $LASTEXITCODE
Exit-NativeStep
if ($rc -ne 0) { throw "pip install failed" }

# 4c. nltk_data 随包 (批次74): g2p-en 首次处理英文文本时 G2p.__init__ 调
#     cmudict.dict(), nltk.pos_tag 首次调用加载 PerceptronTagger — 两条都会
#     触发 nltk 在线下载。无外网/被墙机器下载失败后英文合成直接抛 LookupError
#     (2026-11-01 server.log 实锤)。目标目录固定为 <runtime>\python\nltk_data:
#     nltk.data.path 默认含 sys.prefix/nltk_data, 运行时零代码命中。
#     新旧标注器名都下 (nltk>=3.9 用 _eng, 旧版用原名), 体积合计 ~3MB。
$nltkDir = Join-Path $OutDir "python\nltk_data"
# nltk>=3.10 的 pathsec 默认拒绝"经代理"的 urlopen (CWE-918 SSRF 防护), 构建机经
# 代理出网时连 index.xml 都拉不到 (报 Security Violation)。这里只在构建进程内
# 放行, 运行时 (用户机) 不涉及, 不影响客户端防护。
$env:NLTK_ALLOW_PROXIED_URLOPEN = "1"
# 逐包下载, 且喂空 stdin: downloader 遇到哈希/尺寸不符会交互式问 "Retry? [n/y/e]",
# 无人应答时 input() 抛 EOFError 直接整条构建失败。这里只认必装包 (cmudict +
# _eng, 由下方 Test-Path 硬校验), 旧版兼容包 averaged_perceptron_tagger 失败只告警
# (镜像曾给出尺寸不符的包 — 2026-11-08 实测)。
# downloader 启动会打一条 runpy RuntimeWarning 到 stderr, 用 Enter/Exit-NativeStep
# 夹住避免被 Stop 升级成终止性错误, 成败只看退出码。
Enter-NativeStep
foreach ($pkg in @("cmudict", "averaged_perceptron_tagger_eng", "averaged_perceptron_tagger")) {
    "" | & $py -m nltk.downloader -d $nltkDir $pkg
    if ($LASTEXITCODE -ne 0) { Log ("WARN nltk package skipped: {0}" -f $pkg) }
}
Exit-NativeStep
# 下载不到的资源回落到构建机已有 nltk_data (同一份官方包; 代理镜像偶发尺寸
# 不符时 build 仍可出包), 仍由下方硬校验兜底。
$localNltk = Join-Path $env:APPDATA "nltk_data"
foreach ($res in @("corpora\cmudict", "taggers\averaged_perceptron_tagger_eng")) {
    $dst = Join-Path $nltkDir $res
    $src = Join-Path $localNltk $res
    if (-not (Test-Path $dst) -and (Test-Path $src)) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
        Copy-Item $src $dst -Recurse -Force
        Log ("nltk fallback from local: {0}" -f $res)
    }
    if (-not (Test-Path $dst)) { throw "nltk data missing: $res" }
}
Log ("nltk_data bundled: {0}" -f $nltkDir)

Step "5/7 copy adr + scripts + GSV sources (excluding models / artifacts / caches)"
robocopy (Join-Path $RepoRoot "adr") (Join-Path $OutDir "adr") /E /NFL /NDL /NJH /NJS /NP /XD __pycache__ /XF *.pyc | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy adr failed ($LASTEXITCODE)" }
# 训练 preset yaml 必须随包: adr.core.config 按 REPO/configs/*.yaml 加载,
# 缺失会静默回落内置默认值 (安装机上 doctor 报 preset yaml 缺失, v1.0.1 前实装)
robocopy (Join-Path $RepoRoot "configs") (Join-Path $OutDir "configs") /E /NFL /NDL /NJH /NJS /NP /XF *.pyc | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy configs failed ($LASTEXITCODE)" }
# 训练脚本必须随包 (批次71, 复盘§一: v1.0.1/v1.3.1 双包漏拷 scripts/,
# console.py 以子进程调 REPO/scripts/gsv_finetune.py —
# 缺失即用户点训练 500「缺少训练脚本」, check_runtime 第 8 项防复发)。
# 构建脚本 (build_*.ps1 等) 一并随包: 体积可忽略, 排障时可用。
robocopy (Join-Path $RepoRoot "scripts") (Join-Path $OutDir "scripts") /E /NFL /NDL /NJH /NJS /NP /XD __pycache__ /XF *.pyc | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy scripts failed ($LASTEXITCODE)" }
# 内置文档中心 (/api/adr/v1/docs) 的白名单 = README.md + 下列 12 个路径
# (10 篇中文 + 3 篇英文译文, 与 adr/server/docs_api.py 的 DOC_CATALOG 对齐)。
# 不随包 = 装完文档中心空、/check 还会误报"已是最新"(2026-11-14 深审 A1)。
# 只拷白名单, 不整目录拷: docs/ 里还有 m*-plan / audit / 路线图等内部材料,
# 整目录随包等于把内部计划发到用户磁盘上 (与 docs_api "内部材料不进白名单" 的口径矛盾)。
$docFiles = @(
    "demos.md",
    "guide.md",
    "troubleshooting.md",
    "gpu-compat.md",
    "update-manifest.md",
    "tts-api-spec.md",
    "architecture-design.md",
    "release-notes-v1.2.0.md",
    "architecture\windows-install-acceptance.md",
    "en\quickstart.md",
    "en\guide.md",
    "en\troubleshooting.md"
)
foreach ($rel in $docFiles) {
    $src = Join-Path (Join-Path $RepoRoot "docs") $rel
    $dst = Join-Path (Join-Path $OutDir "docs") $rel
    if (-not (Test-Path $src)) { throw "doc missing: docs/$rel" }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
    Copy-Item $src $dst -Force
}
Copy-Item (Join-Path $RepoRoot "README.md") (Join-Path $OutDir "README.md") -Force
if (-not (Test-Path (Join-Path $OutDir "README.md"))) { throw "README.md copy failed" }
Log ("docs bundled: {0} files" -f ($docFiles.Count + 1))
$gsvSrc = Join-Path $RepoRoot "third_party\gpt_sovits"
$gsvDst = Join-Path $OutDir "third_party\gpt_sovits"
robocopy $gsvSrc $gsvDst /E /NFL /NDL /NJH /NJS /NP `
    /XD __pycache__ .git .github Docker docs logs output TEMP pretrained_models G2PWModel `
         GPT_weights GPT_weights_v2 GPT_weights_v3 GPT_weightsSoVITS SoVITS_weights SoVITS_weights_v2 SoVITS_weights_v3 `
         (Join-Path $gsvSrc "tools\asr\models") `
    /XF *.pyc G2PWModel_1.1.zip | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy gpt_sovits failed ($LASTEXITCODE)" }
Log "sources copied"

Step "6/7 fetch bundled ffmpeg + VC++ redist (批次71 依赖自包含)"
# 2026-11-01 会议拍板 (依赖自包含): 精简系统缺 ffmpeg / VC++ CRT 时装完即坏。
#   ffmpeg → bin\ffmpeg.exe + ffprobe.exe (GSV 训练切片 my_utils.py 与 uvr5
#             裸调 "ffmpeg" 只认 PATH; settings.ensure_runtime_bins_on_path
#             在 server 启动时把 runtime\bin 注入 PATH)
#   redist  → redist\vc_redist.x64.exe (PBS Python 的 CRT 依赖, 精简系统
#             缺它 python.exe 直接起不来; installer.iss [Run] 静默安装,
#             已装则微软安装器自行秒过)
# 钉版直链 + SHA256 (与 Step1 PBS 同口径, 缓存命中同样校验):
#   - ffmpeg: gyan.dev 8.1.2 essentials (两个 exe 各 ~97MB, 闭审计附录 A-1)
#   - VC redist: 微软 download 直链, URL 路径内嵌 SHA256 (aka.ms/vs/17 是
#     滚动指针, 每次解析都可能换文件 — 不可钉; 此直链长期有效)
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$FfVer    = "8.1.2"
$FfZip    = "ffmpeg-$FfVer-essentials_build.zip"
$FfUrl    = "https://www.gyan.dev/ffmpeg/builds/packages/$FfZip"
$FfSha256 = "DB580001CAA24AC104C8CB856CD113A87B0A443F7BDF47D8C12B1D740584A2EC"
$VcUrl    = "https://download.visualstudio.microsoft.com/download/pr/990c720a-46ce-4dc7-87b9-c2eb615d34b9/CC0FF0EB1DC3F5188AE6300FAEF32BF5BEEBA4BDD6E8E445A9184072096B713B/VC_redist.x64.exe"
$VcSha256 = "CC0FF0EB1DC3F5188AE6300FAEF32BF5BEEBA4BDD6E8E445A9184072096B713B"

function Get-VerifiedFile([string]$Url, [string]$OutFile, [string]$Sha256) {
    # 缓存命中即校验哈希; 哈希不符时保留文件让下面的 curl 续传 (弱网下几十 MB
    # 白扔太贵), 完整但损坏的极端情况由下载后的哈希失败整文件重下兜底。
    if (Test-Path $OutFile) {
        if ((Get-FileHash $OutFile -Algorithm SHA256).Hash -eq $Sha256) {
            Log ("cached+verified: {0} — skip download" -f (Split-Path -Leaf $OutFile))
            return
        }
        Log ("cache SHA256 mismatch ({0}) — resume/refetch" -f (Split-Path -Leaf $OutFile))
    }
    # 断点续传 + 重试: gyan.dev 这类源在弱网/高负载下会中途被对端 RST
    # (2026-11-14 实测在 86 MB 处 "connection forcibly closed"), 用
    # Invoke-WebRequest 只能从 0 重来。curl.exe 是 Win10+ 自带; 缺失时退回
    # Invoke-WebRequest (不可续传, 但至少能跑)。
    $curl = Join-Path $env:SystemRoot "System32\curl.exe"
    for ($attempt = 1; $attempt -le 2; $attempt++) {
        if (Test-Path $curl) {
            Log ("downloading {0} (curl, resume, attempt {1}) ..." -f (Split-Path -Leaf $OutFile), $attempt)
            Enter-NativeStep
            & $curl -L --fail --retry 5 --retry-all-errors --retry-delay 3 -C - -o $OutFile $Url
            $rc = $LASTEXITCODE
            Exit-NativeStep
            if ($rc -ne 0) { throw "curl failed for $(Split-Path -Leaf $OutFile) (exit $rc)" }
        } else {
            Log ("downloading {0} ..." -f (Split-Path -Leaf $OutFile))
            Invoke-WebRequest -Uri $Url -OutFile $OutFile
        }
        $h = (Get-FileHash $OutFile -Algorithm SHA256).Hash
        if ($h -eq $Sha256) { Log "download verified (SHA256 ok)"; return }
        Log ("SHA256 mismatch ({0}) — full re-download" -f $h)
        Remove-Item -Force $OutFile
    }
    throw "SHA256 mismatch for $(Split-Path -Leaf $OutFile) (expected $Sha256)"
}
$FfZipPath = Join-Path $DlDir $FfZip
$VcPath    = Join-Path $DlDir "vc_redist.x64.exe"
Get-VerifiedFile $FfUrl $FfZipPath $FfSha256
Get-VerifiedFile $VcUrl $VcPath $VcSha256

# ffmpeg: 只抽取 bin\ffmpeg.exe + ffprobe.exe (zip 内其余载荷不随包)
$binDir = Join-Path $OutDir "bin"
New-Item -ItemType Directory -Force -Path $binDir | Out-Null
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($FfZipPath)
try {
    foreach ($exe in @("ffmpeg.exe", "ffprobe.exe")) {
        $entry = $zip.Entries | Where-Object { $_.Name -eq $exe } | Select-Object -First 1
        if (-not $entry) { throw "ffmpeg zip missing entry: $exe" }
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $binDir $exe), $true)
        Log ("extracted: bin\{0} ({1:N0} MB)" -f $exe, ($entry.Length / 1MB))
    }
} finally { $zip.Dispose() }

# VC redist: 整文件落 redist\ (installer.iss [Run] 从这里静默安装)
$redistDir = Join-Path $OutDir "redist"
New-Item -ItemType Directory -Force -Path $redistDir | Out-Null
Copy-Item $VcPath (Join-Path $redistDir "vc_redist.x64.exe") -Force
Log "redist staged: redist\vc_redist.x64.exe"

Step "7/7 smoke test"
Push-Location $OutDir
# 冒烟是"跑 python 看能不能 import", python 往 stderr 打的 SyntaxWarning
# (GSV 的 GPT_SoVITS/utils.py 有 '\d' 转义) 不该判构建失败 —— 整段降级,
# 成败仍由各步 $LASTEXITCODE 判定。
Enter-NativeStep
try {
    $env:PYTHONPATH = $OutDir
    & $py -c "import sys; print('python', sys.version.split()[0]); import torch, torchaudio; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { throw "torch smoke failed" }
    if ($AllowCuda) {
        # GPU 包的立身之本: cu130 轮子必须真的能见到 CUDA, 否则装出的
        # "GPU 包" 与 CPU 包无异 (静默回退正是要防的坑)
        & $py -c "import torch; assert torch.version.cuda is not None, 'not a CUDA torch build'; assert torch.cuda.is_available(), 'CUDA torch installed but cuda unavailable (driver/runtime mismatch?)'; print('GPU variant smoke: CUDA OK,', torch.cuda.get_device_name(0))"
        if ($LASTEXITCODE -ne 0) { throw "GPU smoke failed: CUDA not usable in built runtime" }
    }
    & $py -c "import adr, adr.server, adr.models.gsv_engine, adr.models.rvc_engine, adr.models.hub, adr.data.asr; print('adr imports OK')"
    if ($LASTEXITCODE -ne 0) { throw "adr import smoke failed" }
    # GSV import 需与 gsv_engine._gsv_context 同款上下文: cwd=GSV 根
    # (sv.py 用 os.getcwd() 拼 eres2net 路径) + 双 sys.path (根 + GPT_SoVITS)
    $gsvPath = $gsvDst.Replace("'", "''")
    & $py -c "import sys, os; g = r'$gsvPath'.replace('/', '\\'); os.chdir(g); sys.path.insert(0, g); sys.path.insert(0, g + '/GPT_SoVITS'); from GPT_SoVITS.TTS_infer_pack.TTS import TTS_Config; print('GSV imports OK')"
    if ($LASTEXITCODE -ne 0) { throw "GSV import smoke failed" }
    & $py -m pip check
    $rc = $LASTEXITCODE
    if ($rc -ne 0) { Log "pip check reported issues (see above)" }
} finally {
    Exit-NativeStep
    Pop-Location
}

Step "8 strip bytecode caches (批次75: 安装包不应带 .pyc / __pycache__ 临时文件)"
# pip install 会为每个模块生成 __pycache__/*.pyc (cp313 默认开), runtime-cpu
# 实测 17768 个 .pyc / 2297 个 __pycache__ 目录 — 用户装完看到一堆"临时文件",
# 安装包也白涨体积。它们是纯缓存: Python 首次 import 会自行重建 (安装目录
# per-user 可写, 重建无碍)。必须排在所有 python 步骤 (pip install / smoke
# test) 之后执行, 否则会被重新生成。
$stripPy = Join-Path $DlDir "strip_bytecode.py"
@'
import os, shutil, sys

root = sys.argv[1]
dirs = files = 0
for dirpath, dirnames, filenames in os.walk(root, topdown=True, onerror=lambda e: None):
    if os.path.basename(dirpath) == "__pycache__":
        shutil.rmtree(dirpath, ignore_errors=True)
        dirs += 1
        dirnames[:] = []
        continue
    for fn in filenames:
        if fn.endswith(".pyc"):
            try:
                os.remove(os.path.join(dirpath, fn))
                files += 1
            except OSError:
                pass
print("removed __pycache__ dirs=%d, .pyc files=%d" % (dirs, files))
'@ | Set-Content -Path $stripPy -Encoding ASCII
$stripOut = & $py -B $stripPy $OutDir
if ($LASTEXITCODE -ne 0) { throw "bytecode strip failed" }
Log ("bytecode caches stripped: {0}" -f $stripOut)

Step "done"
$variant = if ($AllowCuda) { "GPU (cu130)" } else { "CPU" }
$sz = (Get-ChildItem $OutDir -Recurse -File | Measure-Object Length -Sum).Sum
Log ("runtime total: {0:N0} MB  [{1}]  ->  {2}" -f ($sz / 1MB), $variant, $OutDir)
