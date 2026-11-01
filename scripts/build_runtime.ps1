#Requires -Version 5.1
# =====================================================================
# ADR portable runtime builder (batch 44; batch 52 dual-variant)
#
# Output: <repo>/build/runtime
#   python\               CPython 3.13 (python-build-standalone, win64)
#                         + torch stack + pinned deps
#   adr\                  adr package source (PYTHONPATH-friendly flat copy)
#   third_party\gpt_sovits\  GSV source tree
#   scripts\              训练脚本随包 (批次71, 复盘§一: console.py/webui 以
#                         子进程调 REPO/scripts/*.py, v1.0.1/v1.3.1 双包漏拷
#                         → 用户点训练即 500「缺少训练脚本」)
#   bin\                  ffmpeg.exe + ffprobe.exe (gyan 8.1.2, 批次71 依赖
#                         自包含; 启动时 settings.ensure_runtime_bins_on_path
#                         把本目录注入 PATH — GSV 训练切片裸调 "ffmpeg")
#   redist\               VC++ 2015-2022 x64 (批次71: PBS Python 的 CRT 依赖,
#                         installer.iss [Run] 静默安装)
#
# Variants (双包拍板, docs/m10-plan.md G 节):
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
fastapi==0.129.0
uvicorn[standard]==0.41.0
python-multipart==0.0.24
gradio==6.4.0
gradio_client==2.0.3
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
foreach ($pin in "pyopenjtalk==", "jamo==", "ToJyutping==", "ko_pron==", "nltk==") {
    if ($written -notmatch [regex]::Escape($pin)) { throw "requirements file missing $pin (stale write?)" }
}
Log ("requirements written: {0}" -f $ReqFile)

Step "4/7 pip install (this downloads ~1 GB, progress bars below)"
# 4a. torch 栈先行: 后续 requirements 中的 pytorch-lightning 等看到已装即满足
if ($AllowCuda) {
    Log "GPU variant: torch/torchaudio 2.10.0+cu130 from download.pytorch.org (~2.5 GB)"
    & $py -m pip install --prefer-binary --cache-dir $PipCache --no-warn-script-location `
        --index-url https://download.pytorch.org/whl/cu130 `
        torch==2.10.0+cu130 torchaudio==2.10.0+cu130
    if ($LASTEXITCODE -ne 0) { throw "pip install torch cu130 failed" }
} else {
    Log "CPU variant: torch/torchaudio 2.10.0 from PyPI (Windows PyPI wheels are CPU builds)"
    & $py -m pip install --prefer-binary --cache-dir $PipCache --no-warn-script-location `
        torch==2.10.0 torchaudio==2.10.0
    if ($LASTEXITCODE -ne 0) { throw "pip install torch failed" }
}
# 4b. 其余钉版依赖 (torch 已满足, 不会重复解析)
& $py -m pip install --prefer-binary --cache-dir $PipCache --no-warn-script-location -r $ReqFile
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

# 4c. nltk_data 随包 (批次74): g2p-en 首次处理英文文本时 G2p.__init__ 调
#     cmudict.dict(), nltk.pos_tag 首次调用加载 PerceptronTagger — 两条都会
#     触发 nltk 在线下载。无外网/被墙机器下载失败后英文合成直接抛 LookupError
#     (2026-11-01 server.log 实锤)。目标目录固定为 <runtime>\python\nltk_data:
#     nltk.data.path 默认含 sys.prefix/nltk_data, 运行时零代码命中。
#     新旧标注器名都下 (nltk>=3.9 用 _eng, 旧版用原名), 体积合计 ~3MB。
$nltkDir = Join-Path $OutDir "python\nltk_data"
& $py -m nltk.downloader -d $nltkDir cmudict averaged_perceptron_tagger_eng averaged_perceptron_tagger
if ($LASTEXITCODE -ne 0) { throw "nltk data download failed" }
foreach ($res in @("corpora\cmudict", "taggers\averaged_perceptron_tagger_eng")) {
    if (-not (Test-Path (Join-Path $nltkDir $res))) { throw "nltk data missing: $res" }
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
# console.py:386 / webui 以子进程调 REPO/scripts/gsv_finetune.py —
# 缺失即用户点训练 500「缺少训练脚本」, check_runtime 第 8 项防复发)。
# 构建脚本 (build_*.ps1 等) 一并随包: 体积可忽略, 排障时可用。
robocopy (Join-Path $RepoRoot "scripts") (Join-Path $OutDir "scripts") /E /NFL /NDL /NJH /NJS /NP /XD __pycache__ /XF *.pyc | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy scripts failed ($LASTEXITCODE)" }
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
    # 缓存命中即校验哈希, 损坏/换文件重下 (与 Step1 PBS 逻辑同口径)
    $need = $true
    if (Test-Path $OutFile) {
        if ((Get-FileHash $OutFile -Algorithm SHA256).Hash -eq $Sha256) {
            Log ("cached+verified: {0} — skip download" -f (Split-Path -Leaf $OutFile))
            $need = $false
        } else {
            Log ("cache SHA256 mismatch ({0}) — re-download" -f (Split-Path -Leaf $OutFile))
            Remove-Item -Force $OutFile
        }
    }
    if ($need) {
        Log ("downloading {0} ..." -f (Split-Path -Leaf $OutFile))
        Invoke-WebRequest -Uri $Url -OutFile $OutFile
        $h = (Get-FileHash $OutFile -Algorithm SHA256).Hash
        if ($h -ne $Sha256) {
            throw "SHA256 mismatch for $(Split-Path -Leaf $OutFile): $h (expected $Sha256)"
        }
        Log "download verified (SHA256 ok)"
    }
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
    & $py -c "import adr, adr.server, adr.models.gsv_engine, adr.models.rvc_engine, adr.models.hub, adr.data.asr, adr.webui; print('adr imports OK')"
    if ($LASTEXITCODE -ne 0) { throw "adr import smoke failed" }
    # GSV import 需与 gsv_engine._gsv_context 同款上下文: cwd=GSV 根
    # (sv.py 用 os.getcwd() 拼 eres2net 路径) + 双 sys.path (根 + GPT_SoVITS)
    $gsvPath = $gsvDst.Replace("'", "''")
    & $py -c "import sys, os; g = r'$gsvPath'.replace('/', '\\'); os.chdir(g); sys.path.insert(0, g); sys.path.insert(0, g + '/GPT_SoVITS'); from GPT_SoVITS.TTS_infer_pack.TTS import TTS_Config; print('GSV imports OK')"
    if ($LASTEXITCODE -ne 0) { throw "GSV import smoke failed" }
    & $py -m pip check
    if ($LASTEXITCODE -ne 0) { Log "pip check reported issues (see above)" }
} finally {
    Pop-Location
}

Step "done"
$variant = if ($AllowCuda) { "GPU (cu130)" } else { "CPU" }
$sz = (Get-ChildItem $OutDir -Recurse -File | Measure-Object Length -Sum).Sum
Log ("runtime total: {0:N0} MB  [{1}]  ->  {2}" -f ($sz / 1MB), $variant, $OutDir)
