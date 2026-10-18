#Requires -Version 5.1
# =====================================================================
# ADR portable runtime builder (batch 44; batch 52 dual-variant)
#
# Output: <repo>/build/runtime
#   python\               CPython 3.13 (python-build-standalone, win64)
#                         + torch stack + pinned deps
#   adr\                  adr package source (PYTHONPATH-friendly flat copy)
#   third_party\gpt_sovits\  GSV source tree
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

Step "0/6 prepare directories"
New-Item -ItemType Directory -Force -Path $OutDir, $DlDir, $PipCache | Out-Null

Step "1/6 fetch python-build-standalone (CPython 3.13 win64)"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$tarball = Get-ChildItem $DlDir -Filter "cpython-3.13*install_only.tar.gz" -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($tarball) {
    # 缓存命中直接用: 免 API 调用 (GitHub 匿名限流 60/h, 曾因此构建中断)
    Log ("PBS archive cached: {0} — skip API resolve" -f $tarball.Name)
    $tarball = $tarball.FullName   # FileInfo → 绝对路径 (ToString 只有文件名, tar 会找不到)
} else {
    $rel = Invoke-RestMethod -Uri "https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest"
    $asset = $rel.assets |
        Where-Object { $_.name -match "^cpython-3\.13\.[0-9]+\+[0-9]{8}-x86_64-pc-windows-msvc-install_only\.tar\.gz$" } |
        Select-Object -First 1
    if (-not $asset) { throw "no matching PBS asset in latest release" }
    Log ("latest PBS asset: {0} ({1:N0} MB)" -f $asset.name, ($asset.size / 1MB))
    $tarball = Join-Path $DlDir $asset.name
    Log "downloading (progress bar below) ..."
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $tarball
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

Step "2/6 bootstrap pip"
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

Step "3/6 write pinned requirements"
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
chardet
"@
# ASCII on purpose: PowerShell 5.1 reads BOM-less files as ANSI.
# torch/torchaudio 不在此清单: 按变体单独安装 (见下一步), 避免解析器
# 从 PyPI 拉回 CPU 轮子覆盖 cu130 本地版本。
Set-Content -Path $ReqFile -Value $requirements -Encoding ASCII
Log ("requirements written: {0}" -f $ReqFile)

Step "4/6 pip install (this downloads ~1 GB, progress bars below)"
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

Step "5/6 copy adr + GSV sources (excluding models / artifacts / caches)"
robocopy (Join-Path $RepoRoot "adr") (Join-Path $OutDir "adr") /E /NFL /NDL /NJH /NJS /NP /XD __pycache__ /XF *.pyc | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy adr failed ($LASTEXITCODE)" }
# 训练 preset yaml 必须随包: adr.core.config 按 REPO/configs/*.yaml 加载,
# 缺失会静默回落内置默认值 (安装机上 doctor 报 preset yaml 缺失, v1.0.1 前实装)
robocopy (Join-Path $RepoRoot "configs") (Join-Path $OutDir "configs") /E /NFL /NDL /NJH /NJS /NP /XF *.pyc | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy configs failed ($LASTEXITCODE)" }
$gsvSrc = Join-Path $RepoRoot "third_party\gpt_sovits"
$gsvDst = Join-Path $OutDir "third_party\gpt_sovits"
robocopy $gsvSrc $gsvDst /E /NFL /NDL /NJH /NJS /NP `
    /XD __pycache__ .git .github Docker docs logs output TEMP pretrained_models G2PWModel `
         GPT_weights GPT_weights_v2 GPT_weights_v3 GPT_weightsSoVITS SoVITS_weights SoVITS_weights_v2 SoVITS_weights_v3 `
         (Join-Path $gsvSrc "tools\asr\models") `
    /XF *.pyc G2PWModel_1.1.zip | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy gpt_sovits failed ($LASTEXITCODE)" }
Log "sources copied"

Step "6/6 smoke test"
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
