# REPRODUCE — 复现指南

> 所有命令与数据均为实测（开发环境 Windows 11 / Python 3.13.7，基线日期 2026-10-10）。
> 环境冻结清单见 [build_freeze.txt](build_freeze.txt)（pip freeze 全量快照）。
> 第三方逐项验证（声明 → 步骤 → 预期）见 [VERIFY.md](VERIFY.md)。

## 0. 环境准备

要求：Windows 10/11，Python 3.10+（实测 3.13.7）；NVIDIA GPU 可选（CPU 可推理，训练建议 GPU）。

```powershell
pip install -e .[m1]        # 基础 + WebUI
pip install -e .[m2]        # 训练加速 (LoRA/QLoRA), 按需

$env:NUMBA_CACHE_DIR = "E:\adr_numba_cache"   # 任意可写目录, 避免 numba 缓存落只读路径
```

模型资源（均走 `third_party/gpt_sovits/GPT_SoVITS/pretrained_models/`）：

- **GSV v2 底模**（gsv-v2final-pretrained，~4.35GB）：ADR Studio 桌面端首启 bootstrap 自动下载（进度透传）；服务端源码运行首次合成时同样拉取。
- **声纹门禁模型**（`sv/pretrained_eres2netv2w24s4ep4.ckpt`，ERes2NetV2）：随 GPT-SoVITS 官方预训练包就位，评估打分（SECS）依赖它。
- ADR 自研栈备料（WavLM/BigVGAN，仅旧管线用）：`python scripts/download_pretrained.py all`。

## 1. 安装自检（实测）

```powershell
python -c "import adr; print(adr.__version__)"   # 1.0.0
adr doctor                                        # 环境体检; 实测 8 ok 1 警告, --fix 后清零
```

## 2. 测试套件（实测：431 passed + 1 skipped，294.55s）

```powershell
$env:NUMBA_CACHE_DIR = "F:\ADR_data\numba_cache"
python -m pytest --basetemp=F:\ADR_data\tmp_pytest -p no:cacheprovider
```

默认跳过需下载模型的 ASR 真实推理用例（`pytest --run-asr` 显式开启）。

Rust 壳单测（实测 4 passed，增量 ~15s）：

```powershell
$env:PATH = "C:\Users\Admin\.cargo\bin;" + $env:PATH
cd desktop/src-tauri ; cargo test --lib
```

## 3. 服务启动与克隆合成（实测）

```powershell
python -m adr.server          # 默认 127.0.0.1:9881 (-p 改端口 / -a 改地址)
```

健康检查与合成（OpenAI 兼容面，音色档案即 model）：

```powershell
Invoke-RestMethod http://127.0.0.1:9881/api/adr/v1/health

Invoke-RestMethod -Uri http://127.0.0.1:9881/v1/audio/speech -Method Post `
  -ContentType "application/json" `
  -Body (@{ model = "我的声音V2"; input = "你好，这是克隆测试。" } | ConvertTo-Json) `
  -OutFile out.wav
# out.wav 应为 RIFF 头 wav (首次合成含模型冷启动数十秒, 之后流式秒级)
```

需要局域网调用：`$env:ADR_TTS_API_KEY = "<key>"` + `--addr 0.0.0.0`（无 key 时启动横幅打安全警告）。

## 4. 声音克隆训练复现（微调 + 门禁，实测链路）

```powershell
# 一条龙: 切片→ASR→特征→s2→s1 (产物在 third_party/gpt_sovits/{SoVITS,GPT}_weights_v2/)
python scripts/gsv_finetune.py <录音文件> --exp my_voice

# 另开终端, 训练门禁并行 (逐轮克隆打分 + 达标早停):
python scripts/train_gate.py --exp my_voice --target 0.80
#   曲线: output/train_gate_<exp>.jsonl (含逐句分 + zero_shot 基线行)
#   达标自动在 logs/<exp>/ 放 STOP → s2_train 每轮末检查后退出

# 训练完重打分 / 验证 (不轮询):
python scripts/train_gate.py --exp my_voice --once
```

要点：

- 门禁零样本基线**钉死官方预训练 s1/s2**（`adr/eval/probes.py` 的 `PRE_T2S/PRE_S2G`），不受 `tts_infer.yaml` custom 段漂移影响。
- 探测句集 5 句收敛在 [adr/eval/probes.py](adr/eval/probes.py)（长/中/短/混合/叙事），训练门禁 / held-out 评估 / 档案评估三处同源；`--texts` 可换自定义句集（基线自动同句集）。
- 参考音频超 10s 自动裁剪（去静音取前 8s，GSV 合成硬限制 3~10s）。
- 权重版本化：控制台档案选择器可见 vN/相似度/⚠退化标注；`GET /profiles/<name>/history`、`POST /profiles/<name>/rollback` 逐版回退。

## 5. 评估三件套（SECS / MOS / CER）

> **口径修正（批次 51）**：早期文档规划的 PESQ/ESTOI 不适用于零样本 TTS 评估——它们需要平行语料，度量的是信号保真而非音色保持。实际口径为：
> - **SECS**（客观，音色保持）：ERes2NetV2 说话人余弦相似度
> - **MOS**（主观，盲测）：A/B 盲听人评
> - **CER**（客观，可懂度）：faster-whisper 回译字错率

### 5.1 SECS 档案评估（实测 2026-10-10，CPU）

```powershell
python scripts/eval_secs.py --voice 我的声音V2 --cpu `
  --anchor third_party/gpt_sovits/output/slicer_opt/user_voice/8月15日.mp3_0000222400_0001656640.wav
```

实测输出（`output/eval_secs_<voice>.json` + `output/eval_secs_wavs/` 可试听）：

| 条件 | SECS |
| --- | --- |
| 真实语音锚点（真人切片 vs ref） | **0.9324** |
| zero_shot（官方预训练 s1+s2） | 0.7729 |
| finetuned（user_holdout_e8_s176） | 0.8052 |
| **Δ（微调增益）** | **+0.0323** |

经验区间（ERes2Net 中文）：`>0.70` 明显同一人；`0.55-0.70` 接近；`<0.45` 基本不同人。锚点分数高于两条合成条件，验证打分链路灵敏度正常。

### 5.2 MOS A/B 盲测（协议：随机混淆 + 盲听 + 揭晓）

```powershell
python scripts/mos_ab_test.py --voice <档案名>            # generate: 两系统 × 6 句, 随机 A/B 混淆 + score_sheet.csv
python scripts/mos_ab_test.py --score output/mos_ab/<时间戳目录>   # 听完打分填表后聚合揭晓
```

零样本侧同样钉死官方预训练权重，与门禁口径一致；`--texts` 换自定义句集，`--seed` 固定合成种子。

### 5.3 CER 可懂度（实测口径 CER ≤ 0.15 达标）

```powershell
python scripts/render_sample_pack.py    # 10 文本 × 2 参考音色, faster-whisper tiny (CPU/int8) 回译
```

CER 与打分解耦：`--score-only` 复用已合成 wav，不重跑合成。

## 6. 打包复现（实测）

```powershell
# 双包构建线 (批次 52): CPU/GPU runtime 变体隔离目录, 重建互不覆盖
powershell -File scripts\build_runtime.ps1             # → build\runtime-cpu (PyPI torch, Windows 轮子即 CPU 版)
powershell -File scripts\build_runtime.ps1 -AllowCuda  # → build\runtime-gpu (torch 2.10.0+cu130 + CUDA DLL, ~+2.5GB; 冒烟断言 CUDA 可用)

powershell -File scripts\build_installer.ps1           # CPU 包: 体检(runtime-cpu) + tauri + Inno Setup
powershell -File scripts\build_installer.ps1 -Gpu      # GPU 包: 体检(runtime-gpu) --allow-cuda + ISCC /DGPU_BUILD → 文件名带 -gpu
```

`check_runtime.py` 也可单独跑（build_installer.ps1 已内置为打包前置步骤，按变体自动指目录）：根级裸 .pth / 个人数据面 / pretrained_models / CUDA payload（nvidia_* 目录 + torch/lib/cudart64_*.dll）/ 体积告警（CPU >2.2GB / GPU >4.6GB）。

实测体积：CPU runtime **1.80GB** / GPU runtime **4.00GB**（v1.2.0 实测；GPU = +cu130 torch/CUDA DLL，RTX A2000 上 CUDA 冒烟通过）；安装包实测：v1.2.0 CPU **459.5MB** / GPU **1926.4MB**（v1.1.0 基线 473MB/1923.8MB；LZMA 对 CPU 轮子压得更狠，CUDA DLL 已预压缩难压所致差距）。安装包发布件见 [README](README.md) 123 云盘链接。

## 7. 环境冻结

- Python 3.13.7 + `build_freeze.txt`（开发环境全量版本钉子：torch 2.10.0+cu130、faster-whisper、librosa 等；安装包 runtime 的 CPU/GPU torch 装法见第 6 节双包构建线）
- 关键约定：`NUMBA_CACHE_DIR` 可写目录；pytest `--basetemp` 绕开沙箱 Temp 权限；HF 走 hf-mirror（`HF_ENDPOINT` 内建于 download 脚本）
