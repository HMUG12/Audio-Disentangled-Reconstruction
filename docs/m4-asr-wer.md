# M4 ASR-Based 早停 (WER/CER 可懂度评估)

## 概述

ADR 框架的 **M4 模块** 在 M3 wav 质量监控的基础上, 引入了 **基于 ASR (自动语音识别) 的可懂度评估**,
解决了 wav 信号层面"指标好看但听不清"的问题。

M4 的核心是 `ASRCallback`, 通过把合成 wav 转成文本, 计算 **CER (字符错误率)** 或 **WER (词错误率)**,
直接衡量"合成语音被 ASR 听对的程度", 这是更贴近用户感知的核心指标。

| 子模块 | 功能 | 状态 |
|--------|------|------|
| **M4** | ASRCallback (Faster-Whisper + WER/CER 早停) | ✅ |
| **M4.1** | 与 wav 早停 + 早停 3 级联动 (loss → wav → ASR) | ✅ |
| **M4.2** | `compute_cer` / `compute_wer` 独立工具函数 | ✅ |

---

## 1. 动机: 为什么需要 ASR-based 早停?

**问题**: Wav 质量指标 (RMS / 谱质心 / F0) 是 **信号层** 的客观度量, 但信号质量好 ≠ 可懂度高。

TTS 训练中常见的"伪高质量"陷阱:

| 模型状态 | wav_rms | 谱质心 | 听感 |
|----------|---------|--------|------|
| 收敛良好 | 0.41 | 4643Hz | 清晰可懂 |
| 局部模式崩坏 | 0.40 | 4600Hz | 含糊不清, ASR 完全识别不出 |
| 静音退化 | 0.05 | 0Hz | 完全无声 |

第三种情况, RMS 和谱质心都"看起来不对", 但前两种情况 wav 指标可能**完全无法区分**。
只有 ASR 转录出的文字 vs 真实文字的差异 (CER/WER) 才能直接量化可懂度。

**核心结论**: WER/CER 是 TTS 质量的"金标准"指标之一, 应该纳入早停信号。

---

## 2. 核心指标: CER vs WER

### 2.1 CER (Character Error Rate, 字符错误率)

```text
CER = edit_distance(ref_chars, hyp_chars) / len(ref_chars)
```

- **适用语言**: 中文/日文/韩文 (无空格分隔)
- **范围**: [0, +∞), 0 = 完全匹配
- **典型值**:
  - 优秀 TTS: CER < 0.05 (5% 以内)
  - 合格 TTS: CER 0.05-0.20
  - 不可懂: CER > 0.50

### 2.2 WER (Word Error Rate, 词错误率)

```text
WER = edit_distance(ref_words, hyp_words) / len(ref_words)
```

- **适用语言**: 英文等有空格分隔的语言
- **范围**: 同 CER

### 2.3 自动选择 (`compute_text_similarity`)

ADR 自动根据文本内容判断:
- 含中文字符 → CER
- 否则 → WER

```python
from adr.training.callbacks import compute_cer, compute_wer, compute_text_similarity

# 手动计算 CER
cer = compute_cer("你好世界", "你好世x")  # = 0.25 (1 char wrong / 4 chars)

# 手动计算 WER
wer = compute_wer("hello world", "hello there")  # = 0.5 (1 word wrong / 2 words)

# 自动选择
result = compute_text_similarity("你好世界", "你好世x", lang="auto")
# {"rate": 0.25, "mode": "cer", "n_ref": 4, "dist": 1}
```

---

## 3. 架构

### 3.1 ASRCallback 工作流

```
每个 epoch 结束:
  ↓
取 N 个 val sample (默认 5, 来自 trainer.val_loader)
  ↓
每个 sample:
  - 拿 reference text (batch.texts[i])
  - 跑推理 → pred_mel
  - vocoder (placeholder / BigVGAN) → wav
  - ASR (faster-whisper tiny) → hypothesis text
  - compute_cer(reference, hypothesis)
  ↓
聚合所有 sample 的 CER → 平均 asr_cer
  ↓
按 metric_name (默认 asr_cer) 跟踪最佳
  ↓
patience 轮未改善 → should_stop = True
  ↓
Trainer 跳出 epoch 循环
  ↓
final.pt = best_asr.pt (CER 最低的 checkpoint)
```

### 3.2 早停 3 级联动 (与 M3 协同)

ADR 支持 **3 级早停**, 用户可同时启用:

```bash
python scripts/smoke_train_opencpop.py \
    --asr-wer-patience 2 \      # ASR CER 早停 (M4)
    --wav-quality-patience 2 \  # wav 质量早停 (M3)
    --early-stop-patience 3 \   # val loss 早停 (M1)
    --vocoder-path F:/DR_data/bigvgan  # 真实 vocoder (M3.5)
```

**优先级** (final.pt 选择):
1. `best_asr.pt` (ASR CER 最低) - 最贴近可懂度
2. `best_wav.pt` (wav 质量最高) - 信号层
3. `final.pt` (最后一个 epoch) - 兜底

### 3.3 实现位置

[callbacks.py](file:///E:/新创意构思/新建文件夹/ADR/adr/training/callbacks.py) 的 `ASRCallback` 类 (line 566+):
- `_get_vocoder()`: 懒加载 BigVGAN
- `_get_asr()`: 懒加载 faster-whisper
- `_mel_to_wav()`: mel → wav (复用 WavQualityCallback 逻辑)
- `_asr_transcribe()`: wav → text (含 16kHz 重采样)
- `_compute_cer_metrics()`: 批量评估 CER/WER
- `on_epoch_end()`: 早停判定

---

## 4. 7 维输出指标

每个 epoch 结束后, ASRCallback 会输出以下指标到 `metrics`:

| 指标 | 含义 | 典型范围 |
|------|------|----------|
| `asr_cer` | 字符错误率 (主指标) | 0.0 - 1.0 |
| `asr_cer_n` | 参与 CER 计算的样本数 | 0 - n_samples |
| `asr_wer` | 词错误率 (英文场景) | 0.0 - 1.0 |
| `asr_wer_n` | 参与 WER 计算的样本数 | 0 - n_samples |
| `asr_primary` | 主指标值 (随 metric_name 变) | - |
| `asr_samples` | 前 3 个 sample 的 ref/hyp 详情 | list[dict] |

**训练日志示例**:

```text
=== Epoch 1/5 ===
  Epoch 1 done (5.0s): train/loss=1.4500
  [ASR] Loaded: tiny on cpu
  [ASR] epoch 1: asr_cer=0.4500 (init)
         ref='你好 世界' -> hyp='你 豪 世 介' (cer=0.500)
         ref='今天 天气 真 好' -> hyp='金 天 天 气 真 豪' (cer=0.300)

=== Epoch 2/5 ===
  [ASR] epoch 2: asr_cer=0.3800 (best)

=== Epoch 3/5 ===
  [ASR] epoch 3: asr_cer=0.3500 (best)

=== Epoch 4/5 ===
  [ASR] epoch 4: asr_cer=0.4200 (no improve 1/2, best=0.3500)

=== Epoch 5/5 ===
  [ASR] epoch 5: asr_cer=0.5000 (no improve 2/2, best=0.3500)
  [ASR] STOP at epoch 5, best asr_cer=0.3500
```

---

## 5. CLI 用法

### 5.1 快速启用 (推荐)

```bash
python scripts/smoke_train_opencpop.py \
    --data-dir data/opencpop_npz/train \
    --preset medium \
    --n-samples 1000 \
    --epochs 10 \
    --val-ratio 0.1 \
    --vocoder-path F:/DR_data/bigvgan \
    --asr-wer-patience 2 \
    --asr-model-size tiny
```

参数说明:
- `--asr-wer-patience N`: 启用 ASR 早停, N 个 epoch CER 不降则停 (0=关闭)
- `--asr-model-size {tiny,base,small,medium}`: ASR 模型大小, 默认 `tiny` (速度优先)
- `--asr-language zh`: ASR 语言, 默认 `zh` (中文)
- `--val-ratio 0.1`: 必须 ≥ 0.05 (ASR 需要 val 集), 默认 0.1
- `--vocoder-path`: 真实 BigVGAN 路径, 推荐提供以获得真实 ASR 评估

### 5.2 不同 ASR 模型的权衡

| 模型 | 大小 | 速度 (5 samples CPU) | 准确度 | 推荐场景 |
|------|------|----------------------|--------|----------|
| `tiny` | 39M | ~10s | 一般 | smoke test / 快速迭代 |
| `base` | 74M | ~30s | 较好 | 中等规模训练 |
| `small` | 244M | ~1min | 良好 | 精细调优 |
| `medium` | 769M | ~3min | 优秀 | 生产环境 |

**默认推荐 `tiny`**: smoke 训练每个 epoch 只需 ~10s, 可接受。

### 5.3 与 wav 早停 + loss 早停并用

```bash
# 3 级早停全开
python scripts/smoke_train_opencpop.py \
    --early-stop-patience 3 \      # loss 早停
    --wav-quality-patience 2 \     # wav 早停 (M3)
    --asr-wer-patience 2 \         # ASR 早停 (M4)
    --vocoder-path F:/DR_data/bigvgan \
    --val-ratio 0.1
```

任何一个触发都会停止训练, final.pt 会选最优先的 (best_asr > best_wav > last)。

---

## 6. 编程接口

### 6.1 基础用法

```python
from adr.training import Trainer, TrainerConfig
from adr.training.callbacks import ASRCallback, CheckpointCallback

# 1. 基础早停
cb = ASRCallback(
    n_samples=5,
    n_timesteps=10,
    patience=2,
    asr_model_size="tiny",
    asr_language="zh",
    use_real_vocoder=True,
    vocoder_path="F:/DR_data/bigvgan",
)
trainer = Trainer(model, ds, config, callbacks=[cb])
trainer.fit()
```

### 6.2 配合自定义 Checkpoint

```python
# 同时保存 best_asr.pt
asr_ckpt = CheckpointCallback(
    save_dir="output/checkpoints",
    save_every_n_epochs=99,
    save_best=True,
    metric_name="asr_cer",
    mode="min",
)
asr_ckpt.best_path_name = "best_asr.pt"

# 替换 on_epoch_end 强制存为 best_asr.pt
def _save_asr_best(trainer, epoch, metrics, **kw):
    if "asr_cer" in metrics and (
        asr_ckpt.best_value is None
        or metrics["asr_cer"] < asr_ckpt.best_value - 0.001
    ):
        asr_ckpt.best_value = metrics["asr_cer"]
        from pathlib import Path
        Path("output/checkpoints/best_asr.pt").parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint("output/checkpoints/best_asr.pt")

asr_ckpt.on_epoch_end = _save_asr_best
trainer.callbacks = [cb, asr_ckpt]
```

### 6.3 手动计算 CER/WER

```python
from adr.training.callbacks import compute_cer, compute_wer

# CER (中文)
cer = compute_cer("今天天气真好", "金天天气真好")
# = 1/6 = 0.1667

# WER (英文)
wer = compute_wer("the quick brown fox", "the quick brown box")
# = 1/4 = 0.25
```

---

## 7. 性能开销

### 7.1 时间开销 (典型配置)

| 组件 | CPU | GPU |
|------|-----|-----|
| 推理 5 samples (medium model) | 3s | 1s |
| BigVGAN vocoder 5 samples | 2s | 0.5s |
| ASR tiny 5 samples | 10s | 2s |
| ASR small 5 samples | 60s | 10s |
| **总计 (tiny)** | **~15s/epoch** | **~3s/epoch** |
| **总计 (small)** | **~65s/epoch** | **~12s/epoch** |

**建议**:
- smoke test / 5 min 训练: 用 `tiny` (5 samples/epoch)
- 长时间训练: 用 `small` (更准确, 但 1min/epoch 仍可接受)

### 7.2 内存开销

- faster-whisper `tiny`: ~400MB RAM
- faster-whisper `small`: ~1GB RAM
- faster-whisper `medium`: ~3GB RAM

ASR 模型**不会常驻 GPU**, 用 int8 (CPU) / float16 (GPU) 加载, 评估完即释放 cache。

---

## 8. 验证与测试

### 8.1 单元测试 (32 tests)

```bash
python -m pytest tests/test_asr_callback.py -v
```

覆盖:
- `TestCER`: CER 各种边界 (完美匹配, 1 字符错, 全部错, 空字符串)
- `TestWER`: WER 各种边界 (分词, 标点, 大小写)
- `TestASRCallbackConstruction`: 参数验证, 旧别名兼容
- `TestASRCallbackRunWithoutASR`: 无 ASR 环境 (fallback skip)
- `TestASRCallbackWithMockASR`: Mock ASR, 验证 CER 改善逻辑
- `TestASRCallbackEarlyStop`: 早停触发逻辑

### 8.2 端到端验证 (已执行)

**真实验证结果** (2026-08-11, [smoke_m4_real_vocoder.py](file:///E:/新创意构思/新建文件夹/ADR/scripts/smoke_m4_real_vocoder.py)):

| 组件 | 结果 |
|------|------|
| BigVGAN 真实 vocoder | ✅ wav_quality epoch2 峰值 0.4697, 谱质心 3286→4378Hz 持续逼近 4500Hz |
| F0 诊断 | ✅ 正确识别未收敛模型 (f0_pred=86.3Hz 单调 vs ref=278.5Hz) |
| ASR 早停 | ✅ CER 恒 1.0 → patience=2 → epoch 3 正确停止 |
| CUDA→CPU 回退 | ✅ 真实环境触发 (cublas 缺失) 并自动恢复 |

**ASR 管线 sanity check** ([check_asr_sanity.py](file:///E:/新创意构思/新建文件夹/ADR/scripts/check_asr_sanity.py)):
用真实 OpenCpop 人声声乐走同一 ASR 路径:

| utt | ref | hyp | CER |
|-----|-----|-----|-----|
| 2001000001 | 感受停在我发端的指尖 | 感受听在我发短的之间 | 0.40 |
| 2001000003 | 记住望着我坚定的双眼 | 记住忘着我肩定的声言 | 0.40 |

CER 0.4-0.6 是 whisper-tiny 在**歌声**上的正常水平 (歌声比说话难识别, 错误均为同音字: 停→听, 端→短, 尖→间)。
**结论**: e2e 中 CER=1.0 纯粹是 80 样本 × 3 epoch 的 2.4M 未收敛模型的真实质量反映, 管线本身正确。

**注意**: ASR 评估歌声场景建议 `small` 以上模型, `tiny` 仅适合 smoke test。

---

## 9. 故障排查

### 9.1 `ModuleNotFoundError: No module named 'faster_whisper'`

**解决**: `pip install faster-whisper`

### 9.2 ASR 模型下载慢 / 失败

**症状**: 首次运行卡在 "Downloading model..." 几分钟

**解决**:
1. 设置代理: `export HF_ENDPOINT=https://hf-mirror.com`
2. 手动下载到 `~/.cache/huggingface/hub/`
3. 离线使用: `local_files_only=True` (TODO)

### 9.3 CER 永远是 1.0 (完全没改善)

**可能原因**:
- vocoder 加载失败, ASR 转录的是 placeholder 噪声
- ASR 语言不对 (中文文本用 `asr_language="en"`)
- 模型本身没收敛 (loss 还在高位)

**排查**:
```python
# 手动测试 ASR
from faster_whisper import WhisperModel
asr = WhisperModel("tiny", device="cpu", compute_type="int8")
segs, _ = asr.transcribe("test.wav", language="zh")
print(" ".join(s.text for s in segs))
```

### 9.4 OOM (CUDA Out of Memory)

**症状**: 训练 GPU OOM, 但 M3 训练不 OOM

**原因**: ASR 模型占用额外 GPU 显存

**解决**:
- 用 `tiny` 模型 (最小显存)
- 设置 `compute_type="int8"` 强制 CPU
- 减少 `n_samples`

### 9.5 `RuntimeError: Library cublas64_12.dll is not found`

**症状**: CUDA 上 faster-whisper 转录时报 cublas/cudnn 缺失

**解决**: 已内置自动回退 — `ASRCallback` 检测到 `RuntimeError` 后会**自动重载 CPU int8 版 ASR 并重试一次** (只回退一次, 不会反复重试)。
若 CPU 版也失败, 检查 `pip install faster-whisper` 是否完整。

---

## 10. 与 M3 的对比

| 维度 | M3 (Wav Quality) | M4 (ASR) |
|------|------------------|----------|
| 指标 | RMS / 谱质心 / F0 | CER / WER |
| 评估内容 | wav 信号质量 | 可懂度 (语义) |
| 速度 | 极快 (~1s/epoch) | 慢 (~15s/epoch with tiny) |
| 额外依赖 | 无 (只用 numpy) | faster-whisper (~40MB) |
| 适用场景 | 信号层早停 (首选) | 语义层早停 (高保真) |
| 推荐度 | ⭐⭐⭐⭐⭐ 必开 | ⭐⭐⭐⭐ 中长时间训练开启 |

**最佳实践**:
- **5 sec 克隆 / 5 min 微调**: M3 足够, M4 可选
- **长时间训练 (>1h)**: M3 + M4 双开, 互补信号
- **生产环境**: M3 + M4 + loss 早停全开

---

## 11. 相关文件

| 文件 | 作用 |
|------|------|
| [callbacks.py](file:///E:/新创意构思/新建文件夹/ADR/adr/training/callbacks.py) | ASRCallback + CER/WER 工具函数 |
| [smoke_train_opencpop.py](file:///E:/新创意构思/新建文件夹/ADR/scripts/smoke_train_opencpop.py) | CLI 集成 (`--asr-wer-patience`) |
| [test_asr_callback.py](file:///E:/新创意构思/新建文件夹/ADR/tests/test_asr_callback.py) | 32 个单元测试 |
| [m3-wav-quality.md](file:///E:/新创意构思/新建文件夹/ADR/docs/m3-wav-quality.md) | M3 wav 早停文档 (互补) |
| [implementation-plan.md](file:///E:/新创意构思/新建文件夹/ADR/docs/implementation-plan.md) | 总体路线图 |

---

## 12. 下一步

- **M6**: CER 改善幅度作为学习率 warm restart 触发器
- **M7**: 多 ASR 集成 (tiny + small 投票), 提升评估稳定性
- **M8**: 端到端 5min finetune demo 集成 ASR 早停
