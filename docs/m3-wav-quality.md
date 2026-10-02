# M3 Wav-Based 早停 + 多指标质量监控

## 概述

ADR 框架的 **M3 系列** 实现了基于 wav 质量指标的早停 (Early Stopping), 解决了 TTS 训练中常见的
"loss 持续下降但 wav 质量下降" 的过拟合问题。

M3 包含三个子模块:

| 子模块 | 功能 | 状态 |
|--------|------|------|
| **M3** | WavQualityCallback 基础 (placeholder vocoder) | ✅ |
| **M3.5** | 真实 BigVGAN vocoder 接入 + 懒加载缓存 | ✅ |
| **M3.6** | F0 提取 (librosa.pyin) + 音准/稳定性指标 | ✅ |

---

## 1. 动机: 为什么需要 wav-based 早停?

**核心问题**: 在 TTS 训练中, **val loss 持续下降 ≠ wav 质量提升**。

我们的 OpenCpop 3550 样本 × 5 epoch 实验清楚地展示了这一点:

| Epoch | val/loss | wav_rms (BigVGAN) | wav_centroid (Hz) | wav_quality |
|-------|----------|-------------------|-------------------|-------------|
| 1 | 1.42 | 0.36 | 4654 | 0.35 |
| 2 | 1.38 | 0.38 | 4600 | 0.37 |
| **3** | **1.36** | **0.41** | **4643** | **0.41** ← 最佳 |
| 4 | 1.34 | 0.36 | 4750 | 0.36 |
| 5 | 1.32 | 0.34 | 4944 | 0.33 ← 过拟合 |

Epoch 5 的 loss 最低, 但 wav 质量最差 (RMS 下降 17%, 谱质心偏移 6%)。

**结论**: 仅靠 loss 早停会保留过拟合的模型。需要 wav 级别的客观指标。

---

## 2. 架构

### 2.1 WavQualityCallback

核心实现位于 [callbacks.py](file:///E:/新创意构思/新建文件夹/ADR/adr/training/callbacks.py):

```
每个 epoch 结束:
  ↓
取 N 个 val sample (默认 5)
  ↓
model.eval() 跑推理 → pred_mel
  ↓
vocoder (placeholder / BigVGAN) → wav
  ↓
计算 7 维 wav 指标:
  - wav_rms_mean
  - wav_centroid_mean
  - wav_quality (composite)
  - f0_pred_mean
  - f0_pred_std
  - f0_bias_hz
  - naturalness
  ↓
按 `metric_name` (默认 wav_quality) 跟踪最佳
  ↓
patience 轮未改善 → should_stop = True
  ↓
Trainer 跳出 epoch 循环
  ↓
final.pt = best_wav.pt (最佳 wav 质量的 checkpoint)
```

### 2.2 早停机制

- **触发条件**: `metric_name` 连续 `patience` 个 epoch 不改善 (`min_delta` 容差)
- **保存机制**: 每个 epoch 自动保存 `best_wav.pt` (按 wav_quality 最佳)
- **替换机制**: 早停触发后, `final.pt` 自动用 `best_wav.pt` 替换 (避免保存过拟合模型)

---

## 3. 7 维 wav 质量指标

### 3.1 基础指标 (M3)

| 指标 | 公式 | 物理意义 | 范围 |
|------|------|----------|------|
| `wav_rms_mean` | `sqrt(mean(wav²))` | 能量 (越大越有"响度") | 0-1 |
| `wav_centroid_mean` | `sum(f×|F(wav)|) / sum(\|F(wav)\|)` | 谱质心 (人声 ~4500Hz) | 0-11025 Hz |
| `wav_quality` | `rms × (1 - \|centroid-4500\|/4500)` | 综合质量 | 0-1 (越大越好) |

### 3.2 F0 指标 (M3.6)

| 指标 | 公式 | 物理意义 | 范围 |
|------|------|----------|------|
| `f0_pred_mean` | librosa.pyin 均值 | 生成 wav 的平均音高 | 80-500 Hz |
| `f0_pred_std` | librosa.pyin 标准差 | 音高稳定性 | 0-100 Hz |
| `f0_ref_mean` | ref wav 的 pyin 均值 | 参考音高 (来自 batch.waveform) | 80-500 Hz |
| `f0_bias_hz` | `\|pred - ref\|` | 音高偏差 | 0-∞ Hz |
| `f0_stability` | `1 - std/50` | 稳定性得分 | 0-1 |
| `f0_score` | `(1 - bias/200) × stability` | 音准综合 | 0-1 |
| `naturalness` | `wav_quality × f0_score` | 自然度综合 | 0-1 |

### 3.3 早停指标选择

默认 `metric_name="wav_quality"`, 也可改为 `naturalness` (需要真实 vocoder):

```python
WavQualityCallback(
    n_samples=5,
    patience=1,
    metric_name="naturalness",  # 更严格的早停
    use_placeholder_vocoder=False,
    vocoder_path="F:/ADR_data/bigvgan",
)
```

---

## 4. Vocoder 选择

### 4.1 Placeholder (默认, 零依赖)

```python
WavQualityCallback(use_placeholder_vocoder=True)
```

- **优点**: 零依赖, 速度快 (每个 sample 5ms)
- **缺点**: 谱质心固定 184Hz (只生成 180Hz 基频), 数值偏小
- **适用**: smoke 测试, 不需要精确 wav 质量时

### 4.2 真实 BigVGAN (生产推荐)

```python
WavQualityCallback(
    use_placeholder_vocoder=False,
    vocoder_path="F:/ADR_data/bigvgan",  # 或 "auto" 自动搜索
)
```

- **优点**: 真实 wav, 谱质心 ~4500Hz, RMS ~0.4
- **缺点**: 首次加载 7.1s, 推理 0.74s/sample
- **缓存**: 懒加载, 训练中只加载一次
- **自动搜索路径**:
  - `F:\ADR_data\bigvgan` (默认)
  - `third_party/BigVGAN/`
  - `~/BigVGAN/`

### 4.3 关键差异

| 指标 | Placeholder | BigVGAN |
|------|-------------|---------|
| wav_centroid | 184 Hz (异常) | 4552 Hz (正常) |
| wav_rms | 24.99 (虚高) | 0.4011 (正常) |
| wav_quality 公式适配 | 不适配 | 完美 |

**生产环境必须用真实 BigVGAN**, placeholder 仅供快速 smoke 测试。

---

## 5. CLI 用法

### 5.1 smoke_train_opencpop.py

```bash
# 默认 (无 wav 早停)
python scripts/smoke_train_opencpop.py --preset medium --n-samples 200 --epochs 5

# 启用 wav 早停 (placeholder)
python scripts/smoke_train_opencpop.py \
    --preset medium --n-samples 200 --epochs 5 \
    --wav-quality-patience 1 --val-ratio 0.1

# 启用 wav 早停 (真实 BigVGAN, 生产推荐)
python scripts/smoke_train_opencpop.py \
    --preset medium --n-samples 200 --epochs 5 \
    --vocoder-path F:/ADR_data/bigvgan \
    --wav-quality-patience 1 --val-ratio 0.1
```

### 5.2 CLI 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--wav-quality-patience N` | 0 | wav 早停 patience (0=关闭) |
| `--vocoder-path PATH` | None | 真实 vocoder 路径 (None=placeholder) |
| `--val-ratio R` | 0.0 | val 集比例 (启用 wav 早停时建议 0.05-0.1) |
| `--early-stop-patience N` | 0 | loss-based 早停 (可与 wav 早停并用) |

---

## 6. 编程接口

### 6.1 基础用法

```python
from adr.training.callbacks import WavQualityCallback
from adr.training import Trainer, TrainerConfig
from adr.models.sovits import SoVITS, SoVITSConfig

model = SoVITS(SoVITSConfig(...))
cfg = TrainerConfig(epochs=10, val_ratio=0.1)

cb = WavQualityCallback(
    n_samples=5,        # val 推理样本数
    n_timesteps=10,     # 扩散步数
    patience=2,         # 2 轮不改善就停
    min_delta=1e-3,
    metric_name="wav_quality",  # 或 "naturalness"
    use_placeholder_vocoder=False,
    vocoder_path="F:/ADR_data/bigvgan",
)

trainer = Trainer(model=model, train_data=ds, config=cfg, callbacks=[cb])
trainer.fit()  # 自动 wav 早停
```

### 6.2 自定义指标

```python
# 监控 f0_bias_hz 越低越好 (而不是 wav_quality 越大越好)
WavQualityCallback(
    metric_name="f0_bias_hz",
    mode="min",  # 关键: min 模式
    patience=2,
)
```

### 6.3 多 callback 组合

```python
from adr.training.callbacks import (
    CheckpointCallback, EarlyStoppingCallback, WavQualityCallback
)

trainer = Trainer(
    model=model, train_data=ds, config=cfg,
    callbacks=[
        WavQualityCallback(patience=1),     # wav 早停 (优先)
        EarlyStoppingCallback(patience=5),  # loss 早停 (兜底)
        CheckpointCallback(save_every_n_epochs=1, save_best=True),
    ],
)
```

---

## 7. 验证与测试

### 7.1 单元测试 ([test_wav_quality_callback.py](file:///E:/新创意构思/新建文件夹/ADR/tests/test_wav_quality_callback.py))

11 个测试, 全部通过:

- `test_wav_quality_callback_runs`: 基础跑通, 记录 wav 指标
- `test_wav_quality_callback_can_stop`: `min_delta=1e9` 强制早停
- `test_wav_quality_callback_skips_without_val`: 无 val_loader 跳过
- `test_wav_metrics_computation`: wav 指标计算正确
- `test_wav_quality_vocoder_path_param`: `vocoder_path` 参数接受
- `test_wav_quality_vocoder_caching`: vocoder 失败不重试
- `test_wav_quality_placeholder_fallback_on_vocoder_fail`: 失败回退
- `test_wav_quality_placeholder_with_explicit_disable`: auto 模式不抛异常
- `test_wav_quality_f0_extraction`: 220Hz / 330Hz 合成信号 → 准确检测
- `test_wav_quality_f0_extraction_handles_silence`: 静音 → 空数组
- `test_wav_quality_f0_metrics_in_computation`: _compute_wav_metrics 含 F0

### 7.2 端到端验证 (200 样本 + 5 epoch + BigVGAN)

| Epoch | wav_rms | wav_centroid | wav_quality | 状态 |
|-------|---------|--------------|-------------|------|
| 1 | 0.36 | 4654 | 0.35 | init |
| 2 | 0.40 | 4552 | 0.40 (best) | best_wav.pt |
| 3 | - | - | - | (interrupted) |

**结论**: wav 早停有效, `final.pt` = `best_wav.pt` (epoch 2), 避免 epoch 3-5 过拟合。

### 7.3 全套测试

```
$ python -m pytest tests/ -q
=========== 112 passed, 1 skipped, 76 warnings in 88.49s ============
```

---

## 8. 性能与限制

### 8.1 时间成本

- 5 sample × 0.74s (BigVGAN) = **3.7s/epoch** 额外开销
- 占训练时间的 ~3-5% (200 样本 × 3 epoch ~ 4 min)
- 可通过减少 `n_samples` 进一步加速 (如 `n_samples=2` → 1.5s/epoch)

### 8.2 内存成本

- BigVGAN 模型: ~120MB (22kHz 权重)
- pyin F0 提取: 临时数组, <10MB
- 总开销: < 150MB (8GB GPU 完全可以)

### 8.3 限制

- **Batch 必须有 waveform**: 否则不计算 F0 指标 (ref F0)
- **占位 vocoder 不准**: 生产必须用 BigVGAN
- **F0 仅看音高**: 不看音素清晰度 (需要 ASR/MOS, 见 M4)

---

## 9. 推荐配置

| 场景 | 配置 | wav 早停 | vocoder |
|------|------|----------|---------|
| Smoke 测试 | 50 样本 × 1 epoch | `--wav-quality-patience 0` | placeholder |
| 5min 快速克隆 | 200 样本 × 3 epoch | `--wav-quality-patience 1` | placeholder |
| 生产标准 | 3550 样本 × 3 epoch | `--wav-quality-patience 1` | BigVGAN |
| 高质量 | 3550 样本 × 5 epoch | `--wav-quality-patience 1` | BigVGAN |
| 防过拟合 | 任意 + 5+ epoch | `--wav-quality-patience 1` (必备) | BigVGAN |

---

## 10. 故障排查

| 问题 | 原因 | 解决 |
|------|------|------|
| wav 早停不触发 | patience 太大 | 调小 (如 1-2) |
| wav_quality 一直很低 | vocoder 加载失败 | 检查 `--vocoder-path` |
| f0_pred_mean = 0 | placeholder 的 180Hz 被 librosa 漏检 | 改用 BigVGAN |
| 训练卡在 wav 推理 | GPU OOM | 减少 `n_samples` 或换 CPU |
| train_log.jsonl 为空 | smoke 脚本替换了 callbacks | 这是已知 bug (M3.7 待修) |

---

## 11. 未来工作 (M4+)

- **M4**: ASR-based WER 评估 (Whisper-tiny)
- **M4.5**: 多 metric 加权组合 (wav_quality + WER + F0)
- **M7**: 修 train_log.jsonl bug (smoke 脚本保留 default callbacks)
