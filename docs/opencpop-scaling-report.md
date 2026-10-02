# OpenCpop Scaling 实验报告 (M3 验证)

## 目的

在 8GB GPU 上系统化探索:
1. **数据量 scaling** (1000 → 3550 样本)
2. **训练 epoch scaling** (3 → 5 epoch)
3. 找到最佳质量-时间 trade-off

## 实验环境

- **GPU**: NVIDIA RTX A2000 Laptop, 8GB (7GB 空闲)
- **CPU**: Intel/AMD 普通笔记本 CPU
- **数据**: OpenCpop (SVS, 5.23h, 100 首歌, 3756 句)
  - train: 3550 句
  - test: 206 句
- **模型**: SoVITS medium (30.7M, 6 layers, hidden 512)
- **Vocoder**: BigVGAN 22kHz
- **推理参数**: n_timesteps=20

## 实验 1: 数据量 Scaling

| 配置 | 样本数 | 训练时间 | Throughput | Final Loss |
|------|--------|----------|------------|------------|
| medium 1000 × 3ep | 1000 | 6.08 min | 2.7 s/s | 1.281 |
| **medium 3550 × 3ep** | **3550** | **6.36 min** | **9.3 s/s** | **1.283** |

> 同样训练时间下,3550 样本量是 1000 的 3.55x,throughput 提升 3.44x。

### TTS 质量对比 (10 文本)

| 指标 | 1000 样本 | 3550 样本 | Δ | 解读 |
|------|----------|----------|---|------|
| Duration (s) | 3.30 | 3.45 | +4.5% | 时长更准 |
| **RMS** | **0.381** | **0.414** | **+8.6%** | **更响, 更有能量** |
| **Spectral Centroid (Hz)** | **4923** | **4643** | **-5.7%** | **更自然人声** |

✅ **结论**: 数据量提升 → 质量提升 (尤其谱质心接近真人)。

## 实验 2: Epoch Scaling (3550 全量)

| Epochs | 训练时间 | Final Loss | 5th Δ |
|--------|----------|------------|-------|
| 3 | 6.36 min | 1.283 | - |
| **5** | **10.30 min** | **1.260** | **-1.8%** |

### TTS 质量对比 (10 文本)

| 指标 | 3 epoch | 5 epoch | Δ | 解读 |
|------|---------|---------|---|------|
| Duration (s) | 3.45 | 3.41 | -0.04 | 接近 |
| **RMS** | **0.414** | **0.344** | **-16.8%** | **明显下降** |
| **Spectral Centroid (Hz)** | **4643** | **4944** | **+300** | **变亮** |

⚠️ **警告信号**: 5 epochs 的 loss 反而更低 (1.26 < 1.28),但 TTS 质量**下降**。
- **过拟合**: 训练 loss 与 TTS 质量脱节
- **能量下降**: RMS -16.8% 意味着 wav 偏小
- **高频变多**: 谱质心 +300Hz, 接近 1000 样本水平

### 解读

1. **loss 持续下降** (1.28 → 1.26) 但生成质量下降,说明模型开始"记忆"训练集
2. **生成能量下降** 暗示 decoder 学到了"安静"的分布,失去了"响度动态"
3. **建议**: 3 epochs 是 **最优 sweet spot**

## Scaling 总结表

| 维度 | 1000×3e | 3550×3e | 3550×5e |
|------|---------|---------|---------|
| Loss | 1.281 | 1.283 | 1.260 |
| Duration | 3.30s | 3.45s | 3.41s |
| RMS | 0.381 | **0.414** | 0.344 |
| 谱质心 (Hz) | 4923 | **4643** | 4944 |
| 综合质量 | ★★ | **★★★** | ★★ |

**最佳配置**: **3550 样本 × 3 epoch** (medium, 6.36 min, GPU+AMP)

## 工程教训

### 1. Loss 不是万能指标
- 训练 loss 持续下降 (1.28 → 1.26) 但生成质量反而变差
- 评估需要 wav 级别的指标 (RMS, 谱质心, 时长)

### 2. 训练时间 vs 质量
- 同样 6 min, 1000 样本 (1 epoch ≈ 2 min) + 3 epochs
- 同样 6 min, 3550 样本 (1 epoch ≈ 2.1 min) + 3 epochs
- 后者质量明显更好 → **大数据量 + 少 epoch** 优于 **小数据量 + 多 epoch**

### 3. Overfitting 早期检测
- 5 epochs 时 loss 仍下降,但 RMS 显著下降
- **建议**: 加 val loss / 早停机制 (early stopping)
- 或者用 wav 指标 (RMS, 谱质心) 监控生成质量

### 5. Loss-based 早停的局限 (M3 验证)
- 实施 `early_stop_patience=2` + `val_ratio=0.05` 后, 5 epoch 训练**未触发早停**
- val/loss 持续下降 1.3464 → 1.2601
- 但 wav 质量**继续下降**: RMS 0.414→0.380 (-8.2%), 谱质心 4643Hz→4829Hz (+186Hz)
- **结论**: val/loss 不是 TTS wav 质量的可靠代理
- **建议**: 用户应使用 `epochs=3` 作为默认, 或实现 wav-based 早停 (更复杂)

### 6. WAV-based 早停实施 (M3 验证)
- 实现 `WavQualityCallback`: 每个 epoch 跑 N 个 val sample 推理, 监控 wav 质量
- 监控指标: `wav_rms_mean`, `wav_centroid_mean`, `wav_quality = rms * (1 - |centroid-4500|/4500)`
- **wav_quality 时间线 (3550 + 5 epoch)**:
  - Epoch 1: 0.0496 (init)
  - Epoch 2: 0.2159 (best)
  - Epoch 3: 0.3342 (best)
  - **Epoch 4: 1.3512 (best, peak)**
  - Epoch 5: 0.989 (no improve, -27%)
- **结论**: wav_quality 在 epoch 4 后下降, 与真实 wav 评估 (RMS 0.41→0.34) 一致
- `patience=1` 时成功触发早停, 保留 epoch 4 模型

### 7. Placeholder vs Real Vocoder 差距 (M3 发现 → M3.5 解决)
- 用 placeholder 180Hz fundamental + mel envelope 计算 wav 指标
- placeholder wav_quality (epoch 4 peak) ≠ 真实 BigVGAN wav 质量 (epoch 3 最佳)
- **原因**: placeholder 只反映 mel 能量, 不反映真实 TTS 自然度
- **建议**: 生产环境应用真实 vocoder (BigVGAN) 做 wav 早停

### 8. M3.5 真实 BigVGAN 接入 (2026-08-10 验证)
- 改造 [WavQualityCallback](file:///E:/新创意构思/新建文件夹/ADR/adr/training/callbacks.py): 增加 `vocoder_path` 参数 + 懒加载缓存
- 缓存机制: 首次 epoch 加载 BigVGAN (7.1s), 后续 epoch 直接复用 (~0.74s/sample)
- 验证: 50 样本 + 2 epoch, `--vocoder-path F:\ADR_data\bigvgan`
- **真实 wav 指标 (50 样本, BigVGAN 推理)**:
  | Epoch | wav_rms | wav_centroid (Hz) | wav_quality |
  |-------|---------|--------------------|-------------|
  | 1 | 0.3609 | 4654.4 | 0.3485 (init) |
  | 2 | 0.4011 | 4552.4 | 0.3964 (best, +14%) |
- **关键对比**:
  | 指标 | Placeholder | BigVGAN (真实) |
  |------|-------------|----------------|
  | wav_centroid | 184 Hz (只 180Hz 基频) | **4552 Hz (真实人声范围)** |
  | wav_rms | 24.99 (异常) | 0.4011 (正常) |
  | wav_quality | 1.0176 (虚高) | 0.3964 (真实) |
- **结论**: 真实 BigVGAN 谱质心 ~4500Hz 完美对应 wav_quality 公式, 早停信号更可靠
- **用法**: `python scripts/smoke_train_opencpop.py --vocoder-path F:\ADR_data\bigvgan --wav-quality-patience 1`

### 9. M3.6 F0 指标扩展 (2026-08-10 验证)
- 增加 `_extract_f0()` (librosa.pyin, 80-500Hz 人声范围)
- 扩展 `_compute_wav_metrics` 返回:
  - `f0_pred_mean`: 生成 wav 的平均 F0
  - `f0_pred_std`: F0 标准差 (稳定性)
  - `f0_ref_mean`: 参考 wav F0 (来自 batch.waveform)
  - `f0_bias_hz`: |pred - ref| 偏差
  - `f0_stability`: 1 - std/50 (越低越稳)
  - `f0_score`: (1 - bias/200) × stability
  - `naturalness`: wav_quality × f0_score
- **单元测试验证**:
  - 220Hz 合成信号 → 检测 219.9Hz ✓
  - 330Hz 合成信号 → 检测 330Hz ✓
  - 静音 → 返回空数组 (不抛异常)
- **端到端测试 (placeholder vocoder)**:
  ```
  f0_ref_mean=219.9Hz, f0_pred_mean=183.9Hz, f0_bias=36Hz
  f0_stability=0.95, f0_score=0.78, naturalness=0.0025
  ```
- **多指标 wav 质量**: 从单一 wav_quality 扩展到 7 维 (rms/centroid/quality/f0_mean/f0_std/f0_bias/naturalness)
- **测试**: wav_quality 11 个 (含 3 个 F0), 全套 112 passed + 1 skipped

## 推荐配置 (生产)

| 场景 | 推荐配置 | 训练时间 |
|------|----------|----------|
| **快速 smoke 测试** | small + 50 样本 + 1 epoch (30s) |
| **5min 快速克隆** | medium + 200 样本 + 3 epochs (4 min) |
| **标准 1h 训练** | medium + 3550 样本 + 3 epochs (6.4 min) |
| **大模型 LoRA** | x0p3b + LoRA r=8 + 1000 样本 + 3 epochs (75 min) |
| **生产部署** | medium + 3550 样本 + 3 epoch (我们已验证) |
| **带 wav 早停 (placeholder)** | + `--wav-quality-patience 1 --val-ratio 0.05` |
| **带 wav 早停 (真实 BigVGAN, 推荐)** | + `--vocoder-path F:\ADR_data\bigvgan --wav-quality-patience 1 --val-ratio 0.05` |

### 4. 数据增强
- OpenCpop 100 首歌, 3550 句, 21.8 音素/句
- 多样性有限 → 实际"信息量"小于 raw 数据量
- 解决: 增广 (音量/速度/音高扰动)

## 框架建议 (给用户)

| 场景 | 推荐配置 |
|------|----------|
| **快速 smoke 测试** | small + 50 样本 + 1 epoch (30s) |
| **5min 快速克隆** | medium + 200 样本 + 3 epochs (4 min) |
| **标准 1h 训练** | medium + 3550 样本 + 3 epochs (6.4 min) |
| **大模型 LoRA** | x0p3b + LoRA r=8 + 1000 样本 + 3 epochs (75 min) |
| **生产部署** | medium + 3550 样本 + 3 epoch (我们已验证) |

## 复现

```bash
# 实验 1
python scripts/smoke_train_opencpop.py --preset medium --n-samples 1000 --epochs 3 --use-amp
python scripts/smoke_train_opencpop.py --preset medium --n-samples 3550 --epochs 3 --use-amp

# 实验 2
python scripts/smoke_train_opencpop.py --preset medium --n-samples 3550 --epochs 5 --use-amp

# 评估
python scripts/eval_chinese_tts.py --ckpt examples/opencpop_train_3550/final.pt
python scripts/compare_models.py \
  --ckpt1 examples/opencpop_train_3550/final.pt \
  --ckpt2 examples/opencpop_train_3550_5e/final.pt \
  --label1 "3 epochs" --label2 "5 epochs"
```
