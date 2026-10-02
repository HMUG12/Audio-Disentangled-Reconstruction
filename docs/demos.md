# ADR 端到端 Demo 指南

> **目标**: 让用户在不读源码的情况下,5 分钟内跑通任意 demo 并理解输出。

ADR 框架提供 **3 个递进式 demo**,覆盖从"零训练基线"到"LoRA 训练 + 量化部署"的完整链路:

| Demo | 训练? | 模型规模 | 耗时 (CPU) | 显存 | 适用场景 |
|------|-------|---------|-----------|------|---------|
| [01_5sec_clone](#1-5sec_clone-5-秒克隆) | 否 (基座) | 默认 | < 5s | < 1GB | 验证 pipeline 通畅 |
| [02_5min_finetune](#2-5min_finetune-5-分钟微调) | 是 (3 epoch) | small (1-2M) | ~50s | < 2GB | 一键体验克隆效果 |
| [03_lora_finetune](#3-lora_finetune-lora-微调) | 是 (LoRA r=8) | medium→x0p3b | 1-3 min | 0.1-1.9GB (QLoRA 0.8GB) | 低显存训练演示 |

---

## 0. 通用准备

### 0.1 依赖

```bash
# 已包含在 requirements.txt
pip install -e .             # 安装 ADR 框架
# 大型依赖(可选,影响性能):
pip install torch torchvision torchaudio
pip install librosa          # F0 提取 (M3.6)
pip install faster-whisper   # ASR (可选,01/02/03 默认不依赖)
```

### 0.2 验证安装

```bash
python -c "from adr.core import setup_device; setup_device(verbose=True)"
# 应输出: ADR device auto-selected: cuda / cpu
```

### 0.3 数据集

- **不需要任何外部数据**: 所有 demo 自带 ref.wav 或可生成。
- **真实数据实验 (进阶)**: `examples/opencpop_train_3550/` 是用 OpenCpop 3550 句 (5.23h) 训出的模型,可作为"高质量参考"。

---

## 1. 01_5sec_clone (5 秒克隆)

### 1.1 作用

**最简端到端**: ref + text → wav,不训练。验证 pipeline 跑通。

### 1.2 命令

```bash
# 1) 准备 ref (≥ 5 秒 wav)
python examples/02_5min_finetune/gen_ref.py --out ref.wav --duration 5

# 2) 一行克隆
python examples/01_5sec_clone/main.py \
    --ref ref.wav \
    --text "你好世界,欢迎使用 ADR 框架" \
    -o cloned.wav
```

### 1.3 输出

| 项目 | 值 |
|------|---|
| 耗时 (CPU) | 3-5 秒 |
| 显存占用 | < 1GB |
| 输出文件 | `cloned.wav` (mono, 22050Hz) |
| wav 质量 | 噪声/占位 (基座未训练) |

### 1.4 进阶:用训练好的 checkpoint

```bash
# 用 02_5min_finetune 训出的 best.pt 推理
python examples/01_5sec_clone/main.py \
    --ref ref.wav \
    --text "你好世界" \
    -c examples/02_5min_finetune/output/workdir/train/checkpoints/epoch_0003.pt
```

### 1.5 源码结构 ([main.py](file:///e:/新创意构思/新建文件夹/ADR/examples/01_5sec_clone/main.py))

```
Phase 1: 构造 pipeline (InferPipeline.from_checkpoint 或 base)
Phase 2: synthesize(text, ref_path)
Phase 3: save_wav(wav, output_path)
```

---

## 2. 02_5min_finetune (5 分钟微调)

### 2.1 作用

**完整 finetune 流程**: 切片 → G2P → F0/mel → 训练 → 推理。CPU 50 秒跑通。

### 2.2 命令

```bash
# 一键演示 (含数据生成)
python examples/02_5min_finetune/main.py \
    --ref ref.wav \
    --text "你好世界" \
    --train-texts "你好世界,这是 ADR 框架,声纹克隆很简单"
```

参数:

| 参数 | 默认 | 说明 |
|------|------|------|
| `--ref` | 必填 | 参考音频 (≥ 5s) |
| `--text` | 必填 | 要克隆的文本 |
| `--train-texts` | 默认 10 句 | 训练文本 (逗号分隔) |
| `--n-segments` | 6 | 切片数 |
| `--segment-sec` | 3.0 | 每段秒数 |
| `--epochs` | 3 | 训练 epoch |
| `--batch-size` | 2 | batch size |
| `--n-timesteps` | 10 | 扩散步数 |
| `--output` | output/cloned.wav | 输出路径 |
| `--monitor` | wav | **M8** 监控级别: `none` / `wav` / `asr` / `full` |
| `--monitor-patience` | 2 | 监控早停 patience |
| `--asr-ensemble` | 无 | **M7** 多 ASR 集成 (如 `tiny,small`) |
| `--max-restarts` | 1 | **M6** 最大热重启次数 (仅 full) |
| `--vocoder-path` | 自动 | BigVGAN 路径 (默认自动搜索) |

### 2.3 M8 四级监控

`--monitor` 控制训练质量监控 (M3+M4+M6+M7 一体化):

| 级别 | 启用模块 | 效果 |
|------|----------|------|
| `none` | 无 | 固定 epochs (旧行为) |
| `wav` | **M3** | wav_quality 早停 + `best_wav.pt` |
| `asr` | M3 + **M4**(+M7) | + CER 早停 + `best_asr.pt` |
| `full` | M3 + M4 + **M6** | + wav_quality 平台期热重启 |

```bash
# 完整监控示例 (推荐 GPU + BigVGAN)
python examples/02_5min_finetune/main.py \
    --ref ref.wav --text "你好世界" \
    --monitor full --epochs 8
```

**实测日志** (6 epochs, monitor=full, BigVGAN):

```text
[M3] WavQuality 监控: patience=2, vocoder=BigVGAN
[M4] ASR CER 监控: patience=3 (tiny)
[M6] 热重启: metric=wav_quality, max_restarts=1
=== Epoch 2/6 ===
  [WavQuality] wav_quality=0.4137 (best)     ← 质量峰值
=== Epoch 3/6 ===
  [WavQuality] STOP at epoch 3, best=0.3899  ← M3 正确检测过拟合
  [WarmRestart] #1/1 at epoch 3: lr 1.2e-3 -> 6.0e-4
[M8] 选用 best_wav.pt (质量峰值 checkpoint)
[M8] CER 未改善 (恒 1.0), best_asr.pt 不优先, 回退 best_wav/best
```

**checkpoint 选择优先级**: `best_asr.pt` > `best_wav.pt` > `best.pt` > last。
特例: CER 恒 1.0 (模型未收敛到可懂) 时 best_asr 只是 epoch1 存档,
自动回退到 best_wav。

### 2.4 实际验证数据 (CPU)

| 阶段 | 耗时 | 备注 |
|------|------|------|
| 数据准备 | 38.7s | 切片 + G2P + F0 + mel |
| 训练 (3 epoch) | 6.0s | small model (~1M params) |
| 推理 (10 timesteps) | 0.6s | BigVGAN vocoder |
| **总计** | **51.0s** | ✅ ≤ 1 min |

输出 wav 听感验证 ([validate.py](file:///e:/新创意构思/新建文件夹/ADR/examples/02_5min_finetune/validate.py)):

| 指标 | 阈值 | 实测 |
|------|------|------|
| RMS > 0.05 | 有能量 | 0.182 ✅ |
| Peak > 0.5 | 有响度 | 0.949 ✅ |
| Spectral centroid 接近 ref | ± 500Hz | 差 138Hz ✅ |
| 谐波比 | 非噪声 | 0.007 ✅ |

### 2.5 输出

- `output/cloned.wav` — 合成结果
- `output/workdir/samples/*.npz` — 训练样本
- `output/workdir/train/checkpoints/*.pt` — checkpoint
- `output/workdir/train/train_log.jsonl` — 训练日志
- `output/comparison.png` — ref vs cloned 频谱对比
- `output/validation_report.txt` — 听感验证报告
- `output/metadata.json` — 完整元数据

### 2.6 源码结构 ([main.py](file:///e:/新创意构思/新建文件夹/ADR/examples/02_5min_finetune/main.py))

```python
# Phase 1: 数据准备
prepare_training_data(ref, train_texts, output_dir, n_segments=6, segment_sec=3.0)
  # 1.1 加载 ref audio
  # 1.2 初始化 G2P (pypinyin) + F0 (pyin)
  # 1.3 循环切 6 段 × 3s, 配文本, 生成 TrainSample.npz

# Phase 2+3: 模型构建 + 训练
train_finetune(npz_paths, output_dir, epochs=3, batch_size=2)
  # SoVITS(hidden_dim=64, n_layers=2)  # ~1M params
  # Trainer.fit()  # AMP bf16 + grad ckpt

# Phase 4: 推理 (BigVGAN 真实 vocoder 优先)
synthesize_clone(checkpoint, ref, text, output, n_timesteps=10)
```

### 2.7 故障排查

| 现象 | 原因 | 解决 |
|------|------|------|
| BigVGAN 未加载 | F 盘数据未挂载 | 自动 fallback 到 placeholder, 不影响 demo |
| 训练 loss 不下降 | small model 容量小 | 增加 `--n-segments 12` 或 `--epochs 5` |
| wav 是噪声 | ref 质量差/未训练完 | 换更清晰的 ref, 增加训练轮数 |

---

## 3. 03_lora_finetune (LoRA 微调)

### 3.1 作用

**M2 训练栈演示**: LoRA 注入、QLoRA 4-bit 量化、显存自适应、LoRA state dict 持久化。

### 3.2 命令

```bash
# 默认 medium preset (~30M) + LoRA r=8
python examples/03_lora_finetune/main.py \
    --ref ref.wav \
    --text "你好世界,这是 LoRA finetune 测试" \
    --preset medium \
    --lora-rank 8 \
    --epochs 2
```

参数:

| 参数 | 默认 | 说明 |
|------|------|------|
| `--preset` | medium | small/medium/large/x0p3b/x0p4b |
| `--lora-rank` | 8 | LoRA rank |
| `--lora-alpha` | 16 | LoRA alpha |
| `--quantize-4bit` | False | QLoRA 4-bit 量化基座 |
| `--no-finetune` | False | 跳过训练,只对比基线 |
| `--n-samples` | 8 | 训练样本数 |
| `--epochs` | 2 | 训练 epoch |
| `--device` | auto | cpu/cuda/auto |

### 3.3 预设规模 (GPU 显存基准实测)

| Preset | 参数量 | 全量峰值 | LoRA r=8 峰值 | QLoRA r=8 峰值 |
|--------|--------|----------|---------------|-----------------|
| **medium** | 30.7M | 620 MB | 227 MB | **136 MB** |
| **x0p3b** | 338M | **6470 MB** | **1928 MB** | **809 MB** |
| x0p4b | ~450M | (8GB 极限) | ~3GB | ~1.2GB |

> **结论**: **0.3B 模型 + 真 QLoRA 在 8GB GPU 仅占 0.8GB**,有 7GB+ 余量给 batch。
> 全量训练 0.3B 需要 6.5GB,卡在边界。

### 3.4 实际验证数据 (CPU medium)

- 8 样本 × 2 epoch × 30M 模型 ≈ 30-60s
- LoRA state dict: < 1 MB
- 完整 LoRA checkpoint: ~50 KB (vs 全量 117 MB)

### 3.5 输出

- `output/lora_cloned.wav` — 合成结果
- `output/workdir/lora.pt` — **LoRA-only checkpoint** (< 1MB)
- `output/workdir/samples/*.npz` — 训练样本
- `output/workdir/train/checkpoints/*.pt` — base + LoRA 完整 ckpt
- `output/workdir/train/train_log.jsonl` — 训练日志
- `output/workdir/metadata.json` — 完整元数据

### 3.6 关键代码 ([main.py](file:///e:/新创意构思/新建文件夹/ADR/examples/03_lora_finetune/main.py))

```python
# Phase 1: 构建 + LoRA + 可选量化
model = build_medium_model(preset="medium")  # 30M
apply_lora(model, LoRAConfig(rank=8, alpha=16,
                              target_modules=["out_proj","linear1","linear2"]))
if quantize_4bit: model = quantize_4bit(model)  # 真 QLoRA (先 LoRA 后量化!)
# → 训练参数从 30M 降到 ~0.4M (1.4%)

# Phase 3: 训练
trainer = Trainer(model, train_data, TrainerConfig(use_lora=True, ...))
trainer.fit()  # 仅 LoRA 参数更新

# Phase 4: LoRA 持久化 (小!)
lora_state = get_lora_state_dict(model)
torch.save({"lora_state": lora_state, ...}, "lora.pt")  # < 1MB
```

### 3.7 LoRA vs 全量微调

| 维度 | 全量 | LoRA r=8 | QLoRA r=8 |
|------|------|----------|-----------|
| 训练参数 | 100% | ~1% | ~1% |
| GPU 显存 (0.3B) | 6.5GB | 1.9GB | **0.8GB** |
| Checkpoint 大小 | 1.3GB | ~10MB | ~10MB |
| 速度 | 1x | 1.0-1.2x | 0.9-1.0x |
| 质量 (5min 训练) | 基线 | ≈ 基线 | ≈ 基线 |

---

## 4. 进阶实验 (OpenCpop 真实数据)

### 4.1 训练记录

| 目录 | 数据 | Epochs | 训练时长 | 关键指标 |
|------|------|--------|----------|----------|
| `opencpop_train_200/` | 200 句 | 3 | 753.2s (12.6 min) | final_loss 1.52 |
| `opencpop_train_1000/` | 1000 句 | 3 | 364.6s (6.1 min) | final_loss 1.28 |
| `opencpop_train_3550/` | 3550 句 | 3 | **381.6s (6.4 min)** | **RMS=0.414 最佳** |
| `opencpop_train_3550_5e/` | 3550 句 | 5 | 618.1s (10.3 min) | epoch 5 RMS=0.34 过拟合 |
| `opencpop_train_3550_5e_es/` | 3550 句 | 5 + loss 早停 | ~520s | loss 早停救回 (但不是最优) |
| `opencpop_train_3550_5e_wq_es_p1/` | 3550 句 | 5 + wav 早停 | ~520s | **wav 早停 (M3) 救回最佳** |
| `wav_es_test/` | 100 句 | 5 | ~3 min | wav 早停能力验证 |
| `wav_es_real_v2/` | 200 句 | 5 | ~5 min | 真实 BigVGAN 接入 |
| `opencpop_lora_0p3b/` | 200 句 | 2 + LoRA | ~30s | 0.3B + LoRA 显存验证 |

### 4.2 关键发现

1. **大数据 + 少 epoch > 小数据 + 多 epoch**
   - 3550×3 = RMS 0.414 (最佳)
   - 1000×3 = RMS 0.381
2. **Loss 下降 ≠ wav 质量提升**
   - 5 epoch 时 val/loss 持续下降,但 wav RMS 从 0.41 (epoch 3) 跌到 0.34 (epoch 5)
3. **Wav-based 早停可挽救过拟合**
   - `--wav-quality-patience 1 --val-ratio 0.1` 在 epoch 4 触发停训,保留 epoch 3 最佳模型

详见 [opencpop-scaling-report.md](file:///e:/新创意构思/新建文件夹/ADR/docs/opencpop-scaling-report.md) 和 [m3-wav-quality.md](file:///e:/新创意构思/新建文件夹/ADR/docs/m3-wav-quality.md)。

### 4.3 复现实验

```bash
# 复现 wav-based 早停 (M3 + M3.5 + M3.6)
python scripts/smoke_train_opencpop.py \
    --n-samples 200 --epochs 5 \
    --wav-quality-patience 1 --val-ratio 0.1 \
    --vocoder-path F:\ADR_data\bigvgan

# 复现 0.3B + LoRA (M2)
python examples/03_lora_finetune/main.py \
    --ref ref.wav --preset x0p3b --lora-rank 8 --epochs 2

# 评估 wav 质量 (M3.6 F0 指标)
python scripts/eval_chinese_tts.py --checkpoint final.pt --ref ref.wav
```

---

## 5. 常见问题 (FAQ)

### Q1: 训练要多久?
**A**: 取决于数据量 × 模型规模:
- 6 样本 3 epoch small (1-2M, [02 demo](#2-5min_finetune-5-分钟微调)) ≈ 6s (CPU)
- 200 句 3 epoch medium (30M) ≈ 12.6 min (CPU) / 2-3 min (GPU + AMP)
- 1000 句 3 epoch medium (30M) ≈ 6.1 min (CPU) / 1-2 min (GPU)
- 3550 句 3 epoch medium (30M) ≈ 6.4 min (CPU) / 1-2 min (GPU + bf16)

### Q2: 显存不够怎么办?
**A**: 三档自适应:
1. 改 preset: `small` (10M) → `medium` (30M) → `large` (180M) → `x0p3b` (340M)
2. 加 LoRA: 训练参数从 100% → 1%,显存降 2/3
3. 加 4-bit: `--quantize-4bit`, 0.3B 显存从 6.5GB → 0.8GB

### Q3: 输出 wav 是噪声?
**A**: 三种可能:
1. 基座未训练: 用 `01_5sec_clone` 会这样,需先 finetune
2. Vocoder 是 placeholder: 看 log 是否显示 `BigVGAN loaded`, 没显示就是 placeholder
3. ref 质量差: 换 ≥ 5s 单说话人 wav,避免背景音乐

### Q4: G2P OOV 怎么办?
**A**: DiffSinger 607 音素表覆盖中文拼音 + 英文音素。
- 中英文混合: 0 OOV (实测 OpenCpop 3756 句)
- 生僻字: fallback 到 pypinyin 字符级

### Q5: 怎么接入自己的数据集?
**A**: 用 `adr.data.pipeline.TrainSample`:
```python
from adr.data.pipeline import TrainSample
sample = TrainSample(
    sample_id="my_001",
    waveform=wav_22k, sample_rate=22050,
    text="你好世界", phonemes=g2p("你好世界"),
    f0=f0_ext(wav_22k), mel=compute_mel(wav_22k, 22050),
)
sample.save("data/my_001.npz")
```
然后 `Trainer(train_data="data/", ...)`

---

## 6. 下一步

| 想做 | 看 |
|------|-----|
| WebUI | `adr/webui/` (4 Tab: Data/Train/Clone/Models) |
| 量化导出 | (M3 规划中) GGUF / AWQ |
| 跨语言 | (M2+) 接入 WavLM 多语种 |
| ASR WER 早停 | (M4 规划) |
| 模型 Hub | (M5 规划) |

---

## 附录: 文件清单

| 文件 | 行数 | 说明 |
|------|------|------|
| [examples/01_5sec_clone/main.py](file:///e:/新创意构思/新建文件夹/ADR/examples/01_5sec_clone/main.py) | 115 | 5-sec 克隆 |
| [examples/02_5min_finetune/main.py](file:///e:/新创意构思/新建文件夹/ADR/examples/02_5min_finetune/main.py) | 430 | 5-min finetune |
| [examples/02_5min_finetune/gen_ref.py](file:///e:/新创意构思/新建文件夹/ADR/examples/02_5min_finetune/gen_ref.py) | 50 | 合成 ref (无真实音频时) |
| [examples/02_5min_finetune/validate.py](file:///e:/新创意构思/新建文件夹/ADR/examples/02_5min_finetune/validate.py) | 100 | 听感验证 |
| [examples/03_lora_finetune/main.py](file:///e:/新创意构思/新建文件夹/ADR/examples/03_lora_finetune/main.py) | 350 | LoRA finetune |
| [scripts/smoke_train_opencpop.py](file:///e:/新创意构思/新建文件夹/ADR/scripts/smoke_train_opencpop.py) | - | OpenCpop 训练 smoke (含 wav 早停) |
| [scripts/smoke_infer_opencpop.py](file:///e:/新创意构思/新建文件夹/ADR/scripts/smoke_infer_opencpop.py) | - | OpenCpop 推理 smoke |
| [scripts/eval_chinese_tts.py](file:///e:/新创意构思/新建文件夹/ADR/scripts/eval_chinese_tts.py) | - | 10 句中文 TTS 评估 |
| [scripts/compare_models.py](file:///e:/新创意构思/新建文件夹/ADR/scripts/compare_models.py) | - | 双模型对比 |
