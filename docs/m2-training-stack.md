# M2 Training Stack (Low-Resource)

> 状态: ✅ M2 完成 (2026-08-09), QLoRA 硬化完成 (2026-08-13)
> 测试: 167/167 总测试通过 (1 skip 数据相关)
> 验证: 8GB RTX A2000 实测

## 目标

在 **8GB 显存** 下训练 **0.3B+ 参数模型**,5 秒到几分钟完成克隆。

## 核心组件

| 组件 | 文件 | 作用 |
|------|------|------|
| **LoRA** | `adr/training/lora.py` | 零依赖低秩适配 (1.4% 参数量) |
| **Trainer** | `adr/training/trainer.py` | AMP + 梯度累积 + 8bit 优化器 + LoRA 集成 |
| **Optimizer** | `adr/training/optimizer.py` | AdamW / 8-bit AdamW + warmup_cosine |
| **Efficient** | `adr/training/efficient.py` | SDPA FlashAttn + 4/8bit 量化 + 显存统计 |
| **GradCkpt** | `adr/training/grad_ckpt.py` | Gradient Checkpointing |

## 显存对比 (0.3B 模型, 8GB GPU 实测)

| 模式 | 峰值显存 | 训练参数 | 8GB 适配 |
|------|----------|----------|----------|
| **全量 finetune** | 6.47 GB | 338M (100%) | ✓ (吃满) |
| **LoRA r=8** | 1.93 GB | 2.6M (0.75%) | ✓ (大量余量) |
| **QLoRA (4bit 基座 + LoRA)** | **0.81 GB** | 2.8M (0.83%) | ✓ (余量 7GB+) |

> 数据来源: `examples/03_lora_finetune/gpu_benchmark.json` (RTX A2000 Laptop 8GB)
> 测量口径: peak 只计训练 step 显存 (setup 完成后 reset peak)。

**核心收益**:
- LoRA: 6.5GB → 1.9GB,节省 **70%** 显存
- 真 QLoRA: 再降到 **0.81GB**,相比全量节省 **88%**

## 用法

### 一行启动

```python
from adr.training import Trainer, TrainerConfig
from adr.training.lora import LoRAConfig, apply_lora

model = SoVITS(SoVITSConfig(...))
apply_lora(model, LoRAConfig(rank=8, alpha=16, target_modules=["out_proj", "linear1", "linear2"]))
trainer = Trainer(model, "data/samples/", config=TrainerConfig(use_lora=True, epochs=2))
trainer.fit()
```

### CLI

```bash
# 5min finetune (medium 30M, CPU 可跑)
python examples/02_5min_finetune/main.py

# 0.3B LoRA (需要 8GB GPU)
python examples/03_lora_finetune/main.py --preset x0p3b --epochs 1
```

### 预设模型规模

```python
PRESET_SIZES = {
    "small":  (256, 4, 4, ...),     # ~10M, debug
    "medium": (512, 6, 8, ...),     # ~30M, 默认 5min demo
    "large":  (1024, 12, 16, ...),  # ~180M
    "x0p3b":  (1024, 24, 16, ...),  # ~340M, 0.3B-ish, 8GB 全量训练
    "x0p4b":  (1280, 20, 16, ...),  # ~450M, 8GB 极限
}
```

## 设计决策

### 1. 零依赖 LoRA (不强制 PEFT)

- ✅ 自研 `LoRALinear` (单文件,~200 行)
- ✅ 提供 `try_apply_peft()` 可选切换到 HuggingFace PEFT
- ✅ 不引入 `peft / accelerate` 依赖
- ✅ 与 bnb `Linear4bit` 共存 (互斥模式)

### 2. 8bit 优化器 (bitsandbytes)

- ✅ `AdamW8bit` 替代 AdamW
- ✅ 优化器状态内存从 8 bytes/param 降到 2 bytes/param
- ✅ bitsandbytes 不可用时自动 fallback 到 AdamW

### 3. SDPA FlashAttention

- ✅ PyTorch 2.0+ 自带 SDPA (无需额外包)
- ✅ 启用 `flash_sdp` + `mem_efficient_sdp` 后端
- ✅ 真实减少 30-50% 注意力显存

### 4. 4-bit 量化 (真 QLoRA)

- ✅ `bitsandbytes.nn.Linear4bit` (NF4 量化)
- ✅ `LoRALinear` 冻结基座直接 4bit 量化 (LoRA 旁路挂载在量化基座上, 同 PEFT 语义)
- ✅ `nn.MultiheadAttention` 的裸 `in_proj_weight` 由 `_QuantizedSelfAttention` 量化 (SDPA 计算)
- ✅ `dequantize_4bit` 支持保存 checkpoint 前还原, 避免 uint8 权重 shape/keys 不匹配

## 已知限制

| 限制 | 影响 | 解决 |
|------|------|------|
| bitsandbytes 0.49+ API 变动 | 偶尔兼容问题 | 用标准 workflow (Linear4bit + .to(device)) |
| 数据加载慢 (CPU) | 5min demo 90% 时间在数据准备 | 缓存 npz, 或用 DataLoader workers |
| 量化层前向略慢 | 4bit matmul 需在线反量化 | 训练仍受 LoRA 小参数主导, 影响 < 10% |

## QLoRA 实施细节 (真 QLoRA)

我们实现**真 QLoRA**: LoRA 旁路直接挂载在 4bit 量化基座上, 无需 PEFT。

### 实施流程

```python
# 1. 先应用 LoRA (替换目标 nn.Linear 为 LoRALinear)
apply_lora(model, LoRAConfig(rank=8, target_modules=["out_proj", "linear1", "linear2"]))

# 2. 再 4bit 量化:
#    - LoRALinear 的冻结基座 → Linear4bit (LoRA 旁路保留 fp32)
#    - nn.MultiheadAttention 的 qkv 融合投影 → _QuantizedSelfAttention
#    - 其余 nn.Linear → Linear4bit
model = quantize_4bit(model, quant_type="nf4", compute_dtype=torch.float16)

# 3. 优化器只对 LoRA 参数
optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-3)
```

### Checkpoint 兼容

量化模型的 `Linear4bit` 权重是 uint8 packed 形式 (`[N/2, 1]`),
与未量化模型 (`[out, in]`) shape/keys 均不一致。
`Trainer.save_checkpoint` 在全量保存前自动调用 `dequantize_4bit` 还原为普通权重;
LoRA-only 保存 (`get_lora_state_dict`) 只含 lora_A/lora_B, 天然兼容。

### 显存组成 (0.3B 模型, 真 QLoRA)

```
- 量化基座:  338M × 0.5 byte + quant_state ≈ 190 MB
- LoRA 参数: 2.8M × (2 byte param + 8 byte AdamW) ≈ 28 MB
- 激活/中间: ≈ 590 MB
- 总计:      ≈ 0.81 GB  (vs 全量 6.47 GB, vs LoRA 1.93 GB)
```

## 故障排查 (Troubleshooting)

### 1. `optimizer got an empty parameter list`

**症状**: 训练启动时报 `ValueError: optimizer got an empty parameter list`。

**原因**: `apply_lora` 跳过了量化层,导致所有目标 Linear 都被量化,没有 trainable LoRA 旁路。

**解决**:
```python
# 错误顺序: 先量化, 后 LoRA
model = quantize_4bit(model)
apply_lora(model, lora_cfg)  # ← 目标 Linear 已变 Linear4bit, 全被跳过!

# 正确顺序: 先 LoRA, 后量化 (真 QLoRA: 量化 LoRA 冻结基座 + MHA qkv + 其余 Linear)
apply_lora(model, lora_cfg)
model = quantize_4bit(model)
```

### 2. CUDA `device-side assert triggered` (vocab_size)

**症状**: 训练/推理时 `RuntimeError: CUDA error: device-side assert triggered`。

**原因**: 模型 `vocab_size` 与音素字典大小不匹配 (我们用 DiffSinger 607 音素)。

**解决**:
```python
# 确保模型用 607 音素
cfg = SoVITSConfig(vocab_size=607, ...)

# 确保音素字典表也用 607
from adr.data.phoneme_dict import PhonemeDict
pd = PhonemeDict()  # 默认 607 音素 (DiffSinger)
```

### 3. `LayerNorm input size mismatch`

**症状**: `RuntimeError: expected input to have N channels, but got M`。

**原因**: `DurationPredictor` 的 LayerNorm 期望的 `hidden_dim` 与上游不一致。

**解决**: 在 `adr/models/sovits.py` 的 DurationPredictor 初始化时,直接读 config 的 `hidden_dim`:
```python
self.norm = nn.LayerNorm(config.hidden_dim)  # 而不是固定值
```

### 4. LoRA state_dict keys mismatch

**症状**: 加载 LoRA 时部分键不匹配,loss 不下降。

**原因**: 保存/加载时模块路径不一致。

**解决**:
```python
# 始终用 get_lora_state_dict / load_lora_state_dict 包装函数
# 它们内部用 .lora_A / .lora_B 命名,保证一致性
from adr.training.lora import get_lora_state_dict, load_lora_state_dict
state = get_lora_state_dict(model)
torch.save({"lora_state": state, ...}, ckpt_path)
```

### 5. `bitsandbytes 0.49+` 量化 API 变化

**症状**: `TypeError: quantize_4bit() got an unexpected keyword argument 'compute_dtype'`。

**原因**: bnb 0.49 重构了内部 API,部分参数路径变化。

**解决**: 我们用标准 workflow 而非内部函数:
```python
from bitsandbytes.nn import Linear4bit
new_layer = Linear4bit(in_features, out_features, bias=..., quant_type="nf4")
new_layer.weight.data = module.weight.detach().to(dtype=compute_dtype)
new_layer = new_layer.to(module.weight.device)  # ← 触发量化
```

### 6. 8GB OOM (0.3B 全量)

**症状**: `OutOfMemoryError` 在 batch_size=2 时。

**原因**: 0.3B × 16 bytes/param = 4.8GB + 激活 + 缓冲 > 8GB。

**解决**:
- (a) 用 LoRA (降到 1.9GB) 或 QLoRA (降到 0.8GB, 推荐)
- (b) `batch_size=1` + `grad_accum=8`
- (c) `use_gradient_checkpointing=True` (牺牲 30% 速度换 40% 显存)
- (d) `use_amp=True` (fp16 训练, 减半激活显存)

### 7. CPU 训练太慢

**症状**: 5min demo 中数据准备占 90% 时间。

**解决**:
- 缓存 `.npz` 训练样本 (我们已做)
- `DataLoader(num_workers=2)`
- 用小模型 preset (`small` 10M)
- 跳过 `--epochs` 训练 (`--no-finetune`)

## 验证结果

### 单元测试
- `tests/test_lora.py`: 11/11 通过
- 全套: 167/167 通过 (1 skip 数据相关)

### End-to-end
- `examples/03_lora_finetune/main.py --preset medium`: 1.8s 训练, 1.7MB ckpt
- `examples/03_lora_finetune/main.py --preset x0p3b`: 2.7s 训练, 10MB ckpt, 0.93s wav
- `... --quantize-4bit`: QLoRA 训练/保存/加载/推理全链路通过

### 真实 GPU 内存
- 0.3B 全量: 6.5GB / 8GB (80% 利用)
- 0.3B LoRA: 1.9GB / 8GB (24% 利用)
- 0.3B QLoRA: 0.8GB / 8GB (10% 利用)
- 中等 (30M) QLoRA: 0.14GB / 8GB (2% 利用)

## 下一步 (M3)

- A. 真实数据接入 (OpenCpop / GPT-SoVITS 中文)
- B. Vocoder 优化 (HiFi-GAN / NSF-HiFiGAN)
- C. 5min finetune 升级到 5min 全流程 (5min 数据 + 训练 + 推理)
- D. M2 训练栈硬化 (DDP, 错误恢复, 自动超参)
