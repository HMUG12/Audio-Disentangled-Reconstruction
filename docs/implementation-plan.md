# ADR 训练框架 · 实施规划

> **目标**: 把"低资源快速克隆"做成一个开箱即用的 Python 框架(类似 llama.cpp 之于 LLM)
> **核心原则**: 一切以 **8GB 显存用户 5 分钟克隆出可用模型** 为北极星指标

---

## 0. 北极星指标 (North Star)

| 指标 | 目标 |
|---|---|
| **最小显存** | 4GB 推理 / 6GB 训练 |
| **参考音频** | 5 秒起,10 分钟封顶 |
| **训练时间** | ≤ 10 分钟 (8GB GPU) |
| **克隆质量** | 音色相似度 MOS ≥ 3.5 |
| **安装体验** | `pip install adr-clone` 即可 (plug-and-play) |
| **命令简洁** | `adr clone --ref voice.wav --text "你好"` 一行搞定 |

---

## 1. 目录结构 (最终形态)

```
ADR/                                  # 框架根目录
│
├── adr/                              # 核心 Python 包
│   ├── __init__.py                   # 暴露 adr.clone / adr.train / adr.infer
│   ├── cli.py                        # CLI 入口 (click 框架)
│   ├── webui.py                      # WebUI 入口 (Gradio)
│   │
│   ├── core/                         # 框架核心
│   │   ├── __init__.py
│   │   ├── config.py                 # 全局配置 (dataclass + YAML)
│   │   ├── device.py                 # 显存自适应 (RVC 范式)
│   │   ├── registry.py               # 插件注册 (Backbone/Vocoder 注册表)
│   │   ├── logging.py                # 统一日志
│   │   └── exceptions.py             # 自定义异常
│   │
│   ├── data/                         # 数据流水线 (M1 重点)
│   │   ├── __init__.py
│   │   ├── pipeline.py               # 编排: ref.wav → trainable dataset
│   │   ├── separate.py               # UVR5 人声/伴奏分离
│   │   ├── denoise.py                # 降噪 (可选 step)
│   │   ├── slice.py                  # 音频切片 (5-10 秒)
│   │   ├── asr.py                    # Faster-Whisper 自动标注
│   │   ├── g2p.py                    # G2P (中文 g2pW / 英文 CMU)
│   │   ├── phoneme_dict.py           # (已有,迁移)
│   │   ├── f0.py                     # 音高提取 (RMVPE/FCPE)
│   │   └── dataset.py                # PyTorch Dataset
│   │
│   ├── models/                       # 模型 Backend (可插拔)
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseTTS 抽象接口
│   │   ├── sovits.py                 # SoVITS (M1 默认)
│   │   ├── mamtra.py                 # MamTra (M2 集成)
│   │   ├── content_encoder.py        # (已有,迁移为 WavLM)
│   │   ├── timbre_encoder.py         # (已有,迁移为 256 维 adapter)
│   │   ├── melody_encoder.py         # (保留,可选)
│   │   └── backbone.py               # (已有,迁移为可被 LoRA 注入)
│   │
│   ├── vocoder/                      # 声码器
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseVocoder 接口
│   │   ├── bigvgan.py                # BigVGAN v2 44.1kHz
│   │   └── hifigan.py                # HiFi-GAN 备选
│   │
│   ├── training/                     # 训练栈 (M2 重点)
│   │   ├── __init__.py
│   │   ├── trainer.py                # 训练器主类
│   │   ├── lora.py                   # LoRA 注入 (peft 封装)
│   │   ├── qlora.py                  # QLoRA (bitsandbytes 4-bit)
│   │   ├── flash_attn.py             # FlashAttn 加速
│   │   ├── grad_ckpt.py              # Gradient Checkpointing
│   │   ├── deepspeed.py              # DeepSpeed Zero-Offload (M2 可选)
│   │   └── callbacks.py              # 训练回调 (save/eval/log)
│   │
│   ├── inference/                    # 推理
│   │   ├── __init__.py
│   │   ├── pipeline.py               # 端到端: text → wav
│   │   ├── streaming.py              # 流式推理 (M3)
│   │   └── batch.py                  # 批处理推理
│   │
│   ├── export/                       # 量化导出 (M3 重点)
│   │   ├── __init__.py
│   │   ├── gguf.py                   # GGUF 导出 (Ollama 兼容)
│   │   ├── awq.py                    # AWQ 量化 (vLLM 兼容)
│   │   ├── onnx.py                   # ONNX 导出 (M3 可选)
│   │   └── bundle.py                 # 打包: 权重 + 配置 + license 一键
│   │
│   └── utils/                        # 工具
│       ├── __init__.py
│       ├── audio.py                  # 音频 I/O (librosa/torchaudio 封装)
│       ├── download.py               # 模型下载 (huggingface_hub 封装)
│       └── progress.py               # 进度条 (rich/tqdm 封装)
│
├── configs/                          # 配置文件
│   ├── default.yaml                  # 默认配置
│   ├── vram_4gb.yaml                 # 4GB 显存档 (保守)
│   ├── vram_6gb.yaml                 # 6GB 显存档 (推荐)
│   ├── vram_8gb.yaml                 # 8GB+ 显存档 (激进)
│   └── presets/
│       ├── sovits_base.yaml          # SoVITS 预设
│       ├── mamtra_base.yaml          # MamTra 预设
│       └── mave_base.yaml            # MAVE 预设
│
├── examples/                         # 端到端示例
│   ├── 01_5sec_clone/                # 5 秒克隆
│   ├── 02_10min_finetune/            # 10 分钟微调
│   ├── 03_cross_lang/                # 跨语言克隆
│   └── 04_export_gguf/               # 量化导出示例
│
├── scripts/                          # 辅助脚本 (从 scripts/ 迁移)
│   ├── download_pretrained.py        # (已有,迁移到 adr/utils/download.py)
│   ├── verify_install.py             # 安装验证 (新)
│   └── verify_pipeline.py            # (已有,迁移到 adr/inference)
│
├── tests/                            # 单元测试
│   ├── test_data_pipeline.py
│   ├── test_trainer.py
│   ├── test_export.py
│   └── conftest.py
│
├── docs/                             # 文档
│   ├── low-resource-framework-research.md   (已有)
│   ├── architecture-design.md                (已有)
│   ├── a-track-summary.md                    (已有)
│   ├── implementation-plan.md                (本文件)
│   ├── quickstart.md              # 快速上手 (5 分钟跑通)
│   ├── tutorials/
│   │   ├── 01_install.md
│   │   ├── 05sec_clone.md
│   │   ├── 10min_finetune.md
│   │   └── export_ollama.md
│   └── api.md                      # API 文档 (自动生成)
│
├── third_party/                      # 第三方依赖 (已有)
│   ├── DiffSinger/                  # 音素表
│   ├── BigVGAN/                     # 声码器
│   └── fish-speech/                 # 参考实现
│
├── .github/
│   └── workflows/
│       ├── ci.yml                   # CI: lint + test
│       └── docker.yml                # Docker 自动构建
│
├── docker/
│   ├── Dockerfile                   # CUDA + 依赖 + ADR
│   └── docker-compose.yml           # 一键起 WebUI
│
├── pyproject.toml                    # 项目配置 (PEP 621)
├── requirements.txt                  # 运行时依赖
├── requirements-dev.txt              # 开发依赖
├── README.md                         # 项目说明
├── LICENSE                           # MIT
└── CHANGELOG.md
```

---

## 2. M1: 框架骨架 (2 周)

### 2.1 验收标准

- [ ] 用户 5 秒音频 → 一行命令 `adr clone` → 输出 wav
- [ ] 显存占用 < 6GB (8GB GPU 流畅)
- [ ] 完整数据流水线跑通 (UVR5 → 切片 → ASR → G2P)
- [ ] 至少 1 个 backbone (SoVITS 简化版) 可用
- [ ] 显存自适应: 4G/6G/8G 三个配置档位自动切换
- [ ] CLI 4 个子命令全部跑通
- [ ] WebUI 4 个 Tab (数据/训练/推理/导出) 跑通
- [ ] 1 个完整 example: 5 秒克隆

### 2.2 周计划

#### Week 1: 核心 + 数据流水线 (5 天)

| Day | 任务 | 关键文件 | 验证 |
|---|---|---|---|
| **Day 1** | 框架初始化 | `pyproject.toml`, `adr/__init__.py`, `adr/core/config.py` | `pip install -e .` 成功 |
| **Day 2** | 设备管理 + 插件注册 | `adr/core/device.py`, `adr/core/registry.py` | `python -c "from adr.core import device; print(device.config())"` |
| **Day 3** | 数据流水线 1/2 | `adr/data/separate.py`, `adr/data/denoise.py`, `adr/data/slice.py` | 处理 1 分钟音频成功 |
| **Day 4** | 数据流水线 2/2 | `adr/data/asr.py`, `adr/data/g2p.py`, `adr/data/pipeline.py` | 端到端跑通: wav → 标注 |
| **Day 5** | CLI 骨架 | `adr/cli.py` (click), 4 个子命令 stub | `adr --help` 跑通 |

#### Week 2: 训练 + 推理 + 端到端 (5 天)

| Day | 任务 | 关键文件 | 验证 |
|---|---|---|---|
| **Day 6** | 模型 Backend 抽象 | `adr/models/base.py`, `adr/models/sovits.py` (简化) | 加载基座模型成功 |
| **Day 7** | 声码器集成 | `adr/vocoder/bigvgan.py` (集成已有 BigVGAN) | mel → wav 跑通 |
| **Day 8** | 训练器 (无 LoRA) | `adr/training/trainer.py`, `adr/training/grad_ckpt.py` | 5 分钟微调成功 |
| **Day 9** | 推理 Pipeline | `adr/inference/pipeline.py` | text + ref → wav 端到端 |
| **Day 10** | WebUI + Example | `adr/webui.py` (Gradio), `examples/01_5sec_clone/` | 5 秒克隆 example 跑通 |

### 2.3 M1 文件产出清单 (27 个新文件)

**核心 (4)**:
- `adr/core/config.py` - 全局配置
- `adr/core/device.py` - 显存自适应
- `adr/core/registry.py` - 插件注册
- `adr/core/logging.py` - 统一日志

**数据 (6)**:
- `adr/data/pipeline.py` - 编排
- `adr/data/separate.py` - UVR5
- `adr/data/slice.py` - 切片
- `adr/data/asr.py` - Whisper
- `adr/data/g2p.py` - G2P
- `adr/data/f0.py` - 音高

**模型 (4)**:
- `adr/models/base.py` - 抽象
- `adr/models/sovits.py` - SoVITS 简化版
- `adr/vocoder/base.py` - 抽象
- `adr/vocoder/bigvgan.py` - BigVGAN 集成

**训练 (3)**:
- `adr/training/trainer.py` - 训练器
- `adr/training/grad_ckpt.py` - 梯度检查点
- `adr/training/callbacks.py` - 回调

**推理 (2)**:
- `adr/inference/pipeline.py` - 端到端
- `adr/inference/batch.py` - 批处理

**入口 (3)**:
- `adr/cli.py` - CLI
- `adr/webui.py` - WebUI
- `adr/utils/audio.py` - 音频 I/O

**配置 (4)**:
- `configs/default.yaml`
- `configs/vram_4gb.yaml`
- `configs/vram_6gb.yaml`
- `configs/vram_8gb.yaml`

**Example + 文档 (3)**:
- `examples/01_5sec_clone/main.py`
- `docs/quickstart.md`
- `docs/tutorials/01_install.md`

### 2.4 关键技术决策 (M1)

| 决策 | 选择 | 理由 |
|---|---|---|
| **CLI 框架** | Click | 比 argparse 优雅,生态成熟 |
| **WebUI 框架** | Gradio 4.x | GPT-SoVITS 同款,简单 |
| **配置格式** | YAML + dataclass | 易读 + 类型安全 |
| **TTS 主干** | SoVITS 简化版 (VITS + GPT) | 成熟,非自回归,推理快 |
| **声码器** | BigVGAN v2 44.1kHz | GPT-SoVITS v4 同款 |
| **音素表** | DiffSinger 607 (已有) | 中文覆盖好 |
| **ASR** | Faster-Whisper (small) | 中文 90%+ 准确率,CPU 也能跑 |
| **G2P** | g2pW (中文) + CMU (英文) | 开源、稳定 |
| **音高** | RMVPE | 精度高,速度可接受 |
| **人声分离** | UVR5 (Mel-Band Roformer) | GPT-SoVITS 同款 |
| **训练加速** | FP16 + Gradient Ckpt | 8GB 显存够用 |
| **训练监控** | tqdm + rich (M1) / Tensorboard (M2) | 渐进式 |

### 2.5 风险与缓解

| 风险 | 概率 | 缓解措施 |
|---|---|---|
| **G2P 中文分词失败** | 中 | fallback 到字符级 + 强制对齐 |
| **Faster-Whisper 标注错误** | 中 | 允许用户手动校正 |
| **UVR5 安装复杂** | 中 | 提供 Docker 镜像 + 预下载 |
| **8GB 训练 OOM** | 中 | M1 只用 SoVITS 简化版 (~200M 参数) |
| **参考音频质量差** | 高 | 强制 ≥ 5 秒 + 自动 SNR 检测 |

---

## 3. M2: 训练栈深化 (2 周)

### 3.1 验收标准

- [ ] QLoRA 4-bit 集成完成 (bitsandbytes + peft)
- [ ] FlashAttn 加速集成
- [ ] LoRA 注入到 SoVITS attention 层
- [ ] MamTra / MAVE backend 集成 (可选启用)
- [ ] 8GB 训练 7B 模型 demo
- [ ] 训练监控 (Tensorboard / W&B)
- [ ] 显存分析工具 (打印各模块占用)

### 3.2 周计划

| Week | 任务 | 关键文件 |
|---|---|---|
| **Week 3** | QLoRA + LoRA | `adr/training/qlora.py`, `adr/training/lora.py` |
| **Week 4** | FlashAttn + DeepSpeed + Backend | `adr/training/flash_attn.py`, `adr/models/mamtra.py`, `adr/models/mave.py` |

### 3.3 M2 文件产出清单 (10 个)

- `adr/training/qlora.py` - QLoRA 集成
- `adr/training/lora.py` - LoRA 注入
- `adr/training/flash_attn.py` - FlashAttn
- `adr/training/deepspeed.py` - DeepSpeed Zero
- `adr/models/mamtra.py` - MamTra backend
- `adr/models/mave.py` - MAVE backend
- `adr/training/monitor.py` - 显存/指标监控
- `examples/02_10min_finetune/main.py`
- `docs/tutorials/10min_finetune.md`
- `tests/test_trainer.py`

---

## 4. M3: 量化导出 (1 周)

### 4.1 验收标准

- [ ] GGUF 导出 (兼容 Ollama / llama.cpp)
- [ ] AWQ 量化 (兼容 vLLM)
- [ ] 一键打包: 权重 + 配置 + license
- [ ] `adr export` 命令支持 3 种格式
- [ ] Ollama / vLLM 部署示例

### 4.2 文件产出 (6 个)

- `adr/export/gguf.py`
- `adr/export/awq.py`
- `adr/export/bundle.py`
- `examples/04_export_gguf/main.py`
- `docs/tutorials/export_ollama.md`
- `tests/test_export.py`

---

## 5. M4: 社区与生态 (持续)

### 5.1 任务

- [ ] 完整 README + 中文文档
- [ ] 5 个 example 场景
- [ ] 视频教程 (B 站)
- [ ] 持续迭代 (跟进 GPT-SoVITS v5 / OpenVoice V3)
- [ ] 模型 Hub (类似 Hugging Face)

### 5.2 长期规划

| 阶段 | 时间 | 目标 |
|---|---|---|
| **M1** | 2 周 | 5 秒克隆可跑通 |
| **M2** | 2 周 | QLoRA + 多 backbone |
| **M3** | 1 周 | 量化导出 |
| **M4** | 持续 | 社区 + 生态 |
| **v1.0 发布** | M1-M3 后 1 周 | 稳定版,正式推广 |

---

## 6. 即时行动 (本轮)

如果你确认这个规划,本轮我会按以下顺序开始 M1:

1. **Day 1 (今天)** - 框架初始化
   - 创建 `pyproject.toml`, `adr/__init__.py`
   - 创建 `adr/core/` 4 个核心文件
   - 创建 `configs/` 4 个 YAML
   - 验证 `pip install -e .` 成功

2. **Day 2 (明天)** - 设备管理 + 插件注册
3. **Day 3** - 数据流水线 1/2
4. **Day 4** - 数据流水线 2/2
5. **Day 5** - CLI 骨架
6. **Day 6-10** - 训练/推理/WebUI/Example

每完成一天,我会在群里(此对话)汇报进度并提交验收。

---

## 7. 关键问题 (请你回答)

1. **进度节奏**: 2 周 M1 是不是合适?你希望更快(1 周)还是更宽松(3 周)?
2. **优先级**: 哪个子模块你最关心?
   - A) 数据流水线 (UVR5/Whisper/G2P)
   - B) 训练栈 (QLoRA/LoRA)
   - C) WebUI (Gradio)
   - D) 端到端 example (5 秒克隆)
3. **是否保留 ADR 旧模型代码**: 我建议保留作为 `adr/models/_legacy/` 供对比,你同意吗?
4. **是否需要 Docker**: plug-and-play 用户可能更喜欢 Docker 一键启动,要不要 Day 1 就准备?

---

> **请确认规划无误后,我立即开始 Day 1 实施。**
