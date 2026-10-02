# A 轨推进总结

> 完成日期：2026-08-07
> 阶段：A 轨 — 本地能力建设

## 目标

在不依赖云端的情况下，建立 ADR 项目的"本地可做"基础：
- 数据/工具就绪
- 代码骨架可跑
- 关键预训练权重可用
- 端到端管线打通

## 成果汇总

### 1. ffmpeg 安装
- ✅ D:\tools\ffmpeg\bin\ffmpeg.exe（2026-08-06 最新版 N-125978）
- ✅ 永久加到 PATH（环境变量 User 级）
- ✅ 集成 libvmaf, cuda-llvm, libx264, libx265, libopus 等

### 2. F 盘数据目录
| 子目录 | 内容 | 用途 |
|---|---|---|
| F:\ADR_data\opencpop | （HF 镜像下载失败，待 wenet 官方申请） | 100 首歌声数据 |
| F:\ADR_data\wavlm | 377MB pytorch_model.bin + config | 音色编码器（M1 接入） |
| F:\ADR_data\bigvgan | 449MB generator + 1.4GB discriminator + 代码 | 声码器 ✅ 已验证加载 |
| F:\ADR_data\_cache | HF 缓存 | 内部缓存 |

### 3. 预训练权重
- ✅ **WavLM-base**（377MB）：microsoft/wavlm-base
- ✅ **BigVGAN-base 22kHz 80band**（449MB）：nvidia/bigvgan_22khz_80band
- ✅ 补全 BigVGAN 缺失文件：utils.py + alias_free_activation/torch/{filter,resample}.py

### 4. ADR 自研代码新增
| 文件 | 行数 | 作用 |
|---|---|---|
| [adr/data/__init__.py](file:///e:/新创意构思/新建文件夹/ADR/adr/data/__init__.py) | 1 | 数据模块入口 |
| [adr/data/phoneme_dict.py](file:///e:/新创意构思/新建文件夹/ADR/adr/data/phoneme_dict.py) | ~95 | 音素表加载（607 音素） |
| [adr/data/opencpop.py](file:///e:/新创意构思/新建文件夹/ADR/adr/data/opencpop.py) | ~190 | OpenCpop 数据加载器（mock + 真实双模式） |
| [scripts/download_pretrained.py](file:///e:/新创意构思/新建文件夹/ADR/scripts/download_pretrained.py) | 50 | WavLM + BigVGAN 下载脚本 |
| [scripts/verify_bigvgan.py](file:///e:/新创意构思/新建文件夹/ADR/scripts/verify_bigvgan.py) | 60 | BigVGAN 加载验证 |
| [scripts/verify_0p3b_inference.py](file:///e:/新创意构思/新建文件夹/ADR/scripts/verify_0p3b_inference.py) | 100 | 0.3B 模型 OOM 测试 |
| [scripts/verify_e2e_pipeline.py](file:///e:/新创意构思/新建文件夹/ADR/scripts/verify_e2e_pipeline.py) | 110 | 端到端 ADR→BigVGAN→wav |

### 5. ADR 模型修复
- [adr/model/backbone.py](file:///e:/新创意构思/新建文件夹/ADR/adr/model/backbone.py) 增加 `timbre_proj` Linear 层，解决 timbre_dim(512) 与 hidden_dim(1024) 维度不匹配

## 验证结果

| 测试 | 结果 | 关键指标 |
|---|---|---|
| 音素表加载 | ✅ | 607 音素（601 业务 + 6 特殊）|
| OpenCpop 数据加载（mock 模式） | ✅ | batch (4, 256/128) 输出正常 |
| BigVGAN 加载 | ✅ | 112M 参数，mel→wav 0.7s |
| 0.3B 模型训练 | ✅ | **5.28GB / 8GB，损失 2.00** |
| 0.3B 模型推理 | ✅ | **2.02GB / 8GB，5 步 CFM 0.8s** |
| **端到端 wav 输出** | ✅ | 5.94s 音频，[output/adr_bigvgan_e2e.wav](file:///e:/新创意构思/新建文件夹/ADR/output/adr_bigvgan_e2e.wav) |

## 关键发现：8GB 显存足以本地训练

**A 轨最大的成果**：原本设计文档说 0.3B 训练必须 24GB+ 显存，但实测：

| 阶段 | 显存 |
|---|---|
| ADR 0.3B 训练（batch=1, T_mel=1500, 28 层 DiT） | 5.28GB |
| ADR 0.3B 推理（CFM 5 步） | 2.02GB |
| BigVGAN 22kHz 推理 | 1.5GB |

**结论：8GB RTX A2000 可以本地做 0.3B 模型 LoRA 微调或小规模全量训练**。这彻底改变了 M1 的计划——**M1 不必上云**，仅在数据下载和数据量需要时再上云。

## 仍未解决

### OpenCpop 真实数据下载
**状态**：阻塞

**已尝试**：
- ❌ ModelScope AmyZTY/Opencpop（API 错误，参数 `target_dir` 已废弃）
- ❌ HF Saaaxman/Opencpop（webdataset 格式不兼容 hf-mirror.com）
- ❌ HF espnet/ace-opencpop-segments（网络 fetch 失败）
- ❌ HF mitsudate/DiffSinger_opencpop_JPN（仅日文版）

**根因**：OpenCpop 需从 https://wenet.org.cn/opencpop/ 官方申请（发邮件给作者），HF 镜像质量差。

**替代方案**：
1. **A 轨期间**用 mock 数据完成所有骨架测试（已完成）
2. **B 轨（云端）期间**上 wenet.org.cn 申请 OpenCpop
3. **备选数据集**：M4Singer（多歌手多风格）、PopCS、OpenSinger（HF 上有更完整镜像）

## A 轨 → M1 衔接

### M1 修订计划

| 原始 M1 任务 | 修订 |
|---|---|
| 1×A800 24h 冒烟训练 | ❌ 改为**本地 8GB 直接训练** |
| Emilia + OpenCpop 数据加载 | ✅ mock 模式已通，需真实数据替换 |
| 0.3B 全量 backbone 训练 | ✅ **可直接在本机跑 1000 step 验证 loss 收敛** |
| 接 BigVGAN 验证 mel→wav | ✅ 已完成 |
| M0 修复重跑 | ⏳ 推迟到 M1 末，本地空闲时再跑 |

### 接下来这周可继续

1. **申请 OpenCpop 真实数据**（发邮件到 wenet）
2. **接入 WavLM 真实音色编码器**（替换 mel+CNN 占位）
3. **写 mel 提取工具**（用 torchaudio 把 wav 转为 mel）
4. **写真实数据加载器**（需要 OpenCpop 真实数据）
5. **开始 1000 step 冒烟训练**（用 mock 数据先跑通，验证 loss 曲线）

## 项目目录变化

```
ADR/
├── adr/                          # 自研代码（11 个文件）
│   ├── configs.py
│   ├── __init__.py
│   ├── README.md
│   ├── data/                     # 新增
│   │   ├── __init__.py
│   │   ├── phoneme_dict.py
│   │   └── opencpop.py
│   └── model/                    # 5 个核心模块
├── docs/                         # 3 篇文档 + 本总结
│   ├── architecture-design.md
│   ├── m0-report.md
│   ├── m0-diagnose.md
│   └── a-track-summary.md        # 本文件
├── scripts/                      # 10 个脚本
├── output/                       # 多个 wav + 日志
│   ├── bigvgan_smoke.wav         # BigVGAN 单独验证
│   ├── fish_smoke.wav            # M0
│   └── adr_bigvgan_e2e.wav       # A 轨端到端 ✅
└── third_party/                  # Fish Speech + DiffSinger
```

## 关键数字回顾

| 指标 | 数值 |
|---|---|
| 音素表大小 | 607 音素 |
| ADR 0.3B 参数量 | 497M |
| ADR 训练显存 | 5.28GB / 8GB |
| ADR 推理显存 | 2.02GB / 8GB |
| BigVGAN 参数量 | 112M |
| 端到端推理时间 | 0.8s（ADR）+ 0.64s（BigVGAN）= 1.4s |
| 端到端音频时长 | 5.94s（mock 512 帧 mel） |

## 下一步建议

按优先级：
1. **最优先**：从 wenet 申请 OpenCpop 真实数据（邮件，几天回复）
2. **并行**：WavLM 真实音色编码器集成（不依赖真实数据）
3. **并行**：mel 提取工具 + 真实数据加载器（不依赖 OpenCpop）
4. **最后**：用 mock 数据先跑 1000 step 冒烟训练验证 loss 收敛

何时开始下一步？
