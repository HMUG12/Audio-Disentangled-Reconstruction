# ADR 二期规划书 (Phase 2 Plan)

> 版本: v1.1 (2026-08-13, 修订: 路线改为自研架构升级 ADR-2)
> 背景: 一期 (M1-M7) 完成后, 端到端链路打通但**合成音质不达标** (近似噪声)。
> 本文档包含: 一期复盘 (自我总结) + 二期目标 + 路线决策 + ADR-2 架构设计 + 里程碑计划 + 验收标准。

---

## 第一部分: 一期复盘

### 1.1 成绩 (真实、已验证的部分)

| 模块 | 状态 | 证据 |
|------|------|------|
| 训练栈 (Trainer/AMP/GradCkpt/8bit AdamW) | ✅ 稳定 | 51 项训练相关测试全过 |
| LoRA / 真 QLoRA | ✅ 业界同等水平 | 0.3B 模型 8GB 显存仅占 0.8GB (vs 全量 6.5GB) |
| 数据流水线 (切片/ASR/G2P/F0) | ✅ 可用 | OpenCpop 3550 句已 npz 化 |
| WebUI + CLI + 外部 API | ✅ 完整 | 4 Tab / 10+ 命令 / 4 个命名 API |
| checkpoint 兼容 (LoRA/量化/全量) | ✅ 已加固 | 自动识别三种格式 |

### 1.2 核心失败: 自研声学模型不成熟

**事实**: 训练近一周, 所有产出权重合成均为噪声。根因不是训练量不够, 而是:

1. **音素编码断裂存在了近一周未被发现**
   - smoke 脚本: 音素列表 join 成字符串 → 走了 G2P 文本路径 → 200/200 样本编码为空
   - Trainer collate: 原始音素直接查音节字典 → 100% UNK
   - **模型从未见过文本**, 却一路"训练成功、loss 下降、测试通过"
2. **简化 SoVITS 声学天花板低**
   - mel 预测是单次前向, 无扩散/流匹配迭代, 无 VAE+flow+对抗
   - 时长靠 "mel 长度均摊", 无强制对齐
   - 无韵律建模, 唱歌靠外部 F0, 说话没有 prosody predictor
3. **验证闭环缺失 (流程问题, 比技术问题更严重)**
   - 训练指标 (loss/mel corr) 与听感完全脱节: corr 0.43 的"垃圾权重"和 corr 0.48 的修复权重, 听感可能都是噪声
   - 没有"每周合成样张听审"机制, 导致致命 bug 存活一周

### 1.3 教训 (写进二期流程)

> **任何训练改动, 必须以"合成样张可听"为验收终点, 指标只是辅助。**
> 单元测试绿 ≠ 功能正确; loss 降 ≠ 能听。

### 1.4 一期资产盘点 (二期可复用的)

- ✅ 低资源训练栈 (LoRA/QLoRA/Trainer) — **核心资产, 直接保留**
- ✅ WebUI / CLI / API 外壳 — 保留, backbone 可插拔
- ✅ OpenCpop npz 数据 (3550 句, 编码已修复) — 保留
- ✅ BigVGAN 声码器接入 — 保留
- ⚠️ 简化 SoVITS 声学模型 — **降级为实验分支, 不再作为主线**
- ❌ 历史全部权重 — 作废

---

## 第二部分: 二期目标

**总目标: 8GB 显存, 10 分钟数据, 合成"能听清歌词、听出音色"的声音。**

| 指标 | 一期现状 | 二期目标 | 测量方式 |
|------|----------|----------|----------|
| 可懂度 | 噪声 | ASR CER ≤ 15% | whisper small 回译合成音频 |
| 音色相似度 | 不可辨 | 主观可辨 + speaker-cos ≥ 0.6 | resemblyzer / 听审 |
| 自然度 (MOS) | ~1 | ≥ 3.0 | 5 人听审打分 |
| 训练门槛 | 8GB / 5 min | 保持 8GB / ≤ 15 min | gpu_benchmark |
| 验收机制 | 无 | 每周合成样张集 + 听审记录 | 流程 |

---

## 第三部分: 路线决策 (v1.1 修订)

### 路线: **ADR-2 自研架构升级** — 学习 VITS / GPT-SoVITS 的设计原理, 但不照抄

一期复盘后有两个选择: 换成熟底座 (嫁接) 或升级自研架构。
**决策: 升级自研架构**, 理由:

1. 我们的差异化优势是**快速低资源训练** (8GB / QLoRA / 分钟级)——换底座会把这个优势稀释成"别人的模型 + 我们的壳"
2. 一期声学模型的失败根因已查明 (编码断裂 + 简化过度), 不是"自研"本身的失败
3. VITS / GPT-SoVITS 的核心设计原理是公开且可学习的, 取其神而非抄其形

**借鉴清单 (学什么, 不抄什么)**:

| 来源 | 学 (设计原理) | 不抄 (实现形态) |
|------|--------------|----------------|
| VITS | ① VAE 后验编码 + 标准化流 (flow) 提升 mel 表现空间<br>② **MAS 单调对齐搜索**: 训练时自动学音素-mel 对齐, 免 MFA<br>③ 随机时长预测器 (对抗) | 不抄其完整对抗训练管线 (太重, 与低资源目标冲突) |
| GPT-SoVITS | ① 参考音频的全局+局部双路音色条件<br>② 文本-语音语义先验的对齐思路 | 不抄 AR decoder + 语义 token 体系 (推理慢, 训练重) |
| 流式 TTS (VITS 流式变体) | 分块流式生成: decoder 按 chunk 输出, vocoder 流式消费 | 不牺牲离线合成质量换流式 |

**ADR-2 创新点 (我们自己的)**:

1. **统一双模架构**: TTS 与唱歌 (SVS) 共用同一 backbone, 用 mode embedding 切换——TTS 模式走韵律预测头, SVS 模式走外部旋律 (F0/音符) 条件。一份训练栈, 两种能力
2. **MAS + 轻量 flow 解码器**: 对齐用 VITS 的 MAS (免 MFA 依赖), 解码用 4-8 层耦合 flow (比扩散快 10x, 支持流式), 不用重扩散
3. **QLoRA-ready 设计**: 架构从设计起就保证所有大权重矩阵是 nn.Linear/MHA, 可以被现有 LoRA/QLoRA 栈无损包裹——8GB 训练不动摇
4. **流式推理**: chunk 边界对齐音素边界, 首包延迟目标 < 300ms

### 降级方案 (保底)

若 ADR-2 在 M9 末 (第 4 周) 样张集验收仍 CER > 25%, 则启动保底:
接入 DiffSinger 官方模型做底座 (OpenCpop 原生, 适配成本最低), ADR 训练栈套壳——这条路的工程准备 (插件接口) 在 M9.1 就会做好, 切换成本可控。

---

## 第四部分: ADR-2 架构设计

### 4.1 总览 (统一双模)

```
                        ┌─────────── 共用 backbone ───────────┐
文本 ──► G2P ──► 音素 id ──► ContentEncoder (Transformer)      │
参考音频 ─► mel ─► PosteriorEncoder (VAE, 训练时) /             │
                  TimbreEncoder (全局+局部双路, 推理时)          │
                                                              ▼
                              MAS 单调对齐 ◄──► DurationPredictor (随机)
                                                              │
                ┌── mode=TTS:  ProsodyPredictor (韵律/基频/能量) ─┤
                └── mode=SVS:  外部旋律 (F0/音符) 条件 ──────────┘
                                                              ▼
                              FlowDecoder (4-8 耦合层, 可流式)
                                                              ▼
                        mel ──► BigVGAN (流式 vocoder) ──► wav
```

### 4.2 关键模块设计决策

| 模块 | 一期 | ADR-2 | 来源/创新 |
|------|------|-------|-----------|
| 对齐 | mel 长度均摊 (错) | **MAS 单调对齐搜索** (训练自动学, 免 MFA) | 学 VITS |
| 先验 | 单次前向 mel 回归 | VAE 后验 + flow 增强先验 | 学 VITS |
| 解码 | Transformer 直出 mel | 轻量耦合 flow (4-8 层), 可流式 | **创新** (替代重扩散) |
| 时长 | 确定性 log 预测 | 随机时长预测 (flow-based) | 学 VITS |
| 音色 | 单一全局向量 | 全局 + 局部片段双路条件 | 学 GPT-SoVITS |
| 韵律 | 无 (SVS 靠外部 F0) | TTS 模式韵律预测头; SVS 模式外部旋律 | **创新·双模** |
| 模式 | TTS/SVS 不分 | mode embedding 统一切换 | **创新·双模** |
| 推理 | 整句离线 | 音素边界 chunk 流式, 首包 <300ms | **创新** |
| 训练 | LoRA/QLoRA ✅ | 不变 (架构保证大矩阵全 Linear/MHA) | 保持优势 |

### 4.3 参数预算 (8GB 训练约束)

- medium 档目标 ≤ 60M (声学) + BigVGAN (推理才加载)
- MAS 只对齐不增参数; flow 4 层 ≈ 8M; 韵律头 ≈ 2M
- LoRA r=8 可训练参数 ≤ 1M, QLoRA 峰值目标 ≤ 1.5GB

---

## 第五部分: 里程碑计划

### M8: 验证闭环先行 (第 1 周) — ✅ 已完成 (2026-08-13)

> 动机: 一期的失败首先是流程失败。先把"防噪声防线"建好, 再动模型。

- [x] **8.1 合成样张集**: `scripts/render_sample_pack.py` — 10 文本 × 2 参考 = 20 条 wav, 支持 `--score-only` 打分解耦
- [x] **8.2 可懂度自动指标**: whisper tiny (CPU int8) 回译 CER 接入样张集报告 (复用 M4 `compute_cer`)
- [x] **8.3 音色相似度指标**: TimbreEncoder embedding 余弦 (代理指标; resemblyzer 因 webrtcvad 无法安装)
- [x] **8.4 回归门禁**: `tests/test_encoding_health.py` 4 项 (非空率 100% / UNK<10% / G2P 非空 / 短序列不崩)
- [x] **8.5 基线报告**: `output/sample_pack/baseline_3550_fixed/report.json` — **CER=1.00 (全部不可懂), RMS=0.51**, 实证一期权重不可用
- [x] **8.6 backbone 插件接口定型**: `BackboneCapability` 位掩码 + `sample_stream` 流式接口 + `supports_mode`, SoVITS 声明 TTS|SVS|F0_COND|LORA_READY

**验收**: ✅ 一条命令出「20 条 wav + CER + 音色分」报告; 接口单测 8/8; 全量回归 181 passed。

**已知局限**: 基线 CER 用 whisper tiny 测量, 对歌声识别偏弱, 数字偏保守; 二期复测建议换 small。

### M9: ADR-2 架构实现 (第 2-4 周) — 进行中 (9.1/9.2 已完成并验证)

- [x] **9.1 MAS 对齐模块** ✅ (2026-08-13): `adr/models/alignment.py` — maximum_path DP + durations + gaussian_logp; 6 单测全过 (含已知分段精确恢复); OpenCpop 真实形状验证 (353帧×11音素, 5ms)
- [x] **9.2 VAE 后验 + flow 先验** ✅ (2026-08-13): `adr/models/vae.py` (PosteriorEncoder + ResidualCouplingFlow) + `adr/models/adr2.py` (端到端可训骨架)
  - **关键修复**: KL warmup (1000 步) + prior_proj 小初始化 — 无 warmup 时训练中后期发散 (loss 33k→24)
  - **公平对照** (3550 句 × 6 epoch, medium): teacher-forced mel corr **0.675** vs 一期 SoVITS **0.481** (+40%), L1 1.18 vs 1.36
  - 采样路径现状: 结构糊 (mel_head 3 层卷积占位太弱 + 时长预测偏短) → 正是 9.3/9.5 要解决的
  - 可视化: `output/mas_check/adr2_sample_vs_gt.png`; 样张集: `output/sample_pack/adr2_3550_6e_warm/`
- [x] **9.5 FlowDecoder** ✅ (2026-08-13, 流式推理留到 M10): `FlowDecoder` (mel 空间条件流, 精确 NLL 免 GAN) 替换 mel_head 占位
  - teacher-forced corr **0.961** (vs 一期 0.481, **+100%**), L1 0.44; 采样 mel 出现清晰谐波结构
  - **三连坑与修复实录** (都写入代码注释, 防复发):
    1. KL warmup 1000 步 → 后验 z 失控膨胀, warmup 结束后 MAS 必崩 → 固定先验 std=1 + warmup=1
    2. prior_proj 小初始化 → mu_p 趋同 → MAS 冷启动退化 → 撤回
    3. **MAS 与 KL 空间不一致** (MAS 用 raw z, KL 推拉 flow(z)) → 随 flow 训练漂移错位 → MAS 改到 z_p 空间 (VITS 原文做法) ✅ 对齐恢复健康 (最大时长占比 0.90 → 0.39)
  - 诊断工具: `scripts/eval_adr2.py` (TF corr / MAS 健康度 / 采样统计一键出)
- [x] **9.3 随机时长预测器** ✅ (2026-08-13): 高斯随机时长头 (NLL 训练/采样推理) + 双向 clamp + Glow-TTS 式输入 detach
  - **又两个坑与修复**: ① 均摊对齐用 batch T_max 而非逐样本真实长度 → padding 摊进时长, 时长头学到 63 帧/音素 (v9/v10 bug) ② 小实验跑不出 warmup 区导致误判 (train 集歌声 55 帧/字 vs 认知的 12)
  - 结果 (3550×6ep): TF corr **0.966**, MAS 健康度 0.231, 采样时长 311 vs GT 353 ✅
  - 样张集: 3/20 条 ASR 可转录 (CER 1.59-3.54), 其余仍空 — 有语音内容但不可懂
- [x] **9.4 双模头 + F0 旋律注入** ✅ (2026-08-13): mode embedding (TTS/SVS) + F0 注入 FlowDecoder 逐帧条件 + 推理自动从参考音频提取 F0 (F0Extractor); 4 项新单测
- [x] **9.6a 声码器域对齐 (计划外重大发现)** ✅: **BigVGAN 配置 fmax=8000 vs 我方 mel fmax=11025** — 声码器与 mel convention 不匹配, GT mel 直送都乱语, 此前所有"噪声"评估全部被此污染!
  - 修复: `compute_mel_bigvgan` (slaney norm / fmax=8000 / center=False reflect pad / log clamp 1e-5, 与 BigVGAN 训练严格一致)
  - 验证: GT wav → mel_v2 → BigVGAN → ASR 转录与原始音频一致 ✅
  - 全量 npz 重建 (3756 句) + pipeline 推理同步切换
- [ ] **9.6b 端到端联调 + LoRA/QLoRA 实测 8GB**

**当前最佳 (v13 = ADR2 + F0 + mel_v2, 3550×6ep, 已固化 `examples/adr2_best/`)**: TF corr 0.966, MAS 0.353, 采样 T 305/353, CER 1.157 (5/20 可转录)。
**v14 (20ep) 实验证明过拟合**: CER 反升至 1.667, TF corr 降至 0.954 — 6ep 左右是甜点, 不要盲目加 epoch。
**人耳验收 (2026-08-13)**: 用户确认质量提升, TF 重建已能听出歌词片段, 但**电音感严重** (声码器域差二期实锤)。
**已知问题**: ① 自动 F0 提取慢 (10-16s/条, pyin CPU) — 合成延迟主因, 待优化 (缓存/可选关闭) ② 输出 RMS 经归一化修复 (gain cap 10x) ③ 剩余差距: 声码器对抗微调 (需自写 MPD/MRD) 或保底路线。

### 保底路线启动 (2026-08-13, 用户选定路线 2): DiffSinger v1 官方预训练套壳

- [x] **B1 可行性刺探** ✅: MoonInTheRiver/DiffSinger v1 源码 + 官方 OpenCpop 预训练三件套 (声学 0831_opencpop_ds1000 / 声码器 0109_hifigan_bigpopcs_hop128 / PE 0102_xiaoma_pe) 在本机 (py3.13 + torch2.x) 跑通
  - 兼容补丁: scipy.kaiser 迁移、load_ckpt Windows 路径正则、补装 h5py/pycwt/skimage/pyloudnorm/webrtcvad-wheels/pytorch-lightning
  - 下载: ghproxy.net 镜像 + Range 断点续传 (声学 360MB / 声码器 899MB)
- [x] **B2 官方推理基线** ✅: `output/ds_baseline_2044001628.wav` — ASR "余林世了天空 回得更强久" vs 参考 "雨淋湿了天空灰得更讲究", 近似全对 (同音字级), **比 whisper 对真实录音的转录还准** → 这就是质量天花板参照
  - 推理入口: `third_party/diffsinger_v1_code/run_adr_probe.py` (吃 OpenCpop 标注 json)
  - 速度: 1000 步扩散 26s/4s 音频 (PNDM speedup=40 可调)
- [x] **B3 套壳集成** ✅: `adr/models/diffsinger_engine.py` (进程内单例引擎, 懒加载+锁+cwd 上下文) + WebUI 新增 **Sing Tab** (选 OpenCpop 句子 / 自定义 ph+note+dur 高级区); 回调端到端实测 ('你说你不懂为何在这时牵手' 3.91s 合成成功)
- [x] **B4 任意文本桥接** ✅ (2026-08-15): `adr/models/melody_bridge.py` (F0→音符量化: 时间轴按音节均分 + 浊音中位 midi + rest 检测) + 引擎 word 级通道 + `sing_like(歌词, 参考音频)` 一键接口 + WebUI Sing Tab 接入 (歌词+参考音频输入)
  - 端到端实测: 歌词'今天的天气真不错我们一起去公园散步' + 无关歌曲旋律 → ASR '今天的天氣全部錯我每一期去公園散播' (13/16 字正确或同音) ✅
  - 音色注记: 官方模型无克隆能力, 输出为 OpenCpop 训练歌手音色; 克隆属 ADR-2 路线
- [ ] **B5 ADR-2 并行迭代不停**: 自研架构继续 (保底不等于放弃); 声码器电音感可借鉴官方歌声 NSF-HiFiGAN (hop128/24k) 替换 BigVGAN

### 保底第二腿 (2026-08-15): GPT-SoVITS 说话克隆套壳

- [x] **C1 GPT-SoVITS 集成** ✅: 官方 v2final 预训练 (hf-mirror) + `adr/models/gsv_engine.py` 套壳 (同 DiffSinger 模式: 单例/锁/cwd 上下文) + **Clone Tab 引擎选择器** (GPT-SoVITS 保底默认 / 自研 ADR-2 实验)
  - 兼容补丁: jieba_fast→jieba 回退 (py3.13 无轮子)、torchaudio.load→soundfile shim (torchcodec 无 Win 轮)、int16 量程归一化
  - 依赖补装: x-transformers/fast-langdetect/split-lang/cn2an/jieba/opencc/wordsegment/g2p-en + nltk cmudict/tagger (ghproxy 手动)
  - 验收: 10s 朗读 ASR 大体正确 (output/gsv_acceptance.wav); WebUI 回调端到端通过
- [x] **C2 差异化保住** ✅ (2026-08-16): `scripts/gsv_finetune.py` 微调一条龙 (切片→FunASR→BERT/HuBERT/语义→s2/s1) + 引擎热换微调权重; **用户真人录音 (3.8min) s1 微调 8ep 完成**, 验收 `output/user_clone_finetuned_s1.wav` ASR 近全对
  - 环境攻坚战记录: gloo 分布式在本机彻底不可用 (hosts 劫持 localhost→kubernetes.docker.internal + 多网卡) → s1/s2 单卡直通补丁 (跳过 DDP); torch 2.6 weights_only 兼容; PL 残留 ckpt 需清理; 权重输出根目录需预建
  - **未完成**: s2 (VITS 声学) 微调在 epoch2 中段挂死 (GPU 100% CPU 冻结, 疑似 CUDA kernel 死锁) — 音色细节微差的主因; 后续可换机/容器重试或查 eval 阶段
  - s1-only 微调已验证可出权重并热换推理

### C3 流式 + 音色持久化 (2026-08-16)

- [x] **流式生成** ✅: `GSVEngine.synthesize_stream()` 逐块 yield (官方 streaming_mode), 实测 3 块/6.7s 音频, 热机首块延迟 ~17s (冷启动含权重加载 ~90s); WebUI 边生成边播放属增强项未做
- [x] **音色档案库** ✅: `adr/models/voice_library.py` (data/voices/<name>/ = ref.wav + meta.json 含微调权重路径) + Clone Tab「已存音色」下拉 +「保存当前参考为音色档案」按钮; 已建档 '我的声音' (绑 s1-e8 微调权重), 回调实测: 不传参考音频直接合成成功 (47s)
- 注: DiffSinger 唱歌腿为 1000 步扩散, 天然不流式

### C4 客观相似度门禁 (2026-08-16)

- [x] **ERes2Net 声纹相似度** ✅: `adr/eval/speaker_sim.py` (官方 sv 权重, 16k fbank 余弦); 「像不像」从耳朵变成数字
- **实测结论 (用户录音)**:
  - 零样本克隆 0.641 > s1 微调 0.546 — **s1 微调降音色相似度** (s1 管韵律, 音色是挂死的 s2 的活), 微调权重暂从档案摘除
  - 负对照 0.17-0.21, 门禁区分度正常
  - **参考段选择是最大杠杆**: 同一录音不同 7s 段, 克隆保真 0.341~0.749! 扫段后最优段 sim=0.749, 最终验收 9.9s 合成 **sim=0.779** (>0.70 目标达成) → `output/user_clone_best.wav`, 档案「我的声音」已更新为零样本最优段

**产品形态**: Clone Tab (说话+克隆, GPT-SoVITS 保底) + Sing Tab (唱歌, DiffSinger 保底) + 自研 ADR-2 并行迭代。

**验收 (硬门禁)**: 样张集 CER ≤ 15% (当前 1.23, 从 1.0 改善但未达标), mel corr ≥ 0.7 (TF 已达 0.966); **M9 末不达标 → 启动保底路线 (DiffSinger 套壳)**。

### M10: 体验收口 (第 5-6 周)

- [x] **10.1 WebUI 双模接入** ✅ (经保底引擎超额交付: Clone/Sing 双 Tab + 引擎选择器)
- [x] **10.2 唱歌模式** ✅ (B4 旋律桥: F0 自动提取 → 音符量化 → DiffSinger)
- [x] **10.3 长文本流式** ✅ 引擎层 (首块 17.4s, **<300ms 未达**, 列入遗留)
- [x] **10.4 8GB 全链路复测** ✅ (2026-08-16): `output/gpu_benchmark.json` — GSV 推理 1.2GB / DiffSinger 推理 0.3GB / s1 微调 7.9GB 贴顶通过 / s2 微调环境阻塞
- [x] **10.5 文档** ✅ (2026-08-16): 二期验收报告 `docs/m8-phase2-acceptance.md`

**验收**: 新用户从安装到出声 ≤ 30 分钟 (含训练), 样张集 MOS ≥ 3.0。

---

## 第六部分: 风险与对策

| 风险 | 概率 | 对策 |
|------|------|------|
| MAS 实现复杂度高 (动态规划 + 可微近似) | 高 | 参考 VITS 论文伪代码自实现, 先用 toy case 验证对齐; 卡住则用 CTC 对齐替代 |
| flow 解码器音质不及扩散 | 中 | 增层数到 8; 仍不行则在 flow 后接 4 步轻量扩散精修 (混合方案) |
| 双模训练互相干扰 | 中 | mode 分批采样训练; 必要时分头不共干 |
| 流式与离线质量不一致 | 低 | chunk 边界对齐音素边界, 首 chunk 加 lookahead |
| 二期再次"指标绿但难听" | **低 (M8 防线)** | 样张集听审是硬门禁, 一票否决 |
| M9 末仍不达标 | 中 | 保底路线已备好 (DiffSinger 套壳, 共用插件接口) |

## 第七部分: 明确不做的

- ❌ 不照抄 VITS 完整对抗管线 / GPT-SoVITS 语义 token 体系 (与低资源目标冲突)
- ❌ 不做 GGUF/AWQ 导出 (M3 继续延期, 等音质达标)
- ❌ 不做多语言/多说话人 (先把中文单说话人做能听)
- ❌ 不重写 WebUI 框架 (Gradio 够用)
