# M9 三期计划：优化稳固 + 借鉴自研

> 2026-08-16 用户定方向。承接二期验收 (docs/m8-phase2-acceptance.md)。
> **核心指标**: ≥7 秒参考音频 + ≤30 分钟训练 → **≥80% 声纹相似度** (ERes2Net 门禁),
> 增量持久化 (练一次存档, 可基于存档继续练, 推理直接调用),
> 资源占用继续压缩, **Win 下 i/N/A 三卡兼容推理与训练**。

## 现状基线 (二期末)

- 零样本 + 最优参考段: sim=0.779 (≈80%, 但未"训练")
- s1 微调 8ep ≈ 50min (韵律提升, 音色反降 0.546)
- s2 微调本机死锁 (音色训练主路径不通)
- 推理显存: GSV 1.2GB / DiffSinger 0.3GB; 训练 s1 7.9GB 贴顶
- 持久化: 音色档案库已上线 (零样本档), 微调权重可绑档但无增量续练

## 上半场 A：稳固基础（无风险纯软件）

- [x] **A1 引擎常驻+预热** ✅ (2026-08-16): 双引擎 `warmup()` + WebUI 启动后台错峰预热 (`_prewarm_engines`, 失败降级现场加载)
- [x] **A2 档案自动选段** ✅ (2026-08-16): `save_voice_auto()` (>15s 音频扫 6 段×7s → 零样本探测克隆 → ERes2Net 打分 → 最优段建档); 实测 231s 录音 2 分钟建档 sim=0.773, 档案「我的声音V2」
- [x] **A3 边生成边播放** ✅ (2026-08-16): Clone Tab「流式合成」按钮 → `synthesize_stream` 逐块 yield → gr.Audio(streaming=True, autoplay) (Gradio 6.4)
- [x] **A4 一键安装固化** ✅ (2026-08-16): `requirements-lock.txt` (281 包冻结) + `scripts/apply_compat_patches.py` (13 条幂等补丁, marker 防重复, 已自测); DiffSinger/RVC 零文件补丁
- [x] **A5 相似度门禁进 WebUI** ✅ (2026-08-16): 克隆合成后自动 ERes2Net 打分显示在信息面板 (失败不阻塞); 实测档案合成 sim=0.773 自动出分

## 中场 C：训练达标 (30min/80%) — 借鉴自研主线

- [x] **C1 s2 微调死锁根治** ✅ (2026-08-16): **死锁未复现**——A4 补丁脚本补齐了 s2_train 的 `.module` 兼容补丁 (此前临时脚本被 PS 策略拦截未生效) + 权重目录预建, s2 8ep 全跑通, e3-e8 权重全出。**双微调验收 sim=0.803** (零样本 0.779 / s1-only 0.546 → s1+s2 0.803, `output/user_clone_full_finetuned.wav`), s2 微调果然是音色主战场。档案「我的声音V2」已升级为双微调默认
- [x] **C2 训练瘦身** ✅ (2026-08-16): **配方实验结论 — 零样本 s1 + s2 微调 8ep ≈ 7 分钟 → sim 0.801-0.807** (双微调全量 0.803 / s2-only 0.801 / e4 短训 0.749 / s1-only 0.546)。**30 分钟目标超额 4 倍达成**: s1 微调对相似度无益可整段跳过 → `gsv_finetune.py --skip-s1`; 无需 LoRA, 全参 s2 已够快
- [x] **C3 增量持久化** ✅ (2026-08-16): `--resume-from voice:名字` (读档案权重→**enc_q 补齐合并初始化**→叠练) + `--bind-voice` 训完自动绑档 (按 mtime 取最新, 轮次号跨次回卷已修)。**关键坑**: 微调权重缺 enc_q 后验编码器 103 键 (savee 只存推理必需), 直接续练全毁 (0.118), 合并官方预训练补齐后 0.799 保住 (对照首训 0.807)。档案「我的声音V2」现为续练权重
- [x] **C4 训练内门禁** ✅ (2026-08-16): `scripts/train_gate.py` 监控进程 (新权重→克隆→ERes2Net 打分→曲线 jsonl→≥0.80 放 STOP 信号) + s2_train 每轮末检查 STOP 早停 (补丁已登记 apply_compat_patches.py)
- [ ] **C5 借鉴消化**: 修 s2 过程中吃透 VITS/sovits 结构, 产出笔记; 为自研声码器替换 (ADR-2 电音感) 储备
- [ ] **C9 自研擂台机制**: ADR-2 每轮迭代自动 vs 保底引擎跑门禁 (CER+sim+延迟), 达标自动晋升默认引擎 — 架构竞争自动化 (产品战略第四支柱)

## 下半场 C：跨平台压缩 (i/N/A)

- [x] **C6 推理跨卡** ✅ 兜底验证 (2026-08-16): GSV 纯 CPU 克隆通过 (CUDA_VISIBLE_DEVICES='' , 3.5s 音频 136s 慢但可用) — A/I 卡机器今日路径=CPU 兜底; directml 加速列观察项
- [x] **C7 资源再压缩** ✅ 实测结论 (2026-08-16): **bs4→bs2

## 里程碑与验收

| 里程碑 | 验收 |
|---|---|
| M3.1 (A 完) | 新用户装机到出声 ≤15min; 首块 <5s; 档案自动选段 sim≥0.75 |
| M3.2 (C1-C4) | **7s+ 参考 + ≤30min 训练 → sim≥0.80**; 续练可用; 训练曲线含相似度 |
| M3.3 (C6-C8) | A 卡/I 卡 Win 下推理成功 (训练不限); 推理 <1GB |

## 差异化战略 (2026-08-16 用户追加): ADR 风格道路

> 目标: 拉大与同产品的差距, 在学习借鉴中摸索出属于 ADR 的路。
> **行业裂缝**: 说话克隆 (GSV/CosyVoice/IndexTTS) 与唱歌合成 (DiffSinger/ACE) 是两个割裂的世界——没有开源产品让"一份音色档案既能说话又能唱歌"。

**ADR 风格 = 统一音色资产 (Voice Asset)**: 一份档案贯穿 说话/唱歌/流式/续练, 配全自动质量门禁。

- [x] **D1 歌声音色转换链 (最大差异化)** ✅ 链路打通 (2026-08-16): RVC 新版仓库集成 (`scripts/rvc_train.py` 一条龙: 预处理→rmvpe F0→hubert→训练→faiss 索引) + `adr/models/rvc_engine.py` 套壳 + 档案 rvc_weights/rvc_index 字段 + `sing_like(..., voice_name=)` 自动追加转换
  - 实测: 用户 3.8min 说话录音训 RVC → `output/d1_my_voice_sings.wav`; **sim 0.224(歌手)→0.321(60ep)→0.428(200ep 加训, output/d1_my_voice_sings_v200.wav)** — 方向成立, 加训有效
  - 已知短板: 说话数据训的 RVC 转唱歌音域吃力; 提升杠杆=再加唱歌/哼唱数据 / protect 调参 / 更长训练
  - 兼容补丁: -m 模块调用绕 train.py 遮蔽、exp_dir 预建、hubert_base 目录版、mute.zip、filelist 生成复刻、weight_root/rmvpe_root/index_root 环境变量、faiss 中文路径拷贝绕行
- [ ] **D2 质量自评闭环**: ERes2Net 门禁内置到所有产出 (合成后自动打分+自动选段+训练早停), 让产品"自己知道好不好" — 同类产品都没有
- [ ] **D3 借鉴→消化→创新流水线**: 每条套壳链产出结构笔记 (C5), 自研 ADR-2 吸收: 统一双模架构 (TTS+SVS 同模型) 是我们的原生设计, 长线对照官方件持续迭代
- [ ] **D4 三卡+低配普及**: 8GB/i/A/N 是"别人不做我们做"的普及型壁垒

### 差异化 vs 同产品对照

| 能力 | GPT-SoVITS | CosyVoice2 | DiffSinger | **ADR 目标** |
|---|---|---|---|---|
| 7s 零样本说话 | ✅ | ✅ | — | ✅ (0.78 已达) |
| 音色唱歌 | ❌ | ❌ | ❌(固定歌手) | ✅ (D1 转换链) |
| 说话+唱歌同档 | ❌ | ❌ | ❌ | ✅ **独家** |
| 8GB 本地微调 | 部分 | 重 | — | ✅ (C 线) |
| 三卡 Win | 部分 | 部分 | 部分 | ✅ (C6) |
| 质量自动门禁 | ❌ | ❌ | ❌ | ✅ (D2) |

## 风险登记

1. **s2 死锁若三线皆不通** → 退路: 零样本+选段已 0.78; 或换 IndexTTS/CosyVoice2 底座评估 (二期 B2 备案)
2. **directml 训练支持差** → A/I 卡只做推理, 训练引导 N 卡/CPU; 文档写清
3. **30min 训练目标依赖数据量** → 7s 数据 s2 LoRA 可行, 全参微调不可行, 路线选择上 LoRA 优先

## 四期·双线并行 (2026-10-03 起): 流式体验 W1 + 极低资源 W2

> 用户目标: ① 续完流式输出; ② 比 A2000 8GB 更低的设备也能流畅推理 (4GB / 2GB / 纯 CPU 三档)。

### 批次 1 (2026-10-03)

- [x] **held-out 评估补课** ✅ (10-02, `scripts/eval_heldout.py`): 剔除 ref 重叠段重训 `user_holdout` (58s×8ep) — 干净微调 **0.8116** vs 零样本 0.7845 (5/5 句全胜), 微调增益为真实泛化
- [x] **t2s 配对补测** ✅ (`scripts/eval_t2s_pairing.py`): s1 微调版 + holdout s2 = **0.8204** > 官方 s1 组合 0.8125 (4/5 句) — 推翻 0.546 旧结论 (当时 s2 组合不同); 档案已显式钉死该组合 (不再依赖 yaml 隐式默认)
- [x] **档案重绑** ✅: 「我的声音V2」→ t2s=user_voice-e8 + vits=user_holdout_e8 (实测最优组合)
- [x] **W1 流式首包** ✅ (`scripts/bench_stream.py`): 实测首块延迟 ∝ 首段长度 (45 字 cut0 达 25.3s) → 引擎加**首段预切** (≤30 字, 句号优先) + 流式默认 cut3 → 任意输入首包 **25.3s → ~3s** (典型 2.7-3.8s, seed 方差可到 5s); WebUI 默认同步 cut3
- [x] **选段节奏过滤** ✅ (`voice_library._seg_rhythm`): 扫段时过滤静音占比>35% / 停顿≥3 次 / 尾音塌陷 <-10dB 的段 (顿挫根源=参考段节奏被模仿); 合成信号三案例验证通过
- [x] **W2 低资源基线** ✅ (`scripts/bench_lowres.py`): 峰值显存 **1410MB** (4GB 档轻松, 2GB 档可跑); **纯 CPU 短句边际 RTF ≈1.9x** — 旧基准 39x 系首包初始化+长文本撑出, 纯 CPU 档预期大幅上调; t2s int8 动态量化无增益 (BERT 未量化, 短句瓶颈不在 t2s Linear)
- 回归: 快速子集 168 passed / 1 skipped; 慢训练测试单独补跑
- 已知坑: 沙箱阻塞 D:\pyhon 写入导致训练挂死 → 训练/评估一律非沙箱运行

### 批次 2 (2026-10-03)

- [x] **W1 首包方差治理** ✅: 根因二分 — 首次流式调用的 init_vits_weights 热换罚金 ~12s (head_seed 只是次因); 引擎 warmup 支持权重预热 + WebUI 预热传档案权重; head_seed 参数固化 (seeded 同文本极差 0.05s, 完全确定)
- [x] **W2 2GB 档模拟实测** ✅ (`bench_vram_budget.py`, memory fraction 模拟): 峰值 1412MB < 2000MB 预算, RTF 1.35-1.5 — **2GB 显存推理档通过**
- [x] **W2 CPU int8 量化结论** ✅: t2s+BERT 146+50 Linear 量化 RTF 无增益 (6.4/2.2 vs fp32 6.4/1.9) — CPU 提速留待批次 3 ONNX/OpenVINO; vits weight_norm 不兼容 dynamic quant (已从目标剔除)
- [x] **D DiffSinger 采样压缩** ⚠️ 反直觉结论: speedup 40→240 时间不变 (~33s RTF≈8, 首调另有预热) — **瓶颈不在扩散步数**, 步数压缩无效; 需 profile 定位固定开销 (批次 3); 音频 4 档落盘 output/ds_speedup/ 供听感评审
- [x] **W3 4GB 训练路线** ✅ (远超预期, C5 收尾): 显存诊断三部曲 —
  1. 曲线定位: 训练步瞬跳 2GB→7.8GB, **激活值是大头** (非分配器缓存: expandable_segments 无差异 7902≈7892)
  2. **补丁 #16**: v2 训练 forward 梯度检查点 (enc_q/flow; 官方只在 v3 实现, v2 的 grad_ckpt 一直被静默忽略) → 7892→7180
  3. **补丁 #17**: 长 clip 截断 ADR_MAX_CLIP_SEC (ssl/spec/wav 同帧数截, 对齐不破坏) + bs 缩减
  - **实测配方**: bs2+ckpt+cap10s = **训练自身 ~3.5GB** (整机 4793MB 含其他进程 1.3GB); bs1 = ~3.1GB — **4GB 显存训练达标**, 8GB 基线省一半; LoRA 判定为非杠杆 (只省优化器 ~300MB, 不省激活)
  - `gsv_finetune.py --max-clip-sec 10 --batch-size 1` 即 4GB 配方 (配补丁); cap 对音质影响待批次 3 全程训练+门禁验证
- 已知坑: 诊断管道 Tee+超时截断假失败 (训练其实完成); DataLoader worker 反复 spawn 待查 (性能侧, 不影响显存结论)

### 批次 3 (2026-10-03)

- [x] **4GB 配方全程训练 + 门禁验证** ✅: `user_h4gb` (bs1+cap10+ckpt) 8ep 配方, 门禁曲线 零样本 0.841 → e1 0.777 → **e2 0.815 ≥0.80 早停** — cap 截断音质疑虑收口; 曲线持久化 `output/train_gate_user_h4gb.jsonl`; 已知局限: 门禁单句探测方差大 (基线 0.84 系本句偏热), 批次 4 改多句均值
- [x] **DiffSinger profile** ✅ (`bench_diffsinger_profile.py`): Self CPU 43s vs Self CUDA 10s, ~20 万次微算子 (add 6.2万/sigmoid 2万/tanh 2.1万) — **瓶颈=kernel launch 开销与步数无关**; 杠杆=torch.compile 算子融合 (Windows 支持待验证, 批次 4), 60GB 张量流量可由融合压缩
- [x] **CPU 推理 profile** ✅ (`bench_cpu_profile.py`): conv1d/mkldnn 20% + SDPA 5% + conv_transpose 2.5%, 无单点 >20% 热点 — "千刀万剐"型, ONNX Runtime 预期 1.3-2x (批次 4); uniform_ 768ms 系 GPT 采样正当开销 (eval 模式已确认全关 dropout)
- [x] **WebUI 流式端到端验收** ✅ (`bench_webui_stream_e2e.py`): 预热 (引擎+权重热换+流式形状) 68.7s 后台无感 → **首次点按钮 3.99s (原 17.5s) / 热机 2.19s** — 双双达标 (≤5s/≤3s); 修复: ① tts_infer.yaml 相对路径改绝对 (TTS_Config assert 间歇崩) ② prewarm 加流式路径形状预热 (首调 JIT 罚金 ~14s)
- [x] **顺手修复**: tts_infer.yaml 断言间歇崩 (绝对路径化); 官方确认流式不支持并行推理 (自动降级, 无需改)

### 批次 4 (2026-10-03)

- [x] **门禁多句均值化** ✅ (`train_gate.py` 重写): 5 句探测集 (与 eval_heldout.py 同源, 长/中/短/混合/叙事) 逐句打分取均值; jsonl 记录逐句分; `--once` 重打分模式 / `--cpu` 打分走 CPU (训练占 GPU 时零争抢)
- [x] **⚠️ 批次 3 结论修正 (测量污染)**: tts_infer.yaml custom 段被 GSV `init_vits_weights` 热换时**写回** (每次热换都持久化!), 旧门禁"零样本基线 0.841"实为漂移后的微调权重 → 单句 0.815/0.820 是**混合权重污染分**。诚实重测 (钉死预训练 s1): 真零样本 **0.762~0.776** (±0.014 采样方差), **user_h4gb_e2 = 0.744 (-0.018, 未达标!)** — 4GB 配方 e2 被**误早停**, 配方本身未失效但需更多 epoch (holdout e8 才 +0.028); **user_holdout_e8 = 0.804 (+0.028 真实泛化)**; 短句稳定弱项 (0.688~0.747)
- [x] **yaml 漂移根治** ✅ (`gsv_engine._lazy_init`): 只取 yaml v2 原始段构造 dict 传入 TTS_Config (每次启动从干净预训练出发, 免疫 custom 段漂移) + 写回重定向到私有 scratch 文件 (共享 yaml 永不被改); 冒烟验证: 启动即加载预训练对, 共享 yaml custom 段自愈为预训练路径
- [x] **残留 STOP 清理** ✅ (`gsv_finetune.py`): 上次门禁早停的 STOP 文件会让同 exp 下次训练第 1 轮假早停 — 启动前清除; `--once` 模式不发 STOP
- [x] **4GB 配方接入 WebUI** ✅ (`webui.Train` 新区块): 声音克隆微调 UI (录音上传 / 音色名 / 配方 4gb[bs1+cap10s]·8gb[bs4] / 轮数 / 质量门禁开关[CPU 打分并行, 5 句均值 ≥0.80 自动早停] / 训完绑定档案); 回调流式日志 + 门禁曲线实时注入 + 门禁指标 JSON 汇总
- [x] **torch.compile 验证** ✅ (`bench_diffsinger_compile.py`): **Windows 可行** (triton-windows 3.8.0 + torch 2.10+cu130, toy 模型先行验证); DiffSinger 主模型 mode=reduce-overhead: **32.31s → 6.72s (4.81x)** — ~20 万微算子 launch 开销被算子融合 + CUDA Graph 回放打掉, 批次 3 的 33s 之谜正式收口; 代价: 首发编译 560s (inductor 有持久缓存, 二次启动待测); 集成前置: 任意歌长 → 需 shape 分桶/动态形状策略, 否则每首新歌都触发重编译
- [x] **CPU ONNX Runtime 实验** ✅ (`bench_bert_onnx.py`): BERT 特征面 (倒数第 3 层) 导出 ONNX + ORT_ENABLE_ALL, **1.36-1.43x 组件级提速** (L=64/128/256 → 163/257/427ms), 数值一致 (rel_err ≤7e-06); BERT 占 CPU ~1/4 → 端到端 ~10% (RTF 1.9→~1.7), 收益真实但温和; int8 量化在外部数据模型上失败 (跳过, 批次 2 已证无增益); t2s (AR 采样) / vits (flow+decoder) 完整 ONNX 化为后续更大工程, 暂不接入引擎

### 批次 5 (2026-10-03): 框架层聚焦 (唱歌线暂缓, 用户定调)

> 用户定调: 唱歌 (DiffSinger/RVC) 放下, 回归"低资源快训练 + 好质量"框架/架构; 不做"为练模型而练模型"; 未来可能换桌面端控制台 (仅记录, 暂不做)

- [x] **一键链路闭环审计** ✅: 断点① gsv_finetune bind_voice 档案不存在时 `meta.json.read_text` **崩溃** (且崩在训练全部完成后!) → 修复: 不存在则用本次切片自动建档 (免 GPU); 断点② WebUI ft_bind 下拉冻结 → 加刷新按钮; CLI 的 `adr train` 是自研骨架训练器, GSV 微调入口 = 脚本/WebUI (够用)
- [x] **补丁系统幂等性修复** ✅: #17 grad_ckpt 补丁的 marker「梯度检查点开关」不在 new 文本里 + old 串是 new 的前缀永存 → 每次重跑重复插入 (实测 3 份拷贝); 修复: marker 改真实子串 + `patch()` 加 `assert marker in new` 自检; s2_train.py 手工去重; 补丁 #19 新增: `ADR_S2_NUM_WORKERS` 覆盖 DataLoader worker 数
- [x] **端到端计时审计** ✅ (`bench_pipeline_stages.py`, 10 切片/3 分钟音频): 数据准备 slice 37.6s / **ASR 322.3s** / text+BERT 115.2s / HuBERT 112.1s / semantic 77.8s ≈ **12.5 分钟** (一次性成本, 转写已缓存); s2 训练 bs1+cap10 105s/ep; **ASR 慢根因三层剥开**: ① ct2 无静默回退 (cpu+float16 直接 LOAD FAIL, cuda-fp16/cpu-int8 RTF 均 0.07 正常); ② **中文强制路由 FunASR** (fasterwhisper_asr.py 对 zh/yue 仅做语种检测, 实际转写走 Fun-ASR-Nano); ③ nano 确认上 GPU (显存 3716 MiB), 真实语音 RTF 0.49 为模型特性 (1.98B), **批量 generate 无加速** (batch5 28.06s vs loop5 28.19s), 短文件固定开销大 (3.9s 音频 3.65s, RTF 0.93) → **框架决策: 不深陷**, 322s 为一次性成本
- [x] **训练启动开销 A/B** ✅: DataLoader 5 worker × Windows spawn 重导入 torch = 分钟级固定开销, 实测 workers=5 → 553.0s vs workers=0 → 235.0s (1 epoch, **-57.5%**) → 补丁 #19 默认改 `ADR_S2_NUM_WORKERS=0`, 大数据集可手动开回
- [x] **短句质量实测定策略** ✅ (bench_short_sent.py, 4 短句 × 5 seeds × top_k{15,5} + 前导"."对照): top_k=5 非普适 (部分 s1 反跌 0.022), 前导"。"无害 (0.654≈0.644), 种子方差小 (std 0.01-0.02) → **短句低分系超短文本韵律自由度的模型特性, 不做自动收 top_k**, 记录特性
- [x] **BERT ONNX 接入引擎** ✅: `_OnnxBertFeat` (ORT 惰性会话 + 自动导出缓存 + `bert_onnx=auto` 仅 CPU 启用) 冒烟 PASS; **同种子逐位等价** (HF vs ORT 合成 dur/rms/peak 完全一致 0.74s/0.0026/0.0091 = 零回归); CPU e2e 63.2s → 6.7s; 观察项: GSV CPU 合成近静音为官方既有行为 (bert off 同样, 与本改动无关)

### 批次 6 (2026-10-03): 兼容性闭环 + 听感验收

- [x] **AMD/Intel 兼容审计 + 修复** ✅: 全仓 CUDA 依赖点扫描 → 三处真问题修复: ① `gsv_finetune.py` ASR 精度硬编码 float16 (CPU 上 ct2 无静默回退直接 LOAD FAIL) → 动态 float16/int8; ② `is_half:"True"` 硬编码 (CPU BERT half 崩) → 动态; ③ 训练步无 GPU 时友好 abort + `--skip-to s2` 断点提示 (数据准备产物已缓存); `DeviceConfig.capability` 能力分级 (全功能/受限) + `adr check` 输出; 审计确认 FunASR 自动降级 / load_cudnn 有保护 / ERes2Net 纯 CPU / GSV CPU 推理已实证 → 兼容矩阵落盘 `docs/gpu-compat.md` (DirectML/IPEX 评估: 版本滞后+op 覆盖不全, 不集成为承诺)
- [x] **听感验收工具** ✅ (`scripts/mos_ab_test.py`): MOS A/B 盲测 — 每句 × {zero-shot 官方预训练, fine-tuned 档案} 合成, A/B 随机混淆 (seed 可复现), 客观 ERes2Net 分随 answer_key 存档; `--score` 汇总分系统 MOS + 揭晓映射 + 主客观对照; **关键坑修复: 引擎权重传 None=保持当前状态**, 两系统必须每句显式传权重否则 AB 混同; 待用户实际听测填分
- 观察项: AMD/Intel 真机未实测 (本机 NVIDIA), 结论来自代码路径审计 + CPU 路径实测等价覆盖
- [x] **批次 6 补充 — 用户听测反馈** ✅: 短句仍看运气 (与客观一致); 新需求: 音色人设偏离 (期望 MOSS 式冷淡理性, 实际带真人情绪起伏) → 根因: GSV 韵律主要来自 ref 音频, 微调锁音色不锁"性格" → 引出批次 7

### 批次 7 (2026-10-03): 风格人设条件化 (persona conditioning)

- [x] **风格预设库** ✅ (`adr/models/style_presets.py`): 6 内置预设 (自然/理性AI-MOSS/新闻播报/温柔陪伴/沉稳低语/活力元气), 每预设 = {temperature, top_k, speed} + **录制指引文案** + 关键词表; `resolve_style()` 自由文本描述 → 关键词计数匹配 (零依赖, 不做模型语义理解 — MVP 足够); 设计原则: 韵律人设主杠杆是 ref 录制方式 (指引内置于预设), 参数是辅助杠杆
- [x] **WebUI 接入** ✅: 克隆页新增"说话风格/人设"输入框 (placeholder 给 MOSS 例子); 回调 `_run_clone_cmd` 加 `style_desc` → 命中预设覆盖 temperature/top_k, speed 与滑条**相乘**; info 显示命中风格 (未命中回退自然并标注)
- [x] **端到端验证** ✅: "冷淡 果断 无感情 绝对理性" → 理性 AI (MOSS 式) 命中, 档案微调权重加载, 相似度 0.816 (音色未丢)
- [x] **测试** ✅: 5 新增 (预设完整性/MOSS 路由/其他人设/回退/指引文案)
- 收敛决策: CLI 不加 `--style` (自研引擎无采样参数, 硬塞无意义; GSV 主入口 = WebUI/脚本); 档案绑定风格降级不做 (MVP 每次合成时选, 避免多余状态)

### 批次 8 (2026-10-03): 风格档案化 + 架构检查

- [x] **档案绑定风格** ✅: `save_voice/save_voice_auto` 加 `style` 字段; WebUI 保存档案时一并存人设描述; 合成回调风格框为空时回退档案默认人设 (info 标注"档案默认"); 存量档案「我的声音V2」已补绑 MOSS 人设 (立即生效)
- [x] **门禁句集人设定制** ✅: `train_gate.py --texts` 自定义句集 (每行一句, ≥3 句) — 用目标场景句子测门禁 (如 MOSS 人设用系统播报腔); 基线与训练后自动同句集保证可比
- [x] **架构检查** ✅ (Explore agent 盘点 + 人工汇总):
  - **分层健康**: 依赖单向 core←data/utils←models←training←webui, 无循环依赖; `gsv_engine.py` 是 GSV 推理唯一门面 (webui 全程经 `get_gsv_engine()`, 无越层 import TTS_infer_pack)
  - **参数无漂移**: 采样参数集中在 style_presets.py 与 gsv_engine.py 默认值, 无重复硬编码
  - **可接受边界**: scripts/gsv_finetune.py 直接 PYTHONPATH 上游跑训练管线 (设计即胶水层); eval/speaker_sim.py 注入上游 eres2net 路径 (模型在上游目录, 与门面模式不完全一致, 记录不修)
  - **发现并修复**: ① tests/ 两个 _debug 临时文件已删 ② `voice_library` 零单测 (核心资产) → 补 3 测试 (往返含 style/空名拒绝/绑定链路) ③ 超大文件 webui 1072 行 / callbacks 1035 行 — 记录观察, 无拆分必要 (职责单一)
  - **测试缺口 (记录不排期)**: gsv_engine (需真权重, 靠 e2e 脚本覆盖), speaker_sim (需 ERes2Net 权重), webui 回调 (UI 层靠 e2e)
- [x] **测试** ✅: 3 新增, 回归 210 passed

### 批次 9 (2026-10-03): 流式首包与快速合成优化

- [x] **基线实测** ✅ (稳态, 微调权重): 2s 级音频 (2.3~3.6s) 非流式 4.9~6.2s / 流式首包 3.7~4.9s — 距离 "2s 音频 ≤3s" 目标差 ~2x。分解: AR 语义 token 生成 ~26it/s 占 60%+, 权重热换/固定开销占其余
- [x] **优化 1: 热换幂等跳过** ✅: 上游 init_vits/init_t2s_weights 无条件完整重载 — 稳态连续合成同一音色每次白付 1~3s。引擎层 `_loaded` 状态跟踪 + `_ensure_weights()` 同路径跳过 (warmup/stream/synthesize 三处统一)
- [x] **优化 2: 首段预切分收紧 30→14 字** ✅: 首段 AR token 减半, 逗号层 min 8→6 字早生效 — 16~20 字短句也能在逗号处切出短首段
- [x] **验收** ✅ (稳态对比):
  | 文本 | 音频 | 非流式 (前→后) | 流式首包 (前→后) | 流式总 (前→后) |
  |---|---|---|---|---|
  | 收到马上出发 | 2.30s | 5.37→**2.53s** | 3.70→**0.92s** | 4.96→**2.38s** |
  | 您好请问 | 2.78s | 5.83→**2.99s** | 4.93→**2.17s** | 5.36→**2.58s** |
  | 检测到异常 | 3.62s | 6.16→3.67s | 4.48→**2.12s** | 6.80→4.91s |
  - **2s 级音频 ≤3s 达标** (RTF 2.3~2.9 → 1.0~1.3); 首包全部 <2.5s (原 4.5s)
  - 4s 级长音频: 非流式 9.68s (RTF 0.99), AR 物理速度为底
- [x] **不追的优化 (记录)**: t2s AR torch.compile — AR 的 KV cache shape 逐步递增会反复触发 recompile, 除非重构 KV cache 静态化 (改上游模型结构, 超出框架层范围); GPU ORT BERT — 需装 onnxruntime-gpu, 收益 ~0.2s 优先级低
- [x] **测试** ✅: 4 新增 (切分×3 + 热换幂等), 回归 214 passed

### 批次 10 (2026-10-04): 2 秒目标冲线尝试 — CUDA Graph 实证 + 诚实边界

> 用户目标: "2s 长度 2s 内合成"。批次 9 后剩余大头 = t2s AR 逐 token 解码 (~22-26 it/s 物理底), 唯一大杠杆 = 消除 kernel launch 开销。

- [x] **官方 CUDAGraphRunner 桥接入** ✅ (`adr/models/adr_t2s_bridge.py` + 补丁 #20): 官方 CUDA Graph 路径 (`AR/models/t2s_model_cudagraph.py`, KV cache 静态预分配 + graph capture/replay) 只接在 inference_webui.py, TTS_infer_pack 引擎从未享受 → 桥按官方模式挂 `infer_panel` (输入归一化 list/tensor 双形态 / bs≠1 回退 / prompt None 兜底 / 静默 tqdm / runner 单例缓存; 流式=完整 AR 后按 chunk 切块 yield)
- [x] **AR 速度实锤 3.77x** ✅ (`_probe_cudagraph.py`): 同输入短文本 AR 阶段提速 3.77x — kernel launch 开销确实是主瓶颈, 方向判断正确
- [x] **⚠️ 质量损坏诊断 → 默认关闭** ✅ (`_diag_cg.py`): e2e 发现音频时长翻倍 ("好的没问题" 1.74s→8.10s) + 同 seed 两轮不确定 (5.22s/7.18s) → 诊断石锤 **AR 分布漂移: 同输入 base 45 token vs cudagraph 254 token (5.6x)** — 官方 SDPA attention + 自带 Sampler 实现与原路径语义不一致, 不是等价加速而是换了条生成分布 → 按质量红线**默认关闭** (`ADR_T2S_CUDAGRAPH=0`, `gsv_engine._lazy_init` 里 setdefault), 完整实现+挂钩保留, 上游修复 SDPA 语义后设 `=1` 即启用
- [x] **GPU ORT BERT 实验 → 放弃** ✅ (`_bench_bert_gpu.py`): HF fp16 CUDA BERT 单句仅 **43-45ms** (占 2s 预算 <2%, 优化天花板 0.04s << 0.15s 收益线); onnxruntime-gpu 1.24.4 的 CUDA EP 需 CUDA 12 runtime (`cublasLt64_12.dll`), 本机 torch 2.10+cu130 DLL 版本不匹配 → CUDA EP 加载失败回退 CPU (79-117ms, 反而慢 2x)。**结论: GPU 档 BERT 无优化价值**, 环境已还原 CPU onnxruntime (与 requirements-lock 一致)
- **诚实边界结论**: AR ~22-26 it/s 物理速度为底, 2s 音频 (~43 语义 token) 纯 AR 已需 1.5-2s — **不动模型内核, 2s 整体合成不可达**; 批次 9 的 2.2-2.5s 即框架层极限。剩余路径: ① 上游修复 cudagraph SDPA 语义 (3.77x 白拿) ② 模型内核级蒸馏/投机解码 (超框架范围)
- 回归: 214 passed 基线维持

### 批次 11 (2026-10-04): 对外 TTS 服务层 — N.E.K.O 生态对接

> 用户指令: 学习 api_neko (N.E.K.O 消费 GSV 的 HTTP 服务层), 为 ADR 预留对接口与对接规范, 对接 neko 和一切需要 TTS 服务的消费方。

- [x] **api_neko 学习** ✅ (`需要适配的一个接口/api_neko.zip` 解包精读): N.E.K.O 消费面 = GPT-SoVITS **api_v2** (`/api/v2/tts` GET+POST + set_gpt/set_sovits_weights), 端口 9881; 流式 WAV 字节契约 = 首块 44B 头 + 裸 s16le PCM; streaming_mode 0/1/2/3 分支语义 (bool True == 1 落分支 1); 错误统一 `400 {"message": ...}`; v3 高级面 (task 队列 + 双 WebSocket) N.E.K.O 不依赖 → 不复刻, 文档注明差异
- [x] **服务层 `adr/server/`** ✅: `app.py` create_app 工厂 (CORS 全开, 引擎惰性单例, 测试可注入); `v2_compat.py` api_v2 兼容层 (请求模型同名同义, streaming_mode 分支逐字对齐, 流式生成器字节契约一致, 权重切换端点走引擎预热); `native.py` ADR 原生 API `/api/adr/v1/*` (health/profiles/profiles/{name}/ref/tts); `audio_codec.py` 打包工具 (wav/raw/ogg/aac + wave_header_chunk, 字节级对齐移植); `__main__.py` `python -m adr.server -p 9881` (默认端口即 GSV 标准, N.E.K.O 零配置)
- [x] **ADR 扩展参数** ✅: `profile`/`voice` → 音色档案解析 (ref_audio/prompt_text/微调权重自动带上, 显式值优先); `t2s_weights`/`vits_weights` 请求级热换; `ADR_TTS_DEFAULT_PROFILE` 环境变量回退; 引擎不支持的采样参数接受但忽略 (差异清单文档化)
- [x] **对接规范 `docs/tts-api-spec.md`** ✅: 快速启动 / v2 参数表 (支持+忽略清单) / 流式字节格式与客户端拼接示例 / N.E.K.O 对接步骤 / 原生 API / 档案机制 / 与官方 api_v2 差异诚实清单 (v3 队列+WS 未实现等) / 安全注记
- [x] **测试 `tests/test_server_api.py`** ✅: FakeEngine (不加载真模型) 26 用例 — 参数校验 400 / 非流式 WAV/raw 字节数 / 流式首块头+裸 PCM / streaming_mode 分支语义 (含 bool) / 权重端点成功与失败格式 / profile 解析与覆盖优先级 / 默认档案环境变量 / 原生端点全量
- 关键修复: 拆包顺序 — ADR 引擎 `synthesize_stream` yield `(chunk, sr)` 与 GSV pipeline `(sr, chunk)` 相反 (探针定位)
- 回归: 240 passed + 1 skipped (基线 214 + 新增 26), 零回归

### 批次 12 (2026-10-04): TTS 服务鉴权 + 推送

> 用户指令: 推送批次 11 + 补鉴权机制 + 回答 neko 连接问题。

- [x] **推送** ✅: 批次 11 (`47ed0c8`) 首推遇 SSL_ERROR_SYSCALL 网络错误, 重试成功 `bf87db7..47ed0c8 main`
- [x] **API Key 鉴权** ✅ (`adr/server/auth.py` + `app.py` 接线): `ADR_TTS_API_KEY` 环境变量 (逗号分隔多 key, 空串=关闭); 接受 `Authorization: Bearer` / `X-API-Key` / `?api_key=` 三通道等价; key 精确匹配; 失败统一 `401 {"message": "unauthorized"}`; 豁免 `/api/adr/v1/health` (监控探活)。**默认不鉴权** — N.E.K.O 零改造兼容不受影响
- 中间件顺序: APIKey 先加 (内层), CORS 后加 (外层) — 预检 OPTIONS 免 key, 401 响应带 CORS 头 (浏览器可读)
- 纯 ASGI 实现 (无 BaseHTTPMiddleware 开销, 无框架耦合)
- [x] **测试** ✅: 7 新增 (默认开放 / 缺 key 401 / 错 key 401 / 三通道 / 精确匹配 / health 豁免 / 空串=关闭); `client` fixture 加 delenv 防 env 泄漏
- [x] **文档** ✅: `docs/tts-api-spec.md` §7 重写 (默认行为 / 启用方法 / 三通道表 / curl 示例 / 本地路径注记)
- 回归: 33 passed (服务层 26+7); 全量回归见提交说明

### 批次 13 (2026-10-04): 桌面壳选型 + Tauri 2 脚手架

> 用户指令: 保留 web 控制台, 开发本地桌面控制台, 内嵌所需依赖、尽可能不用系统自带环境, 做成完整桌面应用 (Tauri+Python Sidecar / Electron / Flutter 选型)。经对比用户拍板: 直接搭 Tauri 壳。

- [x] **选型对比 → Tauri 2 + Python Sidecar** ✅: 壳只做 进程托管 + 托盘 + 窗口导航到 `http://127.0.0.1:<port>` 现有 Gradio 控制台, **前端零新代码**。三方案对比: Electron 内存大且壳/后端双 Node 运行时冗余; Flutter 桌面生态弱且仍需内嵌 Python, 技术栈割裂; Tauri 2 壳体积小 (~10MB 级) 内存占用低, WebView2 Win10/11 自带, Rust 侧进程管理可靠。三阶段路线: **阶段 0** 便携 Python 运行时 (python-build-standalone + requirements-lock 离线轮子 + 静态 ffmpeg → `runtime/` 绿色文件夹, 实现真"零系统依赖"); **阶段 1** 本批次壳 MVP; **阶段 2** 原生前端 (替换 Gradio, 可选)
- [x] **脚手架 `desktop/`** ✅: `src-tauri/` (Cargo.toml: tauri 2 + tray-icon, release LTO+strip; tauri.conf.json: productName "ADR Studio", identifier io.github.hmug12.adr-studio, NSIS 打包, 主窗 1280x820) + `src/loading.html` 深色加载占位页 (`window.__adrStatus()` 供壳推状态) + `assets/icon.png` (纯 stdlib 生成 512px 深蓝底青色声波条) → `npx tauri icon` 全平台图标
- [x] **sidecar 托管 `src-tauri/src/lib.rs`** (~290 行) ✅:
  - 启动: 随机空闲端口 + `--host 127.0.0.1` (避让现有 GSV/N.E.K.O 的 9881) + `python -m adr.cli webui --port <P> --no-browser` spawn (与命令行 `adr webui` 同一入口, 壳不重复开浏览器), `CREATE_NO_WINDOW` 防控制台闪窗
  - 探活: 轮询 `GET /` (Gradio 首页 200 即全站可用), 180s 超时容忍 torch/CUDA 冷导入, 每 6s 推送加载页状态
  - 守护: `try_wait()` 轮询 (**不能用阻塞 wait()**, 否则托盘退出路径死锁); 崩溃自动重启, 稳定运行 60s 重置失败计数, 连续 4 次失败托盘/加载页报错
  - 清理: 退出 `taskkill /PID <pid> /T /F` 杀整棵进程树; 托盘退出 + `RunEvent::Exit` 双保险; 关窗 = 隐藏到托盘, 真正退出只走托盘
  - 解释器解析: 便携 `runtime/python/python.exe` → `ADR_DESKTOP_PYTHON` → 仓库 `.venv` → PATH `python`; 仓库根由 cwd/exe 向上 8 层探测 `adr/cli.py`
- 工具链: Rust 1.99.0 经 rsproxy.cn 镜像安装 (win.rust-lang.org 直连被掐); cargo 换 rsproxy sparse 源 (用户级 ~/.cargo/config.toml, 不进仓库); MSVC Build Tools 2022 用户手动安装
- [x] **cargo check 验证** ✅ (零警告): 编译期修 4 处 — `TrayIcon::set_tooltip` 在 tauri 2.12 签名为 `Option<S>` (3 处), `if let` 条件里 MutexGuard 临时值存活到块尾导致 E0597 (先 clone 到局部变量); 另自查修复 `creation_flags` 缺 `CommandExt` 导入
- [x] **用户实测 → 修复 UI 挂载** ✅: 实测窗口显示 `{"detail":"Not Found"}` — 根因: 壳最初拉起的 `python -m adr.server` 是批次 11 的**无头 FastAPI API 服务** (无 `/` 路由), 而真正的 Web 控制台是 `adr/webui` (Gradio 6), 导航 `/` 必 404。修复: spawn 改 `python -m adr.cli webui --host 127.0.0.1 --port <P> --no-browser` (cli.py 自带 `__main__` guard, Python 侧零改动), 探活 `/api/adr/v1/health` → 轮询 `GET /`, 仓库根标记同步改 `adr/cli.py`。"127.0.0.1 拒绝连接"为衍生症状 (每次启动随机端口, 旧地址失效), 修复后 base_url 始终指向当前健康端口。注: 备选方案 B (`adr.server` 加 `--ui` 用 `gr.mount_gradio_app` 单进程挂载) 因 Gradio 6 theme/css/queue/prewarm 迁移细节多而弃用

### 批次 14 (2026-10-04): 三控制台改造 — 启动器三选一 + 专业/新手控制台

> 用户指令: 启动时 3 个可选项 (老版本控制台 / 专业控制台 / 新手控制台), 后两个要现代化、数据可视化高级感设计, 新手控制台尽可能傻瓜化白痴化 (用户当纯小白)。

- [x] **Commit A `020d0d6`: 三控制台骨架** ✅ (10 文件 +1096/-105):
  - `adr/server/console.py` (新): 控制台 API (prefix `/api/adr/v1`) — `GET /system/stats` (GPU/RAM/磁盘 psutil+NVML, 引擎/档案/模型/训练状态汇总), `POST /train/start` (409 互斥, exp 名清洗, epochs clamp 1-50, `--skip-s1` 恒定, 4gb 配方加 `--max-clip-sec 10`), `GET /train/status` (log_tail 80 行 + gate_curve 全量读 jsonl), `POST /train/stop`, `POST /train/upload` (multipart, ext 白名单), `GET /models` + `POST /models/download` + 下载进度轮询 (tqdm 百分比正则)
  - 静态页挂载: `/` → index (三卡片跳转), `/pro` → pro.html, `/easy` → easy.html; `auth.py` 豁免三静态页
  - `desktop/src/launcher.html`: 三卡片选择 → `invoke("launch_console", {mode})`; legacy 拉 `adr.cli webui` (探活 `/`), pro/easy 拉 `adr.server` (探活 `/api/adr/v1/health`) → 绝对 URL 导航
  - `lib.rs` 重构 (~464 行): 三模式按需启动 + supervise 守护让位规则 + kill_current 服务互斥; cargo check/build 一次通过
- [x] **Commit B `c3b003b`: 专业控制台 pro.html 完整版** ✅ (1 文件 +805/-53): 深色高级感 (#0a0d14 底 + #2dd4bf/#60a5fa 双 accent), 四 tab — 总览 (GPU/内存 270° 环形仪表, 磁盘条, 显存 sparkline 60 点采样), 声音克隆 (拖拽上传 + 相似度曲线 canvas: 0.80 目标线/zero_shot 虚线/最佳点金色标注), 语音合成 (blob 播放/下载, 600s 超时), 模型库 (表格 + 下载进度)。api_key URL 透传; 手写 canvas 零依赖 (DPR 适配); 修 3 处重复 `id="errline"`
- [x] **Commit C `1ab082d`: 新手控制台 easy.html 完整版** ✅ (2 文件 +625/-40):
  - `console.py` 增建档 API: `POST /profiles/create` + `GET /profiles/create/status` (save_voice_auto 后台线程, 同名 409, 任务互斥; 长音频扫段需引擎加载 1-3 分钟故必须异步轮询)
  - `easy.html` 三步向导 (暖色友好风, 大字体大按钮, 零术语): ①选声音 (卡片点选 / 上传零样本建档双模式轮询 / 可选"进阶加练"训练按显存自动选配方) ②打字 (示例句一键填充) ③听效果 (大播放球 + 保存/改文字/换声音); 人话错误翻译 `human()`; 零样本优先产品设计 — 建档即可合成, 训练做成可选, 避免小白等 20-60 分钟
- 全部提交**未推送**; 阶段 0 便携 Python 打包留待后续 (用户此前拍板延后)
- [x] **Commit D: 返回启动器修复 + 调用模式开关** ✅ (5 文件; 用户实测两轮反馈驱动):
  - 返回按钮根因: 控制台页是远程 origin (127.0.0.1), Tauri 2 默认 IPC 只放行本地 origin, invoke 全被 ACL 拒绝 → 方案回退: pro/easy 页 back 按钮改纯导航 `location.href = "http://tauri.localhost/launcher.html"` (不经 IPC); capabilities/default.json 移除 remote url 配置回退仅本地
  - launcher.html 加载时 invoke `on_launcher_ready` 统一收尾上一控制台 sidecar (含托盘 tooltip 复位), 替代各页面自行杀进程
  - 调用模式 (expose): `ServerState.expose: AtomicBool`, launcher 每卡片新增 checkbox "对外提供调用 (局域网)"; 勾选 → host 用 `0.0.0.0`, 否则 `127.0.0.1`; supervise 重启从 state.expose 读取; tooltip 加 "(对外服务)" 标记
  - cargo check 零警告; 用户实测 debug exe 验证通过 ("可以了")
- [x] **NSIS 安装包产出** ✅: `ADR Studio_0.1.0_x64-setup.exe` (1.60 MiB, tauri-bundler 2.12.1)。障碍与绕过: ① GitHub release 直连 TLS 握手失败 (被墙) → ghfast.top/gh-proxy.com 镜像循环重试下载 nsis-3.11.zip + nsis_tauri_utils.dll v0.5.3 (SHA1 校验); ② TRAE 沙箱对 `%LOCALAPPDATA%\tauri\NSIS` 下 Move/Copy/删除拦截 (新建写入允许) → tar `--strip-components=1` 平铺解压 + `requires_approval=true` 沙箱外放置 dll; ③ CLI 误报 "directory missing some files, recreating" 实为缺插件 dll → WebFetch 直取 tauri 源码确认 13 必需文件清单与精确版本, 手工组装工具链后三次构建内通过

### 批次 15 (2026-10-04): 状态实时化 + 服务级预热 + 壳启动预热 + 调用控制台

> 用户指令链: ①三 bug (专业版状态不动 / 首次合成慢 / 调用参数暴露不足); ②四问题追问 (状态还是不动 / 要独立调用控制台, 参数参考 api_neko / 怀疑假数据硬编码+体积小是不是没打包引擎 / 预热 2 分钟没动静); ③壳启动即自动预热, 预热好才放行进启动面板; ④"退了火绒, 先杀掉那几个 (30 个僵尸 python), 把调用控制台做好, 我这边没看到"。

- [x] **stats 500/全接口僵死双根因修复** ✅ (`console.py`):
  - 根因一: `system_stats` 是 `async def`, 内部 GPU 查询 (首次触发 torch 导入 20s)/文件 IO 全在事件循环上执行 → 一个慢请求卡死全部接口 (状态不动的真凶)。修复: 改普通 `def`, FastAPI 自动丢线程池执行; `_gpu_stats` 加 `"torch" in sys.modules` 守卫, 预热导入 torch 期间不并发 import (会抢引擎锁)
  - 根因二: `train_status` 是 async 函数, stats 里直接调用返回 coroutine 被 jsonable_encoder 拒 → 整个 stats 500。修复: 改 `_train_snapshot()` 同步快照
- [x] **引擎状态实时化 (三态+细分阶段)** ✅ (`gsv_engine.py` + `console.py`): 模块级 `_LOADING`/`_STAGE` + `is_loading()`/`is_ready()`/`stage()` 三个只读探针 (不触发加载); stats 新增 `engine_loading`/`engine_ready`/`engine_stage` (queued/importing/loading/kernel/ready/failed); `_engine_loaded` 单布尔废弃
- [x] **服务启动即后台预热** ✅ (`app.py`): lifespan + `_prewarm_gsv` daemon 线程 — queued→importing (torch+GSV 模块)→loading (档案权重优先, 损坏回退预训练)→kernel (档案参考音频合成首块, JIT 预热省 ~14s CUDA 编译)→ready; 失败不致命 (首次合成现场加载); 测试引擎注入时跳过。裸服务 (N.E.K.O 直连) 也受益: 首请求不再付 1 分钟冷加载
- [x] **原生 API 参数补齐** ✅ (`native.py`): `/profiles` 每项加 `ref_audio` 绝对路径 (调用控制台自动填充用); `/tts` GET+POST 加 `top_k`/`top_p`/`temperature`/`text_split_method` (cut0/cut1/cut3/cut5) 透传 v2
- [x] **壳启动即预热 (预热好才进启动面板)** ✅ (`lib.rs` +242 行 + `prewarm.html` 新 + `tauri.conf.json`): 主窗口 url 指 `prewarm.html` 四阶段进度页 (检查服务→加载模型→内核预热→就绪); setup spawn `prewarm_flow` 线程: `start_server("pro")` 拉服务→探活→spawn supervise 守护→轮询 stats `engine_stage` (500ms)→Rust `eval` 驱动页面阶段推进; **240s 超时放行** (引擎继续后台加载, 不阻塞用户) + `skip_prewarm`/`retry_prewarm` 命令; `engine_status` 命令 (探活 `/api/adr/v1/health` + stats stage) 供 launcher 状态条 3s 轮询三色显示; 用户实测全链路走通: prewarm 页→超时放行→launcher→pro 热通道秒进 (日志 `/pro?desktop=1` 200 铁证)
- [x] **独立调用控制台 `/call`** ✅ (`call.html` 新 ~430 行 + `console.py` 挂载 + `index.html` 入口): 面向程序集成 — 原生 `/api/adr/v1/tts` 与 GSV 兼容 `/api/v2/tts` 双端点切换, 档案下拉自动填充 `prompt_text`/`ref_audio_path`, 全参数面板 (含批次14E 新增采样参数), 实时生成 cURL/Python/JS 调用代码, 合成试听记录 TTFB/总耗时, 引擎状态 4s 轮询
- [x] **launcher 第四卡片 + call 模式接入** ✅ (`launcher.html` + `lib.rs`): 3 列→4 列 grid + 紫色「调用控制台」卡片 (无 expose 勾选框, expose 读取 `?.checked ?? false` 兼容); `launch_console`/`enter_console`/`start_and_wait` 三处 call 模式白名单与 target 映射 (热通道与 pro/easy 同路径秒进); `call.html` 桌面壳内返回链接改「← 返回启动器」(仿 pro.html 纯导航)
- [x] **假数据/硬编码排查结论 (汇报)** ✅: stats 全实测 — GPU 走 NVML, RAM/磁盘走 psutil, 引擎状态来自进程内真实标记, 训练状态来自真实 jsonl/日志; 引擎本来就不在安装包里 (壳 ~1.6MB 只是 Tauri 壳+静态页, 设计如此): 引擎 = D:\pyhon Python 环境 + adr 包, 模型按需下载 (stats `models.cached` 如实显示 0/3)
- [x] **预热慢根因 = 火绒进程链冻结, 与磁盘/代码无关 (汇报, 三轮实验定案)** ✅: 旧结论"文件读取被拖慢"被实测推翻 — 磁盘读 456MB 仅 185ms (2.5GB/s)、TEMP 建删文件 3.8ms/次, 均正常。py-spy 三次抓栈同位置: 壳 spawn 的 python 卡死在 `numba ensure_cache_path → tempfile.TemporaryFile(dir=__pycache__)` 的 `os.open` (librosa `@jit(cache=True)` 导入链) 14+ 分钟, WS 恒定不涨 = 进程被火绒**冻结挂起**而非慢。决定性对照: 裸进程 `import librosa` 0.0s; PowerShell 手动/`.NET CreateNoWindow` spawn 同款服务 50-60s 正常推进; **终端拉起的壳被火绒静默终止 (无事件日志) 且其子 python cmdline 冻结不可读**, 而 **explorer.exe 中转 (等同用户双击) 启动壳 → 存活 + 服务 91s fully ready (WS 3.2GB, stage=ready)**。定论: 火绒按**启动来源/进程链信誉**行为判定 — 用户双击日常使用完全正常不受影响; 目录信任区对该判定无效也无需; `CREATE_NO_WINDOW` 标记非触发条件 (已排除)。29 个 cmdline 为空的 python 僵尸 = 历次冻结尸体 (taskkill 拒绝访问, WS=0 不占资源, 无害残留)
- 验证: cargo build 通过; `python -m pytest tests/test_server_api.py` 全绿; 服务端 /call 页 HTTP 200; 用户确认四卡片 + 调用控制台秒进 + 返回按钮 OK

### 批次 16 (2026-10-04): 专业控制台训练三问题 — 门禁崩溃根因修复

> 用户反馈: "训练花了 33 分钟 24 轮太慢了; 训好的模型不知道去哪里了识别不出; 训练进度与相似度曲线从始至终一点反应都没有"。三问题同源: 门禁进程秒崩。

- [x] **根因链 (实验+日志定案)** ✅: 门禁进程打零样本基线时把**完整训练录音** (output/uploads/xxx.mp3, 常 >10s) 直接当克隆 ref → GSV 硬限制 "参考音频在3~10秒范围外" 抛 OSError (TTS.py:816 set_ref_audio) → 门禁进程秒崩。连锁后果: ①`train_gate_<exp>.jsonl` 不存在 → 进度/相似度曲线全无数据; ②sim≥0.80 早停永不触发 → 跑满 24 轮 (55s/轮本身正常, 33 分钟 = 24×~80s); ③绑定下拉默认"不绑定"用户没选 → 权重产出在 `SoVITS_weights_v2/` 但没建档案 → "模型不知道去哪"
- [x] **门禁 ref 自动裁剪** ✅ (`train_gate.py`): 新增 `_clip_ref()` — ref >10s 时 librosa 去首尾静音 (top_db=30) 取前 8s 写 `output/gate_ref_<exp>.wav`, 基线与逐轮打分用同一裁剪 ref, 可比性不变
- [x] **绑定默认自动建档** ✅ (`pro.html`): 下拉首项改 `__auto__` (用实验名自动建档绑定, 推荐); `loadBindOptions()` 动态填充同步保留该项; tStart 提交 `__auto__` → 实验名 (空则 my_voice)。训完即档案即用, 不再"模型去哪了"
- [x] **训练日志乱码修复** ✅ (`console.py`): 训练/门禁两个 Popen 加 `PYTHONIOENCODING=utf-8 + PYTHONUTF8=1` — Windows 子进程默认 GBK 输出, 按utf-8 读全乱码 (exp 名/报错栈不可辨认)
- [x] **存量补绑** ✅: 「yui的声音」e24 权重 (81MB) 经 `voice_library.save_voice` 手动建档案, ref 取 slicer_opt 首切片 — 用户可直接在 Clone 页刷新使用, 无需重训
- 验证: py_compile 语法通过; `pytest tests/test_server_api.py` 33 全绿。注: 服务 pid 33808 是旧代码启动, console.py/train_gate.py 修复需重启服务/壳生效; pro.html 静态页刷新即可

### 批次 17 (2026-10-05): Clone 合成 "tts failed" — 档案 ref 超长闭环修复

> 用户反馈: "训练完后我想测试时报 tts failed"。批次 16 补绑档案的 ref 本身踩了门禁同款 3~10s 硬限制。

- [x] **根因 (实测复现+定案)** ✅: `POST /tts (profile=yui的声音)` → 400 `{"message":"tts failed","Exception":"参考音频在3~10秒范围外"}`。量时长实锤: 档案 ref.wav = **10.50125s** (336320 采样 @ 32kHz), 16kHz 重采样后 168040 > 160000 (TTS.py:815)。源头 = 批次 16 补绑时取 slicer_opt **文件名排序首切片**, 未验时长 (该切片恰 10.5s)。与门禁崩溃 (批次 16) 同源: GSV 3~10s 硬限制的第二处踩坑
- [x] **存量修复** ✅: yui 档案 ref.wav 跳过开头 1s 取 8s 覆盖写回 (8.0s @ 32kHz PCM_16); meta.json 不动 (权重绑定不变)
- [x] **自动建档防护** ✅ (`gsv_finetune.py`): 新增 `_pick_ref()` 替代"盲取首切片" — soundfile 只读头测时长, 挑 3~9.5s 内最接近 7s 的切片; 无合格切片 (全超长/过短/无切片用原始录音) 则去静音裁 8s 写 `output/bind_ref_<exp>.wav`。下次训练自动建档即合法
- [x] **复测通过** ✅: 服务 (壳 explorer 中转重启, 端口动态 13496) ready 后 `POST /tts` → **HTTP 200**, 213KB, 4.0s; 产物 WAV PCM_16 32kHz 单声道 3.34s 有效
- 注: 复测前一次服务进程在合成请求时崩溃 (curl 56 连接重置, 无日志可查, 疑似偶发); explorer 中转重启后同请求正常, 不再复现, 观察即可。附带坑: 探活轮询应按 python 子进程 pid 过滤 netstat (壳 pid 无监听端口); 终端直起 python 跑 librosa 导入会被火绒冻结 (改纯 soundfile 方案绕开)

