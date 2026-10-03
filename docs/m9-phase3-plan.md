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
