"""A4: 三方仓库兼容补丁一键应用 (幂等, marker 防重复)。

新机器复装流程:
    1. clone 三个 third_party 仓库 (见各 scripts/*_train.py 头部注释)
    2. pip install -r requirements-lock.txt
    3. python scripts/apply_compat_patches.py
    4. 下载各引擎权重 (scripts 目录下的下载逻辑 / hf-mirror)

所有补丁都是环境兼容 (Windows / torch 2.10 / gloo 不可用), 不改算法。

批次56: 大块补丁 (整函数重写级) 走 PATCH_FILES 机制 — git diff 落成
scripts/patches/*.patch, 在对应仓库根 `git apply` (钉版 commit 上行号
精确; 上游改版时重新生成)。容器镜像构建 (docker/Dockerfile) 依赖本
脚本全量可复现: 干净 clone + 本脚本 == 本地 third_party 树。
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TP = REPO / "third_party"

PATCHES = []

# patch 文件补丁: (仓库名, patch 文件名, 目标文件相对仓库根, 幂等特征串)
# 特征串 = 补丁引入的代码, 已在目标文件中即视为已应用
PATCH_FILES = [
    (
        "gpt_sovits",
        "track_e_ar_sync.patch",
        "GPT_SoVITS/AR/models/t2s_model.py",
        # 批次33 Track E: AR 推理 EOS 旗标设备侧累积 (GPU->CPU 同步 4x/步 -> 1x/8步),
        # 推理性能核心补丁 — 本地手工改动, 批次56 收编 (容器可复现性)
        "ADR_AR_SYNC_EVERY",
    ),
]


def patch(rel, marker, old, new):
    # 自检: marker 必须是 new 文本的真实子串, 否则幂等检查失效,
    # 重跑脚本会重复插入 (实测 #17 三连插的根因)
    assert marker in new, f"补丁 marker 不在 new 文本里 (幂等会失效): {rel} / {marker}"
    PATCHES.append((rel, marker, old, new))


# ---------- GPT-SoVITS ----------
# jieba_fast py3.13 无 wheel → 回退纯 jieba (3 处)
for _f, _old, _new in [
    ("text/tone_sandhi.py",
     "import jieba_fast as jieba",
     "try:\n    import jieba_fast as jieba\nexcept ImportError:\n    import jieba  # py3.13 无 jieba-fast 轮子, 回退纯 Python 版 (ADR 兼容补丁)"),
    ("text/chinese.py",
     "import jieba_fast\n",
     "try:\n    import jieba_fast\nexcept ImportError:\n    import jieba as jieba_fast  # py3.13 兼容回退 (ADR 兼容补丁)\n"),
    ("text/chinese2.py",
     "import jieba_fast\n",
     "try:\n    import jieba_fast\nexcept ImportError:\n    import jieba as jieba_fast  # py3.13 兼容回退 (ADR 兼容补丁)\n"),
    ("text/chinese.py",
     "import jieba_fast.posseg as psg",
     "try:\n    import jieba_fast.posseg as psg\nexcept ImportError:\n    import jieba.posseg as psg  # py3.13 兼容回退 (ADR 兼容补丁)"),
    ("text/chinese2.py",
     "import jieba_fast.posseg as psg",
     "try:\n    import jieba_fast.posseg as psg\nexcept ImportError:\n    import jieba.posseg as psg  # py3.13 兼容回退 (ADR 兼容补丁)"),
]:
    patch("gpt_sovits/GPT_SoVITS/" + _f, "ADR 兼容补丁", _old, _new)

# s2_train: 单卡跳过分布式 (本机 gloo 彻底不可用)
GSV_S2 = "gpt_sovits/GPT_SoVITS/s2_train.py"
patch(
    GSV_S2, "_single_gpu_no_dist",
    '''    dist.init_process_group(
        backend="gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl",
        init_method="env://?use_libuv=False",
        world_size=n_gpus,
        rank=rank,
    )''',
    '''    # ADR 兼容补丁: 本机 gloo 不可用 (hosts 劫持+多网卡), 单卡直通
    _single_gpu_no_dist = (n_gpus == 1)
    if not _single_gpu_no_dist:
        dist.init_process_group(
            backend="gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl",
            init_method="env://?use_libuv=False",
            world_size=n_gpus,
            rank=rank,
        )''',
)
patch(
    GSV_S2, "net_g = net_g.cuda(rank)   # 单卡直通",
    '''    if torch.cuda.is_available():
        net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
        net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)''',
    '''    if torch.cuda.is_available():
        if _single_gpu_no_dist:
            net_g = net_g.cuda(rank)   # 单卡直通, 不包 DDP
            net_d = net_d.cuda(rank)
        else:
            net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
            net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)''',
)
patch(  # .module 访问兼容裸模型
    GSV_S2, 'getattr(net_g, "module", net_g)',
    "net_g.module.", 'getattr(net_g, "module", net_g).',
)
patch(
    GSV_S2, 'getattr(net_d, "module", net_d)',
    "net_d.module.", 'getattr(net_d, "module", net_d).',
)
patch(
    GSV_S2, 'getattr(generator, "module", generator)',
    "generator.module.", 'getattr(generator, "module", generator).',
)

# s1_train: 单卡 strategy auto
patch(
    "gpt_sovits/GPT_SoVITS/s1_train.py", "单卡用 auto",
    '''        strategy=DDPStrategy(process_group_backend="nccl" if platform.system() != "Windows" else "gloo")
        if torch.cuda.is_available()
        else "auto",''',
    '''        strategy=(  # ADR 兼容补丁: 本机 gloo 不可用, 单卡用 auto
            DDPStrategy(process_group_backend="nccl" if platform.system() != "Windows" else "gloo")
            if torch.cuda.is_available() and torch.cuda.device_count() > 1
            else "auto"
        ),''',
)

# bucket_sampler: 无 dist 时按单副本
patch(
    "gpt_sovits/GPT_SoVITS/AR/data/bucket_sampler.py", "dist.is_initialized()",
    '''            num_replicas = dist.get_world_size() if torch.cuda.is_available() else 1''',
    '''            # ADR 兼容补丁: 单卡无 dist 时按 1 处理
            num_replicas = (dist.get_world_size()
                            if torch.cuda.is_available() and dist.is_initialized()
                            else 1)''',
)
patch(
    "gpt_sovits/GPT_SoVITS/AR/data/bucket_sampler.py", "dist.is_initialized()",
    '''            rank = dist.get_rank() if torch.cuda.is_available() else 0''',
    '''            rank = (dist.get_rank()
                    if torch.cuda.is_available() and dist.is_initialized()
                    else 0)''',
)

# C4 训练门禁: s2_train 每轮末检查 STOP 信号
patch(
    GSV_S2, "收到训练门禁早停信号",
    '''        scheduler_g.step()
        scheduler_d.step()
    print("training done")''',
    '''        scheduler_g.step()
        scheduler_d.step()
        # ADR 兼容补丁 (C4 训练门禁): train_gate.py 达标后放 STOP 文件, 早停
        _stop = os.path.join(hps.s2_ckpt_dir, "STOP")
        if os.path.exists(_stop):
            print("收到训练门禁早停信号: %s" % open(_stop, encoding="utf-8").read())
            break
    print("training done")''',
)

# W3 显存补丁: v2 训练 forward 梯度检查点 (enc_q/flow 全 clip 激活 ~5.6GB,
# 实测 1 epoch 显存曲线 2GB→7.8GB 瞬跳; v3 有 use_grad_ckpt, v2 从未实现)
patch(
    "gpt_sovits/GPT_SoVITS/module/models.py",
    "ADR 显存补丁",
    '''        x, m_p, logs_p, y_mask, _, _ = self.enc_p(quantized, y_lengths, text, text_lengths, ge512 if self.is_v2pro else ge)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=ge)
        z_p = self.flow(z, y_mask, g=ge)''',
    '''        x, m_p, logs_p, y_mask, _, _ = self.enc_p(quantized, y_lengths, text, text_lengths, ge512 if self.is_v2pro else ge)
        # ADR 显存补丁: use_grad_ckpt=True 时对全 clip 的 enc_q/flow 做梯度检查点
        if getattr(self, "use_grad_ckpt", False) and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            z, m_q, logs_q, y_mask = checkpoint(
                lambda _y, _yl, _g: self.enc_q(_y, _yl, g=_g),
                y, y_lengths, ge, use_reentrant=False)
            z_p = checkpoint(
                lambda _z, _m, _g: self.flow(_z, _m, g=_g),
                z, y_mask, ge, use_reentrant=False)
        else:
            z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=ge)
            z_p = self.flow(z, y_mask, g=ge)''',
)
patch(
    GSV_S2,
    "把 train.grad_ckpt 透传给 v2 SynthesizerTrn",
    '''    else:
        net_g = net_g.to(device)
        net_d = net_d.to(device)''',
    '''    else:
        net_g = net_g.to(device)
        net_d = net_d.to(device)
    # ADR 显存补丁: 把 train.grad_ckpt 透传给 v2 SynthesizerTrn (官方只接 v3)
    getattr(net_g, "module", net_g).use_grad_ckpt = bool(
        getattr(hps.train, "grad_ckpt", False))''',
)

# W3 显存补丁 2: 长 clip 截断 (t=0 起 ssl/spec/wav 同步截断, 对齐不破坏)
# bs1 实测 4832MB, 大头=最长 clip (16s) 的 enc_q/flow/判别器激活; ADR_MAX_CLIP_SEC 环境变量控制
patch(
    "gpt_sovits/GPT_SoVITS/module/data_utils.py",
    "ADR 显存补丁",
    '''                ssl = torch.load("%s/%s.pt" % (self.path4, audiopath), map_location="cpu")
                if ssl.shape[-1] != spec.shape[-1]:
                    typee = ssl.dtype
                    ssl = F.pad(ssl.float(), (0, 1), mode="replicate").to(typee)''',
    '''                ssl = torch.load("%s/%s.pt" % (self.path4, audiopath), map_location="cpu")
                if ssl.shape[-1] != spec.shape[-1]:
                    typee = ssl.dtype
                    ssl = F.pad(ssl.float(), (0, 1), mode="replicate").to(typee)
                # ADR 显存补丁: ADR_MAX_CLIP_SEC>0 时长 clip 从 t=0 截断 (ssl/spec/wav 同帧数截)
                _max_sec = float(os.environ.get("ADR_MAX_CLIP_SEC", "0") or 0)
                _max_frames = int(_max_sec * self.sampling_rate / self.hop_length)
                if _max_frames > 0 and spec.shape[-1] > _max_frames:
                    spec = spec[:, :_max_frames]
                    wav = wav[:, : _max_frames * self.hop_length]
                    ssl = ssl[..., :_max_frames]''',
)

# 批次5 启动开销补丁: DataLoader worker 数可覆盖 (ADR_S2_NUM_WORKERS)
# Windows spawn × 5 worker 各自重导入 torch ≈ 分钟级固定开销; 小数据集
# (百级切片, 特征已预计算) 数据加载毫秒级, worker 纯浪费。
# 实测 (10 切片/3min 音频, 1 epoch 全程): workers=5 = 553.0s, workers=0 = 235.0s
# → 默认 0 省 318s (-57.5%); 大数据集可 ADR_S2_NUM_WORKERS=5 手动开回
# 注意 persistent_workers/prefetch_factor 必须联动 (workers=0 时二者非法)
patch(
    "gpt_sovits/GPT_SoVITS/s2_train.py",
    "ADR 启动开销补丁: ADR_S2_NUM_WORKERS 覆盖 worker 数",
    '''    train_loader = DataLoader(
        train_dataset,
        num_workers=5,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=3,
    )''',
    '''    # ADR 启动开销补丁: ADR_S2_NUM_WORKERS 覆盖 worker 数
    # (小数据集默认 0: Windows spawn 重导入 torch 的固定开销 >> 数据加载收益,
    #  实测 1ep 全程 553s→235s; 大数据集可设 5 开回 worker;
    #  persistent_workers/prefetch_factor 随 workers=0 联动关闭, 否则 DataLoader 报错)
    _adr_workers = int(os.environ.get("ADR_S2_NUM_WORKERS", "0"))
    train_loader = DataLoader(
        train_dataset,
        num_workers=_adr_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=_adr_workers > 0,
        prefetch_factor=3 if _adr_workers > 0 else None,
    )''',
)

# 批次10 t2s CUDA Graph 桥挂钩 (AR 3.77x): ADR_T2S_CUDAGRAPH=1 时把
# infer_panel 换成官方 CUDAGraphRunner 桥 (桥逻辑在 adr/models/adr_t2s_bridge.py,
# 不在 fork 里 — 本补丁只挂一行钩子)。官方 webui 有同款能力, 引擎路径没有。
patch(
    "gpt_sovits/GPT_SoVITS/TTS_infer_pack/TTS.py",
    "t2s CUDA Graph 桥挂钩",
    '''        else:
            print(i18n("朴素推理模式已开启"))
            self.t2s_model.model.infer_panel = self.t2s_model.model.infer_panel_naive_batched
''',
    '''        else:
            print(i18n("朴素推理模式已开启"))
            self.t2s_model.model.infer_panel = self.t2s_model.model.infer_panel_naive_batched

        # ADR 兼容补丁: t2s CUDA Graph 桥挂钩 (AR 3.77x, ADR_T2S_CUDAGRAPH=1 启用)
        if os.environ.get("ADR_T2S_CUDAGRAPH") == "1":
            from adr.models.adr_t2s_bridge import install_cudagraph_infer_panel
            self.t2s_model.model.infer_panel = install_cudagraph_infer_panel(
                self.t2s_model.model, self)
''',
)

# 批次63 t2s 静态 KV Cache 桥挂钩 (AR 3.77x): ADR_T2S_STATIC_KV=1 时把 infer_panel
# 换成静态 KV 桥 (桥逻辑在 adr/models/adr_t2s_static_kv.py, 不在 fork 里 — 本补丁
# 只挂一行钩子)。排在 CUDA Graph 之后安装, 同开时本桥覆盖生效。
patch(
    "gpt_sovits/GPT_SoVITS/TTS_infer_pack/TTS.py",
    "t2s 静态 KV 桥挂钩",
    '''        # ADR 兼容补丁: t2s CUDA Graph 桥挂钩 (AR 3.77x, ADR_T2S_CUDAGRAPH=1 启用)
        if os.environ.get("ADR_T2S_CUDAGRAPH") == "1":
            from adr.models.adr_t2s_bridge import install_cudagraph_infer_panel
            self.t2s_model.model.infer_panel = install_cudagraph_infer_panel(
                self.t2s_model.model, self)
''',
    '''        # ADR 兼容补丁: t2s CUDA Graph 桥挂钩 (AR 3.77x, ADR_T2S_CUDAGRAPH=1 启用)
        if os.environ.get("ADR_T2S_CUDAGRAPH") == "1":
            from adr.models.adr_t2s_bridge import install_cudagraph_infer_panel
            self.t2s_model.model.infer_panel = install_cudagraph_infer_panel(
                self.t2s_model.model, self)

        # ADR 兼容补丁: t2s 静态 KV 桥挂钩 (AR 3.77x, ADR_T2S_STATIC_KV=1 启用;
        # 排在 CUDA Graph 之后安装, 同开时本桥的 infer_panel 覆盖生效)
        if os.environ.get("ADR_T2S_STATIC_KV") == "1":
            from adr.models.adr_t2s_static_kv import install_static_kv_infer_panel
            self.t2s_model.model.infer_panel = install_static_kv_infer_panel(
                self.t2s_model.model, self)
''',
)

# 批次65 t2s per-seq RNG 批采样桥挂钩: ADR_T2S_BATCH_RNG=1 时把 infer_panel
# 换成逐行采样桥 (桥逻辑在 adr/models/adr_t2s_batchrng.py — 行独立 generator,
# 跨请求合批的内核前置)。排在前两桥之后安装, row_seeds 缺席时透传 fallback。
patch(
    "gpt_sovits/GPT_SoVITS/TTS_infer_pack/TTS.py",
    "ADR_T2S_BATCH_RNG",
    '''        # ADR 兼容补丁: t2s 静态 KV 桥挂钩 (AR 3.77x, ADR_T2S_STATIC_KV=1 启用;
        # 排在 CUDA Graph 之后安装, 同开时本桥的 infer_panel 覆盖生效)
        if os.environ.get("ADR_T2S_STATIC_KV") == "1":
            from adr.models.adr_t2s_static_kv import install_static_kv_infer_panel
            self.t2s_model.model.infer_panel = install_static_kv_infer_panel(
                self.t2s_model.model, self)
''',
    '''        # ADR 兼容补丁: t2s 静态 KV 桥挂钩 (AR 3.77x, ADR_T2S_STATIC_KV=1 启用;
        # 排在 CUDA Graph 之后安装, 同开时本桥的 infer_panel 覆盖生效)
        if os.environ.get("ADR_T2S_STATIC_KV") == "1":
            from adr.models.adr_t2s_static_kv import install_static_kv_infer_panel
            self.t2s_model.model.infer_panel = install_static_kv_infer_panel(
                self.t2s_model.model, self)

        # ADR 兼容补丁: t2s per-seq RNG 批采样桥 (批次 65 第二子批,
        # ADR_T2S_BATCH_RNG=1 启用; 排在前两桥之后安装, row_seeds 缺席时
        # 透传 fallback = stock 全局 RNG 行为)
        if os.environ.get("ADR_T2S_BATCH_RNG") == "1":
            from adr.models.adr_t2s_batchrng import install_batched_rng_infer_panel
            self.t2s_model.model.infer_panel = install_batched_rng_infer_panel(
                self.t2s_model.model, self)
''',
)

# 批次65 非流式逐段行种子派生: 桥启用时每段独立 generator (段 0 = 请求种子,
# 其余 set_seed 重播的全局 random → 请求内确定性); stock 路径忽略此字段。
patch(
    "gpt_sovits/GPT_SoVITS/TTS_infer_pack/TTS.py",
    "derive_row_seeds",
    '''            data = self.text_preprocessor.preprocess(text, text_lang, text_split_method, self.configs.version)
            if len(data) == 0:
                yield 16000, np.zeros(int(16000), dtype=np.int16)
                return
''',
    '''            data = self.text_preprocessor.preprocess(text, text_lang, text_split_method, self.configs.version)
            if len(data) == 0:
                yield 16000, np.zeros(int(16000), dtype=np.int16)
                return

            # 批次 65 第二子批: per-seq RNG 行种子 — 桥启用时逐段独立 generator
            # (段 0 = 请求种子, 其余 set_seed 重播的全局 random → 请求内确定
            # 性); stock 路径 (桥未装/row_seeds 透传 fallback) 忽略此字段零影响
            if os.environ.get("ADR_T2S_BATCH_RNG") == "1":
                from adr.models.adr_t2s_batchrng import derive_row_seeds
                for it, s in zip(data, derive_row_seeds(len(data), actual_seed)):
                    it["seed"] = s
''',
)

# 批次65 to_batch 行种子透传 (两处: item 收集 + batch dict 字段) —
# split_bucket 重排后种子跟随行走到 t2s 调用点。
patch(
    "gpt_sovits/GPT_SoVITS/TTS_infer_pack/TTS.py",
    "item_seed_list",
    '''            all_bert_features_list = []
            norm_text_batch = []
            all_bert_max_len = 0
            all_phones_max_len = 0
            for item in item_list:
''',
    '''            all_bert_features_list = []
            norm_text_batch = []
            item_seed_list = []  # 批次 65 第二子批: per-seq RNG 行种子透传
            all_bert_max_len = 0
            all_phones_max_len = 0
            for item in item_list:
                item_seed_list.append(item.get("seed"))
''',
)
patch(
    "gpt_sovits/GPT_SoVITS/TTS_infer_pack/TTS.py",
    '"row_seeds": (item_seed_list',
    '''                "all_bert_features": all_bert_features_batch,
                "norm_text": norm_text_batch,
                "max_len": max_len,
            }
''',
    '''                "all_bert_features": all_bert_features_batch,
                "norm_text": norm_text_batch,
                "max_len": max_len,
                "row_seeds": (item_seed_list
                              if any(s is not None for s in item_seed_list) else None),
            }
''',
)

# 批次65 t2s 调用点传行参数 (stock infer_panel_batch_infer **kwargs 吸收零影响;
# cudagraph / static_kv 桥 **kw 吸收透传 fallback)。
patch(
    "gpt_sovits/GPT_SoVITS/TTS_infer_pack/TTS.py",
    "row_seeds=item.get",
    '''                        early_stop_num=self.configs.hz * self.configs.max_sec,
                        max_len=max_len,
                        repetition_penalty=repetition_penalty,
                    )
''',
    '''                        early_stop_num=self.configs.hz * self.configs.max_sec,
                        max_len=max_len,
                        repetition_penalty=repetition_penalty,
                        # 批次 65 第二子批: per-seq RNG 行参数 (桥消费; stock
                        # infer_panel_batch_infer **kwargs 吸收, 零影响)
                        row_seeds=item.get("row_seeds"),
                        row_sampling=item.get("row_sampling"),
                    )
''',
)

# 批次20 O3 bf16 混合精度 (s2 训练降显存 30~40%): ADR_S2_BF16=1 且显卡支持
# (Ampere+, torch.cuda.is_bf16_supported) 时 autocast 用 bfloat16 并关 GradScaler
# (bf16 动态范围大, 无需 loss scale, 数值更稳)。未开环境变量 → dtype=fp16,
# 行为与上游完全一致; 模型主权重始终 fp32, 权重产物格式不变。
patch(
    "gpt_sovits/GPT_SoVITS/s2_train.py",
    "ADR bf16 O3",
    "from torch.cuda.amp import GradScaler, autocast",
    '''from torch.cuda.amp import GradScaler, autocast

# ADR 兼容补丁 ADR bf16 O3: 混合精度开关 — ADR_S2_BF16=1 且 Ampere+ 显卡生效
_ADR_BF16 = (os.environ.get("ADR_S2_BF16") == "1"
             and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
_ADR_AMP_DTYPE = torch.bfloat16 if _ADR_BF16 else torch.float16''',
)
patch(
    "gpt_sovits/GPT_SoVITS/s2_train.py",
    "GradScaler bf16",
    "    scaler = GradScaler(enabled=hps.train.fp16_run)",
    "    scaler = GradScaler(enabled=hps.train.fp16_run and not _ADR_BF16)  # GradScaler bf16",
)
# 第一处 autocast (前向) — 用上文 sv_emb 锚定, 避免命中第二处
patch(
    "gpt_sovits/GPT_SoVITS/s2_train.py",
    "autocast fwd bf16",
    '''                sv_emb = sv_emb.to(device)
        with autocast(enabled=hps.train.fp16_run):''',
    '''                sv_emb = sv_emb.to(device)
        with autocast(enabled=hps.train.fp16_run, dtype=_ADR_AMP_DTYPE):  # autocast fwd bf16''',
)
# 第二处 autocast (生成器) — 上一个补丁应用后此处已唯一
patch(
    "gpt_sovits/GPT_SoVITS/s2_train.py",
    "autocast gen bf16",
    "        with autocast(enabled=hps.train.fp16_run):",
    "        with autocast(enabled=hps.train.fp16_run, dtype=_ADR_AMP_DTYPE):  # autocast gen bf16",
)

# ---------- DiffSinger / RVC ----------
# 均无需文件补丁: DiffSinger 直接可跑; RVC 新版用 -m 模块调用 +
# 环境变量 (weight_root/rmvpe_root/index_root, 见 rvc_engine.py/_rvc_context)


def _apply_patch_files():
    """PATCH_FILES: 在对应仓库根 git apply (幂等: 目标文件已含特征串则跳过)。"""
    applied = skipped = failed = 0
    for repo, patch_name, target_rel, marker in PATCH_FILES:
        repo_dir = TP / repo
        fp = repo_dir / target_rel
        pf = REPO / "scripts" / "patches" / patch_name
        if not pf.exists():
            print(f"[MISS] scripts/patches/{patch_name} (补丁文件缺失)")
            failed += 1
            continue
        if not fp.exists():
            print(f"[MISS] {repo}/{target_rel} (仓库未 clone)")
            failed += 1
            continue
        if marker in fp.read_text(encoding="utf-8", errors="replace"):
            print(f"[skip] {repo} <- {patch_name} (已打过)")
            skipped += 1
            continue
        r = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(pf)],
            cwd=repo_dir, capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"[FAIL] {repo} <- {patch_name}: {r.stderr.strip()}")
            failed += 1
            continue
        if marker not in fp.read_text(encoding="utf-8", errors="replace"):
            print(f"[FAIL] {repo} <- {patch_name}: apply 返回 0 但特征串未现 (锚点漂移?)")
            failed += 1
            continue
        print(f"[ok]   {repo} <- {patch_name}")
        applied += 1
    return applied, skipped, failed


def main():
    applied = skipped = failed = 0
    for rel, marker, old, new in PATCHES:
        fp = TP / rel
        if not fp.exists():
            print(f"[MISS] {rel} (仓库未 clone)")
            failed += 1
            continue
        src = fp.read_text(encoding="utf-8", errors="replace")
        if marker in src:
            print(f"[skip] {rel} (已打过)")
            skipped += 1
            continue
        if old not in src:
            print(f"[FAIL] {rel} 找不到锚点 (上游可能已改版)")
            failed += 1
            continue
        fp.write_text(src.replace(old, new, 1), encoding="utf-8")
        print(f"[ok]   {rel}")
        applied += 1
    _a, _s, _f = _apply_patch_files()
    applied += _a
    skipped += _s
    failed += _f
    print(f"\n应用 {applied} / 跳过 {skipped} / 失败 {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
