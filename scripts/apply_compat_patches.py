"""A4: 三方仓库兼容补丁一键应用 (幂等, marker 防重复)。

新机器复装流程:
    1. clone 三个 third_party 仓库 (见各 scripts/*_train.py 头部注释)
    2. pip install -r requirements-lock.txt
    3. python scripts/apply_compat_patches.py
    4. 下载各引擎权重 (scripts 目录下的下载逻辑 / hf-mirror)

所有补丁都是环境兼容 (Windows / torch 2.10 / gloo 不可用), 不改算法。
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TP = REPO / "third_party"

PATCHES = []


def patch(rel, marker, old, new):
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
    "梯度检查点开关",
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

# ---------- DiffSinger / RVC ----------
# 均无需文件补丁: DiffSinger 直接可跑; RVC 新版用 -m 模块调用 +
# 环境变量 (weight_root/rmvpe_root/index_root, 见 rvc_engine.py/_rvc_context)


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
    print(f"\n应用 {applied} / 跳过 {skipped} / 失败 {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
