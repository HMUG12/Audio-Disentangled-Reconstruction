"""ADR 命令行入口。

完整功能在 Day 5 实现,Day 1 先提供 stub 让 console_scripts 可用。
"""

from __future__ import annotations

import sys

import click


@click.group()
@click.version_option(package_name="adr-clone")
def main() -> None:
    """ADR: 低资源快速克隆训练框架。

    一行命令克隆你的声音:
        adr clone --ref voice.wav --text "你好世界"
    """
    pass


@main.command()
def info() -> None:
    """显示环境信息。"""
    from adr.core import setup_device

    click.echo("ADR Framework v0.1.0")
    click.echo("=" * 50)
    setup_device(verbose=True)


@main.command()
def check() -> None:
    """运行安装检查。"""
    from adr.core import get_logger, setup_logging
    from adr.core.config import load_config

    setup_logging(level="INFO")
    log = get_logger("adr.cli")

    log.info("Running installation check...")

    # 设备能力分级 (AMD/Intel 用户明确预期)
    from adr.core.device import detect_device
    dev = detect_device()
    log.info(f"  [DEV] {dev.device} ({dev.gpu_name or 'CPU'}) — {dev.capability}")

    # 加载所有 4 档配置
    for preset in ["default", "vram_4gb", "vram_6gb", "vram_8gb"]:
        try:
            cfg = load_config(preset=preset)
            log.info(f"  [OK] {preset}: batch_size={cfg.train.batch_size}, "
                     f"precision={cfg.train.precision}, "
                     f"qlora={cfg.train.use_qlora}")
        except Exception as e:
            log.error(f"  [FAIL] {preset}: {e}")
            sys.exit(1)

    log.info("All presets loaded successfully.")


@main.command()
def list() -> None:
    """列出已注册的插件 (Backbone/Vocoder/ASR/G2P/...)。"""
    from adr.core import REGISTRY

    click.echo("Registered plugins:")
    click.echo("=" * 50)
    for name, reg in REGISTRY.all().items():
        items = reg.keys() or ["(empty)"]
        click.echo(f"  {name}: {', '.join(items)}")


# ===== 占位命令 (Day 5 完整实现) =====
@main.command()
@click.option("--input", "-i", required=True, help="输入音频路径")
@click.option("--output", "-o", default="./output/processed", help="输出目录")
@click.option("--preset", "-p", default="default", help="配置预设 (default/vram_4gb/...)")
@click.option("--no-asr", is_flag=True, help="跳过 ASR (手动提供文本)")
@click.option("--no-f0", is_flag=True, help="跳过 F0 提取")
def process(input: str, output: str, preset: str, no_asr: bool, no_f0: bool) -> None:
    """数据流水线: 音频 → 训练样本 (Day 2 完整实现)。"""
    from adr.core import get_logger, setup_logging
    from adr.data.pipeline import DataPipeline, PipelineConfig

    setup_logging(level="INFO")
    log = get_logger("adr.cli")

    config = PipelineConfig(
        enable_asr=not no_asr,
        enable_f0=not no_f0,
        output_dir=output,
    )

    log.info(f"Running data pipeline...")
    log.info(f"  Input:  {input}")
    log.info(f"  Output: {output}")
    log.info(f"  ASR:    {not no_asr}")
    log.info(f"  F0:     {not no_f0}")

    try:
        pipeline = DataPipeline(config)
        result = pipeline.run(input, output_dir=output)
        log.info(f"✓ {result.summary()}")
    except Exception as e:
        log.error(f"✗ Pipeline failed: {e}")
        sys.exit(1)


@main.command()
@click.option("--input", "-i", required=True, help="输入音频路径")
@click.option("--output", "-o", default="./output/slices", help="切片输出目录")
@click.option("--min-sec", default=3.0, help="最小切片时长(秒)")
@click.option("--max-sec", default=10.0, help="最大切片时长(秒)")
def slice(input: str, output: str, min_sec: float, max_sec: float) -> None:
    """仅切分音频 (Day 2 完整实现)。"""
    from adr.core import get_logger, setup_logging
    from adr.data.slice import SliceConfig, slice_audio, save_slices
    from adr.utils import load_audio

    setup_logging(level="INFO")
    log = get_logger("adr.cli")

    log.info(f"Slicing audio: {input}")
    audio = load_audio(input)
    config = SliceConfig(min_sec=min_sec, max_sec=max_sec)
    slices = slice_audio(audio, config)
    paths = save_slices(slices, output)
    log.info(f"✓ Generated {len(paths)} slices in {output}")


@main.command()
@click.option("--text", "-t", required=True, help="要转音素的文本")
@click.option("--backend", "-b", default="pypinyin", help="G2P 后端 (pypinyin/g2pW/char)")
@click.option("--no-tone", is_flag=True, help="不带声调")
def g2p(text: str, backend: str, no_tone: bool) -> None:
    """G2P 文本转音素 (Day 2 完整实现)。"""
    from adr.core import get_logger, setup_logging
    from adr.data.g2p import G2P, G2PConfig

    setup_logging(level="INFO")
    log = get_logger("adr.cli")

    g2p_obj = G2P(G2PConfig(backend=backend, with_tone=not no_tone))

    phonemes = g2p_obj(text)
    log.info(f"Input:  {text}")
    log.info(f"Output: {phonemes}")
    log.info(f"Count:  {len(phonemes)}")


@main.command()
@click.option("--ref", required=True, help="参考音频路径")
@click.option("--text", required=True, help="要合成的文本")
@click.option("--output", default="output.wav", help="输出 wav 路径")
@click.option("--checkpoint", "-c", default=None,
              help="训练好的 checkpoint 路径 (None=用基座模型)")
@click.option("--g2p-backend", default="pypinyin", help="G2P 后端 (pypinyin/char/g2pW)")
@click.option("--no-tone", is_flag=True, help="G2P 不带声调")
@click.option("--n-timesteps", default=20, help="扩散步数")
def clone(
    ref: str,
    text: str,
    output: str,
    checkpoint: str | None,
    g2p_backend: str,
    no_tone: bool,
    n_timesteps: int,
) -> None:
    """一行命令克隆声音 (M1 Day 9 实现)。

    完整流程: text + ref → 端到端推理 → wav

    用法:
        adr clone --ref voice.wav --text "你好世界"
        adr clone --ref voice.wav --text "你好" --checkpoint model.pt --output out.wav
    """
    from adr.core import get_logger, setup_logging
    from adr.inference.pipeline import InferConfig, InferPipeline

    setup_logging(level="INFO")
    log = get_logger("adr.cli.clone")

    # 检查 ref
    import os
    if not os.path.exists(ref):
        log.error(f"Reference audio not found: {ref}")
        sys.exit(1)

    # 构造 pipeline
    config = InferConfig(
        g2p_backend=g2p_backend,
        g2p_with_tone=not no_tone,
        n_timesteps=n_timesteps,
    )

    try:
        if checkpoint:
            log.info(f"Loading checkpoint: {checkpoint}")
            pipe = InferPipeline.from_checkpoint(checkpoint, config=config)
        else:
            log.info("Using base backbone (untrained) — for quick demo only")
            from adr.models.sovits import SoVITS, SoVITSConfig
            model = SoVITS(SoVITSConfig())
            pipe = InferPipeline(model, config=config)
    except Exception as e:
        log.error(f"[FAIL] Pipeline init failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 合成
    log.info("=" * 60)
    log.info(f"Cloning: ref={ref}")
    log.info(f"  text:  {text}")
    log.info(f"  ->    {output}")
    log.info("=" * 60)

    try:
        wav = pipe.synthesize(text, ref, n_timesteps=n_timesteps)
        pipe.save_wav(wav, output)
        log.info(f"[OK] Saved: {output} (samples={len(wav)})")
    except Exception as e:
        log.error(f"[FAIL] Synthesis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@main.command()
@click.option("--ref", required=True, help="参考音频路径")
@click.option("--text", required=True, help="要合成的文本")
@click.option("--output", default="output.wav", help="输出 wav 路径")
@click.option("--checkpoint", "-c", default=None,
              help="训练好的 checkpoint 路径 (None=用基座模型)")
@click.option("--n-timesteps", default=20, help="扩散步数")
def infer(ref: str, text: str, output: str, checkpoint: str | None, n_timesteps: int) -> None:
    """推理 (M1 Day 9 实现) — 与 clone 相同,保留为独立命令以备扩展。"""
    # 直接复用 clone
    ctx = click.get_current_context()
    ctx.forward(clone)


@main.command()
@click.option("--format", "fmt", default="gguf", help="导出格式 (gguf/awq/onnx)")
def export(fmt: str) -> None:
    """量化导出 (M3 完整实现)。"""
    from adr.core import get_logger, setup_logging

    setup_logging(level="INFO")
    log = get_logger("adr.cli")
    log.warning(f"export to {fmt} is a Day 1 stub.")
    log.info("Full implementation by M3 (end of M2).")


@main.group()
def model() -> None:
    """预训练模型管理 (list/download/info/search)。"""
    pass


@model.command("list")
@click.option("--category", "-c", default=None,
              help="按类别筛选 (vocoder/content_encoder/...)")
@click.option("--all", "show_all", is_flag=True, help="显示所有 (含未下载)")
def model_list(category: str | None, show_all: bool) -> None:
    """列出所有可用预训练模型。"""
    from adr.core import get_logger, setup_logging
    from adr.models.hub import PRETRAINED_REGISTRY, get_hub

    setup_logging(level="INFO")
    log = get_logger("adr.cli.model")
    hub = get_hub()

    click.echo("Available pretrained models:")
    click.echo("=" * 78)
    header = f"{'NAME':<32} {'CATEGORY':<18} {'SIZE':<8} {'STATUS':<10}"
    click.echo(header)
    click.echo("-" * 78)

    for name, info in PRETRAINED_REGISTRY.items():
        if category and info.category != category:
            continue
        status = "[OK] cached" if hub.is_downloaded(name) else "[..] missing"
        size = f"{info.size_mb:.0f}MB"
        click.echo(
            f"{name:<32} {info.category:<18} {size:<8} {status:<10}"
        )

    click.echo("-" * 78)
    click.echo(f"Cache dir: {hub.cache_dir}")
    if category:
        log.info(f"Filtered by category: {category}")


@model.command("download")
@click.argument("name")
@click.option("--force", "-f", is_flag=True, help="强制重新下载 (覆盖)")
def model_download(name: str, force: bool) -> None:
    """下载预训练模型。

    示例:
        adr model download bigvgan_22khz_80band
    """
    from adr.core import get_logger, setup_logging
    from adr.models.hub import get_hub

    setup_logging(level="INFO")
    log = get_logger("adr.cli.model")

    hub = get_hub()
    try:
        path = hub.download(name, force=force, progress=True)
        log.info(f"[OK] {name} -> {path}")
    except Exception as e:
        log.error(f"[FAIL] Download failed: {e}")
        sys.exit(1)


@model.command("info")
@click.argument("name")
def model_info(name: str) -> None:
    """显示预训练模型详情。"""
    from adr.core import get_logger, setup_logging
    from adr.models.hub import PRETRAINED_REGISTRY, get_hub

    setup_logging(level="INFO")
    log = get_logger("adr.cli.model")

    if name not in PRETRAINED_REGISTRY:
        log.error(f"Unknown model: {name}")
        log.info(f"Available: {list(PRETRAINED_REGISTRY.keys())}")
        sys.exit(1)

    info = PRETRAINED_REGISTRY[name]
    hub = get_hub()
    is_dl = hub.is_downloaded(name)

    click.echo(f"Model: {info.name}")
    click.echo("=" * 60)
    click.echo(f"  Category:    {info.category}")
    click.echo(f"  Description: {info.description}")
    click.echo(f"  Version:     {info.version}")
    click.echo(f"  Size:        ~{info.size_mb:.0f} MB")
    click.echo(f"  URL:         {info.url}")
    if info.sha256:
        click.echo(f"  SHA256:      {info.sha256}")
    click.echo(f"  Status:      {'[OK] downloaded' if is_dl else '[..] not downloaded'}")
    if is_dl:
        path = hub.get_path(name)
        actual_mb = path.stat().st_size / 1024 / 1024
        click.echo(f"  Local path:  {path}")
        click.echo(f"  Actual size: {actual_mb:.1f} MB")


@model.command("search")
def model_search() -> None:
    """扫描本地已有的预训练权重 (用于发现未注册的资源)。"""
    from adr.core import get_logger, setup_logging
    from adr.models.hub import get_hub

    setup_logging(level="INFO")
    log = get_logger("adr.cli.model")
    hub = get_hub()

    log.info("Scanning local pretrained weights...")
    results = hub.search_local()

    if not results:
        click.echo("[!] No local pretrained weights found.")
        click.echo("    Searched paths:")
        click.echo(f"      - {hub.cache_dir}")
        click.echo("      - F:/ADR_data/bigvgan")
        click.echo("      - F:/ADR_data/wavlm")
    else:
        click.echo("Found local pretrained weights:")
        click.echo("-" * 78)
        for name, path in results:
            size_mb = path.stat().st_size / 1024 / 1024
            click.echo(f"  {name:<40} {size_mb:>7.1f} MB  {path}")


@model.command("path")
@click.argument("name")
def model_path(name: str) -> None:
    """打印预训练模型的本地路径 (已下载时)。"""
    from adr.models.hub import get_hub

    hub = get_hub()
    try:
        path = hub.get_path(name)
        click.echo(str(path))
    except FileNotFoundError as e:
        click.echo(f"error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option("--data", "-d", required=True, help="训练数据目录 (含 samples/*.npz)")
@click.option("--backbone", "-b", default="sovits", help="Backbone 类型 (sovits)")
@click.option("--config", "-c", default=None, help="YAML 配置文件 (可选)")
@click.option("--output", "-o", default="output/train", help="输出目录")
@click.option("--epochs", "-e", default=10, help="训练轮数")
@click.option("--batch-size", default=None, type=int, help="batch size (None 则按显存自动)")
@click.option("--lr", default=1e-4, help="学习率")
@click.option("--no-amp", is_flag=True, help="关闭混合精度")
@click.option("--no-grad-ckpt", is_flag=True, help="关闭 gradient checkpointing")
@click.option("--resume", default=None, help="从 checkpoint 恢复训练")
@click.option("--lora", is_flag=True, help="启用 LoRA 微调 (只训 ~1% 参数, 显存降 70%%)")
@click.option("--lora-rank", default=8, help="LoRA rank")
@click.option("--lora-alpha", default=16, help="LoRA alpha")
@click.option("--qlora", is_flag=True,
              help="真 QLoRA: 4-bit 量化基座 + LoRA (显存再降 60%%, 需 CUDA + bitsandbytes)")
def train(
    data: str,
    backbone: str,
    config: str | None,
    output: str,
    epochs: int,
    batch_size: int | None,
    lr: float,
    no_amp: bool,
    no_grad_ckpt: bool,
    resume: str | None,
    lora: bool,
    lora_rank: int,
    lora_alpha: int,
    qlora: bool,
) -> None:
    """训练模型 (M1 Day 8 实现)。

    用法:
        adr train -d output/processed
        adr train -d output/processed -b sovits -e 20 -o output/my_model
        adr train -d output/processed --lora              # LoRA 微调
        adr train -d output/processed --qlora             # 4-bit QLoRA (8GB 可训 0.3B)
    """
    from adr.core import get_logger, setup_logging
    from adr.core.registry import REGISTRY
    from adr.training import Trainer, TrainerConfig

    setup_logging(level="INFO")
    log = get_logger("adr.cli.train")

    use_lora = lora or qlora

    # 1. 选 backbone
    log.info(f"Loading backbone: {backbone}")
    bb_cls = REGISTRY.backbone.get(backbone)
    if bb_cls is None:
        log.error(f"Unknown backbone: {backbone}")
        log.info(f"Available: {list(REGISTRY.backbone.keys())}")
        sys.exit(1)
    model = bb_cls()
    log.info(f"  {model}")

    # 2. LoRA / QLoRA (必须先 apply_lora 再 quantize_4bit)
    if use_lora:
        import torch
        from adr.training.lora import LoRAConfig, apply_lora
        # 注意: 本文件定义了 list 子命令遮蔽内置 list, 这里用解包代替 list()
        lora_cfg = LoRAConfig(
            rank=lora_rank,
            alpha=lora_alpha,
            target_modules=[*TrainerConfig().lora_target_modules],
        )
        apply_lora(model, lora_cfg)
    if qlora:
        import torch
        from adr.training.efficient import is_bnb_available, quantize_4bit
        if not is_bnb_available():
            log.error("--qlora 需要 bitsandbytes: pip install bitsandbytes")
            sys.exit(1)
        if not torch.cuda.is_available():
            log.error("--qlora 需要 CUDA (bnb 4bit matmul 不支持 CPU), 请改用 --lora")
            sys.exit(1)
        # 先上 CUDA 再量化 (bnb 标准 workflow: .to(device) 触发量化)
        model = model.cuda()
        model = quantize_4bit(model)
        log.info("  QLoRA: 4-bit 量化基座已应用 (MHA qkv + LoRA 基座 + 其余 Linear)")

    # 3. 配置
    cfg = TrainerConfig(
        epochs=epochs,
        lr=lr,
        output_dir=output,
        use_amp=not no_amp,
        use_gradient_checkpointing=not no_grad_ckpt,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
    if batch_size is not None:
        cfg.batch_size = batch_size
    # 读 YAML (可选)
    if config:
        try:
            import yaml
            with open(config, encoding="utf-8") as f:
                yaml_cfg = yaml.safe_load(f)
            for k, v in yaml_cfg.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            log.info(f"  Config loaded: {config}")
        except Exception as e:
            log.warning(f"  Failed to load config: {e}")

    # 4. 训练
    log.info("=" * 60)
    log.info(f"Training: {backbone} on {data}")
    log.info(f"  Output: {output}")
    log.info(f"  Epochs: {epochs}, lr={lr}")
    log.info(f"  LoRA: {use_lora} (rank={lora_rank}), QLoRA: {qlora}")
    log.info(f"  AMP: {cfg.use_amp}, GradCkpt: {cfg.use_gradient_checkpointing}")
    log.info("=" * 60)

    try:
        trainer = Trainer(model=model, train_data=data, config=cfg)
        if resume:
            log.info(f"Resuming from: {resume}")
            trainer.load_checkpoint(resume)
        metrics = trainer.fit()
        log.info(f"[OK] Training complete: {metrics}")
    except FileNotFoundError as e:
        log.error(f"[FAIL] {e}")
        log.info("Run data pipeline first: adr process -i <audio>")
        sys.exit(1)
    except Exception as e:
        log.error(f"[FAIL] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@main.command()
@click.option("--host", default="127.0.0.1", help="WebUI 监听地址")
@click.option("--port", default=7860, help="WebUI 端口")
@click.option("--share", is_flag=True, help="生成公网分享链接")
@click.option("--no-browser", is_flag=True, help="启动后不自动打开浏览器")
def webui(host: str, port: int, share: bool, no_browser: bool) -> None:
    """启动 WebUI (Day 10 完整实现, 当前为骨架)。"""
    from adr.core import get_logger, setup_logging

    setup_logging(level="INFO")
    log = get_logger("adr.cli")

    log.info("Launching ADR WebUI...")
    log.info(f"  URL:  http://{host}:{port}")

    try:
        from adr.webui import build_ui
    except ImportError as e:
        log.error(f"gradio not installed: {e}")
        log.info("Install with: pip install gradio>=4.0")
        sys.exit(1)

    demo = build_ui()
    from adr.webui import _CSS, _theme
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=not no_browser,
        theme=_theme(),
        css=_CSS,
    )


if __name__ == "__main__":
    main()
