"""ADR WebUI (Gradio 6+)。

设计原则:
1. 简洁克制的界面 — 无渐变/无表情符号/中性色,信息密度优先
2. 外部可调用 — 所有操作暴露命名 API (gradio_client / REST: /gradio_api/)
3. 稳定错误处理 — 输入先校验, 错误给出可操作建议, 不向 UI 泄漏 traceback

四个 Tab:
- Data:   音频 → 训练样本 (process)
- Train:  启动训练 (支持 LoRA / QLoRA)
- Clone:  文本 + 参考音频 → wav
- Models: 预训练模型管理

启动: adr webui --host 127.0.0.1 --port 7860
外部调用:
    from gradio_client import Client
    c = Client("http://127.0.0.1:7860/")
    c.predict(None, "default", "上传音频", api_name="/process")
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]

# 允许的音频扩展名
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

# 训练子进程超时 (秒)
_TRAIN_TIMEOUT = 7200
_CLONE_TIMEOUT = 300


# ============================================================
# UI 主题 (简洁中性, 无 AI 风格渐变)
# ============================================================
_CSS = """
/* 全局: 系统字体栈 + 克制配色 */
.gradio-container {
    font-family: system-ui, -apple-system, "Segoe UI", "PingFang SC",
                 "Microsoft YaHei", sans-serif !important;
    max-width: 1100px !important;
}
/* 标题区: 左对齐纯文本 */
#adr-header h1 { font-size: 1.4rem; font-weight: 600; margin: 0; }
#adr-header p  { color: #666; font-size: 0.9rem; margin: 4px 0 0 0; }
/* 卡片: 细边框, 无阴影 */
.tabitem, .form, .block {
    border: 1px solid #e5e5e5 !important;
    box-shadow: none !important;
    border-radius: 6px !important;
}
/* 主按钮: 深灰实心, 无渐变 */
button.primary, .gr-button-primary {
    background: #1a1a1a !important;
    border: 1px solid #1a1a1a !important;
    color: #fff !important;
}
button.primary:hover { background: #333 !important; }
/* 小节标签: 小号大写字母感 */
.adr-section { font-size: 0.8rem; letter-spacing: 0.06em; color: #888;
               text-transform: uppercase; margin-bottom: 4px; }
/* 错误/日志框: 等宽字体 */
textarea { font-family: ui-monospace, Consolas, monospace !important; }
footer { display: none !important; }
"""


def _theme():
    import gradio as gr

    return gr.themes.Base(
        primary_hue=gr.themes.colors.neutral,
        secondary_hue=gr.themes.colors.neutral,
        neutral_hue=gr.themes.colors.gray,
        font=(gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"),
    )


# ============================================================
# UI 构建
# ============================================================
def _list_opencpop_utts(limit: int = 200) -> list[str]:
    """OpenCpop 句子列表 (utt_id | 文本)。"""
    path = Path("data/opencpop/transcriptions.txt")
    if not path.exists():
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        f = line.strip().split("|")
        if len(f) >= 2:
            out.append(f"{f[0]} | {f[1]}")
        if len(out) >= limit:
            break
    return out


def _run_sing(utt_sel, custom_text, ref_audio, ph_seq, note_seq,
              note_dur_seq, is_slur_seq, speedup):
    """唱歌 Tab 回调: 优先级 自定义旋律 > 参考音频+歌词 (B4 桥) > OpenCpop 标注。"""
    import time

    import soundfile as sf

    try:
        from adr.models.diffsinger_engine import get_engine
        eng = get_engine()
        mode = ""
        if ph_seq and note_seq and note_dur_seq:
            wav, sr = eng.synthesize(
                ph_seq=ph_seq, note_seq=note_seq, note_dur_seq=note_dur_seq,
                is_slur_seq=is_slur_seq or "", speedup=int(speedup))
            label = "自定义旋律"
            mode = "phoneme"
        elif custom_text and ref_audio:
            wav, sr = eng.sing_like(custom_text, ref_audio,
                                    speedup=int(speedup))
            label = custom_text
            mode = "ref-melody (B4)"
        elif utt_sel:
            utt_id = utt_sel.split("|")[0].strip()
            kwargs = eng.load_opencpop_annotation(utt_id)
            label = kwargs.pop("text")
            wav, sr = eng.synthesize(speedup=int(speedup), **kwargs)
            mode = "opencpop-annotation"
        else:
            return None, _err("请选择 OpenCpop 句子, 或填 歌词+参考音频, 或自定义旋律")
        t0 = time.time()
        out_dir = Path("output/webui_sing")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"sing_{int(t0)}.wav"
        sf.write(str(out_path), wav, sr)
        return str(out_path), {
            "text": label,
            "mode": mode,
            "duration_s": round(len(wav) / sr, 2),
            "sample_rate": sr,
            "engine": "DiffSinger v1 (official opencpop pretrained)",
        }
    except Exception as e:
        return None, _err(f"歌声合成失败: {e}",
                          "首次运行需加载 ~1.3GB 权重, 确认 third_party/diffsinger_v1_code/checkpoints 齐全")


def build_ui(share: bool = False, inbrowser: bool = True) -> "gr.Blocks":
    """构建 Gradio UI (懒加载 gradio)。

    Returns:
        gr.Blocks 实例,调用 .launch() 启动。
    """
    import gradio as gr

    with gr.Blocks(title="ADR Voice Clone") as demo:
        # ---- 头部 ----
        with gr.Row(elem_id="adr-header"):
            gr.Markdown(
                "# ADR Voice Clone\n"
                "低资源声音克隆 · 8GB 显存可训练 · LoRA / QLoRA"
            )

        # ---- Tab 1: Data ----
        with gr.Tab("Data"):
            gr.Markdown('<p class="adr-section">数据预处理</p>')
            with gr.Row():
                with gr.Column(scale=1):
                    data_source = gr.Dropdown(
                        choices=["上传自己的音频", "OpenCpop 歌声数据 (3550 句, 免处理)"],
                        value="上传自己的音频",
                        label="数据源",
                    )
                    data_input = gr.Audio(label="参考音频", type="filepath")
                    data_preset = gr.Dropdown(
                        choices=["default", "vram_4gb", "vram_6gb", "vram_8gb"],
                        value="default",
                        label="配置预设",
                    )
                    data_run = gr.Button("开始处理", variant="primary")
                with gr.Column(scale=1):
                    data_log = gr.Textbox(label="处理日志", lines=14, interactive=False)
                    data_summary = gr.JSON(label="摘要")

            data_run.click(
                _run_data_pipeline,
                inputs=[data_input, data_preset, data_source],
                outputs=[data_log, data_summary],
                api_name="process",
            )

        # ---- Tab 2: Train ----
        with gr.Tab("Train"):
            gr.Markdown('<p class="adr-section">模型训练</p>')
            gr.Markdown(
                "模型类型由训练数据决定:OpenCpop 是**唱歌**数据,训出的是唱歌模型;"
                "在 Data 页上传**说话**音频处理后再训练,得到的就是说话模型。"
                "8GB 显存训练 0.3B 及以上规模请勾选 LoRA 或 QLoRA。"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    train_data_dir = gr.Dropdown(
                        choices=[
                            "data/opencpop_npz/train (OpenCpop 3550 句)",
                            "data/opencpop_npz/test (OpenCpop 206 句)",
                            "output/processed (自定义)",
                        ],
                        value="data/opencpop_npz/train (OpenCpop 3550 句)",
                        label="数据目录",
                        allow_custom_value=True,
                    )
                    train_preset = gr.Dropdown(
                        choices=[
                            ("small · 10M", "small"),
                            ("medium · 30M", "medium"),
                            ("large · 180M", "large"),
                            ("x0p3b · 340M (8GB 需 LoRA/QLoRA)", "x0p3b"),
                            ("x0p4b · 450M", "x0p4b"),
                        ],
                        value="medium",
                        label="模型规模",
                    )
                    train_n_samples = gr.Slider(
                        minimum=10, maximum=3550, value=500, step=10,
                        label="训练样本数",
                    )
                    train_epochs = gr.Slider(
                        minimum=1, maximum=20, value=3, step=1,
                        label="训练轮数",
                    )
                    with gr.Row():
                        train_use_lora = gr.Checkbox(label="LoRA", value=False)
                        train_use_qlora = gr.Checkbox(
                            label="QLoRA (4-bit 基座)", value=False
                        )
                        train_use_amp = gr.Checkbox(label="混合精度", value=True)
                    train_run = gr.Button("开始训练", variant="primary")
                with gr.Column(scale=1):
                    train_log = gr.Textbox(label="训练日志", lines=18, interactive=False)
                    train_metrics = gr.JSON(label="指标")

            train_run.click(
                _run_train_cmd,
                inputs=[train_data_dir, train_preset, train_n_samples,
                        train_epochs, train_use_lora, train_use_qlora, train_use_amp],
                outputs=[train_log, train_metrics],
                api_name="train",
            )

            # ---- 声音克隆微调 (GPT-SoVITS · 说话音色, 4GB 配方) ----
            gr.Markdown('<p class="adr-section">声音克隆微调 (GPT-SoVITS · 说话音色)</p>')
            gr.Markdown(
                "上传一段**说话**录音 (≥1 分钟, 越长越稳), 只微调 s2 音色层 (~7 分钟起)。"
                "**4GB 低配** = bs1 + 10s 截断配方, 训练自身 ~3.1-3.5GB 显存;"
                "质量门禁在 CPU 上并行打 5 句相似度均值, ≥0.80 自动早停 (不抢显存)。"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    ft_audio = gr.Audio(label="原始录音 (mp3/wav)", type="filepath")
                    ft_exp = gr.Textbox(label="音色名 (训练标识)", value="my_voice")
                    ft_recipe = gr.Radio(
                        choices=[
                            ("4GB 低配 (bs1 + 10s 截断, 推荐)", "4gb"),
                            ("8GB 标准 (bs4)", "8gb"),
                        ],
                        value="4gb",
                        label="显存配方",
                    )
                    ft_epochs = gr.Slider(
                        minimum=1, maximum=16, value=8, step=1,
                        label="s2 训练轮数 (门禁达标会提前停)",
                    )
                    ft_gate = gr.Checkbox(
                        label="质量门禁自动早停 (5 句均值 sim ≥ 0.80)", value=True,
                    )
                    ft_bind = gr.Dropdown(
                        choices=_list_voices(),
                        value=None,
                        label="训完自动绑定到已有音色档案 (可选)",
                        allow_custom_value=True,
                    )
                    ft_run = gr.Button("开始克隆微调", variant="primary")
                with gr.Column(scale=1):
                    ft_log = gr.Textbox(label="微调日志", lines=18, interactive=False)
                    ft_metrics = gr.JSON(label="门禁曲线 / 指标")

            ft_run.click(
                _run_gsv_finetune,
                inputs=[ft_audio, ft_exp, ft_recipe, ft_epochs, ft_gate, ft_bind],
                outputs=[ft_log, ft_metrics],
                api_name="clone_train",
            )

        # ---- Tab 3: Clone ----
        with gr.Tab("Clone"):
            gr.Markdown('<p class="adr-section">文本转语音 (TTS)</p>')
            gr.Markdown(
                "输入文本,用所选模型的声音朗读。"
                "标注「唱歌」的模型用 OpenCpop 歌声数据训练,朗读会带唱腔。"
                "标注「演示」的权重只训练了几秒,**合成必然是噪声**,"
                "请选用正式训练的权重 (数百样本以上)。"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    clone_voice = gr.Dropdown(
                        choices=_list_voices(),
                        value=None,
                        label="已存音色 (选了就不用传参考音频)",
                        allow_custom_value=True,
                    )
                    clone_voice_refresh = gr.Button("刷新音色列表")
                    clone_engine = gr.Radio(
                        choices=[
                            ("GPT-SoVITS 保底 (推荐, 官方级质量)", "gsv"),
                            ("自研 ADR-2 (实验, 需选权重)", "adr2"),
                        ],
                        value="gsv",
                        label="合成引擎",
                    )
                    clone_prompt = gr.Textbox(
                        label="参考音频对应的文本 (GPT-SoVITS, 可空但建议填)",
                        placeholder="参考音频里说的那句话",
                        lines=1,
                    )
                    clone_ckpt = gr.Dropdown(
                        choices=_scan_checkpoints(),
                        value=None,
                        label="模型权重 (仅自研引擎需要)",
                        allow_custom_value=True,
                    )
                    clone_ckpt_refresh = gr.Button("刷新模型列表")
                    clone_ref = gr.Audio(label="参考音频 (3-10 秒干净人声, 决定音色)", type="filepath")
                    clone_text = gr.Textbox(
                        label="要朗读的文本",
                        placeholder="你好,这是 ADR 框架的演示。",
                        lines=3,
                    )
                    with gr.Row():
                        clone_split = gr.Dropdown(
                            choices=[
                                ("按中文句号切 (推荐, 流式首包最快)", "cut3"),
                                ("凑四句一切", "cut1"),
                                ("不切 (整段一口气)", "cut0"),
                                ("凑50字一切", "cut2"),
                                ("按标点符号切 (每标点一顿)", "cut5"),
                            ],
                            value="cut3",
                            label="切句方式",
                        )
                        clone_speed = gr.Slider(
                            0.6, 1.6, value=1.0, step=0.05,
                            label="语速 (>1 加快)",
                        )
                    clone_run = gr.Button("开始合成", variant="primary")
                    clone_stream = gr.Button("流式合成 (边生成边播)")
                    with gr.Row():
                        clone_save_name = gr.Textbox(
                            label="音色名", placeholder="我的声音", lines=1)
                        clone_save_btn = gr.Button("保存当前参考为音色档案")
                with gr.Column(scale=1):
                    clone_output = gr.Audio(label="合成结果", type="filepath")
                    clone_stream_out = gr.Audio(
                        label="流式播放 (边生成边播)", streaming=True,
                        autoplay=True, interactive=False)
                    clone_info = gr.JSON(label="执行信息")

            clone_ckpt_refresh.click(
                lambda: gr.update(choices=_scan_checkpoints()),
                outputs=[clone_ckpt],
            )
            clone_voice_refresh.click(
                lambda: gr.update(choices=_list_voices()),
                outputs=[clone_voice],
            )
            clone_save_btn.click(
                _save_voice_cmd,
                inputs=[clone_save_name, clone_ref, clone_prompt],
                outputs=[clone_info],
            )
            clone_run.click(
                _run_clone_cmd,
                inputs=[clone_ref, clone_text, clone_ckpt, clone_engine,
                        clone_prompt, clone_voice, clone_split, clone_speed],
                outputs=[clone_output, clone_info],
                api_name="clone",
            )
            clone_stream.click(
                _stream_clone_cmd,
                inputs=[clone_ref, clone_text, clone_prompt, clone_voice, clone_split],
                outputs=[clone_stream_out],
            )

        # ---- Tab 4: Sing (SVS 保底基线) ----
        with gr.Tab("Sing"):
            gr.Markdown('<p class="adr-section">歌声合成 (DiffSinger 官方预训练, 保底基线)</p>')
            gr.Markdown(
                "用官方 OpenCpop 预训练模型唱歌, 音质远好于当前自研模型。"
                "选择一条 OpenCpop 标注 (自带旋律) 即可合成; "
                "高级区可自定义 音素/音符/时长 序列。"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    sing_utt = gr.Dropdown(
                        choices=_list_opencpop_utts(),
                        value=None,
                        label="OpenCpop 句子 (按标注旋律演唱)",
                        allow_custom_value=True,
                    )
                    sing_text = gr.Textbox(
                        label="自定义歌词 (配合下方参考音频的旋律)",
                        placeholder="今天的天气真不错,我们一起去公园散步",
                        lines=2,
                    )
                    sing_ref = gr.Audio(
                        label="参考音频 (提供旋律, 歌声/哼唱均可)", type="filepath")
                    sing_speedup = gr.Slider(
                        10, 100, value=40, step=10,
                        label="PNDM 加速 (越大越快略降质)",
                    )
                    sing_run = gr.Button("开始演唱", variant="primary")
                with gr.Column(scale=1):
                    sing_output = gr.Audio(label="合成歌声", type="filepath")
                    sing_info = gr.JSON(label="合成信息")
            with gr.Accordion("高级: 自定义旋律 (音素/音符/时长)", open=False):
                sing_ph = gr.Textbox(label="ph_seq", placeholder="g an sh ou ... (空格分隔)")
                sing_note = gr.Textbox(label="note_seq", placeholder="C#4/Db4 C#4/Db4 ...")
                sing_ndur = gr.Textbox(label="note_dur_seq", placeholder="0.25 0.25 ...")
                sing_slur = gr.Textbox(label="is_slur_seq (可空)", placeholder="0 0 0 ...")

            sing_run.click(
                _run_sing,
                inputs=[sing_utt, sing_text, sing_ref, sing_ph, sing_note,
                        sing_ndur, sing_slur, sing_speedup],
                outputs=[sing_output, sing_info],
                api_name="sing",
            )

        # ---- Tab 5: Models ----
        with gr.Tab("Models"):
            gr.Markdown('<p class="adr-section">预训练模型</p>')
            model_list_btn = gr.Button("刷新列表")
            model_list_out = gr.Dataframe(
                headers=["Name", "Category", "Size (MB)", "Status"],
                label="模型列表",
            )
            model_list_btn.click(
                _list_pretrained,
                outputs=[model_list_out],
                api_name="list_models",
            )

        # ---- API 说明 ----
        with gr.Accordion("API (外部调用)", open=False):
            gr.Markdown(
                "启动后可通过 REST (`/gradio_api/`) 或 gradio_client 调用:\n"
                "```python\n"
                "from gradio_client import Client\n"
                "c = Client('http://127.0.0.1:7860/')\n"
                "c.predict('data/opencpop_npz/train', 'medium', 500, 3,\n"
                "          True, False, True, api_name='/train')\n"
                "c.predict('ref.wav', '你好', 'ckpt.pt', api_name='/clone')\n"
                "```\n"
                "接口列表见 `http://127.0.0.1:7860/gradio_api/`"
            )

    return demo


# ============================================================
# 错误处理工具
# ============================================================
def _err(msg: str, suggestion: str = "") -> dict:
    """统一错误返回格式。"""
    return {"error": msg, "suggestion": suggestion} if suggestion else {"error": msg}


def _classify_subprocess_error(stderr: str) -> tuple[str, str]:
    """把子进程 stderr 归类为 (可读原因, 建议)。"""
    s = stderr.lower()
    if "outofmemoryerror" in s or "out of memory" in s:
        return (
            "GPU 显存不足 (OOM)",
            "勾选 QLoRA (0.3B 仅需 ~0.8GB), 或换更小 preset / 减少样本数",
        )
    if "device-side assert" in s:
        return ("CUDA device-side assert", "检查音素表 vocab 与模型 vocab_size 是否一致 (607)")
    if "bitsandbytes" in s and ("import" in s or "not" in s):
        return ("bitsandbytes 不可用", "pip install bitsandbytes, 或取消 QLoRA")
    if "filenotfounderror" in s or "no such file" in s:
        return ("文件缺失", "检查数据目录 / checkpoint 路径是否存在")
    tail = stderr.strip().split("\n")[-1][:200] if stderr.strip() else "未知错误"
    return (f"子进程错误: {tail}", "查看日志末尾获取详情")


def _validate_audio(audio_path: Optional[str]) -> Optional[dict]:
    """校验上传音频, 失败返回错误 dict, 成功返回 None。"""
    if not audio_path:
        return _err("请先上传参考音频")
    p = Path(audio_path)
    if not p.exists():
        return _err(f"音频文件不存在: {p.name}", "请重新上传")
    if p.suffix.lower() not in _AUDIO_EXTS:
        return _err(
            f"不支持的音频格式: {p.suffix}",
            f"支持: {', '.join(sorted(_AUDIO_EXTS))}",
        )
    if p.stat().st_size < 10_000:  # < 10KB 基本是空文件
        return _err("音频文件过小 (<10KB)", "参考音频建议 5-30 秒")
    return None


# ============================================================
# checkpoint 扫描
# ============================================================
def _ckpt_kind(rel_path: str) -> str:
    """按路径推断模型类型标签 (唱歌/说话, LoRA, 演示)。"""
    p = rel_path.lower()
    kind = "唱歌" if "opencpop" in p else "说话"
    if "lora" in p:
        kind += "·LoRA"
    # 只跑了几秒/几样本的演示权重 → 合成会是噪声, 明确标注
    if any(k in p for k in ("workdir", "smoke", "_test", "cli_", "_qlora_smoke")):
        kind = "演示·" + kind
    return kind


def _scan_checkpoints() -> list[tuple[str, str]]:
    """扫描 examples/ 与 output/ 下的 .pt, 返回 (显示标签, 相对路径), 新的在前。"""
    found: list[Path] = []
    for root in (REPO / "examples", REPO / "output"):
        if root.exists():
            found.extend(root.rglob("*.pt"))
    found = [p for p in found if p.stat().st_size > 1024]  # 过滤空文件
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in found[:50]:
        rel = p.relative_to(REPO).as_posix()
        out.append((f"[{_ckpt_kind(rel)}] {rel}", rel))
    return out


# ============================================================
# 内部回调
# ============================================================
def _run_data_pipeline(audio_path: Optional[str], preset: str, source: str):
    """数据处理回调。"""
    from adr.core import get_logger
    from adr.data.pipeline import DataPipeline, PipelineConfig

    log = get_logger("adr.webui.data")

    # OpenCpop 预置数据
    if "OpenCpop" in source:
        op_dir = REPO / "data" / "opencpop_npz" / "train"
        n = len(list(op_dir.glob("*.npz"))) if op_dir.exists() else 0
        if n == 0:
            return "[!] OpenCpop 数据未就绪", _err(
                f"目录无 npz: {op_dir}", "先运行数据准备脚本或改用上传音频"
            )
        return (
            f"[OK] OpenCpop 数据集已就绪\n  目录: {op_dir}\n  样本数: {n}\n"
            f"  在 Train Tab 选择 'data/opencpop_npz/train' 即可训练。",
            {"source": "opencpop", "n_samples": n, "dir": str(op_dir)},
        )

    if err := _validate_audio(audio_path):
        return f"[!] {err['error']}", err

    log.info(f"Processing {audio_path} with preset={preset}")
    try:
        config = PipelineConfig(
            enable_asr=True,
            enable_f0=True,
            output_dir=f"output/{preset}",
        )
        pipeline = DataPipeline(config)
        result = pipeline.run(audio_path, output_dir=config.output_dir)
        return result.summary(), {
            "slices": result.num_slices,
            "total_duration_sec": round(result.total_duration, 2),
            "output_dir": config.output_dir,
        }
    except Exception as e:
        log.error(f"Pipeline failed: {e}")
        msg, hint = _classify_subprocess_error(str(e))
        return f"[FAIL] {msg}", _err(msg, hint)


def _run_train_cmd(
    data_dir: str,
    preset: str,
    n_samples: int,
    epochs: int,
    use_lora: bool,
    use_qlora: bool,
    use_amp: bool,
):
    """训练回调 - 子进程流式输出 (生成器, 日志实时刷新到 UI)。

    每次 yield (日志文本, 指标); 最后一次 yield 带最终指标或错误。
    """
    import subprocess
    import sys
    import time

    log_buf = []

    # 解析 data_dir (去掉括号注释)
    if "(" in data_dir:
        data_dir = data_dir.split("(")[0].strip()

    # ---- 输入校验 (失败时 yield 一次即结束) ----
    data_path = REPO / data_dir
    if not data_path.exists():
        yield "[FAIL] 数据目录不存在", _err(
            f"数据目录不存在: {data_path}", "先在 Data Tab 处理音频, 或选 OpenCpop 预置数据"
        )
        return
    n_npz = len(list(data_path.glob("*.npz")))
    if n_npz == 0:
        yield "[FAIL] 数据目录为空", _err(
            f"目录内无 .npz 样本: {data_path}", "先在 Data Tab 跑数据预处理"
        )
        return
    if n_samples > n_npz:
        n_samples = n_npz
        log_buf.append(f"[!] 样本数调整为实际可用: {n_samples}")

    script = REPO / "scripts" / "smoke_train_opencpop.py"
    if not script.exists():
        yield "[FAIL] 训练脚本缺失", _err(f"找不到 {script}")
        return

    # 大模型不挂 LoRA/QLoRA → 提前警告 (8GB 全量 x0p4b 必 OOM)
    if preset == "x0p4b" and not (use_lora or use_qlora):
        yield "[FAIL] x0p4b 全量训练超出 8GB", _err(
            "x0p4b 全量需 >8GB 显存", "勾选 LoRA 或 QLoRA, 或改用 x0p3b"
        )
        return

    batch_size = 4 if preset in ("x0p3b", "x0p4b", "large") else 8

    # -u: 子进程无缓冲, 日志实时流出
    cmd = [
        sys.executable, "-u", str(script),
        "--preset", preset,
        "--n-samples", str(n_samples),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--data-dir", str(data_path),
        "--output", str(REPO / "examples" / "webui_train"),
    ]
    if use_lora:
        cmd.append("--use-lora")
    if use_qlora:
        cmd.append("--quantize-4bit")
    if use_amp:
        cmd.append("--use-amp")

    # 显示完整运行命令 (用户可见, 便于复现/排查)
    log_buf.append(f"[CMD] {' '.join(cmd)}")
    log_buf.append(f"[配置] preset={preset} 样本={n_samples} epochs={epochs} "
                   f"LoRA={use_lora} QLoRA={use_qlora} AMP={use_amp}")
    log_buf.append("=" * 50)
    yield "\n".join(log_buf), {"status": "训练启动中..."}

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            cwd=str(REPO), bufsize=1,
        )
    except Exception as e:
        yield "\n".join(log_buf), _err(f"训练进程启动失败: {e}")
        return

    lines: list[str] = []
    t0 = time.time()
    last_push = 0.0
    timed_out = False

    while True:
        line = proc.stdout.readline()
        if line:
            lines.append(line.rstrip())
            # 限频推送: 每 0.5s 最多刷新一次 UI
            now = time.time()
            if now - last_push >= 0.5:
                last_push = now
                yield "\n".join(log_buf + lines[-120:]), {"status": "训练中..."}
        elif proc.poll() is not None:
            break
        else:
            time.sleep(0.2)
            if time.time() - t0 > _TRAIN_TIMEOUT:
                proc.kill()
                timed_out = True
                break

    rc = proc.wait()
    tail = "\n".join(lines[-30:])
    final_log = "\n".join(log_buf + lines[-120:])

    if timed_out:
        msg = f"训练超时 (>{_TRAIN_TIMEOUT // 3600}h), 已终止"
        yield final_log + f"\n\n[FAIL] {msg}", _err(msg, "减少样本数 / epochs")
        return
    if rc != 0:
        msg, hint = _classify_subprocess_error(tail)
        yield final_log + f"\n\n[FAIL] {msg}" + (f"\n[建议] {hint}" if hint else ""), \
            _err(msg, hint)
        return
    yield final_log + "\n\n[OK] 训练完成", {
        "exit_code": 0, "preset": preset,
        "n_samples": n_samples, "epochs": epochs,
        "lora": use_lora, "qlora": use_qlora,
        "ckpt_dir": "examples/webui_train",
    }


def _run_gsv_finetune(
    audio_path: str,
    exp_name: str,
    recipe: str,
    epochs: int,
    gate_on: bool,
    bind_voice: str,
):
    """声音克隆微调回调 — gsv_finetune.py 子进程 + 可选 train_gate.py 并行早停。

    配方:
      8gb: bs4 (8GB 显存实测峰值 ~7.8GB)
      4gb: bs1 + 10s clip 截断 (实测训练自身 ~3.1-3.5GB, 补丁 #16/#17/#18)
    门禁走 CPU (--cpu): 离线打分不与训练抢显存, 5 句均值 ≥0.80 发 STOP 早停。
    """
    import json
    import subprocess
    import sys
    import time

    log_buf = []
    if not audio_path:
        yield "[FAIL] 请先选择原始录音", _err("缺少音频文件", "上传一段 ≥1 分钟的说话录音")
        return
    audio_path = str(audio_path)
    if not Path(audio_path).exists():
        yield "[FAIL] 音频文件不存在", _err(f"找不到 {audio_path}")
        return
    exp = (exp_name or "my_voice").strip()
    exp = "".join(c for c in exp if c not in '\\/:*?"<>|') or "my_voice"

    script = REPO / "scripts" / "gsv_finetune.py"
    cmd = [
        sys.executable, "-u", str(script), audio_path,
        "--exp", exp,
        "--s2-epochs", str(epochs),
        "--batch-size", "1" if recipe == "4gb" else "4",
        "--skip-s1",  # s1 全量微调 ~50min 且降音色相似度 (实测结论)
    ]
    if recipe == "4gb":
        cmd += ["--max-clip-sec", "10"]
    if bind_voice:
        cmd += ["--bind-voice", bind_voice]

    # 质量门禁: 训练前启动, CPU 打分, 达标往 logs/<exp>/STOP 写信号让 s2 早停
    gate_proc = None
    gate_curve = REPO / "output" / f"train_gate_{exp}.jsonl"
    if gate_on:
        gate_log = REPO / "output" / f"train_gate_{exp}.log"
        gate_log.parent.mkdir(exist_ok=True)
        gate_cmd = [
            sys.executable, "-u", str(REPO / "scripts" / "train_gate.py"),
            "--exp", exp, "--ref", audio_path, "--target", "0.80", "--cpu",
        ]
        try:
            gate_fh = open(gate_log, "a", encoding="utf-8", errors="replace")
            gate_proc = subprocess.Popen(
                gate_cmd, stdout=gate_fh, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", cwd=str(REPO),
            )
            log_buf.append(f"[gate] 质量门禁已启动 (PID {gate_proc.pid}, CPU 打分, "
                           f"曲线 {gate_curve.name})")
        except Exception as e:
            log_buf.append(f"[!] 门禁启动失败 (不影响训练): {e}")

    log_buf.append(f"[CMD] {' '.join(cmd)}")
    log_buf.append(f"[配置] exp={exp} 配方={recipe} (bs={'1' if recipe == '4gb' else '4'}"
                   f"{', clip≤10s' if recipe == '4gb' else ''}) s2轮数={epochs} "
                   f"门禁={'开' if gate_on else '关'}"
                   + (f" 绑定档案={bind_voice}" if bind_voice else ""))
    log_buf.append("=" * 50)
    yield "\n".join(log_buf), {"status": "克隆微调启动中..."}

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            cwd=str(REPO), bufsize=1,
        )
    except Exception as e:
        yield "\n".join(log_buf), _err(f"微调进程启动失败: {e}")
        return

    def _gate_tail(seen_n):
        """读门禁曲线新增行, 转成日志行。返回 (新 seen_n, 日志行列表)。"""
        out = []
        try:
            lines = gate_curve.read_text(encoding="utf-8").splitlines()
        except OSError:
            return seen_n, out
        for ln in lines[seen_n:]:
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            ep = d.get("epoch", "?")
            if d.get("kind") == "zero_shot":
                out.append(f"[gate] 零样本基线 mean={d['sim']:.3f} (逐句 {d.get('sims')})")
            else:
                out.append(f"[gate] epoch {ep}: mean={d['sim']:.3f} (逐句 {d.get('sims')})")
        return len(lines), out

    lines: list[str] = []
    t0 = time.time()
    last_push = 0.0
    gate_seen = 0
    gate_lines: list[str] = []
    timed_out = False

    while True:
        line = proc.stdout.readline()
        if line:
            lines.append(line.rstrip())
            now = time.time()
            if now - last_push >= 0.5:
                last_push = now
                gate_seen, new_gate = _gate_tail(gate_seen)
                gate_lines.extend(new_gate)
                yield "\n".join(log_buf + gate_lines + lines[-100:]), {"status": "微调中..."}
        elif proc.poll() is not None:
            break
        else:
            time.sleep(0.2)
            if time.time() - t0 > _TRAIN_TIMEOUT:
                proc.kill()
                timed_out = True
                break

    rc = proc.wait()
    tail = "\n".join(lines[-30:])
    final_log = "\n".join(log_buf + gate_lines + lines[-100:])

    # 门禁收尾: 训练结束 (含早停) 后门禁也该退场
    if gate_proc and gate_proc.poll() is None:
        gate_proc.terminate()
        try:
            gate_proc.wait(timeout=10)
        except Exception:
            gate_proc.kill()

    if timed_out:
        msg = f"微调超时 (>{_TRAIN_TIMEOUT // 3600}h), 已终止"
        yield final_log + f"\n\n[FAIL] {msg}", _err(msg, "减少轮数 / 样本")
        return

    # 汇总门禁曲线为指标
    metrics = {"exp": exp, "recipe": recipe, "exit_code": rc}
    gate_entries = []
    try:
        for ln in gate_curve.read_text(encoding="utf-8").splitlines():
            try:
                gate_entries.append(json.loads(ln))
            except ValueError:
                pass
    except OSError:
        pass
    if gate_entries:
        scored = [g for g in gate_entries if g.get("kind") != "zero_shot"]
        if scored:
            best = max(scored, key=lambda g: g["sim"])
            metrics["gate"] = {
                "zero_shot": gate_entries[0].get("sim"),
                "best_mean_sim": best.get("sim"),
                "best_epoch": best.get("epoch"),
                "best_weight": best.get("weight"),
                "target_met": bool(best.get("sim", 0) >= 0.80),
                "curve": [{"epoch": g.get("epoch"), "mean_sim": g.get("sim")}
                          for g in scored],
            }
    stopped = "收到训练门禁早停信号" in tail or "收到训练门禁早停信号" in final_log
    if stopped:
        metrics["early_stop"] = "质量门禁触发 (sim ≥ 0.80)"

    if rc != 0:
        msg, hint = _classify_subprocess_error(tail)
        yield final_log + f"\n\n[FAIL] {msg}" + (f"\n[建议] {hint}" if hint else ""), \
            {**_err(msg, hint), **metrics}
        return
    ok_msg = "\n\n[OK] 克隆微调完成"
    if metrics.get("gate", {}).get("target_met"):
        ok_msg += (f" (门禁达标: best mean sim={metrics['gate']['best_mean_sim']:.3f} "
                   f"@ epoch {metrics['gate']['best_epoch']}, "
                   f"权重 {metrics['gate']['best_weight']})")
    yield final_log + ok_msg, metrics


def _prewarm_engines():
    """A1: 后台线程顺序预热 GSV + DiffSinger (错峰加载避免显存瞬时叠加)。

    失败不致命 (首次合成时仍会按需加载), 只打日志。
    """
    import threading

    def _job():
        import logging
        log = logging.getLogger("adr.webui.prewarm")
        try:
            from adr.models.gsv_engine import get_gsv_engine
            from adr.models.voice_library import list_voices, load_voice
            kw = {}
            voices = list_voices()
            if voices:
                prof = load_voice(voices[0])
                kw = {"vits_weights": prof.get("vits_weights"),
                      "t2s_weights": prof.get("t2s_weights")}
                log.info("[prewarm] 用音色档案「%s」的权重预热", voices[0])
            get_gsv_engine().warmup(**kw)
            log.info("[prewarm] GPT-SoVITS 引擎就绪")
            # 流式路径形状预热: 否则首次点"流式合成"要多付 ~14s kernel JIT
            try:
                if voices:
                    for _ in get_gsv_engine().synthesize_stream(
                            "你好。", prof["ref_audio"],
                            vits_weights=prof.get("vits_weights"),
                            split_method="cut3"):
                        break  # 只取首块
                log.info("[prewarm] 流式路径预热完成")
            except Exception as e:
                log.warning("[prewarm] 流式预热失败 (不影响功能): %s", e)
        except Exception as e:
            log.warning("[prewarm] GSV 预热失败 (首次合成时将现场加载): %s", e)
        try:
            from adr.models.diffsinger_engine import get_engine
            get_engine().warmup()
            log.info("[prewarm] DiffSinger 引擎就绪")
        except Exception as e:
            log.warning("[prewarm] DiffSinger 预热失败: %s", e)

    threading.Thread(target=_job, daemon=True, name="adr-prewarm").start()


def _list_voices() -> list[str]:
    from adr.models.voice_library import list_voices
    return list_voices()


def _save_voice_cmd(name, ref_path, prompt_text):
    """保存音色档案回调 (A2: 长音频自动扫段选最优)。"""
    if err := _validate_audio(ref_path):
        yield err
        return
    if not name or not name.strip():
        yield _err("请填写音色名")
        return
    try:
        from adr.models.voice_library import save_voice_auto
        logs = []

        def _progress(msg):
            logs.append(msg)

        yield {"status": "建档中, 长音频将自动扫段选最优 (~1-2 分钟)…"}
        vdir, info = save_voice_auto(
            name.strip(), ref_path, (prompt_text or "").strip(),
            progress=_progress)
        yield {"saved": name.strip(), "path": str(vdir), **info,
               "sweep_log": logs,
               "hint": "以后在「已存音色」里直接选用"}
    except Exception as e:
        yield _err(f"保存失败: {e}")


def _stream_clone_cmd(ref_path, text, prompt_text, voice_name, split_method="cut3"):
    """A3: 流式克隆回调 — 逐块 yield (sr, int16) 给 streaming Audio。

    voice_name 命中档案时用档案参考/微调权重。
    """
    import numpy as np

    if not (text or "").strip():
        return
    t2s_w = vits_w = None
    if voice_name and voice_name.strip():
        try:
            from adr.models.voice_library import load_voice
            prof = load_voice(voice_name.strip())
            ref_path = prof["ref_audio"]
            prompt_text = prompt_text or prof.get("prompt_text") or ""
            t2s_w = prof.get("t2s_weights")
            vits_w = prof.get("vits_weights")
        except Exception:
            return
    if _validate_audio(ref_path):
        return
    try:
        from adr.models.gsv_engine import get_gsv_engine
        eng = get_gsv_engine()
        for chunk, sr in eng.synthesize_stream(
                text, ref_path, prompt_text=(prompt_text or "").strip(),
                t2s_weights=t2s_w, vits_weights=vits_w,
                split_method=split_method or "cut3"):
            pcm = np.clip(chunk, -1.0, 1.0)
            yield sr, (pcm * 32767).astype(np.int16)
    except Exception:
        return


def _run_clone_cmd(ref_path: Optional[str], text: str, ckpt_value: str,
                   engine: str = "gsv", prompt_text: str = "",
                   voice_name: str = "", split_method: str = "cut1",
                   speed_factor: float = 1.0):
    """TTS 回调。engine='gsv' 走 GPT-SoVITS 保底 (进程内), 'adr2' 走自研子进程。
    voice_name 命中已存音色时, 参考音频/参考文本/微调权重全部来自档案。"""
    t2s_w = vits_w = None
    if voice_name and voice_name.strip():
        try:
            from adr.models.voice_library import load_voice
            prof = load_voice(voice_name.strip())
            ref_path = prof["ref_audio"]
            prompt_text = prompt_text or prof.get("prompt_text") or ""
            t2s_w = prof.get("t2s_weights")
            vits_w = prof.get("vits_weights")
        except Exception as e:
            yield None, _err(f"音色档案加载失败: {e}")
            return
    if err := _validate_audio(ref_path):
        yield None, err
        return
    if not text or not text.strip():
        yield None, _err("请输入要朗读的文本")
        return
    text = text.strip()
    if len(text) > 500:
        yield None, _err("文本过长 (>500 字)", "拆分为多段分别合成")
        return

    import time

    if engine == "gsv":
        # GPT-SoVITS 保底路径 (官方预训练, 零样本克隆)
        try:
            import soundfile as sf

            from adr.models.gsv_engine import get_gsv_engine
            t0 = time.time()
            wav, sr = get_gsv_engine().synthesize(
                text, ref_path, prompt_text=(prompt_text or "").strip(),
                t2s_weights=t2s_w, vits_weights=vits_w,
                split_method=split_method or "cut1",
                speed_factor=speed_factor or 1.0)
            out_dir = Path("output/webui_clone")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"clone_{int(t0)}.wav"
            sf.write(str(out_path), wav, sr)
            info = {
                "engine": "GPT-SoVITS v2 (official pretrained)"
                          + (" + 微调" if t2s_w else ""),
                "voice": voice_name or "(临时参考)",
                "duration_s": round(len(wav) / sr, 2),
                "elapsed_s": round(time.time() - t0, 1),
                "sample_rate": sr,
                "timbre_from": Path(ref_path).name,
            }
            # A5: ERes2Net 声纹相似度自动打分 (失败不阻塞)
            try:
                from adr.eval.speaker_sim import similarity
                info["similarity"] = round(similarity(ref_path, str(out_path)), 3)
            except Exception as e:
                info["similarity"] = f"打分失败: {e}"
            yield str(out_path), info
        except Exception as e:
            yield None, _err(f"GPT-SoVITS 合成失败: {e}",
                             "首次需加载 ~600MB 权重; 确认 third_party/gpt_sovits 权重齐全")
        return

    if not ckpt_value or not ckpt_value.strip():
        yield None, _err("自研引擎需要选择模型权重 (checkpoint)",
                         "先在 Train Tab 训练, 或点「刷新模型列表」选择已有权重")
        return

    import subprocess
    import sys

    # ckpt_value 可能是扫描出的相对路径, 也可能是用户手输的路径
    full_ckpt = Path(ckpt_value)
    if not full_ckpt.is_absolute():
        full_ckpt = REPO / ckpt_value
    if not full_ckpt.exists():
        yield None, _err(f"模型权重不存在: {full_ckpt}",
                         "点「刷新模型列表」重新选择")
        return

    rel = full_ckpt.relative_to(REPO).as_posix() if full_ckpt.is_relative_to(REPO) else str(full_ckpt)

    out_path = REPO / "output" / "webui_tts" / f"tts_{int(time.time())}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-u", "-m", "adr.cli", "clone",
        "--ref", str(ref_path),
        "--text", text,
        "-c", str(full_ckpt),
        "--output", str(out_path),
    ]
    yield None, {"status": "合成中 (加载模型 + BigVGAN, 约 20-60 秒)...",
                 "cmd": " ".join(cmd)}
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(REPO), timeout=_CLONE_TIMEOUT,
        )
        if r.returncode == 0 and out_path.exists():
            yield str(out_path), {
                "text": text,
                "ckpt": rel,
                "model_type": _ckpt_kind(rel),
                "vocoder": "BigVGAN",
            }
            return
        msg, hint = _classify_subprocess_error(r.stderr)
        yield None, _err(msg, hint)
    except subprocess.TimeoutExpired:
        yield None, _err(f"合成超时 (>{_CLONE_TIMEOUT}s)",
                         "首次加载 BigVGAN 较慢, 请重试; 或检查 GPU 占用")
    except Exception as e:
        yield None, _err(f"WebUI 内部错误: {e}")


def _list_pretrained():
    """列出预训练模型。"""
    from adr.models.hub import PRETRAINED_REGISTRY, get_hub

    hub = get_hub()
    rows = []
    for name, info in PRETRAINED_REGISTRY.items():
        status = "已下载" if hub.is_downloaded(name) else "未下载"
        rows.append([name, info.category, f"{info.size_mb:.0f}", status])
    return rows


def launch(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    inbrowser: bool = True,
) -> None:
    """启动 WebUI (供 CLI 调用)。"""
    demo = build_ui(share=share, inbrowser=inbrowser)
    _prewarm_engines()  # A1: 后台常驻预热, 首次合成免冷启动
    # Gradio 6: theme/css 在 launch() 传入
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=inbrowser,
        theme=_theme(),
        css=_CSS,
    )
