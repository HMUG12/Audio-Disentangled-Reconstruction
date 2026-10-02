"""ADR 推理流水线 (M1 Day 9)。

端到端:
    text + ref.wav → G2P → 编码 → backbone.sample → mel → vocoder → wav

设计原则:
- 一行调用:  `InferPipeline.synthesize(text="你好", ref_audio="ref.wav")` →  wav
- 自动设备管理 (cuda / cpu)
- 自动 vocoder 加载 (默认 BigVGAN 22kHz,带 placeholder fallback)
- 5 秒内完成 (CPU) / < 1 秒 (GPU)
"""
