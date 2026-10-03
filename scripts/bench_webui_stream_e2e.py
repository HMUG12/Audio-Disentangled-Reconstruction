"""批次3-4: WebUI 流式端到端验收 — 复刻 WebUI prewarm → 流式首包的真实时序。

不含 gradio 传输开销 (浏览器侧另测), 度量:
  A. 预热耗时 (引擎加载 + 档案权重热换)
  B. 预热完成后第一次点"流式合成"的首包 (用户真实首次体验)
  C. 热机第二次的首包

用法: python scripts/bench_webui_stream_e2e.py
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TEXT = ("这是一段用于验收流式首包的文本，它包含多个分句和常见标点，"
        "看看预热完成后的首包到底是多少秒。")


def stream_first(text, prof, tag):
    from adr.models.gsv_engine import get_gsv_engine
    t0 = time.perf_counter()
    first = None
    for chunk, sr in get_gsv_engine().synthesize_stream(
            text, prof["ref_audio"], vits_weights=prof.get("vits_weights"),
            split_method="cut3"):
        if first is None:
            first = time.perf_counter() - t0
    print(f"  [{tag}] 首包 {first:.2f}s", flush=True)
    return first


def main():
    from adr.models.voice_library import list_voices, load_voice

    t0 = time.perf_counter()
    from adr.models.gsv_engine import get_gsv_engine  # 与 webui prewarm 同构
    voices = list_voices()
    assert voices, "无音色档案"
    prof = load_voice(voices[0])
    eng = get_gsv_engine()
    eng.warmup(vits_weights=prof.get("vits_weights"),
               t2s_weights=prof.get("t2s_weights"))
    for _ in eng.synthesize_stream("你好。", prof["ref_audio"],
                                   vits_weights=prof.get("vits_weights"),
                                   split_method="cut3"):
        break  # 流式路径形状预热 (与 webui prewarm 一致)
    t_warm = time.perf_counter() - t0
    print(f"[e2e] A. 预热 (引擎+权重热换+流式形状): {t_warm:.1f}s")

    first = stream_first(TEXT, prof, "B. 首次点按钮")
    second = stream_first(TEXT, prof, "C. 热机二次")
    print(f"[e2e] 验收: 首次 {first:.2f}s | 热机 {second:.2f}s "
          f"(目标 ≤3s / 容忍 ≤5s)")


if __name__ == "__main__":
    main()
