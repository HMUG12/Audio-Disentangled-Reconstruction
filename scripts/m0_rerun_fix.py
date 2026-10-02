"""M0 修复重跑脚本：使用 prompt 音频 + fp32 重推理，验证 EOS 正常触发。

仅在 M1 上云（或本地空闲时）执行。预估算力：5–10 分钟（vs M0 异常版的 44 分钟）。
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FISH = ROOT / "third_party" / "fish-speech"
PYTHON = FISH / ".venv" / "Scripts" / "python.exe"
CHECKPOINT = FISH / "checkpoints" / "openaudio-s1-mini"
PROMPT_WAV = ROOT / "output" / "silent_prompt.wav"
OUTPUT_DIR = ROOT / "output" / "m0_fix"
TEXT_FILE = ROOT / "scripts" / "m0_smoke_text.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="显式限制 token 数，避免再跑满 max_seq_len (默认 2048)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    text = TEXT_FILE.read_text(encoding="utf-8").strip()

    if not PROMPT_WAV.exists():
        print("Prompt wav missing, generating...")
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "m0_make_prompt.py")])

    cmd = [
        str(PYTHON), "-m", "fish_speech.models.text2semantic.inference",
        "--text", text,
        "--prompt-audio", str(PROMPT_WAV),
        "--prompt-text", "你好。",
        "--checkpoint-path", str(CHECKPOINT),
        "--output", str(OUTPUT_DIR / "out.wav"),
        "--max-new-tokens", str(args.max_new_tokens),
        "--output-dir", str(OUTPUT_DIR),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(FISH))


if __name__ == "__main__":
    main()
