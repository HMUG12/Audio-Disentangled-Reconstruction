"""批次75: 运行时依赖随包 + 打包卫生审计。

背景 (2026-11-01 用户报障): 安装版训练失败, 报"缺少 modelscope / ffmpeg
等必需依赖组件"; 同时安装包里塞满 __pycache__/*.pyc 临时文件。

根因:
- modelscope — GSV 的 tools/asr/fasterwhisper_asr.py 与 funasr_asr.py 顶层
  ``from modelscope import snapshot_download``, 而 build_runtime 的钉版清单
  里没有它 → 训练管线 2/7 (ASR) 直接 ModuleNotFoundError。
- ffmpeg — 已由批次71 随包 (bin/ffmpeg.exe + ffprobe.exe), check_runtime 第 9 项
  门禁; 本文件锁住"切片裸调 ffmpeg 只认 PATH"这条链路仍然成立。
- .pyc — pip install 为每个模块生成字节码缓存 (runtime-cpu 实测 17768 个 .pyc
  / 2297 个 __pycache__ 目录), 随包进安装目录 → 用户看到一堆临时文件。
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUILD_RUNTIME = REPO / "scripts" / "build_runtime.ps1"
CHECK_RUNTIME = REPO / "scripts" / "check_runtime.py"
GSV = REPO / "third_party" / "gpt_sovits"


def test_runtime_pins_modelscope():
    """modelscope 必须钉版随包: GSV ASR 脚本顶层硬 import, 缺了训练必挂。"""
    src = BUILD_RUNTIME.read_text(encoding="utf-8")
    assert "modelscope==1.35.3" in src
    # write-back 校验也要带上 (stale requirements 文件的老坑)
    assert '"modelscope=="' in src


def test_gsv_asr_scripts_hard_import_modelscope():
    """事实锁定: 两个 ASR 脚本顶层 import modelscope — 证明它不是可选依赖。"""
    for rel in ("tools/asr/fasterwhisper_asr.py", "tools/asr/funasr_asr.py"):
        src = (GSV / rel).read_text(encoding="utf-8")
        assert "from modelscope import" in src, f"{rel} 不再依赖 modelscope — 可撤掉钉版"


def test_training_pipeline_runs_the_asr_script():
    """事实锁定: 训练管线 2/7 调 fasterwhisper_asr.py → modelscope 是训练硬依赖。"""
    src = (REPO / "scripts" / "gsv_finetune.py").read_text(encoding="utf-8")
    assert "fasterwhisper_asr.py" in src


def test_build_runtime_strips_bytecode_caches():
    """构建末段必须清 __pycache__/*.pyc — 且排在 smoke test 之后 (否则又被生成)。"""
    src = BUILD_RUNTIME.read_text(encoding="utf-8")
    assert "strip_bytecode.py" in src
    assert "__pycache__" in src
    assert "Step \"8 strip bytecode caches" in src
    assert src.index("Step \"8 strip bytecode caches") > src.index("Step \"7/7 smoke test\"")


def test_check_runtime_flags_missing_modelscope():
    """门禁: runtime python 探针 import modelscope — 缺失即 FAIL。"""
    src = CHECK_RUNTIME.read_text(encoding="utf-8")
    assert "import modelscope" in src
    assert "modelscope 缺失" in src
    assert "ModuleNotFoundError" in src


def test_check_runtime_flags_residual_bytecode():
    """门禁: 自有代码树残留 __pycache__ 即 FAIL (用户可见的临时文件)。"""
    src = CHECK_RUNTIME.read_text(encoding="utf-8")
    assert "残留 __pycache__" in src
    assert "残留 .pyc" in src


def test_check_runtime_probes_do_not_write_bytecode():
    """探针必须带 -B: 否则体检自己就把 .pyc 写回 runtime, 卫生门禁自打脸。"""
    src = CHECK_RUNTIME.read_text(encoding="utf-8")
    assert src.count('"-B", "-c"') >= 3


# ===== 批次117 (审计 R5-SCR-01): 训练脚本不得再拼 shell 命令串 =====
def test_train_scripts_pass_argv_list_without_shell():
    """此前音频路径 / exp 用双引号拼进 shell=True 命令串, POSIX 下文件名含
    `$()` / 反引号即可注入 (R4-P3-08 的完整形态)。现一律 argv 列表。"""
    for name in ("gsv_finetune.py", "rvc_train.py"):
        src = (REPO / "scripts" / name).read_text(encoding="utf-8")
        assert "shell=True" not in src, name
        assert "def run(argv: list" in src, name
        assert "subprocess.run(argv" in src, name
        assert "subprocess.run(cmd" not in src, name
