"""批次74 审计回归: nltk_data 随包 (英文合成离线可用) + 缺包降级提示。

背景 (2026-11-01 server.log): g2p-en 首次处理英文文本时 cmudict.dict() 与
nltk.pos_tag 会触发在线下载, 无外网/被墙机器下载失败后英文 G2P 直接抛
LookupError, 合成 400 "tts failed"。
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILD_RUNTIME = REPO / "scripts" / "build_runtime.ps1"
CHECK_RUNTIME = REPO / "scripts" / "check_runtime.py"
V2_COMPAT = REPO / "adr" / "server" / "v2_compat.py"
V3_COMPAT = REPO / "adr" / "server" / "v3_compat.py"


# ===== 打包: nltk_data 随包 =====
def test_build_runtime_bundles_nltk_data():
    src = BUILD_RUNTIME.read_text(encoding="utf-8")
    assert "nltk==3.10.3" in src           # 钉版 (构建可复现, 标注器资源名随之固定)
    assert "nltk.downloader" in src
    # 落点必须是 sys.prefix/nltk_data — nltk.data.path 默认包含它, 运行期零代码命中
    assert 'Join-Path $OutDir "python\\nltk_data"' in src
    for res in ("cmudict", "averaged_perceptron_tagger_eng",
                "averaged_perceptron_tagger"):
        assert res in src
    # 下载失败即构建失败 (不能静默交付缺英文能力的包)
    idx = src.index("nltk.downloader")
    assert "throw" in src[idx:idx + 600]
    assert "Test-Path" in src[idx:idx + 600]


# ===== 体检: 缺失即 FAIL =====
def test_check_runtime_checks_nltk_data():
    src = CHECK_RUNTIME.read_text(encoding="utf-8")
    assert "nltk.data.find('corpora/cmudict')" in src
    assert "averaged_perceptron_tagger_eng" in src
    assert "英文合成将失败" in src


# ===== API: 缺包映射 503/nltk_missing =====
def test_v2_compat_reports_nltk_missing():
    src = V2_COMPAT.read_text(encoding="utf-8")
    assert "is_nltk_missing" in src and "NLTK_MISSING_MESSAGE" in src
    assert '"code": "nltk_missing"' in src
    exc = src.index('"code": "nltk_missing"')
    assert "status_code=503" in src[exc - 300:exc]


def test_v3_compat_reports_nltk_missing():
    src = V3_COMPAT.read_text(encoding="utf-8")
    assert "is_nltk_missing" in src and "NLTK_MISSING_MESSAGE" in src
    # WS 错误帧: (message, code) 双字段都要给, 客户端才能区分
    assert '"nltk_missing"' in src


# ===== 判定助手: 只认 LookupError + 资源名 =====
def test_is_nltk_missing_recognizes_lookup_error():
    from adr.models.gsv_engine import is_nltk_missing

    assert is_nltk_missing(LookupError("Resource 'corpora/cmudict' not found."))
    assert is_nltk_missing(
        LookupError("Resource 'taggers/averaged_perceptron_tagger_eng' not found."))
    # 其他 LookupError (非本依赖) 不误判, 其他异常类型也不吞
    assert not is_nltk_missing(LookupError("Resource 'punkt' not found."))
    assert not is_nltk_missing(ValueError("cmudict broken"))


def test_nltk_missing_reason_never_raises():
    from adr.models import gsv_engine

    assert gsv_engine._NLTK_RESOURCES == (
        "corpora/cmudict", "taggers/averaged_perceptron_tagger_eng")
    reason = gsv_engine.nltk_missing_reason()
    assert isinstance(reason, str)
