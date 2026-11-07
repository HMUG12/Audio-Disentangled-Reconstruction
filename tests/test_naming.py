"""测试 adr.core.naming (批次58 Windows 落盘安全 + 批次65 shell 注入收口)。"""

from adr.core.naming import is_windows_reserved, sanitize_exp_name


def test_strips_windows_invalid_chars():
    """非法文件名字符与尾部点/空格清除, 保留名加下划线。"""
    assert sanitize_exp_name('a/b\\c:d*e?f"g<h>i|j') == "abcdefghij"
    assert sanitize_exp_name("name.  ") == "name"
    assert sanitize_exp_name("con").startswith("_")
    assert sanitize_exp_name("CON.txt").startswith("_")


def test_empty_falls_back():
    assert sanitize_exp_name("") == "my_voice"
    assert sanitize_exp_name("   ", fallback="fb") == "fb"
    # 全是被剔除的字符 → 也回退, 不会产出空目录名
    assert sanitize_exp_name("$(){}[]`;&") == "my_voice"


def test_strips_shell_metacharacters():
    """批次65: exp 名流入 gsv_finetune/rvc_train 的 shell=True 命令串,
    Linux /bin/sh 双引号内仍做命令替换/展开。清洗必须剔除所有命令执行/
    引号/通配元字符, 使 sanitize 后的结果不含任何可注入字符。"""
    payloads = [
        "$(touch pwned)",
        "`id`",
        "a;rm -rf /",
        "a&&whoami",
        "x|nc evil 1",
        "v'inject'",
        "~root",
        "{a,b}",
        "g[0-9]",
        "!!",
    ]
    dangerous = set("$`;&()|'<>\"\\/*?{}[]!~")
    for p in payloads:
        out = sanitize_exp_name(p)
        assert out, p
        assert not (set(out) & dangerous), f"{p!r} -> {out!r} 仍含危险字符"


def test_keeps_legitimate_names():
    """合法中英/数字/下划线/连字符/空格 不被破坏。"""
    assert sanitize_exp_name("测试音_v2") == "测试音_v2"
    assert sanitize_exp_name("my-voice 01") == "my-voice 01"
    assert sanitize_exp_name("speaker.alpha") == "speaker.alpha"


def test_is_windows_reserved():
    for r in ("con", "CON", "nul", "com1", "lpt9", "CON.txt", "aux.log"):
        assert is_windows_reserved(r), r
    for ok in ("console", "com", "com10", "my_con"):
        assert not is_windows_reserved(ok), ok


# ---- 批次89 (#23): 路径安全判定单点化 ----

def test_pathsafe_and_voice_library_share_one_implementation():
    """server.pathsafe / models.voice_library 只是再导出, 规则不会漂移。"""
    from adr.core import naming
    from adr.models import voice_library
    from adr.server import pathsafe

    assert pathsafe.is_safe_name is naming.is_safe_name
    assert voice_library.is_safe_name is naming.is_safe_name
    assert pathsafe.resolve_within is naming.resolve_within


def test_is_safe_name_rules():
    from adr.core.naming import is_safe_name

    for ok in ("我的音色", "spk_01", "voice-2", "a.b"):
        assert is_safe_name(ok), ok
    for bad in ("", "../x", "a/b", "a\\b", "D:evil", "x\x00y", "CON", "nul.wav"):
        assert not is_safe_name(bad), bad


def test_resolve_within_containment(tmp_path):
    from adr.core.naming import resolve_within

    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    assert resolve_within(root, "sub/a.wav") == (root / "sub" / "a.wav").resolve()
    assert resolve_within(root, root / "sub" / "b.wav") == (
        root / "sub" / "b.wav").resolve()
    assert resolve_within(root, "../escape.wav") is None
    assert resolve_within(root, tmp_path / "outside.wav") is None
