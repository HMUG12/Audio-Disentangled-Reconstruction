"""批次73 审计回归: 客户端更新链 (壳 update.rs / 启动器 / 预热页) + 包型互提示
+ 预热失败原因透传。"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UPDATE_RS = REPO / "desktop" / "src-tauri" / "src" / "update.rs"
LIB_RS = REPO / "desktop" / "src-tauri" / "src" / "lib.rs"
LAUNCHER = REPO / "desktop" / "src" / "launcher.html"
PREWARM = REPO / "desktop" / "src" / "prewarm.html"
BUILD_RUNTIME = REPO / "scripts" / "build_runtime.ps1"


# ===== 壳: 命令注册 =====
def test_shell_registers_update_commands():
    src = LIB_RS.read_text(encoding="utf-8")
    assert "mod update;" in src
    for cmd in ("update::check_update", "update::download_update",
                "update::install_update", "update::package_hint"):
        assert cmd in src


# ===== 壳: SHA256 是硬门槛, 主备双源 =====
def test_update_sha256_is_hard_gate():
    src = UPDATE_RS.read_text(encoding="utf-8")
    assert "缺少 sha256" in src       # 清单无校验值 → 拒绝下载
    assert "fallback_url" in src      # 主链失败切备用源
    # 下载/安装链一律走本地文件, 不碰业务接口 (防后续误改成边下边写数据目录)
    assert "/api/adr/v1/" not in src


def test_update_installer_waits_for_shell_exit():
    """Inno 安装器带 AppMutex (批次72): 壳未退出时 /VERYSILENT 会被互斥体
    静默中止 — 助手脚本必须先等壳进程消失, 再拉起安装包。"""
    src = UPDATE_RS.read_text(encoding="utf-8")
    fn = src[src.index("fn spawn_installer"):]
    fn = fn[: fn.index("#[cfg(not(windows))]")]
    assert "tasklist" in fn and "VERYSILENT" in fn
    assert fn.index("goto wait") < fn.index("VERYSILENT")


# ===== 启动器: 更新横幅 / 模态框 / 进度回调接线 =====
def test_launcher_update_wiring():
    src = LAUNCHER.read_text(encoding="utf-8")
    for token in ('"check_update"', '"download_update"', '"install_update"',
                  '"package_hint"', "window.__adrUpdate"):
        assert token in src
    # 设置中心「检查更新」不再打开网页, 改走壳内更新链
    assert 'st-update").addEventListener("click", () => runUpdateCheck(true))' in src
    # 下载完成才允许安装 (按钮初始禁用, 由 __adrUpdate/download 结果解锁)
    assert 'id="upd-inst" disabled' in src


# ===== 预热页: 失败原因 + 复制诊断 =====
def test_prewarm_shows_failure_reason():
    src = PREWARM.read_text(encoding="utf-8")
    assert "window.__adrPrewarmError" in src
    assert 'id="btnCopyErr"' in src and 'id="errtx"' in src


# ===== 打包: 包型标记 =====
def test_build_runtime_writes_variant_marker():
    src = BUILD_RUNTIME.read_text(encoding="utf-8")
    assert "BUILD_VARIANT.txt" in src
    assert 'if ($AllowCuda) { "gpu" } else { "cpu" }' in src


# ===== Python 侧: 失败原因摘要 =====
def test_stage_error_only_when_failed(monkeypatch):
    from adr.models import gsv_engine

    monkeypatch.setattr(gsv_engine, "_STAGE", "loading")
    monkeypatch.setattr(gsv_engine, "_STAGE_ERROR", "boom")
    assert gsv_engine.stage_error() == ""
    monkeypatch.setattr(gsv_engine, "_STAGE", "failed")
    assert gsv_engine.stage_error() == "boom"


def test_error_summary_has_type_and_message():
    from adr.models import gsv_engine

    try:
        raise RuntimeError("权重文件缺失")
    except RuntimeError as e:
        txt = gsv_engine._error_summary(e)
    assert "RuntimeError" in txt and "权重文件缺失" in txt


def test_error_summary_truncates():
    from adr.models import gsv_engine

    txt = gsv_engine._error_summary(RuntimeError("x" * 5000))
    assert len(txt) <= 1201 and txt.startswith("…")
