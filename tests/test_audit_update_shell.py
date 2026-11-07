"""批次73 审计回归: 客户端更新链 (壳 update.rs / 启动器 / 预热页) + 包型互提示
+ 预热失败原因透传。批次76 追加: 更新链信任边界 (#3)。"""
import hashlib
import importlib.util
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UPDATE_RS = REPO / "desktop" / "src-tauri" / "src" / "update.rs"
LIB_RS = REPO / "desktop" / "src-tauri" / "src" / "lib.rs"
LAUNCHER = REPO / "desktop" / "src" / "launcher.html"
PREWARM = REPO / "desktop" / "src" / "prewarm.html"
BUILD_RUNTIME = REPO / "scripts" / "build_runtime.ps1"
TAURI_CONF = REPO / "desktop" / "src-tauri" / "tauri.conf.json"
SIGNER = REPO / "scripts" / "sign_update_manifest.py"


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


# ===== 批次76 (#3): 更新链信任边界 =====
def test_manifest_signature_is_mandatory():
    """清单是唯一信任根: 未签名/验签失败一律拒绝, 不降级、不跳过。"""
    src = UPDATE_RS.read_text(encoding="utf-8")
    assert "UPDATE_PUBKEY_HEX" in src and "ED25519" in src
    assert "更新清单未签名" in src
    assert "更新清单签名校验失败" in src
    # check / download / install 三个入口都必须经过验签函数
    assert src.count("fetch_verified_manifest(") >= 4   # 定义 1 处 + 3 个调用
    # 未验签的缓存不得用于给出任何下载地址
    assert "filter(|m| verify_manifest(m).is_ok())" in src


def test_package_choice_and_install_path_are_server_side():
    """前端不再回传 pkg / path: 包型、地址、哈希由壳从已验签清单取。"""
    src = UPDATE_RS.read_text(encoding="utf-8")
    assert re.search(
        r"pub async fn download_update\(\s*app: tauri::AppHandle,\s*version: Option<String>,?\s*\)",
        src,
    ), "download_update 不得再收前端传来的整个包对象"
    assert "pub fn install_update(app: tauri::AppHandle) -> Result<(), String>" in src
    assert "pkg: UpdatePackage" not in src
    assert "path: String" not in src
    # 安装前: 路径必须仍在 updates/ 内 + 重算 SHA256 比对
    assert "starts_with(&canon_dir)" in src
    assert "安装包校验不符" in src


def test_launcher_sends_only_version():
    src = LAUNCHER.read_text(encoding="utf-8")
    assert '"download_update", { version: updData.latest }' in src
    assert re.search(r'invoke\("install_update"\)', src)
    assert "updPkg" not in src
    assert "pkg: updPkg" not in src


def test_shell_sets_content_security_policy():
    """csp: null + withGlobalTauri 等于给注入脚本开绿灯; 至少要挡住外域。"""
    conf = json.loads(TAURI_CONF.read_text(encoding="utf-8"))
    csp = conf["app"]["security"]["csp"]
    assert isinstance(csp, str) and csp.strip()
    assert "default-src 'self'" in csp
    assert "connect-src" in csp and "ipc:" in csp
    # 本地 API / 音频预览必须放行, 否则页面哑火
    assert "http://127.0.0.1:*" in csp


def test_signer_payload_matches_rust_format():
    """签名脚本与 Rust signing_payload 必须逐字节一致, 否则全部清单验签失败。"""
    pytest = __import__("pytest")
    pytest.importorskip("cryptography")
    spec = importlib.util.spec_from_file_location("sign_update_manifest", SIGNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    manifest = {
        "version": " 1.4.0 ",
        "notes": "fix",
        "packages": [
            {
                "variant": "cpu",
                "name": "a.exe",
                "url": "https://x/a.exe",
                "sha256": "AB12",
                "size": 4096,
            }
        ],
    }
    notes_hash = hashlib.sha256(b"fix").hexdigest()
    assert mod.signing_payload(manifest).decode() == (
        f"adr-update-v1\n1.4.0\n{notes_hash}\ncpu|a.exe|https://x/a.exe||ab12|4096\n"
    )

    # 脚本读到的内置公钥必须与壳里的一致 (签错钥 = 客户端全部拒收)
    src = UPDATE_RS.read_text(encoding="utf-8")
    pk = re.search(r'UPDATE_PUBKEY_HEX:\s*&str\s*=\s*"([0-9a-f]{64})"', src).group(1)
    assert mod.embedded_pubkey() == pk
