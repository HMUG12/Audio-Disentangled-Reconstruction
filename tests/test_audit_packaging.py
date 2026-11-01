"""批次70: 打包 / 容器依赖完整性回归 — 审计 §5 测试缺口。

- 容器依赖清单: pandas (AR 训练数据路径隐式依赖) 与 chardet 必须钉版
- pandas × numpy: 容器钉的组合在本机可构造 DataFrame (ABI 兼容冒烟)
- build_installer.ps1 -DryRun: 门禁链路端到端冒烟 (Windows only)
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CONTAINER_REQS = REPO / "docker" / "requirements-linux.txt"
INSTALLER = REPO / "scripts" / "build_installer.ps1"


def _pins() -> dict[str, str]:
    """requirements 文件 → {包名: 版本} (注释与空行跳过; 未钉版值为空串)。"""
    out: dict[str, str] = {}
    for raw in CONTAINER_REQS.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "@" in line:
            continue
        name, _, ver = line.partition("==")
        out[name.strip().lower()] = ver.strip()
    return out


def test_container_pins_pandas():
    """Q4: 容器含 pytorch-lightning/tensorboard 即训练可用形态, pandas 必须钉版。"""
    pins = _pins()
    assert pins.get("pandas"), "docker/requirements-linux.txt 缺 pandas 钉版"


def test_container_pins_chardet():
    """批次69 遗留: chardet 曾为裸名 (每次构建可能换版本), 现要求钉版。"""
    assert _pins().get("chardet"), "docker/requirements-linux.txt 缺 chardet 钉版"


def test_pandas_numpy_interop():
    """容器钉的 pandas × numpy 组合可实际构造对象 (ABI 不匹配会在此炸)。"""
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    df = pd.DataFrame(np.zeros((2, 3), dtype=np.float32), columns=list("abc"))
    assert df.to_numpy().shape == (2, 3)

    pins = _pins()
    # 本机版本与容器钉版一致时才有"实测组合"意义; 不一致仅提示不失败
    assert pins["pandas"].count(".") == 2


@pytest.mark.skipif(sys.platform != "win32", reason="build_installer.ps1 为 Windows 脚本")
def test_build_installer_dry_run():
    """-DryRun: 版本/合规门禁 + 前置预检全跑, 不做编译, 结尾 DRY RUN OK。"""
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(INSTALLER), "-DryRun"],
        cwd=str(REPO), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=600,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, out[-2000:]
    assert "DRY RUN OK" in out
    # 干跑不得进入编译阶段
    assert "Compiling" not in out


# ===== 批次71: 打包链修复移植 + 依赖自包含 (复盘 v1.0.1 §一/§二/§六) =====

CHECK_RUNTIME = REPO / "scripts" / "check_runtime.py"

# 训练子进程实际调用的脚本 (console.py:386/439 + webui/__init__.py)
TRAIN_SCRIPTS = ("gsv_finetune.py", "train_gate.py",
                 "smoke_train_opencpop.py", "rvc_train.py")


def _make_runtime_skeleton(rt: Path) -> None:
    """最小可通过骨架: configs preset (第 6 项)。无 python.exe 时第 7 项跳过。"""
    (rt / "configs").mkdir(parents=True)
    (rt / "configs" / "default.yaml").write_text("train: {}\n", encoding="utf-8")


def _run_check_runtime(rt: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CHECK_RUNTIME), "--runtime", str(rt)],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=120,
    )


def test_check_runtime_flags_missing_train_scripts(tmp_path):
    """§一防复发: runtime 缺 scripts/ → 体检必须 FAIL (v1.0.1/v1.3.1 双包漏拷)。"""
    rt = tmp_path / "rt"
    _make_runtime_skeleton(rt)
    proc = _run_check_runtime(rt)
    assert proc.returncode == 1
    out = proc.stdout + proc.stderr
    assert "训练脚本缺失" in out
    for s in TRAIN_SCRIPTS:
        assert s in out


def test_check_runtime_flags_missing_bundled_ffmpeg_redist(tmp_path):
    """依赖自包含: 缺 bin/ffmpeg.exe、ffprobe.exe、redist/vc_redist.x64.exe → FAIL。"""
    rt = tmp_path / "rt"
    _make_runtime_skeleton(rt)
    (rt / "scripts").mkdir()
    for s in TRAIN_SCRIPTS:
        (rt / "scripts" / s).write_text("# stub\n", encoding="utf-8")
    proc = _run_check_runtime(rt)
    assert proc.returncode == 1
    out = proc.stdout + proc.stderr
    assert "bin/ffmpeg.exe 缺失" in out
    assert "bin/ffprobe.exe 缺失" in out
    assert "redist/vc_redist.x64.exe 缺失" in out


def test_check_runtime_passes_with_bundled_deps(tmp_path):
    """训练脚本 + ffmpeg + redist 齐备 → 体检通过 (正向路径)。"""
    rt = tmp_path / "rt"
    _make_runtime_skeleton(rt)
    (rt / "scripts").mkdir()
    for s in TRAIN_SCRIPTS:
        (rt / "scripts" / s).write_text("# stub\n", encoding="utf-8")
    for exe in ("ffmpeg.exe", "ffprobe.exe"):
        (rt / "bin").mkdir(exist_ok=True)
        (rt / "bin" / exe).write_bytes(b"MZ stub")
    (rt / "redist").mkdir()
    (rt / "redist" / "vc_redist.x64.exe").write_bytes(b"MZ stub")
    proc = _run_check_runtime(rt)
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-1500:]
    assert "体检通过" in proc.stdout


def test_compat_patches_idempotency_uses_new_text():
    """§二缺陷1防复发: main() 的幂等判据必须是 `new in src` 而非 `marker in src`
    (marker 打补丁后仍在文件中 → 同文件第 2 条补丁被静默跳过, v1.0.1 实测
    漏打 bucket_sampler.py rank 守卫致 s1 启动即崩)。"""
    src = CHECK_RUNTIME.with_name("apply_compat_patches.py").read_text(encoding="utf-8")
    assert "if new in src:" in src
    assert "if marker in src:" not in src


def test_runtime_bins_path_injection(tmp_path, monkeypatch):
    """settings.ensure_runtime_bins_on_path: REPO_ROOT/bin 存在时前置 PATH, 幂等。"""
    import os

    from adr.core import settings
    (tmp_path / "bin").mkdir()
    monkeypatch.setattr(settings, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("PATH", r"C:\Windows\System32")
    settings.ensure_runtime_bins_on_path()
    expected = str(tmp_path / "bin")
    assert os.environ["PATH"].startswith(expected + os.pathsep)
    # 幂等: 重复注入不叠加
    settings.ensure_runtime_bins_on_path()
    assert os.environ["PATH"].count(expected + os.pathsep) == 1


# ===== 批次72: 卸载体验 (运行互斥体 + 清理清单 + 档案保留询问) =====

INSTALLER_ISS = REPO / "scripts" / "installer" / "installer.iss"
SHELL_MAIN = REPO / "adr" / "server" / "__main__.py"
SHELL_RS = REPO / "desktop" / "src-tauri" / "src" / "lib.rs"

APP_MUTEX = "ADR-Studio-Running"


def test_app_mutex_name_consistent_across_three_sides():
    """防漂移: 壳 (lib.rs) / 服务 (__main__.py) / 安装器 ([Setup] AppMutex)
    三端名字必须逐字一致 — 跨语言无法编译期共享, 只能靠本断言锁死。
    名字不一致 = 卸载器检测不到运行中的程序, "先关程序再卸载"直接失效。"""
    assert APP_MUTEX in SHELL_RS.read_text(encoding="utf-8")
    assert APP_MUTEX in SHELL_MAIN.read_text(encoding="utf-8")
    iss = INSTALLER_ISS.read_text(encoding="utf-8")
    assert f"AppMutex={APP_MUTEX}" in iss


def test_installer_uninstall_delete_covers_regenerable_dirs():
    """卸载清理: 运行期可再生成产物必须进 [UninstallDelete] (否则残留)。"""
    iss = INSTALLER_ISS.read_text(encoding="utf-8")
    head, _, tail = iss.partition("[Run]")
    section = head.partition("[UninstallDelete]")[2]
    assert section, "installer.iss 缺 [UninstallDelete] 段"
    for rel in (r"runtime\data\tts_cache",
                r"runtime\third_party\gpt_sovits\output",
                r"runtime\third_party\gpt_sovits\logs",
                r"runtime\third_party\gpt_sovits\TEMP"):
        assert rel in section, f"[UninstallDelete] 缺 {rel}"


def test_installer_asks_before_deleting_user_assets():
    """卸载询问页: 用户资产 (声音档案/训练权重) 默认保留, 静默卸载不弹窗。"""
    iss = INSTALLER_ISS.read_text(encoding="utf-8")
    code = iss.partition("[Code]")[2]
    assert "CurUninstallStepChanged" in code
    assert "usUninstall" in code
    # 静默卸载必须跳过询问 (无人可答, 默认保留)
    assert "UninstallSilent" in code
    assert "MB_DEFBUTTON2" in code, "默认按钮必须是「否」(保留)"
    assert r"runtime\data\voices" in code
    assert r"SoVITS_weights_v2" in code
    assert r"GPT_weights_v2" in code
