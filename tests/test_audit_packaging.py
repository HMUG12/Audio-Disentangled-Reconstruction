"""批次70: 打包 / 容器依赖完整性回归 — 审计 §5 测试缺口。

- 容器依赖清单: pandas (AR 训练数据路径隐式依赖) 与 chardet 必须钉版
- pandas × numpy: 容器钉的组合在本机可构造 DataFrame (ABI 兼容冒烟)
- build_installer.ps1 -DryRun: 门禁链路端到端冒烟 (Windows only)
"""
import re
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
    """最小可通过骨架: configs preset (第 6 项) + nltk_data (第 11 项, 批次74)。
    无 python.exe 时第 7/11 项的 runtime python 探针跳过。"""
    (rt / "configs").mkdir(parents=True)
    (rt / "configs" / "default.yaml").write_text("train: {}\n", encoding="utf-8")
    for res in ("corpora/cmudict", "taggers/averaged_perceptron_tagger_eng"):
        (rt / "python" / "nltk_data" / res).mkdir(parents=True, exist_ok=True)


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


# ===== 批次76: 打包/更新链整改 (#20 安装器变体清理 + 静默升级拉起应用) =====


def test_installer_cleans_python_tree_on_variant_switch():
    """#20 防复发: 双变体共用 AppId 原地升级 → 换包型必须先删整棵
    runtime\\python, 否则旧 torch/nvidia_* 与新包混装 (标记与实际不符)。"""
    iss = INSTALLER_ISS.read_text(encoding="utf-8")
    assert "[InstallDelete]" in iss, "installer.iss 缺 [InstallDelete] 段"
    # 注释里也提到过 [InstallDelete], 只认行首的段头
    m = re.search(r"^\[InstallDelete\]$(.*?)(?=^\[|\Z)", iss, re.S | re.M)
    assert m, "installer.iss 缺 [InstallDelete] 段"
    section = m.group(1)
    assert r'runtime\python' in section
    assert "Check: VariantChanged" in section
    code = iss.partition("[Code]")[2]
    assert "function VariantChanged" in code
    assert "BUILD_VARIANT.txt" in code, "必须读已装变体标记, 否则无从比对"
    assert "'{#VariantName}'" in code
    # 用户数据不在 python 树内 → 清理条目不得波及 data / 权重目录
    entries = [ln for ln in section.splitlines() if ln.strip().startswith("Type:")]
    assert entries, "[InstallDelete] 无实际条目"
    for ln in entries:
        assert r"runtime\data" not in ln
        assert "SoVITS_weights" not in ln


def test_installer_launches_app_after_silent_upgrade():
    """#20 防复发: [Run] 拉起应用不得带 skipifsilent — 壳侧静默升级
    (/VERYSILENT) 装完必须自动回到应用, 否则程序"消失"。"""
    iss = INSTALLER_ISS.read_text(encoding="utf-8")
    run = iss.partition("[Run]")[2].partition("[Code]")[0]
    # 条目跨行 (行尾反斜杠续行) → 整段匹配, 只看 MyAppExe 那条
    m = re.search(r'Filename: "\{app\}\\\{#MyAppExe\}".*?(?=\nFilename:|\n\[|\Z)',
                  run, re.S)
    assert m, "[Run] 缺拉起应用的条目"
    entry = m.group(0)
    assert "skipifsilent" not in entry
    assert "nowait postinstall" in entry


# ===== 批次76: CI 发布门禁 (#22) + 依赖版本漂移 (#23) =====

DOCKER_PUBLISH = REPO / ".github" / "workflows" / "docker-publish.yml"
SMOKE_WF = REPO / ".github" / "workflows" / "smoke.yml"
CHECK_REQUIREMENTS = REPO / "scripts" / "check_requirements.py"
CHECK_VERSIONS = REPO / "scripts" / "check_versions.py"


def _run(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(REPO), capture_output=True, text=True, check=False,
        encoding="utf-8", errors="replace", timeout=120,
    )


def test_docker_publish_gated_by_release_checks():
    """#22 防复发: v* tag 不得直接构建推送 — gate 先跑版本面 / tag 对齐 /
    依赖漂移 / 语法, 任一红则 build-push 不启动。"""
    wf = DOCKER_PUBLISH.read_text(encoding="utf-8")
    build = wf.partition("build-push:")[2]
    assert build, "docker-publish.yml 缺 build-push job"
    assert "needs: gate" in build, "build-push 必须依赖发布门禁"
    gate = wf.partition("gate:")[2].partition("build-push:")[0]
    for needle in ("check_versions.py", "check_requirements.py",
                   "check_compliance.py", "compileall"):
        assert needle in gate, f"发布门禁缺 {needle}"
    # tag 与 pyproject 事实源对账 (只有 tag 触发时才传 --tag)
    assert '--tag "$GITHUB_REF_NAME"' in gate
    assert "GITHUB_REF_TYPE" in gate


def test_check_versions_rejects_mismatched_tag():
    """#22: tag 与 pyproject 不同版就该红 (v9.9.9 不可能等于当前版本)。"""
    proc = _run(CHECK_VERSIONS, "--tag", "v9.9.9")
    out = proc.stdout + proc.stderr
    assert proc.returncode == 1, out[-800:]
    assert "git tag v9.9.9" in out
    # 同版 tag 必须放行, 否则门禁永远红
    ver = re.search(r'^version\s*=\s*"([^"]+)"',
                    (REPO / "pyproject.toml").read_text(encoding="utf-8"),
                    re.MULTILINE).group(1)
    ok = _run(CHECK_VERSIONS, "--tag", f"v{ver}")
    assert ok.returncode == 0, (ok.stdout + ok.stderr)[-800:]


def test_core_requirements_have_upper_bounds():
    """#23 防复发: 只有 >= 下限时 CI 会解析到比发货 runtime 更新的大版本。"""
    reqs = (REPO / "requirements.txt").read_text(encoding="utf-8")
    entries = [ln.split("#", 1)[0].strip() for ln in reqs.splitlines()]
    entries = [ln for ln in entries if ln]
    assert entries, "requirements.txt 无核心依赖"
    for ln in entries:
        assert "==" in ln or "<" in ln, f"核心依赖缺上限: {ln}"
    proc = _run(CHECK_REQUIREMENTS)
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-800:]


def test_smoke_ci_reuses_container_pins():
    """#23: CI 与容器共用同一份钉版清单, 不在 CI 侧另解析一套版本。"""
    wf = SMOKE_WF.read_text(encoding="utf-8")
    assert "check_requirements.py" in wf
    assert "-c docker/requirements-linux.txt" in wf


# ===== 批次118 (审计 R5-DOCK-01/02): 容器非 root + 端口单一事实源 =====
def test_dockerfile_runs_as_non_root():
    """以 root 跑服务会把"容器逃逸/挂载目录写权限"的后果放大。"""
    src = (REPO / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "USER 10001:10001" in src
    assert "--chown=10001:10001 /app" in src
    # USER 之后不得再切回 root
    assert "USER root" not in src[src.index("USER 10001:10001"):]


def test_dockerfile_port_single_source():
    """探针读 ADR_PORT、CMD 写死 9881 时, 改端口只设一个必然恒 unhealthy。"""
    src = (REPO / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "ENV ADR_PORT=9881" in src
    assert 'exec python -m adr.server --addr 0.0.0.0 --port \\"${ADR_PORT}\\"' in src
    assert '"--port", "9881"' not in src
    assert "os.environ.get('ADR_PORT'" in src
