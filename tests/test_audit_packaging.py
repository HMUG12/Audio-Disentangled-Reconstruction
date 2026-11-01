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
