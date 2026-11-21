"""`python -m adr.server` 入口: 启动 ADR TTS 服务。

默认 127.0.0.1:9881 (批次41a 安全收口, 原默认 0.0.0.0) — 与 GPT-SoVITS
api_v2 / api_neko 端口一致, N.E.K.O 本机对接无需改端口配置。

安全提示 (批次41a): 需要局域网访问时显式传 ``--addr 0.0.0.0``, 且**必须**
同时设置 ADR_TTS_API_KEY (未设时所有 API 完全无鉴权, 见 app.py 中间件);
该组合启动时打醒目警告 — 桌面壳 expose=true 也走此入口 (壳传
--addr 0.0.0.0), 同样被覆盖。
"""
from __future__ import annotations

import argparse
import faulthandler
import os
import sys

import uvicorn

from adr.core import settings
from adr.core.logging import setup_logging
from adr.server.app import create_app


def _enforce_terms(args) -> None:
    """无 UI 形态条款 gate (批次62, m10 方案 C)。

    检查顺序:
    1. ``<adr_data_dir()>/.terms_agreed_v1`` 标记已存在 → 放行;
    2. ``--from-shell`` — 桌面壳拉起时声明; 壳的 prewarm 先于用户同意启动,
       由壳层条款视图 (批次61 prewarm.html) 担保 "同意后才可进主界面",
       server 侧不再重复拦截;
    3. ``--agree-terms`` 参数或 ``ADR_AGREE_TERMS`` env — 显式同意,
       写标记 + 审计后放行;
    4. 其余 → 打印条款要点并拒绝启动 (exit 2)。
    """
    from adr.core import terms
    if terms.agreed():
        return
    if getattr(args, "from_shell", False):
        return
    if args.agree_terms or settings.agree_terms():
        if terms.mark():
            from adr.core.audit import log_event
            log_event("terms_agreed",
                      source="cli_arg" if args.agree_terms else "env")
        else:
            # 写失败仍按显式同意放行, 但须留痕 (数据目录不可写等), 否则静默无感知
            print("警告: 同意标记写入失败, 本次按显式同意放行; "
                  "下次启动可能再次要求同意", file=sys.stderr, flush=True)
        return
    print(
        "\n" + "=" * 64,
        "  使用条款未同意, 服务拒绝启动 (exit 2)",
        "",
        "  ADR 为语音克隆工具, 使用前需确认:",
        "  1. 仅克隆你拥有权利或已获书面授权的声音, 不得用于冒充他人;",
        "  2. 上传声纹素材须已获得声音权利人的单独同意;",
        "  3. 合成内容须遵守所在地法律法规, 不得用于欺诈/侵权等非法用途。",
        "",
        "  同意方式 (二选一):",
        "    python -m adr.server --agree-terms [其余参数]",
        "    或设置环境变量 ADR_AGREE_TERMS=1 后启动",
        "  桌面壳用户: 首次启动在条款页点 \"同意并继续\" 即可, 无需本参数。",
        "=" * 64 + "\n",
        sep="\n", file=sys.stderr, flush=True,
    )
    raise SystemExit(2)


def _warn_unauthenticated_listen(addr: str, *, from_shell: bool = False) -> None:
    """监听地址对外 + 无 API Key: 默认拒绝启动 (批次26 告警; 批次87 / SEC-04 收口)。

    create_app 不知道 bind host (uvicorn.run 在本入口传参),
    所以只有这里能同时看到 host 与鉴权配置 — 判定只在这一处, 不在 app.py 重复。

    旧行为只打警告, 默认部署 ``--addr 0.0.0.0`` 就等于把无鉴权的合成/换权/
    管理接口开放给整个局域网。现在:
    - 显式 ``ADR_ALLOW_UNAUTHENTICATED=1`` → 仅告警 (容器/受控内网);
    - 桌面壳 ``--from-shell`` → 仅告警 (局域网开关是用户在设置页的显式操作);
    - 其余 (CLI / 默认容器入口) → 打印原因后以退出码 2 结束。
    """
    if addr not in ("0.0.0.0", "::"):
        return
    if settings.has_api_key():
        return
    allow = os.environ.get("ADR_ALLOW_UNAUTHENTICATED", "").strip().lower()
    if allow in ("1", "true", "yes", "on") or from_shell:
        print(
            "\n" + "=" * 64,
            "⚠  安全警告: 服务监听 0.0.0.0 且未设置 ADR_TTS_API_KEY",
            "   局域网内任何设备均可无鉴权调用合成 API。",
            "   如需暴露到局域网, 请设置环境变量后重启:",
            '     set ADR_TTS_API_KEY=<你的密钥>        (Windows CMD)',
            '     $env:ADR_TTS_API_KEY="<你的密钥>"     (PowerShell)',
            "   仅本机使用可改用: python -m adr.server --addr 127.0.0.1",
            "=" * 64 + "\n",
            sep="\n", file=sys.stderr, flush=True,
        )
        return
    print(
        "\n" + "=" * 64,
        "✗  拒绝启动: 监听 0.0.0.0 且未设置 ADR_TTS_API_KEY",
        "   这会把无鉴权的合成 / 换权 / 管理接口开放给整个局域网。",
        "   请二选一:",
        "     1) 设置 ADR_TTS_API_KEY=<你的密钥> 后重启 (推荐)",
        "     2) 仅本机使用: --addr 127.0.0.1",
        "     3) 明确接受风险 (容器内监听 0.0.0.0 但只发布到回环):",
        "        set ADR_ALLOW_UNAUTHENTICATED=1  (Windows CMD)",
        '        $env:ADR_ALLOW_UNAUTHENTICATED="1"  (PowerShell)',
        "=" * 64 + "\n",
        sep="\n", file=sys.stderr, flush=True,
    )
    raise SystemExit(2)


def _warn_ref_audio_roots() -> None:
    """S3: 设了 API Key 但未设 ADR_TTS_AUDIO_ROOTS 时提示 ref_audio_path 面。

    v2 兼容面按契约接受调用方本机绝对路径 (N.E.K.O), 默认不设目录白名单。
    对局域网/容器部署, 建议显式收口, 否则任何持 Key 的客户端都能让服务读取
    本机任意音频文件并以其音色合成。
    """
    if not settings.has_api_key() or settings.audio_roots():
        return
    print(
        "\n" + "=" * 64,
        "⚠  提示: 已启用 ADR_TTS_API_KEY, 但未设置 ADR_TTS_AUDIO_ROOTS",
        "   v2 的 ref_audio_path 仍可指向本机任意音频文件。",
        "   如需限制可读范围, 设置逗号分隔的根目录后重启:",
        '     set ADR_TTS_AUDIO_ROOTS=D:\\voices,D:\\samples   (Windows CMD)',
        "   仅本机使用可忽略此提示。",
        "=" * 64 + "\n",
        sep="\n", file=sys.stderr, flush=True,
    )


def _print_banner(addr: str, port: int) -> None:
    """启动横幅: 标明这是 ADR TTS 服务而非 GSV 官方 API, 排障时便于区分
    (NEKO 日志只显示 "GPT-SoVITS 连接失败", 需要能确认对端到底是谁)。"""
    print(
        "=" * 64,
        "  ADR TTS 服务 (语音克隆) — 兼容 GPT-SoVITS api_v2/v3 协议",
        f"  监听地址: http://{addr}:{port}",
        f"  NEKO 对接: GPT-SoVITS 模式, API 地址填 http://127.0.0.1:{port} 即可",
        "  音色下拉 = ADR 语音档案 (GET /api/v3/voices 自动返回)",
        "=" * 64,
        sep="\n", flush=True,
    )


def _hold_app_mutex() -> None:
    """批次72 (卸载体验): 创建命名互斥体, 供安装器/卸载器 (Inno AppMutex) 检测
    "服务正在运行" — 覆盖 server.bat / 直接 ``python -m adr.server`` 的无壳场景。

    名字与 desktop/src-tauri/src/lib.rs (壳) / scripts/installer/installer.iss
    三端**逐字一致** (tests/test_audit_packaging.py 有防漂移断言)。

    句柄故意不关: 随进程退出 (含 taskkill /F) 由 OS 回收释放。
    创建失败静默 — 不阻断服务启动, 只是安装器少一道运行中拦截。
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.kernel32.CreateMutexW(None, False, "ADR-Studio-Running")
    except Exception:  # noqa: BLE001 - 沙箱策略拦截等, 卸载拦截是尽力而为
        pass


def _app_log_file():
    """应用日志落点: ``<数据目录>/logs/adr.log`` (按日轮转, 14 份 / 200MB)。"""
    try:
        from adr.core.config import adr_data_dir
        d = adr_data_dir() / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return str(d / "adr.log")
    except OSError:
        return None


def main():
    # 批次57: native 层崩溃 (如 Intel 13/14 代 AVX-512 缺陷触发 oneDNN/MKL 段错误)
    # 通常无 Python traceback、stderr 零输出 — faulthandler 在崩溃瞬间把 Python 侧
    # 调用栈写到 stderr, 随桌面壳泵入 server.log, 是"闪退无痕"的最后兜底。
    faulthandler.enable(file=sys.stderr, all_threads=True)
    # 批次72: 尽早持锁 — 卸载器在服务进程存活期间不得动 runtime 文件
    _hold_app_mutex()
    ap = argparse.ArgumentParser(
        description="ADR TTS 服务 (GSV api_v2 兼容层 + ADR 原生档案 API)")
    ap.add_argument("-p", "--port", type=int, default=9881,
                    help="监听端口 (默认 9881, 与 GSV api_v2 一致)")
    ap.add_argument("-a", "--addr", default="127.0.0.1",
                    help="监听地址 (默认 127.0.0.1 仅本机; 局域网访问用 "
                         "0.0.0.0, 必须同时设置 ADR_TTS_API_KEY)")
    ap.add_argument("--agree-terms", action="store_true",
                    help="确认使用条款并同意 (等同 ADR_AGREE_TERMS=1; "
                         "条款要点见 README「使用条款」节)")
    ap.add_argument("--from-shell", action="store_true",
                    help=argparse.SUPPRESS)
    args = ap.parse_args()
    # 批次56 review: 此前 server 从不调 setup_logging, root logger 默认 WARNING —
    # adr.* 的 INFO 日志 (gsv_engine [prewarm] 就绪/预热等排障关键行) 全部被吞,
    # server.log 与 docker logs 只剩 uvicorn 行。级别可用 ADR_LOG_LEVEL 覆盖。
    # 批次103: 同时写 <数据目录>/logs/adr.log (按日轮转 14 份 / 总量 200MB),
    # 排障不再只能翻会被截断的原始 server.log。
    setup_logging(level=os.environ.get("ADR_LOG_LEVEL", "INFO"),
                  log_file=_app_log_file())
    _enforce_terms(args)
    _warn_unauthenticated_listen(args.addr, from_shell=args.from_shell)
    _warn_ref_audio_roots()
    _print_banner(args.addr, args.port)
    # 批次90 (R3-CON-01 / CON-01): 进程级 watchdog。
    # 卡死的 worker 线程可能同时持有引擎锁与 _CHDIR_LOCK, 进程内任何"自愈"
    # 都拿不到锁 → 永久 503 直到重启。/readyz 只负责把这种状态暴露给编排层,
    # 真恢复必须靠进程重启: 这里在降级持续超过阈值时记审计后 os._exit(75),
    # 由 Docker restart policy / 桌面壳 supervise 拉起新进程。
    # 仅在真实服务入口启动 (不在模块 import / 引擎构造 / 测试中), 阈值可用
    # ADR_ENGINE_WATCHDOG_SEC 调整, 设 0 关闭。
    from adr.models import gsv_engine
    from adr.server import console
    gsv_engine.start_watchdog(on_exit=console.stop_background_procs)
    uvicorn.run(create_app(), host=args.addr, port=args.port)


if __name__ == "__main__":
    main()
