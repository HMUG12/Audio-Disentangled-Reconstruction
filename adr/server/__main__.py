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


def _warn_unauthenticated_listen(addr: str) -> None:
    """监听地址对外 + 无 API Key 时打醒目警告 (批次26, P2-a)。

    create_app 不知道 bind host (uvicorn.run 在本入口传参),
    所以只有这里能同时看到 host 与鉴权配置 — 警告只打这一处, 不在 app.py 重复。
    """
    if addr not in ("0.0.0.0", "::"):
        return
    if settings.has_api_key():
        return
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


def main():
    # 批次57: native 层崩溃 (如 Intel 13/14 代 AVX-512 缺陷触发 oneDNN/MKL 段错误)
    # 通常无 Python traceback、stderr 零输出 — faulthandler 在崩溃瞬间把 Python 侧
    # 调用栈写到 stderr, 随桌面壳泵入 server.log, 是"闪退无痕"的最后兜底。
    faulthandler.enable(file=sys.stderr, all_threads=True)
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
    setup_logging(level=os.environ.get("ADR_LOG_LEVEL", "INFO"))
    _enforce_terms(args)
    _warn_unauthenticated_listen(args.addr)
    _print_banner(args.addr, args.port)
    uvicorn.run(create_app(), host=args.addr, port=args.port)


if __name__ == "__main__":
    main()
