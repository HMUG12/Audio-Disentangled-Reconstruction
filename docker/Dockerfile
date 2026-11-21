# syntax=docker/dockerfile:1
# =============================================================================
# ADR 容器版 (批次56, v1.2 多形态分发第一形态)
#
# 设计要点:
#   - 引擎/引导/档案逻辑锚定 REPO_ROOT/third_party/gpt_sovits (源码布局),
#     因此 ADR 以 pip install -e 装到 /app, GSV clone 到 /app/third_party 同层
#   - GSV 不入 git (third_party 整树被 .gitignore), 容器内按 GSV_COMMIT 钉版
#     clone + scripts/apply_compat_patches.py 打补丁, 与本地树可复现对齐
#   - 双变体: TORCH_VARIANT=cpu (默认) | cuda (cu130, 与 Windows GPU 包对齐)
#     torch 不进 requirements-linux.txt, 按变体单独安装
#   - 无 WebUI: gradio 不装, 经典 Gradio 控制台已移除 (CLI + HTTP API + 静态控制台页形态)
#   - 权重自引导: 首次启动自动下载 pretrained_models (~1.4GB) + G2PWModel
#     (~560MB), 生产环境用 compose 卷持久化 (见 docker-compose.yml)
# =============================================================================

# GSV 钉版 commit (与本地 third_party/gpt_sovits HEAD 对齐, 批次56 文档化)
ARG GSV_COMMIT=d523079fc05d9a8028d6085bffe4a2757c32abb6
# GSV clone 前缀: 网络受限时可传 https://ghproxy.net/ 加速
ARG GSV_PROXY_PREFIX=
# apt 镜像: 默认 deb.debian.org 走 HTTP/80 (部分代理不覆盖 HTTP 会 update 失败),
# 可传 mirrors.tuna.tsinghua.edu.cn 等 (替换为 https://<mirror>)
ARG APT_MIRROR=

# ============================== Stage 1: 依赖构建 ==============================
FROM python:3.11-slim AS builder

ARG GSV_COMMIT
ARG GSV_PROXY_PREFIX
ARG APT_MIRROR
# cpu | cuda
ARG TORCH_VARIANT=cpu

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# git: clone GSV; build-essential+cmake: pyopenjtalk 等无轮子时 sdist 编译兜底
RUN if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|http://deb.debian.org|https://${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources; \
    fi \
    && apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# venv: 构建产物整体搬到 runtime, builder 里的工具链不进最终镜像
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# 依赖清单先行 (代码变更不触发依赖重装, 层缓存友好)
COPY docker/requirements-linux.txt /tmp/requirements-linux.txt

# torch 栈按变体安装 (先于其余依赖: pytorch-lightning 等看到已装即满足;
# 避免 pip 从 PyPI 拉回另一变体轮子覆盖)
RUN if [ "$TORCH_VARIANT" = "cuda" ]; then \
        pip install torch==2.10.0 torchaudio==2.10.0 \
            --index-url https://download.pytorch.org/whl/cu130; \
    else \
        pip install torch==2.10.0 torchaudio==2.10.0 \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi

RUN pip install -r /tmp/requirements-linux.txt

# ADR 本体: --no-deps 跳过依赖解析 (依赖已由上面清单钉版)
# editable 安装保持源码布局 — REPO_ROOT=/app 锚定不被破坏
COPY pyproject.toml README.md LICENSE ./
COPY adr/ ./adr/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
# 文档中心 (/api/adr/v1/docs) 白名单 8 篇 = README.md + 下列 7 篇。
# 只拷白名单: docs/ 下的 m*-plan / audit / 路线图属内部材料, 不进公开镜像。
COPY docs/demos.md docs/gpu-compat.md docs/update-manifest.md docs/tts-api-spec.md \
     docs/architecture-design.md docs/release-notes-v1.2.0.md ./docs/
COPY docs/architecture/windows-install-acceptance.md ./docs/architecture/
RUN pip install -e . --no-deps

# GSV 钉版 clone + 兼容补丁 (含批次33 Track E AR 性能补丁 track_e_ar_sync.patch)
# apply_compat_patches.py 失败即退出非零, 构建失败显式报错
RUN git clone ${GSV_PROXY_PREFIX}https://github.com/RVC-Boss/GPT-SoVITS.git \
        third_party/gpt_sovits \
    && git -C third_party/gpt_sovits checkout --detach "$GSV_COMMIT" \
    && python scripts/apply_compat_patches.py

# ================================ Stage 2: 运行时 ==============================
FROM python:3.11-slim AS runtime

ARG ADR_VERSION=1.3.1-beta.92
ARG APT_MIRROR

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ADR_DATA_DIR=/data \
    ADR_TTS_CACHE_DIR=/data/tts_cache \
    HF_HOME=/data/hf-cache \
    TMPDIR=/tmp

# git: GSV 兼容层运行时需要; ffmpeg: pydub 音频转码; libsndfile1: soundfile
RUN if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|http://deb.debian.org|https://${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources; \
    fi \
    && apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 批次118 (审计 R5-DOCK-01): 非 root 运行 —— 数值 UID/GID 便于挂载目录 chown
# 与 OpenShift 类随机 UID 环境; 应用代码与数据目录都归该用户, venv 仍 root
# 只读 (运行时改不了解释器与依赖)。
RUN groupadd --gid 10001 adr \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin adr

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=10001:10001 /app /app

WORKDIR /app

# 权重自引导目标目录 (首启下载; 生产环境挂卷持久化, 见 docker-compose.yml)
# 新建目录随镜像归属 adr —— named volume 首次挂载会继承该属主; bind mount
# (如 compose 的 ./data) 需宿主机侧 `chown -R 10001:10001`, 见 docker/README.md
RUN mkdir -p /data \
        /app/third_party/gpt_sovits/pretrained_models \
        /app/third_party/gpt_sovits/text/G2PWModel \
    && chown -R 10001:10001 /data /app/third_party/gpt_sovits/pretrained_models \
        /app/third_party/gpt_sovits/text/G2PWModel

# 缓存类目录兜底 (numba/huggingface/matplotlib 等默认写 $HOME)
ENV HOME=/home/adr
USER 10001:10001

EXPOSE 9881

# 批次90 (R3-OBS-01): 用 /readyz 而非 /health 做健康检查。
# /health 恒 200 (只证明 HTTP 能应答), 引擎卡死/永久降级时仍显示 healthy;
# /readyz 在引擎未就绪或已降级时返回 503, 才是"能否提供服务"的真实信号。
# start-period 给足 600s: 首启要下载 ~2GB 权重并初始化模型, 期间 readyz 是 503,
# 不能算失败; 超过 start-period + retries 仍 503 → unhealthy (可配 autoheal 重启)。
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('ADR_PORT','9881') + '/api/adr/v1/readyz', timeout=4)" || exit 1

# 批次118 (审计 R5-DOCK-02): 端口唯一事实源 = ADR_PORT —— 服务监听与探针读
# 同一个变量。原先是探针读 ADR_PORT、CMD 却写死 9881, 改端口只设一个必然
# 探针恒 unhealthy。`exec` 让 python 直接接管 PID 1, SIGTERM 可正常停机。
ENV ADR_PORT=9881

# 设备走向由引擎 auto 自检 (ADR_TTS_DEVICE 可覆盖):
#   cpu 镜像 -> 自动 cpu; cuda 镜像有 GPU -> cuda, 无 GPU 环境回退 cpu
# 容器内必须监听 0.0.0.0 才能被端口映射访问 (0.0.0.0 是容器自身命名空间,
# 非对外暴露); 暴露面由 compose/`docker run -p` 的发布地址决定。
# 批次87 (SEC-04): 对外监听 + 未设 ADR_TTS_API_KEY 时服务默认拒绝启动。
# 批次90 (SEC-04 残留): 不再在镜像里写死 ADR_ALLOW_UNAUTHENTICATED=1 ——
# 那等于把"无鉴权对外"做成镜像默认值, `docker run -p 9881:9881` 直接裸奔。
# 现在由 docker-compose.yml 显式传入 (其默认只发布 127.0.0.1); 手写
# `docker run` 需自行 -e ADR_ALLOW_UNAUTHENTICATED=1 或配置 ADR_TTS_API_KEY。
CMD ["sh", "-c", "exec python -m adr.server --addr 0.0.0.0 --port \"${ADR_PORT}\""]

LABEL org.opencontainers.image.title="ADR Server" \
      org.opencontainers.image.description="ADR: low-resource fast voice cloning (GPT-SoVITS compatible TTS)" \
      org.opencontainers.image.version="${ADR_VERSION}" \
      org.opencontainers.image.licenses="AGPL-3.0" \
      org.opencontainers.image.source="https://github.com/HMUG12/Audio-Disentangled-Reconstruction"
