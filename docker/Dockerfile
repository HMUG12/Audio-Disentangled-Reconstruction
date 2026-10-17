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
#   - 无 WebUI: gradio 不装, `adr webui` 在容器内不可用 (CLI + HTTP API 形态)
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
RUN pip install -e . --no-deps

# GSV 钉版 clone + 兼容补丁 (含批次33 Track E AR 性能补丁 track_e_ar_sync.patch)
# apply_compat_patches.py 失败即退出非零, 构建失败显式报错
RUN git clone ${GSV_PROXY_PREFIX}https://github.com/RVC-Boss/GPT-SoVITS.git \
        third_party/gpt_sovits \
    && git -C third_party/gpt_sovits checkout --detach "$GSV_COMMIT" \
    && python scripts/apply_compat_patches.py

# ================================ Stage 2: 运行时 ==============================
FROM python:3.11-slim AS runtime

ARG ADR_VERSION=1.2.0
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

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

WORKDIR /app

# 权重自引导目标目录 (首启下载; 生产环境挂卷持久化, 见 docker-compose.yml)
RUN mkdir -p /data \
        /app/third_party/gpt_sovits/pretrained_models \
        /app/third_party/gpt_sovits/text/G2PWModel

EXPOSE 9881

# /health 恒 200 (engine_ready 是字段非状态码) — 判 HTTP 可答即进程健康;
# 权重下载/引擎初始化由业务侧轮询 engine_ready 观察, 不影响容器健康态
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('ADR_PORT','9881') + '/api/adr/v1/health', timeout=4)" || exit 1

# 设备走向由引擎 auto 自检 (ADR_TTS_DEVICE 可覆盖):
#   cpu 镜像 -> 自动 cpu; cuda 镜像有 GPU -> cuda, 无 GPU 环境回退 cpu
ENTRYPOINT ["python", "-m", "adr.server"]
CMD ["--addr", "0.0.0.0", "--port", "9881"]

LABEL org.opencontainers.image.title="ADR Server" \
      org.opencontainers.image.description="ADR: low-resource fast voice cloning (GPT-SoVITS compatible TTS)" \
      org.opencontainers.image.version="${ADR_VERSION}" \
      org.opencontainers.image.licenses="AGPL-3.0" \
      org.opencontainers.image.source="https://github.com/HMUG12/Audio-Disentangled-Reconstruction"
