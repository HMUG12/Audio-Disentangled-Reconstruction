# ADR Framework Docker Image
# 多阶段构建,最终镜像约 3GB (含 CUDA + PyTorch + ADR)

# ============ Stage 1: 依赖构建 ============
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS deps

LABEL maintainer="ADR Team"
LABEL description="ADR: low-resource fast voice cloning training framework"
LABEL version="0.1.0"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        git wget curl ffmpeg libsndfile1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 符号链接
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# 升级 pip
RUN python -m pip install --upgrade pip setuptools wheel

# 复制依赖文件
COPY requirements.txt pyproject.toml ./

# 安装 PyTorch (CUDA 12.1)
RUN pip install torch==2.1.0 torchaudio==2.1.0 \
        --index-url https://download.pytorch.org/whl/cu121

# 安装 ADR 基础依赖
RUN pip install -r requirements.txt

# 安装 ADR 完整依赖 (M1)
RUN pip install \
        "gradio>=4.0" \
        "faster-whisper>=1.0" \
        "g2pW>=0.1" \
        "pypinyin>=0.49"

# ============ Stage 2: 代码复制 ============
FROM deps AS code

# 复制项目文件
COPY adr/ /app/adr/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY tests/ /app/tests/
COPY docs/ /app/docs/
COPY README.md LICENSE ./

# 安装 ADR (可编辑模式)
RUN pip install -e .

# 预下载预训练模型 (在构建时缓存)
# RUN python scripts/download_pretrained.py

# ============ Stage 3: 运行时配置 ============
FROM code AS runtime

# 暴露 Gradio 端口
EXPOSE 7860

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import adr; print('OK')" || exit 1

# 默认工作目录
WORKDIR /workspace

# 默认启动 WebUI
ENTRYPOINT ["adr"]
CMD ["webui", "--host", "0.0.0.0", "--port", "7860"]
