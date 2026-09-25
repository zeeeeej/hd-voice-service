# 架构无关：本机 arm64 构建即 arm64 镜像；x86 服务器上构建即得 amd64
# v1 仅 PCM/WAV：不装 ffmpeg（Opus 阶段再加）；healthcheck 用 python stdlib，无需 apt
# 默认官方 PyPI（本机直连可用；清华/阿里源对容器出口 IP 返回 403）
# 如在其他网络需换源: docker build --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
FROM python:3.11-slim-bookworm

ARG PIP_INDEX_URL=
ARG REQUIREMENTS=requirements.txt   # gpu-fp16 profile 用 requirements-gpu.txt

WORKDIR /opt/voice

COPY requirements.txt requirements-gpu.txt ./
RUN pip install --no-cache-dir ${PIP_INDEX_URL:+-i $PIP_INDEX_URL} -r $REQUIREMENTS

COPY app ./app
COPY config.yaml ./config.yaml

ENV VOICE_CONFIG=/opt/voice/config.yaml \
    VOICE_MODELS_DIR=/opt/voice/models \
    PYTHONUNBUFFERED=1

EXPOSE 8090
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090", "--workers", "1"]
