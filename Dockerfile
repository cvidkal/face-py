# face-py runtime image — CUDA 12.8 + cuDNN 9 + Python + onnxruntime-gpu.
#
# Base 跟 host (T4 / CUDA 12.8 / cuDNN 9) 对齐, onnxruntime-gpu 1.23.x 在这上面跑通.
# 用 -runtime tag, 不要 -devel — 没编译需求, 省 ~3GB.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps:
# - python3.10 + pip (Ubuntu 22.04 默认)
# - libgl1 / libglib2.0-0: opencv-python-headless 仍依赖少量动态库
# - tini: PID 1, 转发信号给 python (跟 sibling 容器同款)
# - ca-certificates: HTTPS
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
        libgl1 libglib2.0-0 \
        tini ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依赖先装, 利用 docker layer cache (代码改不重装)
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

# 代码 (models / .git / tests / docs 由 .dockerignore 屏蔽)
COPY module /app/module
COPY service /app/service
COPY run_http_server.sh /app/run_http_server.sh
COPY VERSION /app/VERSION

# Models 跟 sessions 通过 bind mount 进来 (compose volumes 段), 不打镜像
ENV FACE_DETECT_MODEL_PATH=/app/models/face_detection_yunet_2023mar.onnx \
    FACE_RECOGNIZE_MODEL_PATH=/app/models/face_recognition_sface_2021dec.onnx

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python3", "-u", "/app/service/serve.py"]
