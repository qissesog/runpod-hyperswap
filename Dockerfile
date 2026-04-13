FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip git curl libgl1-mesa-glx libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install FaceFusion (pinned to 3.3.0 which introduced HyperSwap)
WORKDIR /opt
RUN git clone --branch 3.3.0 --depth 1 https://github.com/facefusion/facefusion.git

WORKDIR /opt/facefusion

# Install Python deps — skip gradio (not needed for headless)
RUN pip3 install --no-cache-dir \
    numpy==2.2.1 \
    onnx==1.20.1 \
    onnxruntime-gpu==1.20.1 \
    opencv-python==4.10.0.84 \
    tqdm==4.67.3 \
    scipy==1.14.1 \
    filetype \
    runpod

# Pre-download ALL models FaceFusion needs at runtime
RUN mkdir -p /opt/facefusion/.assets/models

# --- models-3.3.0 (HyperSwap 1B + 1C) ---
RUN curl -fSL -o /opt/facefusion/.assets/models/hyperswap_1b_256.onnx \
    "https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1b_256.onnx" && \
    curl -fSL -o /opt/facefusion/.assets/models/hyperswap_1b_256.hash \
    "https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1b_256.hash" && \
    curl -fSL -o /opt/facefusion/.assets/models/hyperswap_1c_256.onnx \
    "https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1c_256.onnx" && \
    curl -fSL -o /opt/facefusion/.assets/models/hyperswap_1c_256.hash \
    "https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1c_256.hash"

# --- models-3.0.0 (face detector, landmarker, recognizer, enhancer, occluder, parser) ---
# YOLOFace detector
RUN curl -fSL -o /opt/facefusion/.assets/models/yoloface_8n.onnx \
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/yoloface_8n.onnx" && \
    curl -fSL -o /opt/facefusion/.assets/models/yoloface_8n.hash \
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/yoloface_8n.hash"

# Face landmarker (2dfan4 + fan_68_5)
RUN curl -fSL -o /opt/facefusion/.assets/models/2dfan4.onnx \
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/2dfan4.onnx" && \
    curl -fSL -o /opt/facefusion/.assets/models/2dfan4.hash \
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/2dfan4.hash" && \
    curl -fSL -o /opt/facefusion/.assets/models/fan_68_5.onnx \
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/fan_68_5.onnx" && \
    curl -fSL -o /opt/facefusion/.assets/models/fan_68_5.hash \
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/fan_68_5.hash"

# CodeFormer face enhancer
RUN curl -fSL -o /opt/facefusion/.assets/models/codeformer.onnx \
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx" && \
    curl -fSL -o /opt/facefusion/.assets/models/codeformer.hash \
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.hash"

# GFPGAN 1.4 face enhancer (sharper than CodeFormer, preserves eyes better)
RUN curl -fSL -o /opt/facefusion/.assets/models/gfpgan_1.4.onnx \
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx" && \
    curl -fSL -o /opt/facefusion/.assets/models/gfpgan_1.4.hash \
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.hash"

# ArcFace face recognizer (needed by HyperSwap for face embedding)
RUN curl -fSL -o /opt/facefusion/.assets/models/arcface_w600k_r50.onnx \
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/arcface_w600k_r50.onnx" && \
    curl -fSL -o /opt/facefusion/.assets/models/arcface_w600k_r50.hash \
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/arcface_w600k_r50.hash"

# Face occluder (xseg_1 — from models-3.1.0)
RUN curl -fSL -o /opt/facefusion/.assets/models/xseg_1.onnx \
    "https://huggingface.co/facefusion/models-3.1.0/resolve/main/xseg_1.onnx" && \
    curl -fSL -o /opt/facefusion/.assets/models/xseg_1.hash \
    "https://huggingface.co/facefusion/models-3.1.0/resolve/main/xseg_1.hash"

# Face parser (bisenet_resnet_34)
RUN curl -fSL -o /opt/facefusion/.assets/models/bisenet_resnet_34.onnx \
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/bisenet_resnet_34.onnx" && \
    curl -fSL -o /opt/facefusion/.assets/models/bisenet_resnet_34.hash \
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/bisenet_resnet_34.hash"

# FairFace (age/gender/race classification)
RUN curl -fSL -o /opt/facefusion/.assets/models/fairface.onnx \
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/fairface.onnx" && \
    curl -fSL -o /opt/facefusion/.assets/models/fairface.hash \
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/fairface.hash"

# NSFW content filter (3 models — from models-3.3.0)
RUN curl -fSL -o /opt/facefusion/.assets/models/nsfw_1.onnx \
    "https://huggingface.co/facefusion/models-3.3.0/resolve/main/nsfw_1.onnx" && \
    curl -fSL -o /opt/facefusion/.assets/models/nsfw_1.hash \
    "https://huggingface.co/facefusion/models-3.3.0/resolve/main/nsfw_1.hash" && \
    curl -fSL -o /opt/facefusion/.assets/models/nsfw_2.onnx \
    "https://huggingface.co/facefusion/models-3.3.0/resolve/main/nsfw_2.onnx" && \
    curl -fSL -o /opt/facefusion/.assets/models/nsfw_2.hash \
    "https://huggingface.co/facefusion/models-3.3.0/resolve/main/nsfw_2.hash" && \
    curl -fSL -o /opt/facefusion/.assets/models/nsfw_3.onnx \
    "https://huggingface.co/facefusion/models-3.3.0/resolve/main/nsfw_3.onnx" && \
    curl -fSL -o /opt/facefusion/.assets/models/nsfw_3.hash \
    "https://huggingface.co/facefusion/models-3.3.0/resolve/main/nsfw_3.hash"

# Kim vocal separator (needed by pipeline even for images)
RUN curl -fSL -o /opt/facefusion/.assets/models/kim_vocal_2.onnx \
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/kim_vocal_2.onnx" && \
    curl -fSL -o /opt/facefusion/.assets/models/kim_vocal_2.hash \
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/kim_vocal_2.hash"

# Copy handler
COPY handler.py /opt/facefusion/handler.py

CMD ["python3", "/opt/facefusion/handler.py"]
