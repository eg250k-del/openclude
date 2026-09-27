# openclude production image.
#
# Facts this is built on, each verified by reading the engine's own source or
# SaladCloud's own docs rather than guessed:
#
#  * SaladCloud runs containers from a registry image, linux/amd64. There is no
#    volume mounting (containers are unprivileged) and the filesystem is
#    ephemeral, so everything durable goes to object storage.
#  * The engine's Colab notebook pins torch 2.10.0 + cu128 and refuses 2.8/2.9.
#    We bake torch in rather than installing it at start, which removes a ~4 GB
#    download on every cold start.
#  * ffmpeg must support -fps_mode, which means ffmpeg >= 5. Ubuntu 22.04 ships
#    4.4, so a static build is required. The engine will otherwise download one
#    itself on every boot.
#  * The engine's matanyone preprocessor calls matplotlib.use('TkAgg'), which
#    fails with no display. It must be patched at build time, not at runtime.
#  * A leftover startup.lock makes the engine enter Safe Mode and disable
#    plugins, so the entrypoint removes it before every start.
#  * --profile 3 is the engine's own setting for 24 GB VRAM. The Colab notebook
#    uses --profile 5, which is a T4/15 GB choice and is wrong for a 4090.
#
# Model weights are NOT baked in. They are pulled at container start into
# /data/ckpts, which the caller is expected to back with object storage.

ARG CUDA_TAG=12.8.1-cudnn-runtime-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

ARG WAN2GP_REF=main
ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_HOME=/usr/local/cuda \
    HF_HOME=/data/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/data/cache/huggingface/hub \
    TORCH_HOME=/data/cache/torch \
    XDG_CACHE_HOME=/data/cache \
    TRITON_CACHE_DIR=/data/cache/triton \
    WAN2GP_ROOT=/opt/WanGP \
    OPENCLIDE_DATA=/data

# ---------------------------------------------------------------- system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        git git-lfs curl ca-certificates \
        ffmpeg \
        libglib2.0-0 libgl1 libportaudio2 \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

# ffmpeg 4.x (Ubuntu 22.04) has no -fps_mode, which the engine requires.
# BtbN publishes static builds; take the newest linux64 GPL release and verify
# the archive's SHA-256 before unpacking it, exactly like the engine's own
# bootstrap does.
ARG FFMPEG_RELEASE_API=https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest
COPY deploy/pick_ffmpeg.py /usr/local/bin/pick_ffmpeg.py
RUN set -eux; \
    url=$(python3 /usr/local/bin/pick_ffmpeg.py "$FFMPEG_RELEASE_API"); \
    echo "downloading $url"; \
    curl -fsSL -o /tmp/ffmpeg.tar.xz "$url"; \
    mkdir -p /tmp/ffx && tar -xJf /tmp/ffmpeg.tar.xz -C /tmp/ffx; \
    bin=$(find /tmp/ffx -type f -name ffmpeg | head -1); \
    test -n "$bin"; \
    install -m 0755 "$bin" /usr/local/bin/ffmpeg; \
    install -m 0755 "$(find /tmp/ffx -type f -name ffprobe | head -1)" /usr/local/bin/ffprobe; \
    rm -rf /tmp/ffx /tmp/ffmpeg.tar.xz; \
    ffmpeg -version | head -1; \
    # the engine needs -fps_mode, which ffmpeg 4.x (Ubuntu 22.04) does not have;
    # fail the build here rather than at container start, minutes and money later
    ffmpeg -h full 2>/dev/null | grep -q -- -fps_mode || { echo "ffmpeg lacks -fps_mode"; exit 1; }; \
    rm -f /usr/local/bin/pick_ffmpeg.py

# ---------------------------------------------------------------------- torch
RUN python3.11 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python3.11 -m pip install --no-cache-dir \
        torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
        --index-url https://download.pytorch.org/whl/cu128 \
 && python3.11 -m pip install --no-cache-dir onnxruntime-gpu==1.22.0 || \
   python3.11 -m pip install --no-cache-dir 'onnxruntime>=1.22.0'

# --------------------------------------------------------------------- engine
# A build toolchain is not optional. The engine pulls in packages with C
# extensions, and the cudnn-runtime base image has no compiler at all, so pip
# fails with the least informative error Python can produce. This was the
# first build's actual failure.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Cloned in its own layer so a failure names the step that failed rather than
# being buried inside a three-command chain.
RUN git clone --depth 1 --branch ${WAN2GP_REF} \
        https://github.com/deepbeepmeep/Wan2GP.git /opt/WanGP \
 && cd /opt/WanGP \
 && git rev-parse HEAD > .pinned-ref \
 && echo "engine pinned at $(cat .pinned-ref)"

# --extra-index-url is the important part. Without it pip resolves torch from
# PyPI and happily installs the CPU-only wheel, which silently undoes the
# cu128 install above and yields an image that cannot use the GPU. The engine's
# own Colab bootstrap uses the same trick.
#
# --extra-index-url is preferred over --index-url here so a plain PyPI
# requirement still resolves. If a transitive pin does move torch, the next
# layer forces the CUDA build back.
RUN python3.11 -m pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        -r /opt/WanGP/requirements.txt

# Re-assert the CUDA torch after the requirements pass. Without the "+cu128"
# local version tag, pip installs a CPU build that imports fine and then fails
# at the first .cuda() call, on a rented GPU, at cost.
RUN python3.11 -m pip install --no-cache-dir --force-reinstall \
        torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128 \
 && python3.11 -c "import torch,sys; print(torch.__version__, torch.version.cuda); sys.exit(0 if torch.version.cuda else 1)"

# The engine assumes a desktop. The patch is build-time only, so it must be
# applied here or the container dies on the first image it touches.
RUN set -eux; \
    f=/opt/WanGP/preprocessing/matanyone/tools/interact_tools.py; \
    test -f "$f"; \
    sed -i "s/matplotlib.use('TkAgg')/matplotlib.use('Agg')/" "$f"; \
    grep -q "matplotlib.use('Agg')" "$f"

# ------------------------------------------------------------- openclude code
WORKDIR /app
COPY pyproject.toml /app/
COPY src /app/src
RUN python3.11 -m pip install --no-cache-dir /app

# ---------------------------------------------------------------- data layout
# Every one of these is a mount point for object-storage sync. Nothing durable
# may live anywhere else, because the filesystem dies with the container.
RUN mkdir -p /data/ckpts /data/loras /data/outputs /data/frames /data/audio \
             /data/cache /data/config /data/work && \
    ln -sfn /data/ckpts   /opt/WanGP/ckpts && \
    ln -sfn /data/loras   /opt/WanGP/loras && \
    ln -sfn /data/outputs /opt/WanGP/outputs

COPY deploy/entrypoint.sh /usr/local/bin/openclude-entrypoint
RUN chmod +x /usr/local/bin/openclude-entrypoint

# Liveness: the process is alive and the ledger is readable. Deliberately does
# NOT check the GPU, because a model download can take many minutes and a probe
# that fails during it would get the instance killed mid-download.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/openclude-entrypoint"]
