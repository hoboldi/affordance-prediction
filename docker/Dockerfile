# syntax=docker/dockerfile:1
# SAM 3D Objects + affordance-prediction pipeline
# ---------------------------------------------------------------------------
# Build with BuildKit so RUN --mount=type=cache reuses conda/pip downloads:
#   DOCKER_BUILDKIT=1 docker compose build
# ---------------------------------------------------------------------------
# Prerequisites (host):
#   - NVIDIA driver + nvidia-container-toolkit (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
#   - SAM3D code: cloned from GitHub during build (no git submodule required). Mounting the repo
#     replaces /workspace/sam-3d-objects with your submodule checkout if you use one.
#   - SAM3D weights:  mount sam-3d-objects/checkpoints/ or download inside the container (HF token)
#
# Official SAM3D setup reference:
#   https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# OS libraries: OpenGL / EGL for pyrender, build tools for flash-attn & git deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates git curl \
    build-essential cmake ninja-build pkg-config \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

# Miniforge3 (conda-forge + mamba)
ENV CONDA_DIR=/opt/conda
RUN wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniforge.sh
ENV PATH="${CONDA_DIR}/bin:${PATH}"

WORKDIR /workspace

# SAM3D source (submodule not required in build context — clone upstream here)
ARG SAM3D_REPO=https://github.com/facebookresearch/sam-3d-objects.git
# Optional: branch, tag, or commit SHA (empty = shallow clone of repo default branch)
ARG SAM3D_REF=
RUN set -eux; \
    if [ -z "${SAM3D_REF}" ]; then \
        git clone --depth 1 "${SAM3D_REPO}" /workspace/sam-3d-objects; \
    else \
        git clone "${SAM3D_REPO}" /workspace/sam-3d-objects; \
        git -C /workspace/sam-3d-objects checkout "${SAM3D_REF}"; \
    fi

# SAM3D conda stack (CUDA 12.1, Python 3.11, compilers)
# First-time expectation: often ~10–25 min (large downloads). Not stuck — mamba output scrolls slowly.
# Cache mount: speeds rebuilds when this layer is invalidated later (clone/pip still re-run).
# Do NOT run `mamba clean -afy` in the same step: it tries to wipe `/opt/conda/pkgs`, which is
# the cache mount, and libmamba fails with "cannot remove: Device or resource busy".
RUN --mount=type=cache,target=/opt/conda/pkgs \
    echo ">>> [1/6] mamba env create (conda packages) $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    && mamba env create --name sam3d --file /workspace/sam-3d-objects/environments/default.yml \
    && echo ">>> [1/6] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Shrink the image: clean tarballs / index on the real `/opt/conda/pkgs` (not the cache mount).
RUN mamba clean -afy

# Pip indexes from upstream setup.md
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

# flash-attn / pytorch3d compile can be memory-heavy
ARG MAX_JOBS=4
ENV MAX_JOBS=${MAX_JOBS}

# No GPU is visible during `docker build`, so PyTorch cannot infer SMs. gsplat (from sam3d
# `[inference]`) then fails in torch.utils.cpp_extension._get_cuda_arch_flags: IndexError on
# empty arch_list. Pin real architectures + PTX for forward compatibility.
# Override at build time, e.g. `--build-arg TORCH_CUDA_ARCH_LIST="8.9+PTX"` for RTX 4090 only.
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Split pip so `docker compose build` shows which phase you are in (single RUN looks “hung” for 30–90 min).
# Rough first-build budgets: [dev] ~10–40 min, [p3d] ~15–60 min (flash-attn compile), [inference] ~5–20 min.

RUN --mount=type=cache,target=/root/.cache/pip \
    mamba run -n sam3d bash -lc 'set -euo pipefail; \
    echo ">>> [2/6] sam3d pip install .[dev] $(date -u +%Y-%m-%dT%H:%M:%SZ)"; \
    cd /workspace/sam-3d-objects && pip install -e ".[dev]"; \
    echo ">>> [2/6] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"'

RUN --mount=type=cache,target=/root/.cache/pip \
    mamba run -n sam3d bash -lc 'set -euo pipefail; \
    echo ">>> [3/6] sam3d pip install .[p3d] (pytorch3d + flash-attn — often the slowest) $(date -u +%Y-%m-%dT%H:%M:%SZ)"; \
    cd /workspace/sam-3d-objects && pip install -e ".[p3d]"; \
    echo ">>> [3/6] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"'

RUN --mount=type=cache,target=/root/.cache/pip \
    mamba run -n sam3d bash -lc 'set -euo pipefail; \
    echo ">>> [4/6] sam3d pip install .[inference] $(date -u +%Y-%m-%dT%H:%M:%SZ)"; \
    cd /workspace/sam-3d-objects && pip install -e ".[inference]"; \
    echo ">>> [4/6] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"'

RUN mamba run -n sam3d bash -lc 'set -euo pipefail; \
    echo ">>> [5/6] sam3d hydra patch $(date -u +%Y-%m-%dT%H:%M:%SZ)"; \
    cd /workspace/sam-3d-objects && python patching/hydra; \
    echo ">>> [5/6] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"'

# This repo (avoid pulling a second PyTorch — SAM3D pins torch 2.5.1+cu121)
COPY pyproject.toml README.md /workspace/
COPY configs /workspace/configs
COPY scripts /workspace/scripts
COPY src /workspace/src
COPY dataset_pipeline.py /workspace/dataset_pipeline.py
COPY notebooks /workspace/notebooks
COPY tests /workspace/tests

RUN --mount=type=cache,target=/root/.cache/pip \
    mamba run -n sam3d bash -lc 'set -euo pipefail; \
    echo ">>> [6/6] affordance-prediction pip deps $(date -u +%Y-%m-%dT%H:%M:%SZ)"; \
    cd /workspace; \
    pip install -e ".[notebooks,dev]" --no-deps; \
    pip install \
        "numpy>=1.24" "Pillow>=10.0" "PyYAML>=6.0" "trimesh>=4.0" \
        "pyrender>=0.1.45" "PyOpenGL>=3.1.0" "tqdm>=4.65" "loguru>=0.7" \
        "transformers>=4.36" "safetensors>=0.4" \
        "jupyter>=1.0" "matplotlib>=3.7" "ipykernel>=6.0" \
        "pytest>=7.0" "ruff>=0.1"; \
    echo ">>> [6/6] done — image ready $(date -u +%Y-%m-%dT%H:%M:%SZ)"'

ENV PYTHONPATH=/workspace/src
ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV CONDA_DEFAULT_ENV=sam3d
# sam3d_wrapper uses CONDA_PREFIX when CUDA_HOME is unset
ENV CONDA_PREFIX=/opt/conda/envs/sam3d
ENV PATH=/opt/conda/envs/sam3d/bin:${PATH}

SHELL ["/bin/bash", "-lc"]

CMD ["bash"]
