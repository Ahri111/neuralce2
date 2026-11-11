# GPU image: CUDA 12 runtime + Python 3.10 + the pinned environment + this package.
#   docker build -t neuralce .
#   docker run --gpus all -it -v $PWD/runs:/work/runs neuralce bash
#   docker run --gpus all neuralce pytest -q                 # tests (CPU parts run without a GPU as well)
#   docker run --gpus all neuralce neuralce-mc --help
# CPU-only machines: the same image works (JAX falls back to CPU; add JAX_PLATFORMS=cpu to silence the warning).
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    JAX_DEFAULT_MATMUL_PRECISION=highest \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    NCE_REPO=/work \
    NCE_RUNS=/work/runs

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-venv python3-pip \
        build-essential git ca-certificates fonts-liberation \
    && rm -rf /var/lib/apt/lists/* \
    && python3.10 -m pip install --upgrade pip setuptools wheel

WORKDIR /work
# dependencies first (cached layer); icet is compiled here (~3 min)
COPY requirements.txt requirements-cuda.txt ./
RUN python3.10 -m pip install -r requirements.txt -r requirements-cuda.txt

# the package, data, models and processed results
COPY . /work
RUN python3.10 -m pip install -e . && mkdir -p /work/runs /work/logs

CMD ["bash"]
