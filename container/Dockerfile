ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
FROM $BASE_IMAGE AS builder

# Set up environment variables
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/a2e6fa7e035ff058fc37fdaaf014707efff2fcf3/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl \
    CFLAGS="-I/usr/include -I/usr/include/x86_64-linux-gnu" \
    CXXFLAGS="${CFLAGS}" \
    NVCCFLAGS="-I/usr/include -I/usr/include/x86_64-linux-gnu"


RUN apt-get update -y && \
    apt-get install python3-venv -y && \
    apt-get install vim tmux -y && \
    apt-get install python3-dev build-essential -y && \
    apt-get install -y cmake build-essential ninja-build && \
    apt install -y gcc g++ gcc-multilib g++-multilib && \
    apt-get install -y git && \
    python3 -m venv $VIRTUAL_ENV && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Activate the venv and install Poetry
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir poetry==1.8.3 && \
    pip install sae_lens==6.13.0

WORKDIR /app

COPY local_reqs/requirements.txt /app/local_reqs/requirements.txt
RUN pip install --no-cache-dir -r local_reqs/requirements.txt

COPY . /app

RUN ls -la /app && echo $VLLM_PRECOMPILED_WHEEL_LOCATION

RUN pip uninstall pynvml -y

RUN cd /app && . /opt/venv/bin/activate && pip install -e .

# env VLLM_PRECOMPILED_WHEEL_LOCATION="$VLLM_PRECOMPILED_WHEEL_LOCATION" \
#     VLLM_COMMIT="$VLLM_COMMIT" \
#     cd /app && \
#     pip install -e .

# Expose the ports
EXPOSE 5001 80 8000 8001 8002 8003
