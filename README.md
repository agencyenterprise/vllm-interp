# vllm-interp

A fork of [vLLM](https://github.com/vllm-project/vllm) with support for SAE (Sparse Autoencoder) steering and activation extraction.

**Base vLLM commit:** [`a2e6fa7e`](https://github.com/vllm-project/vllm/commit/a2e6fa7e035ff058fc37fdaaf014707efff2fcf3)

## What's Added (vllm-interp)

The following files and modifications are specific to vllm-interp and not part of the original vLLM:

### New Files
- `vllm/model_executor/models/goodfire_sae.py` - Goodfire SAE setup and loading
- `vllm/model_executor/models/llama_models_and_saes.py` - SAE configuration for different Llama models
- `example_scripts/async_example_llama.py` - Example script for running Llama with SAE steering
- `local_reqs/requirements.txt` - Additional requirements for SAE functionality

### Modified Files
- `vllm/model_executor/models/llama.py` - Extended with SAE steering support (see details below)
- `vllm/v1/worker/gpu_model_runner.py` - Added `steer_positions` generation for steering

### Key Modifications to `llama.py`

The main additions to the Llama model are:

1. **`init_sae_for_rank`**: Initializes and loads the SAE for each worker
2. **`forward_sae`**: Runs the intervention/steering using the SAE

The `forward` method is extended with two additional arguments:
- `interventions`: A list of intervention inputs with featureID and strength
- `steer_positions`: Generated in `gpu_model_runner.py` (not required when sending requests to the engine)

Key code locations:
- Forward signature extension: `vllm/model_executor/models/llama.py:L722`
- Output modification: `vllm/model_executor/models/llama.py:L509`
- Sparse output handling: `vllm/model_executor/models/llama.py:L356` and `L367`
- Final layer tensor subtraction: `vllm/model_executor/models/llama.py:L536`

The `intervention_enabled` flag determines the steering layer, while `feature_enabled` determines the layer for reading features. Goodfire SAEs have different steering and feature layers (see `llama_models_and_saes.py`).

---

## Installation

Prerequisites: Environment with CUDA 12.8+ driver (vLLM is compiled with CUDA 12.8).

### 1. Prepare the environment

```bash
apt-get update -y
apt-get install ninja-build cmake jq zip -y
apt-get install -y build-essential libstdc++-12-dev libc6-dev
apt-get install -y gcc g++ gcc-multilib g++-multilib
```

### 2. Install uv and create virtualenv

[uv](https://docs.astral.sh/uv/) handles Python version management — no need to install Python 3.12 separately.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env  # or restart shell
uv venv /tmp/vllm_env --python 3.12
source /tmp/vllm_env/bin/activate
```

### 3. Install sae_lens first (uses outdated numpy version)

```bash
uv pip install sae_lens==6.13.0
```

### 4. Install build packages

```bash
uv pip install pip wheel setuptools_scm setuptools --upgrade
```

### 5. Install requirements

```bash
uv pip install -r local_reqs/requirements.txt
```

### 6. Install vllm-interp as editable package with pre-built libraries

```bash
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/a2e6fa7e035ff058fc37fdaaf014707efff2fcf3/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
uv pip install --editable .
```

The wheel URL corresponds to the base vLLM commit that vllm-interp is forked from.

---

## Usage

### Environment Variables

Required for attention backend (especially for Gemma models):
```bash
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_FLASH_ATTN_VERSION=3
```

Set your Hugging Face token to pull Llama models:
```bash
export HF_TOKEN=your_token_here
```

### Example Script

See `example_scripts/async_example_llama.py` for a complete example with batched steering inputs.
