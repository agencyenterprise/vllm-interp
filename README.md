# vllm-interp

A fork of [vLLM](https://github.com/vllm-project/vllm) with support for SAE (Sparse Autoencoder) steering and activation extraction.

**Base vLLM commit:** [`a2e6fa7e`](https://github.com/vllm-project/vllm/commit/a2e6fa7e035ff058fc37fdaaf014707efff2fcf3)

## Installation

Prerequisites: CUDA 12.8 environment, [uv](https://docs.astral.sh/uv/getting-started/installation/), `ninja-build`.

```bash
git clone https://github.com/agencyenterprise/vllm-interp.git
cd vllm-interp
apt-get install -y ninja-build  # if not already installed
uv sync
```

That's it. `uv sync` creates a virtualenv, installs all dependencies (including sae-lens), and builds vllm-interp using precompiled CUDA binaries from the upstream vLLM wheel server.

To build from source instead (slower, requires full CUDA toolkit):

```bash
VLLM_USE_PRECOMPILED=0 uv sync
```

## Usage

### Environment Variables

Set your Hugging Face token to pull Llama models:
```bash
export HF_TOKEN=your_token_here
```

Optional, for attention backend (especially for Gemma models):
```bash
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_FLASH_ATTN_VERSION=3
```

### Example Script

```bash
uv run example_scripts/async_example_llama.py
```

See `example_scripts/async_example_llama.py` for a complete example with batched steering inputs.

## What's Added (vllm-interp)

The following files and modifications are specific to vllm-interp and not part of the original vLLM:

### New Files
- `vllm/model_executor/models/goodfire_sae.py` - Goodfire SAE setup and loading
- `vllm/model_executor/models/llama_models_and_saes.py` - SAE configuration for different Llama models
- `example_scripts/async_example_llama.py` - Example script for running Llama with SAE steering

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

The `intervention_enabled` flag determines the steering layer, while `feature_enabled` determines the layer for reading features. Goodfire SAEs have different steering and feature layers (see `llama_models_and_saes.py`).
