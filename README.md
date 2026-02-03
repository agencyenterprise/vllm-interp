# Installation guide for VLLM-SAE

Prereqs: Environment with CUDA 12.8 as VLLM is compiled with CUDA 12.8. 

- Prepare the environment for insallation, install gcc, and other build packages.

`apt-get update -y`

`apt-get install python3.12 python3.12-venv -y`

`apt-get install python3-dev build-essential -y`

`apt-get install ninja-build cmake jq zip -y`

`apt-get install -y python3.12-dev build-essential ninja-build`

`apt install -y build-essential`

`apt install -y libstdc++-12-dev libc6-dev`

`apt install -y gcc g++ gcc-multilib g++-multilib`

Then, activate the virtualenv, for example via:

`python3.12 -m venv /tmp/vllm_env`

`source /tmp/vllm_env/bin/activate`

We need to install `sae_lens` separately first, as it uses an outdated `numpy` version. For example, via,

`pip install sae_lens==6.13.0`

Then, install specific build packages:

`pip install pip wheel setuptools_scm setuptools --upgrade`

Install the requirements here https://github.com/agencyenterprise/vllm-sae/blob/main/local_reqs/requirements.txt:

`pip install -r local_reqs/requirements.txt`.

Finally, install the vllm as an editable package with pre-built libraries:

`export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/a2e6fa7e035ff058fc37fdaaf014707efff2fcf3/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl`

`pip install --editable .`

The URL from is the pre-built commit that is origin of the vllm-sae fork.

# Main files:
The main files for running Llama or Gemma2 SAEs are as follows:

Llama:

- https://github.com/agencyenterprise/vllm-sae/blob/main/vllm/model_executor/models/llama.py
- https://github.com/agencyenterprise/vllm-sae/blob/main/vllm/model_executor/models/goodfire_sae.py
- https://github.com/agencyenterprise/vllm-sae/blob/main/example_scripts/async_example_llama.py

The first file is the modified llama function, the second one is for the Goodfire SAE setup. The third one is an example file to launch the vLLM engine with llama model and send requests.

The main differences in the llama module is having the following additional functions:

`init_sae_for_rank`: Initializes and loads the SAE for each worker.
'forward_sae`: Runs the intervention/steering using the SAE.

https://github.com/agencyenterprise/vllm-sae/blob/main/vllm/model_executor/models/llama.py#L722

In this line, we extend the `forward` method of the LLM models with two additional arguments:

- `interventions`: A list of intervention inputs with featureID and strength.
- `steer_positions`: This input is generated here: https://github.com/agencyenterprise/vllm-sae/blob/main/vllm/v1/worker/gpu_model_runner.py#L976, which gives the sequences lengths of applying steering as VLLM does not have an explicit batch dimension. This input is not required for sending requests to the engine.

To modify the output, you can read the following part: https://github.com/agencyenterprise/vllm-sae/blob/main/vllm/model_executor/models/llama.py#L509

Specifically, there are two main checks: `intervention_enabled` and `feature_enabled`. The first one is to determine the steering layer, the second one is the layer for reading features. For example, Goodfire SAEs have different steering and feature layers, see https://github.com/agencyenterprise/vllm-sae/blob/main/vllm/model_executor/models/llama_models_and_saes.py for details.

If steering is enabled, we send the hidden states to the `forward_sae` function and apply steering. There are 2 main things that are different compared to the vllm llama setup:

- The forward method of llama is slightly modified to get "sparse" outputs from SAE as it's trained with a different forward method.
The main differences can be found here:
- https://github.com/agencyenterprise/vllm-sae/blob/main/vllm/model_executor/models/llama.py#L356
- https://github.com/agencyenterprise/vllm-sae/blob/main/vllm/model_executor/models/llama.py#L367

Second, we "subtract" the added tensor at the final layer here before returning the final hidden states:

- https://github.com/agencyenterprise/vllm-sae/blob/main/vllm/model_executor/models/llama.py#L536
This generally helps with more stable steering, not yet tested extensively with Gemma2 SAEs.

Finally, see the script for an exmaple with batched steering inputs and printing outputs: https://github.com/agencyenterprise/vllm-sae/blob/main/example_scripts/async_example_llama.py

To run the script, you need 3 env variables:
First two are related to the attention backend, and is required for Gemma models:

- `VLLM_ATTENTION_BACKEND=FLASHINFER`
- `VLLM_FLASH_ATTN_VERSION=3`

Finally, set `HF_TOKEN` to pull the llama model.

