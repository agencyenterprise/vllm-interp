import os
from dotenv import load_dotenv

# Load any values from .env first, then set/override critical cache paths.
load_dotenv(override=True)

# Configure Hugging Face caches BEFORE importing libraries that read them.
HF_ROOT = "/workspace/huggingface"
FI_ROOT = os.path.join(HF_ROOT, "flashinfer")
os.environ["HF_HOME"] = HF_ROOT
os.environ["HF_HUB_CACHE"] = os.path.join(HF_ROOT, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_ROOT, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_ROOT, "transformers")
os.environ["FLASHINFER_WORKSPACE_BASE"] = FI_ROOT

# Ensure directories exist
for _p in [
    os.environ["HF_HUB_CACHE"],
    os.environ["HF_DATASETS_CACHE"],
    os.environ["TRANSFORMERS_CACHE"],
    os.environ["FLASHINFER_WORKSPACE_BASE"],
]:
    os.makedirs(_p, exist_ok=True)

print("HF_HOME", os.environ["HF_HOME"])
print("HF_HUB_CACHE", os.environ["HF_HUB_CACHE"])
print("HF_DATASETS_CACHE", os.environ["HF_DATASETS_CACHE"])
print("TRANSFORMERS_CACHE", os.environ["TRANSFORMERS_CACHE"])
print("FLASHINFER_WORKSPACE_BASE", os.environ["FLASHINFER_WORKSPACE_BASE"])

import torch
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import PromptType, TokenInputs, InterventionInputs
from vllm.model_executor.models.llama_models_and_saes import llama_models_and_saes

import asyncio
import time
import uuid


example_inputs = [
    {
        "prompt": [
            {"role": "system", "content": "You are a helpful assistant who should follow the users requests."},
            {"role": "user", "content": "Tell me about some tourist spots in Tokyo in 100 words."}],
        "intervention": [{
            # 8B model
            # "feature_id": 58644,
            # 70B model
            "feature_id": 28612,
            "value": 6.0,
        },
        ],
        "temperature": 0.6,
    },
    {
        "prompt": [
            {"role": "system", "content": "You are a helpful assistant who should follow the users requests."},
            {"role": "user", "content": "About 100 words, please give me some tourist information about Osaka."}],
        "intervention": [{
            # 8B model
            # "feature_id": 17940,
            # 70B model
            "feature_id": 34737,
            "value": 4.0,
        },
        ],
        "temperature": 0.6,
    },
]



#model_str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_str = "meta-llama/Meta-Llama-3.3-70B-Instruct"
model_config = llama_models_and_saes[model_str]


async def gen(engine: AsyncLLMEngine, example_input: dict, tokenizer, id, activations_layer):
    print(example_input["input"])
    print("-"*100)
    if 'intervention' in example_input.keys():
        interventions = InterventionInputs(intervention=example_input["intervention"])
    else:
        interventions = None
    results_generator = engine.generate(
        prompt=example_input["input"],
        sampling_params=SamplingParams(temperature=example_input["temperature"],max_tokens=1024,repetition_penalty=1.0, seed=324),
        request_id=str(id),
        interventions=interventions,
        get_activations_layer=activations_layer,
        # this is false for generating the text
        # this is true for feature readout routes
        is_feature_decode=False,
    )

    final_output = None

    async for request_output in results_generator:
        final_output = request_output
    print(len(final_output.outputs[0].token_ids))

    text_output = [output.text for output in final_output.outputs]
    print("final_output.feature_tensor", final_output.feature_tensor.shape)
    if len(final_output.activations_output) > 0:
        for idx in final_output.activations_output.keys():
            print("activations_tensor layer", idx, final_output.activations_output[idx].shape)
    else:
        print("final_output.activations_output is empty", final_output.activations_output)
    print(text_output[0])
    print("-"*100)
    return text_output[0]

async def main():

    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=model_config["model_id"],
            dtype=torch.bfloat16,
            enforce_eager=True,
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            tensor_parallel_size=torch.cuda.device_count(),
            sae_name=model_config["sae_id"],
            sae_filepath=model_config["sae_filepath"],
            hidden_size=model_config["d_model"],
            sae_expansion_factor=model_config["expansion_factor"],
            steering_layer=model_config["steering_layer"],
            feature_layer=model_config["feature_layer"],
            quantization=model_config["quantization"],
            # need to disable prefix caching for consistent steering 
            enable_prefix_caching=False,
        )
    )
    tokenizer = await engine.get_tokenizer()

    print("Engine initialized")
    st = time.time()

    results = []
    tasks = []
    for j, example_input in enumerate(example_inputs):
        #if isinstance(example_input["prompt"], list) and isinstance(example_input["prompt"][0], dict):
        example_input["prompt_id"] = tokenizer.apply_chat_template(example_input["prompt"])
        # Add the token ids for the assistant header
        example_input["prompt_id"].extend([128000, 128006, 78191, 128007])
        print(len(example_input["prompt_id"]))
        print('example_input["prompt"]', example_input["prompt"])
        example_input["input"] = TokenInputs(prompt_token_ids=example_input["prompt_id"], prompt=example_input["prompt"])
        # set different activations layer for the first and second example
        if j == 0:
            activations_layer = [13, 15]
        else:
            activations_layer = []
        for i in range(3):

            tasks.append(asyncio.create_task(gen(engine, example_input, tokenizer, uuid.uuid4(), activations_layer)))
    print('length of tasks', len(tasks))
    res = [await task for task in tasks]
    for r in res:
        results.append(r)
    print(len(results))
    for r in results:
        print(r)
        print("-"*100)


    print("Async vLLM inference time: ", time.time() - st)

if __name__ == "__main__":
    asyncio.run(main())
