import torch
import os
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import PromptType, TokenInputs, InterventionInputs
from vllm.model_executor.models.gemma_models_and_saes import gemma_models_and_saes
from dotenv import load_dotenv

import asyncio
import time
import uuid

load_dotenv()

# Flash attention is not supported for gemma-2-2b-it, need ot use flash infer
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

example_inputs = [
    {
        "prompt": [
            # {"role": "system", "content": "You are a helpful assistant who should follow the users requests."},
            {"role": "user", "content": "Explain me how to make a strong password."}],
        "intervention": [{
            "feature_id": 15343,
            "value": 210.0,
        },
        ],
        "temperature": 0.7,
    },
    {
        "prompt": [
            # {"role": "system", "content": "You are a helpful assistant who should follow the users requests."},
            {"role": "user", "content": "About 100 words, please give me some tourist information about Osaka."}],
        # "intervention": [{
        #     "feature_id": 17940,
        #     "value": 1.85,
        # },
        # ],
        "temperature": 0.7,
    },
]

# ger model setup
model_str = "google/gemma-2-2b-it-l16"
model_config = gemma_models_and_saes[model_str]


async def gen(engine: AsyncLLMEngine, example_input: dict, tokenizer, id, activations_layer):
    print(example_input["input"])
    print("-"*100)
    if 'intervention' in example_input.keys():
        interventions = InterventionInputs(intervention=example_input["intervention"])
    else:
        interventions = None
    results_generator = engine.generate(
        prompt=example_input["input"],
        sampling_params=SamplingParams(temperature=example_input["temperature"],max_tokens=1024,repetition_penalty=1.0),
        request_id=str(id),
        interventions=interventions,
        is_feature_decode=True,
        get_activations_layer=activations_layer,
    )

    final_output = None

    async for request_output in results_generator:
        final_output = request_output
    print(len(final_output.outputs[0].token_ids))

    text_output = [output.text for output in final_output.outputs]
    if final_output.feature_tensor is not None:
        print("final_output.feature_tensor", final_output.feature_tensor.shape)
    else:
        print("final_output.feature_tensor is None")
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
            # set the model
            model=model_config["model_id"],
            dtype=torch.bfloat16,
            enforce_eager=True,
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            tensor_parallel_size=model_config["tensor_parallel_size"],
            # set the sae release, id and steering and feature layer
            sae_release=model_config["sae_release"],
            sae_id=model_config["sae_id"],
            steering_layer=model_config["steering_layer"],
            feature_layer=model_config["feature_layer"],
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
        #example_input["prompt_id"].extend([128000, 128006, 78191, 128007])
        print(len(example_input["prompt_id"]))
        print('example_input["prompt"]', example_input["prompt"])
        example_input["input"] = TokenInputs(prompt_token_ids=example_input["prompt_id"], prompt=example_input["prompt"])
        if j == 0:
            activations_layer = [10, 12]
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
