"""Example: using DirectionSetCodec for steering with arbitrary direction vectors.

This script creates a small set of random directions, saves them to a .pt file,
and then launches vLLM with the direction_set codec for steering and readout.

Usage:
    python example_scripts/direction_set_example.py
"""

import asyncio
import os
import tempfile
import uuid

import torch

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import InterventionInputs, TokenInputs

# ---------------------------------------------------------------------------
# 1. Create a small direction set and save to a temp .pt file
# ---------------------------------------------------------------------------
NUM_DIRECTIONS = 64
D_MODEL = 2048  # must match the model's hidden_size
STEERING_LAYER = 12
FEATURE_LAYER = 12
MODEL_ID = "google/gemma-2-2b-it"

# Random unit-normalized directions
directions = torch.randn(NUM_DIRECTIONS, D_MODEL, dtype=torch.bfloat16)
directions = directions / directions.norm(dim=-1, keepdim=True)

tmpdir = tempfile.mkdtemp()
directions_path = os.path.join(tmpdir, "directions.pt")
torch.save(directions, directions_path)
print(f"Saved {NUM_DIRECTIONS} directions to {directions_path}")

# ---------------------------------------------------------------------------
# 2. Set up example inputs with steering via direction indices
# ---------------------------------------------------------------------------
example_inputs = [
    {
        "prompt": [
            {"role": "user", "content": "Tell me about the weather today."},
        ],
        # Steer direction index 5 with value 3.0
        "intervention": [{"feature_id": 5, "value": 3.0}],
        "temperature": 0.7,
    },
    {
        "prompt": [
            {"role": "user", "content": "What is machine learning?"},
        ],
        # No steering for this request
        "temperature": 0.7,
    },
]


async def gen(
    engine: AsyncLLMEngine,
    example_input: dict,
    request_id: str,
    activations_layer: list[int],
) -> str:
    interventions = None
    if "intervention" in example_input:
        interventions = InterventionInputs(
            intervention=example_input["intervention"]
        )

    results_generator = engine.generate(
        prompt=example_input["input"],
        sampling_params=SamplingParams(
            temperature=example_input["temperature"],
            max_tokens=256,
        ),
        request_id=request_id,
        interventions=interventions,
        get_activations_layer=activations_layer,
        is_feature_decode=True,
    )

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    text_output = final_output.outputs[0].text
    if final_output.feature_tensor is not None:
        print(
            f"  feature_tensor shape: {final_output.feature_tensor.shape}"
        )
    else:
        print("  feature_tensor: None")

    if final_output.activations_output:
        for idx, tensor in final_output.activations_output.items():
            print(f"  activations layer {idx}: {tensor.shape}")

    print(f"  output: {text_output[:120]}...")
    return text_output


async def main() -> None:
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")

    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=MODEL_ID,
            dtype=torch.bfloat16,
            enforce_eager=True,
            gpu_memory_utilization=0.90,
            max_model_len=2048,
            tensor_parallel_size=1,
            # Direction set codec configuration
            codec_type="direction_set",
            directions_filepath=directions_path,
            steering_layer=STEERING_LAYER,
            feature_layer=FEATURE_LAYER,
            enable_prefix_caching=False,
        )
    )
    tokenizer = await engine.get_tokenizer()
    print("Engine initialized with direction_set codec")

    tasks = []
    for j, example_input in enumerate(example_inputs):
        prompt_ids = tokenizer.apply_chat_template(example_input["prompt"])
        example_input["input"] = TokenInputs(
            prompt_token_ids=prompt_ids, prompt=example_input["prompt"]
        )
        activations_layer = [STEERING_LAYER] if j == 0 else []
        tasks.append(
            asyncio.create_task(
                gen(engine, example_input, str(uuid.uuid4()), activations_layer)
            )
        )

    results = [await t for t in tasks]
    print("\n--- All results ---")
    for i, r in enumerate(results):
        print(f"[{i}] {r[:200]}")


if __name__ == "__main__":
    asyncio.run(main())
