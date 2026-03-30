"""Test pre-computed vector steering with Llama 3.3 70B.

Picks a few wikipedia vectors by index, steers the model with them,
and checks whether the output relates to the corresponding topic.
"""

import asyncio
import json
import torch

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokenInputs, InterventionInputs


VECTORS_PATH = "/home/user/wikipedia_vectors/wikipedia_contrastive_dataset_l48.pt"
METADATA_PATH = "/home/user/wikipedia_vectors/wikipedia_metadata_l48.json"
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"

# Test a handful of vectors at different scales
TEST_CASES = [
    {"vector_id": 5, "scale": 15.0},   # "Demographics of Canada"
    {"vector_id": 36, "scale": 15.0},   # "Brazilian Armed Forces"
    {"vector_id": 82, "scale": 15.0},   # "Aramaic"
]

PROMPT_MESSAGES = [{"role": "user", "content": "Tell me something interesting."}]


async def main():
    # Load labels
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    titles = metadata["titles"]

    print(f"Loaded metadata: {len(titles)} titles")
    for tc in TEST_CASES:
        print(f"  vector_id={tc['vector_id']}: {titles[tc['vector_id']]}")

    # Initialize engine — no SAE needed, just vector steering
    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=MODEL_ID,
            dtype=torch.bfloat16,
            enforce_eager=True,
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            tensor_parallel_size=2,
            quantization="fp8",
            # Vector steering config — apply at layer 33 (vectors extracted at layer 48)
            steering_layer=33,
            steering_vectors_path=VECTORS_PATH,
            # No SAE needed
            sae_name=None,
            # Disable prefix caching for consistent steering
            enable_prefix_caching=False,
        )
    )

    tokenizer = await engine.get_tokenizer()

    # Run baseline (no steering)
    print("\n" + "=" * 80)
    print("BASELINE (no steering)")
    print("=" * 80)
    prompt_ids = tokenizer.apply_chat_template(PROMPT_MESSAGES)
    token_inputs = TokenInputs(prompt_token_ids=prompt_ids, prompt=PROMPT_MESSAGES)

    final = None
    async for output in engine.generate(
        prompt=token_inputs,
        sampling_params=SamplingParams(temperature=0.6, max_tokens=200, seed=42),
        request_id="baseline",
        interventions=None,
        is_feature_decode=False,
    ):
        final = output
    print(f"Output: {final.outputs[0].text[:500]}")

    # Run with each vector
    for tc in TEST_CASES:
        vid = tc["vector_id"]
        scale = tc["scale"]
        title = titles[vid]
        print("\n" + "=" * 80)
        print(f"VECTOR {vid}: '{title}' (scale={scale})")
        print("=" * 80)

        interventions = InterventionInputs(
            intervention=[{"vector_id": vid, "value": scale}]
        )

        prompt_ids = tokenizer.apply_chat_template(PROMPT_MESSAGES)
        token_inputs = TokenInputs(prompt_token_ids=prompt_ids, prompt=PROMPT_MESSAGES)

        final = None
        async for output in engine.generate(
            prompt=token_inputs,
            sampling_params=SamplingParams(temperature=0.6, max_tokens=200, seed=42),
            request_id=f"vector_{vid}",
            interventions=interventions,
            is_feature_decode=False,
        ):
            final = output
        print(f"Output: {final.outputs[0].text[:500]}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
