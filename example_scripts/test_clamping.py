"""
Script to test SAE latent clamping by setting specific features to 0.
This compares model output with and without clamping interventions.

INSTRUCTIONS:
1. First run gather_firing_latents.py to identify which latents fire for your themed text
2. Replace LATENTS_TO_CLAMP below with the top firing latent IDs
3. Run this script to see the comparison
"""
import torch
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import PromptType, TokenInputs, InterventionInputs
from vllm.model_executor.models.llama_models_and_saes import llama_models_and_saes
from dotenv import load_dotenv
import asyncio

load_dotenv()

# Use the smaller 8B model for testing
model_str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_config = llama_models_and_saes[model_str]

model_config = dict(model_config)  # Make a copy
model_config["steering_layer"] = model_config["feature_layer"]
print(f"SAE trained on layer {model_config['feature_layer']}")
print(f"Using layer {model_config['feature_layer']} for both steering and feature reading")

# Same themed prompt used in gather_firing_latents.py
prompt_text = "Tell me about different types of trees and their characteristics."

# Update these after running gather_firing_latents.py with the corrected layer configuration
LATENTS_TO_CLAMP = [1215, 43309, 40915, 41290, 59404, 52104, 23091, 48542, 28760, 39459, 36708, 25784, 30967, 62066, 59905, 26046, 63558, 59823, 24511, 42886, 56440, 47255, 26272, 58162, 41901, 20634, 13472, 48183, 39122, 63640, 2663, 21051, 18269, 3392, 37346, 25622, 34289, 27356, 7470, 16106, 28814, 46334, 12491, 27869, 57090, 33239, 57190, 63594, 22312, 62826, 39843, 52635, 5369, 40570, 48627, 52017, 57491, 2900, 11534, 4268, 4925, 1653, 42139, 46830, 21056, 49886, 482, 46227, 3089, 52634, 47985, 52917, 7220, 50553, 17101, 1177, 38126, 8011, 48187, 7912, 12215, 1556, 50630, 54361, 50675, 56734, 39503, 25977, 42435, 53271, 11443, 27673, 24411, 51639, 43081, 54399, 47530, 61466, 64436, 5307]


async def generate_text(engine, tokenizer, prompt_text, interventions=None, request_id="test", print_debug=False):
    """Helper function to generate text with optional interventions."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text}
    ]
    prompt_token_ids = tokenizer.apply_chat_template(messages)
    # Add assistant header tokens for Llama 3 style
    prompt_token_ids.extend([128000, 128006, 78191, 128007])

    token_inputs = TokenInputs(prompt_token_ids=prompt_token_ids, prompt=messages)

    if print_debug and interventions:
        print(f"[DEBUG] Applying {len(interventions['intervention'])} interventions:")
        for interv in interventions['intervention']:
            print(f"  - Feature {interv['feature_id']}: {interv['mode']} value={interv['value']}")

    results_generator = engine.generate(
        prompt=token_inputs,
        sampling_params=SamplingParams(
            temperature=0.6,
            max_tokens=200,
            repetition_penalty=1.0,
            seed=324  # Same seed for reproducibility
        ),
        request_id=request_id,
        interventions=interventions,
        get_activations_layer=[],
        is_feature_decode=False,  # Just generating text, not reading features
    )

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    return final_output.outputs[0].text


async def main():
    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=model_config["model_id"],
            dtype=torch.bfloat16,
            enforce_eager=True,
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            tensor_parallel_size=model_config["tensor_parallel_size"],
            sae_name=model_config["sae_id"],
            sae_filepath=model_config["sae_filepath"],
            hidden_size=model_config["d_model"],
            sae_expansion_factor=model_config["expansion_factor"],
            steering_layer=model_config["steering_layer"],
            feature_layer=model_config["feature_layer"],
            quantization=model_config["quantization"],
            enable_prefix_caching=False,
        )
    )
    tokenizer = await engine.get_tokenizer()

    print("Engine initialized")
    print(f"Prompt: {prompt_text}")
    print(f"Clamping {len(LATENTS_TO_CLAMP)} latents to 0: {LATENTS_TO_CLAMP}")
    print("=" * 100)

    # Generate WITHOUT intervention (baseline)
    print("\n1. BASELINE (No intervention):")
    print("-" * 100)
    baseline_text = await generate_text(
        engine, tokenizer, prompt_text,
        interventions=None,
        request_id="baseline"
    )
    print(baseline_text)
    print("-" * 100)

    # Generate WITH clamping intervention
    print("\n2. WITH CLAMPING (Selected latents clamped to 0):")
    print("-" * 100)

    # Create intervention specification: clamp each selected latent to 0
    clamping_interventions = InterventionInputs(
        intervention=[
            {"feature_id": latent_id, "value": 0.0, "mode": "clamp"}
            for latent_id in LATENTS_TO_CLAMP
        ]
    )

    clamped_text = await generate_text(
        engine, tokenizer, prompt_text,
        interventions=clamping_interventions,
        request_id="clamped",
        print_debug=True
    )
    print(clamped_text)
    print("-" * 100)

    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)
    print(f"Baseline length: {len(baseline_text)} chars")
    print(f"Clamped length:  {len(clamped_text)} chars")
    print(f"\nClamped features: {LATENTS_TO_CLAMP}")


if __name__ == "__main__":
    asyncio.run(main())
