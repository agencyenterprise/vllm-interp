"""
Script to gather which SAE latents fire for a given themed text.
This helps identify which features to clamp in subsequent experiments.
"""
import torch
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import PromptType, TokenInputs, InterventionInputs
from vllm.model_executor.models.llama_models_and_saes import llama_models_and_saes
from dotenv import load_dotenv
import asyncio
import numpy as np

load_dotenv()

# Use the smaller 8B model for testing
model_str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_config = llama_models_and_saes[model_str]

model_config = dict(model_config)  # Make a copy
model_config["steering_layer"] = model_config["feature_layer"]
print(f"SAE trained on layer {model_config['feature_layer']}")
print(f"Using layer {model_config['feature_layer']} for both steering and feature reading")

# Themed prompt about trees
prompt_text = "Tell me about different types of trees and their characteristics."

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
    print("-" * 100)

    # Prepare the prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text}
    ]
    prompt_token_ids = tokenizer.apply_chat_template(messages)
    # Add assistant header tokens for Llama 3 style
    prompt_token_ids.extend([128000, 128006, 78191, 128007])

    token_inputs = TokenInputs(prompt_token_ids=prompt_token_ids, prompt=messages)

    # Run generation with feature decoding enabled
    results_generator = engine.generate(
        prompt=token_inputs,
        sampling_params=SamplingParams(
            temperature=0.6,
            max_tokens=200,
            repetition_penalty=1.0,
            seed=324
        ),
        request_id="feature_gather",
        interventions=None,  # No interventions, just observe
        get_activations_layer=[],  # Not using activations, just features
        is_feature_decode=True,  # Enable feature readout
    )

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    # Get the generated text
    text_output = final_output.outputs[0].text
    print("Generated text:")
    print(text_output)
    print("-" * 100)

    # Get the feature tensor
    feature_tensor = final_output.feature_tensor
    print(f"\nFeature tensor shape: {feature_tensor.shape}")

    # Aggregate features across all tokens (sum of absolute values)
    # Shape is [num_tokens, num_features]
    feature_activations = torch.abs(feature_tensor).sum(dim=0)  # Sum across tokens

    # Get top firing features
    top_k = 100
    top_values, top_indices = torch.topk(feature_activations, k=top_k)

    print(f"Top {top_k} feature IDs for clamping experiment:")
    print(top_indices[:top_k].tolist())

if __name__ == "__main__":
    asyncio.run(main())
