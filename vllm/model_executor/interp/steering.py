from __future__ import annotations

import torch

from vllm.inputs import InterventionInputs
from vllm.model_executor.interp.codec import InterpCodec, SteeringResult


def apply_steering(
    codec: InterpCodec,
    hidden_states: torch.Tensor,
    intervention_list: list[InterventionInputs],
    steer_positions: list[int],
) -> SteeringResult:
    """Shared steering logic extracted from forward_sae in llama.py / gemma2.py.

    1. Encode hidden states to features
    2. Compute reconstruction error
    3. Apply add/clamp interventions per request
    4. Decode modified features + error
    5. Project add_tensor to model space
    """
    features = codec.encode(hidden_states)

    add_tensor = torch.zeros_like(
        features, dtype=torch.bfloat16, device=features.device
    )

    reconstructed_acts = codec.decode(features)
    error = hidden_states - reconstructed_acts

    for i, intervention in enumerate(intervention_list):
        if not intervention:
            continue
        pos_beg, pos_end = steer_positions[i], steer_positions[i + 1]
        for steer in intervention["intervention"]:
            mode = steer.get("mode", "add")
            if mode == "clamp":
                features[pos_beg:pos_end, steer["feature_id"]] = steer["value"]
            else:  # mode == "add"
                add_tensor[pos_beg:pos_end, steer["feature_id"]] += steer["value"]

    features += add_tensor

    return SteeringResult(
        hidden_states=codec.decode(features) + error,
        add_tensor_projected=codec.project_to_model_space(add_tensor),
    )
