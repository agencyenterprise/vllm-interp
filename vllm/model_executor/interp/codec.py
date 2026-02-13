from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class InterpCodec(Protocol):
    """Protocol for interpretability codecs (SAEs, direction sets, etc.).

    Each codec can encode hidden states into a feature space, decode features
    back to model space, and project feature-space tensors to model space
    without bias (for stable steering subtraction).
    """

    @property
    def num_features(self) -> int: ...

    @property
    def d_model(self) -> int: ...

    def encode(self, x: torch.Tensor) -> torch.Tensor: ...

    def decode(self, features: torch.Tensor) -> torch.Tensor: ...

    def project_to_model_space(self, features: torch.Tensor) -> torch.Tensor:
        """Project feature-space tensor to model space (bias-free).

        Used for the add_tensor projection in steering subtraction.
        Goodfire SAE: matmul(features, decoder_weight.T)
        sae_lens SAE: sae.decode(features)
        DirectionSet: matmul(features, directions)
        """
        ...


@dataclass
class SteeringResult:
    """Result of apply_steering."""

    hidden_states: torch.Tensor
    add_tensor_projected: torch.Tensor | None
