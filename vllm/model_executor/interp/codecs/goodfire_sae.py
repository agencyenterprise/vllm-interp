from __future__ import annotations

import torch

from vllm.model_executor.models.goodfire_sae import SparseAutoEncoder


class GoodfireSaeCodec:
    """Wraps an existing Goodfire SparseAutoEncoder as an InterpCodec."""

    def __init__(self, sae: SparseAutoEncoder) -> None:
        self._sae = sae

    @property
    def num_features(self) -> int:
        return self._sae.d_hidden

    @property
    def d_model(self) -> int:
        return self._sae.d_in

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._sae.encode(x)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self._sae.decode(features)

    def project_to_model_space(self, features: torch.Tensor) -> torch.Tensor:
        """Bias-free projection: matmul with decoder weight transpose."""
        return torch.matmul(features, self._sae.decoder_linear.weight.T)
