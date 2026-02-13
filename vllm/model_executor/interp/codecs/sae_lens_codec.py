from __future__ import annotations

import torch
from sae_lens import SAE


class SaeLensCodec:
    """Wraps a sae_lens SAE as an InterpCodec."""

    def __init__(self, sae: SAE) -> None:
        self._sae = sae

    @property
    def num_features(self) -> int:
        return self._sae.cfg.d_sae

    @property
    def d_model(self) -> int:
        return self._sae.cfg.d_in

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._sae.encode(x)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self._sae.decode(features)

    def project_to_model_space(self, features: torch.Tensor) -> torch.Tensor:
        """sae_lens decode is used for projection (matches current gemma behavior)."""
        return self._sae.decode(features)
