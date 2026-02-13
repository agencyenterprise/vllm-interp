from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class DirectionSetCodec:
    """InterpCodec backed by a set of direction vectors.

    Holds a ``[n_directions, d_model]`` matrix.  ``encode`` computes dot
    products, ``decode`` computes the weighted sum.
    """

    def __init__(self, directions: torch.Tensor) -> None:
        """
        Args:
            directions: Tensor of shape ``[n_directions, d_model]``.
                        Rows should be unit-normalized for meaningful
                        dot-product activations, but this is not enforced.
        """
        if directions.ndim != 2:
            raise ValueError(
                f"directions must be 2-D [n_directions, d_model], got shape {directions.shape}"
            )
        self._directions = directions  # [n_dirs, d_model]

    @classmethod
    def from_file(cls, path: str, device: torch.device) -> DirectionSetCodec:
        """Load directions from a ``.pt`` or ``.npy`` file."""
        p = Path(path)
        if p.suffix == ".pt":
            directions = torch.load(p, map_location=device, weights_only=True)
        elif p.suffix == ".npy":
            arr = np.load(p)
            directions = torch.from_numpy(arr).to(device)
        else:
            raise ValueError(
                f"Unsupported file format {p.suffix!r}. Use .pt or .npy."
            )
        return cls(directions.to(dtype=torch.bfloat16, device=device))

    @property
    def num_features(self) -> int:
        return self._directions.shape[0]

    @property
    def d_model(self) -> int:
        return self._directions.shape[1]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Dot products: [seq_len, d_model] @ [d_model, n_dirs] -> [seq_len, n_dirs]."""
        return x @ self._directions.T

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Weighted sum: [seq_len, n_dirs] @ [n_dirs, d_model] -> [seq_len, d_model]."""
        return features @ self._directions

    def project_to_model_space(self, features: torch.Tensor) -> torch.Tensor:
        """Same as decode for direction sets (no bias to exclude)."""
        return self.decode(features)
