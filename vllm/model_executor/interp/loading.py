from __future__ import annotations

from typing import Optional

import torch

from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.interp.codec import InterpCodec

logger = init_logger(__name__)


def init_codec_for_rank(
    codec_type: str,
    *,
    # Goodfire SAE params
    sae_name: Optional[str] = None,
    sae_filepath: Optional[str] = None,
    hidden_size: Optional[int] = None,
    sae_expansion_factor: Optional[int] = None,
    # sae_lens params
    sae_release: Optional[str] = None,
    sae_id: Optional[str] = None,
    # direction_set params
    directions_filepath: Optional[str] = None,
) -> dict[int, InterpCodec]:
    """Generic factory replacing per-model init_sae_for_rank functions.

    Returns a dict mapping TP rank -> InterpCodec.
    """
    tp_rank = get_tp_group().local_rank
    device = torch.device(f"cuda:{tp_rank}")

    codec: InterpCodec

    if codec_type == "goodfire_sae":
        from vllm.model_executor.interp.codecs.goodfire_sae import GoodfireSaeCodec
        from vllm.model_executor.models.goodfire_sae import load_sae

        assert sae_name is not None
        assert sae_filepath is not None
        assert hidden_size is not None
        assert sae_expansion_factor is not None

        sae = load_sae(sae_name, sae_filepath, hidden_size, sae_expansion_factor, device)
        new_dict: dict[str, torch.Tensor] = {}
        for key, item in sae.state_dict().items():
            new_dict[key.replace("module._orig_mod.", "")] = item
        sae.load_state_dict(new_dict)
        codec = GoodfireSaeCodec(sae)

    elif codec_type == "sae_lens":
        from vllm.model_executor.interp.codecs.sae_lens_codec import SaeLensCodec
        from vllm.model_executor.models.gemma_sae import load_sae

        assert sae_release is not None
        assert sae_id is not None

        sae = load_sae(sae_release, sae_id, device)
        new_dict = {}
        for key, item in sae.state_dict().items():
            new_dict[key.replace("module._orig_mod.", "")] = item
        sae.load_state_dict(new_dict)
        codec = SaeLensCodec(sae)

    elif codec_type == "direction_set":
        from vllm.model_executor.interp.codecs.direction_set import DirectionSetCodec

        assert directions_filepath is not None
        codec = DirectionSetCodec.from_file(directions_filepath, device)

    else:
        raise ValueError(f"Unknown codec_type: {codec_type!r}")

    logger.info(
        "Loaded %s codec for TP rank %d: %d features, d_model=%d",
        codec_type, tp_rank, codec.num_features, codec.d_model,
    )
    return {tp_rank: codec}
