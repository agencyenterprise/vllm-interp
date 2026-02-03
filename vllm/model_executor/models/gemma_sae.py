import torch
from vllm.logger import init_logger
from sae_lens import SAE
logger = init_logger(__name__)


def load_sae(
    sae_release: str,
    sae_id: str,
    device: torch.device = torch.device("cuda"),
):
    # Ensure device is not meta
    if device.type == 'meta':
        device = torch.device("cuda")
        logger.warning("SAE load_sae received meta device, using cuda")
        
    # load the sae from sae_lens, put it on cpu first
    sae, _, _ = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device="cpu",
    )
    # then put it on the correct device with bfloat16 precision
    sae = sae.to(dtype=torch.bfloat16, device=device)

    return sae
