import torch
from pathlib import Path
from huggingface_hub import snapshot_download
from vllm.logger import init_logger

logger = init_logger(__name__)

class SparseAutoEncoder(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.device = device
        self.encoder_linear = torch.nn.Linear(d_in, d_hidden)
        self.decoder_linear = torch.nn.Linear(d_hidden, d_in)
        self.dtype = dtype
        self.to(self.device, self.dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of data using a linear, followed by a ReLU."""
        return torch.nn.functional.relu(self.encoder_linear(x))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a batch of data using a linear."""
        return self.decoder_linear(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """SAE forward pass. Returns the reconstruction and the encoded features."""
        f = self.encode(x)
        return self.decode(f), f

def load_sae(
    sae_id: str,
    sae_filepath: str,
    d_model: int,
    expansion_factor: int,
    device: torch.device = torch.device("cuda"),
):
    # Ensure device is not meta
    if device.type == 'meta':
        device = torch.device("cuda")
        logger.warning("SAE load_sae received meta device, using cuda")
        
    # Download the repo from HuggingFace Hub
    repo_dir = snapshot_download(repo_id=sae_id)
    full_sae_filepath = Path(repo_dir) / sae_filepath

    sae = SparseAutoEncoder(
        d_model,
        d_model * expansion_factor,
        device,
        dtype=torch.bfloat16,
    )
    sae_dict = torch.load(
        full_sae_filepath, weights_only=True, map_location=device
    )
    sae.load_state_dict(sae_dict)
    return sae
