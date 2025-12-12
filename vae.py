import torch
import torchvision.transforms as T
import PIL.Image as Image
import PIL.Image
from diffusers.models import AutoencoderKL

class VaeEncoder:
    def __init__(self, model_name="stabilityai/sdxl-vae", dtype=torch.bfloat16, device="cuda"):
        """
        Initialize the VAE encoder/decoder.

        Args:
            model_name (str): Hugging Face model identifier for the VAE.
            dtype (torch.dtype): Data type for the model weights (default: bfloat16).
            device (str): Preferred device ('cuda' or 'cpu').
        """
        self.device = device if torch.cuda.is_available() or device != "cuda" else "cpu"
        self.dtype = dtype
        self.vae = AutoencoderKL.from_pretrained(model_name).to(self.dtype).to(self.device)
        self.vae.eval()  # Set to evaluation mode

    def _to_tensor(self, image):
        """Convert PIL image or tensor to normalized 4D tensor."""
        if isinstance(image, Image.Image):
            image = T.ToTensor()(image.convert("RGB"))
        if image.ndim == 3:
            image = image.unsqueeze(0)
        return image

    @torch.no_grad()
    def encode(self, image : PIL.Image.Image | torch.Tensor):
        """
        Encode an image (PIL or tensor) into the VAE latent space.

        Args:
            image (PIL.Image.Image or torch.Tensor): Input image.
                If tensor: shape (C, H, W) or (B, C, H, W), values in [0, 1].

        Returns:
            torch.Tensor: Latent representation, shape (B, 4, H//8, W//8).
        """
        image = self._to_tensor(image)
        im_dev = image.device
        image = image.to(self.device, dtype=self.dtype)

        posterior = self.vae.encode(image)
        latents = posterior.latent_dist.sample() * self.vae.config.scaling_factor
        return latents.to(im_dev)

    @torch.no_grad()
    def decode(self, latents : torch.Tensor):
        """
        Decode latents back to image space.

        Args:
            latents (torch.Tensor): Latent tensor, typically shape (B, 4, H, W).

        Returns:
            torch.Tensor: Reconstructed image, shape (B, 3, H*8, W*8), values in [0, 1].
        """
        dev = latents.device
        latents = latents.to(self.device, dtype=self.dtype)
        recon = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        return recon.clamp(0, 1).to(dev).float()

    def to(self, device):
        """Move VAE to specified device."""
        self.device = device
        self.vae = self.vae.to(device)
        return self