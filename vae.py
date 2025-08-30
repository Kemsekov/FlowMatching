import torch
import torchvision.transforms as T
import PIL.Image as Image
from diffusers.models import AutoencoderKL

# model that we gonna use
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").bfloat16()

def encode(image,device='cuda') -> torch.Tensor:
    if torch.cuda.is_available() and device=='cuda':
        vae.cuda()
    else:
        vae.cpu()
        device='cpu'
    dtype = list(vae.parameters())[0].dtype
        
    if isinstance(image,Image.Image):
        image = T.ToTensor()(image)
    if image.ndim==3: image=image[None,:]
    image=image.to(device).to(dtype)
    # Encode to latent space
    with torch.no_grad():
        posterior = vae.encode(image)           # Diagonal Gaussian
        latents = posterior.latent_dist.sample() * vae.config.scaling_factor
    return latents.cpu()
    # print("Latent shape:", latents.shape)  # e.g. [1, 4, 64, 64]

def decode(latents,device='cuda'):
    if torch.cuda.is_available() and device=='cuda':
        vae.cuda()
    else:
        vae.cpu()
        device='cpu'
    dtype = list(vae.parameters())[0].dtype
    # Decode back to image
    with torch.no_grad():
        recon = vae.decode(latents.to(device).to(dtype) / vae.config.scaling_factor).sample.clip(0,1)

    # print("Reconstruction shape:", recon.shape)  # [1, 3, 512, 512]
    return recon.float()