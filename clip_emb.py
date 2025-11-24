from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import os
from pathlib import Path
import numpy as np

class CLIPEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device='cuda', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.model = CLIPModel.from_pretrained(model_name).to(device).to(dtype)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()  # Important: disable dropout/batchnorm
        
    @torch.no_grad()
    def image_to_embedding(self, image: str | Image.Image) -> torch.Tensor:
        """Extract normalized image embedding."""
        if isinstance(image, str):
            image = Image.open(image)
        
        # Use CLIP processor's built-in preprocessing (deterministic)
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        image_features = self.model.get_image_features(**inputs)
        # Normalize for cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features[0].float()  # Return float32 for stability
    
    @torch.no_grad()
    def text_to_embedding(self, text: str) -> torch.Tensor:
        """Extract normalized text embedding."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        text_features = self.model.get_text_features(**inputs)
        # Normalize for cosine similarity
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features[0].float()  # Return float32 for stability

