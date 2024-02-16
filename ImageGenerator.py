from typing import Optional, List, Any

import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image
from tqdm import tqdm

from models.text_embedding import TextEncoderResult
from utils import torch_utils


class ImageGenerator:
    """
    The image generator is able to create images using an existing image diffusion pipeline.
    """

    def __init__(self, pipeline: Any, device: Optional[str] = None):
        self.device_str = device if device is not None else torch_utils.get_device_string()
        self.device = torch.device(self.device_str)

        device = torch_utils.get_device_string()

        self.pipe = pipeline
        self.pipe.to(device)

        # optimize for speed
        if not torch_utils.is_macosx():
            self.pipe.unet.set_default_attn_processor()
            self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)

        self.pipe.set_progress_bar_config(disable=True)

    @staticmethod
    def from_pretrained(name: str,
                        device: Optional[str] = None,
                        torch_dtype: Optional[torch.dtype] = None) -> "ImageGenerator":
        if torch_dtype is None:
            torch_dtype = torch_utils.get_dtype()

        pipe = AutoPipelineForText2Image.from_pretrained(name, torch_dtype=torch_dtype)
        return ImageGenerator(pipe, device)

    def generate_images(self, embeddings: TextEncoderResult, batch_size: int = 4,
                        guidance_scale: float = 0.0, num_inference_steps: int = 1,
                        seed: int = 42, latents: Optional[torch.Tensor] = None) -> List[Image]:
        batches = embeddings.split(batch_size)

        if latents is not None:
            latent_batches = latents.split(batch_size)

        images = []

        pbar = tqdm(total=embeddings.count, desc="generating")
        for i, batch in enumerate(batches):
            generator = [torch.Generator(self.device).manual_seed(seed) for _ in range(batch.count)]

            outputs = self.pipe(
                prompt_embeds=batch.prompt_embeds,
                pooled_prompt_embeds=batch.pooled_prompt_embeds,
                negative_prompt_embeds=batch.negative_prompt_embeds,
                negative_pooled_prompt_embeds=batch.negative_pooled_prompt_embeds,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                latents=None if latents is None else latent_batches[i]
            )

            images += outputs.images
            pbar.update(len(outputs.images))

        return images
