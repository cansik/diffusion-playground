from typing import Optional, List, Any

import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm

from models.text_embedding import TextEmbedding
from utils import torch_utils


SDXL_TURBO = "stabilityai/sdxl-turbo"


class ImageGenerator:
    """
    The image generator is able to create images using an existing image diffusion pipeline.
    """

    def __init__(self, pipeline: Any, device: Optional[str] = None):
        self.device = torch.device(device or torch_utils.get_device_string())
        self.pipe = pipeline.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    @staticmethod
    def from_pretrained(name: str = SDXL_TURBO,
                        device: Optional[str] = None,
                        torch_dtype: Optional[torch.dtype] = None) -> "ImageGenerator":
        device = device or torch_utils.get_device_string()
        if torch_dtype is None:
            torch_dtype = torch.float32 if torch.device(device).type == "cpu" else torch.float16

        variant = "fp16" if name == SDXL_TURBO else None
        pipe = AutoPipelineForText2Image.from_pretrained(name, dtype=torch_dtype, variant=variant)
        return ImageGenerator(pipe, device)

    def make_generator(self, seed: int) -> torch.Generator:
        # Diffusers can sample on the CPU and move the noise to MPS.
        device = self.device if self.device.type == "cuda" else "cpu"
        return torch.Generator(device).manual_seed(seed)

    def sample_latents(self, generator: torch.Generator) -> torch.Tensor:
        size = self.pipe.default_sample_size
        return randn_tensor(
            (1, self.pipe.unet.config.in_channels, size, size),
            generator=generator,
            device=self.device,
            dtype=self.pipe.unet.dtype,
        )

    def generate_images(self, embeddings: TextEmbedding, batch_size: Optional[int] = None,
                        guidance_scale: float = 0.0, num_inference_steps: int = 1,
                        seed: int = 42, latents: Optional[torch.Tensor] = None) -> List[Image]:
        if batch_size is None:
            batch_size = 4 if self.device.type == "cuda" else 1
        batches = embeddings.split(batch_size)

        if latents is not None:
            latent_batches = latents.split(batch_size)

        images = []

        pbar = tqdm(total=embeddings.count, desc="generating")
        for i, batch in enumerate(batches):
            generator = [self.make_generator(seed) for _ in range(batch.count)]

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

        pbar.close()
        return images
