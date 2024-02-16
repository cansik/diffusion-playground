from typing import Optional

import gradio as gr
import torch
from diffusers.utils.torch_utils import randn_tensor

from generator.ImageGenerator import ImageGenerator
from utils import torch_utils

image_generator: Optional[ImageGenerator] = None


def generate(prompt: str, steps: int, guidance_scale: float, seed: int,
             d0: float, d1: float, d2: float, d3: float):
    generator = torch.manual_seed(int(seed))

    pipe = image_generator.pipe
    latent_shape = 1, pipe.unet.config.in_channels, pipe.default_sample_size, pipe.default_sample_size
    initial_latents = randn_tensor(latent_shape, generator=generator,
                                   device=image_generator.device, dtype=torch_utils.get_dtype())

    # edit latents with slider
    initial_latents[0, 0, :] += d0
    initial_latents[0, 1, :] += d1
    initial_latents[0, 2, :] += d2
    initial_latents[0, 3, :] += d3

    outputs = pipe(generator=generator, prompt=prompt, num_inference_steps=int(steps), guidance_scale=guidance_scale,
                   latents=initial_latents)

    image = outputs.images[0]
    return image


if __name__ == "__main__":
    sr = 0.5

    demo = gr.Interface(
        generate,
        [
            gr.Textbox(label="Prompt"),
            gr.Slider(label="Steps", value=1, minimum=1, maximum=10, step=1),
            gr.Number(label="Guidance Scale", value=0.0),
            gr.Number(label="Seed", value=42, step=1),

            gr.Slider(label="Delta D0", value=0.0, step=0.001, minimum=-sr, maximum=sr),
            gr.Slider(label="Delta D1", value=0.0, step=0.001, minimum=-sr, maximum=sr),
            gr.Slider(label="Delta D2", value=0.0, step=0.001, minimum=-sr, maximum=sr),
            gr.Slider(label="Delta D3", value=0.0, step=0.001, minimum=-sr, maximum=sr),
        ],
        [
            gr.Image(label="Image")
        ],
        live=True,
        allow_flagging='never'
    )

    # hack for progressbar
    demo.dependencies[0]["show_progress"] = False  # the hack

    # load image generator
    image_generator = ImageGenerator.from_pretrained("stabilityai/sdxl-turbo")

    # launch web
    demo.launch(server_name="0.0.0.0")
