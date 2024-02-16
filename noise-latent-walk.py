import torch
from diffusers.utils.torch_utils import randn_tensor

from generator.ImageGenerator import ImageGenerator
from models.text_embedding import TextEmbedding
from utils import torch_utils
from utils.media_utils import generate_video_from_pil_images


def main():
    prompt = "a photo of a rabbit"
    step_count = 20

    d0_delta = -0.01
    d1_delta = 0.0
    d2_delta = 0.0
    d3_delta = 0.0

    image_generator = ImageGenerator.from_pretrained("stabilityai/sdxl-turbo")
    pipe = image_generator.pipe

    embeddings = TextEmbedding(*pipe.encode_prompt(prompt, num_images_per_prompt=1))

    torch_device = torch_utils.get_device()

    generator = torch.Generator(torch_device).manual_seed(42)
    latent_shape = embeddings.count, pipe.unet.config.in_channels, pipe.default_sample_size, pipe.default_sample_size
    initial_latents = randn_tensor(latent_shape, generator=generator, device=torch_device, dtype=torch.float32)

    # latent walk
    latent_list = []
    moving_latents = initial_latents.clone().squeeze()
    for i in range(step_count):
        moving_latents[0, :] += d0_delta
        moving_latents[1, :] += d1_delta
        moving_latents[2, :] += d2_delta
        moving_latents[3, :] += d3_delta
        latent_list.append(moving_latents.clone())

    multi_latents = torch.stack(latent_list)

    # expand text embeddings
    multi_embeddings = embeddings.expand(step_count)

    images = image_generator.generate_images(multi_embeddings, latents=multi_latents, batch_size=16)

    generate_video_from_pil_images("output/rabbit-noise-walk.mp4", images, 10)
    print("done")


if __name__ == "__main__":
    main()
