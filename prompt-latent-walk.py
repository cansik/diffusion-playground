from ImageGenerator import ImageGenerator
from LatentPromptWalker import LatentPromptWalker
from utils.media_utils import generate_video_from_pil_images


def main():
    prompts = [
        "a photograph of a cute cat",
        "a photograph of a cute dog",
        "a photograph of a cute rabbit",
    ]

    generator = ImageGenerator.from_pretrained("stabilityai/sdxl-turbo")
    walker = LatentPromptWalker(generator)

    images = walker.generate_walk(prompts, 10, batch_size=16)
    generate_video_from_pil_images("output/cat-dog-rabbit-walk.mp4", images, 10)


if __name__ == "__main__":
    main()
