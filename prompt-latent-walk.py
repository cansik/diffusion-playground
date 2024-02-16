from generator.ImageGenerator import ImageGenerator
from generator.LatentPromptWalker import LatentPromptWalker
from utils import curve_utils
from utils.media_utils import generate_video_from_pil_images


def main():
    prompts = [
        "a photograph of a cute cat",
        "a photograph of a cute dog",
        "a photograph of a cute rabbit",
    ]

    generator = ImageGenerator.from_pretrained("stabilityai/sdxl-turbo")
    walker = LatentPromptWalker(generator)

    # create interpolation curve
    curve = curve_utils.symmetric_cubic_bezier(0.25, 0.9)

    def bezier_interpolate(x: float) -> float:
        result = curve.evaluate(float(x))
        return float(result[1][0])

    images = walker.generate_walk(prompts, 10, batch_size=16,
                                  interpolation_method=bezier_interpolate, loop=True)
    generate_video_from_pil_images("output/cat-dog-rabbit-walk.mp4", images, 10)


if __name__ == "__main__":
    main()
