import uuid
from pathlib import Path
from typing import Optional

import gradio as gr

from generator.ImageGenerator import ImageGenerator
from generator.LatentPromptWalker import LatentPromptWalker
from utils import curve_utils
from utils.media_utils import generate_video_from_pil_images

walker: Optional[LatentPromptWalker] = None


def generate(prompts_text: str, style: str, interpolation_steps: int, fps: int, loop: bool, seed: int):
    # prepare prompts
    prompts = [p.strip() for p in prompts_text.split("\n") if p.strip() != ""]

    # add style to prompts
    style = style.strip()
    if style != "":
        prompts = [f"{p}, {style}" for p in prompts]

    if len(prompts) < 2:
        raise gr.Error("Enter at least two prompts to interpolate between.")

    # create interpolation curve
    curve = curve_utils.symmetric_cubic_bezier(0.25, 0.9)

    def bezier_interpolate(x: float) -> float:
        result = curve.evaluate(float(x))
        return float(result[1][0])

    images = walker.generate_walk(prompts, int(interpolation_steps),
                                  interpolation_method=bezier_interpolate, loop=loop,
                                  seed=int(seed))

    video_path = Path("output").joinpath(f"{str(uuid.uuid1().hex)}.mp4")
    generate_video_from_pil_images(video_path, images, int(fps))
    return str(video_path)


def main():
    demo = gr.Interface(
        generate,
        [
            gr.Textbox(label="Prompts", placeholder="Write a prompt per line.", lines=5, max_lines=5),
            gr.Textbox(label="Style", placeholder="The style will be added to each prompt."),
            gr.Slider(label="Interpolation Steps", value=10, minimum=1, maximum=30, step=1),
            gr.Slider(label="FPS", value=10, minimum=1, maximum=30, step=1),
            gr.Checkbox(label="Loop"),
            gr.Number(label="Seed", value=42, step=1),
        ],
        [
            gr.Video(label="Latent Walk")
        ],
        title="Latent Walk Creator",
        flagging_mode="never",
        show_progress="minimal",
    )

    global walker
    walker = LatentPromptWalker(ImageGenerator.from_pretrained())
    demo.launch()


if __name__ == "__main__":
    main()
