import gradio as gr
from PIL import Image

from generator.ImageGenerator import ImageGenerator


def main() -> None:
    model = ImageGenerator.from_pretrained()
    pipe = model.pipe

    def generate(prompt: str | None, seed: int | None, steps: float, temperature: float) -> Image.Image:
        prompt = (prompt or "").strip()
        if not prompt:
            raise gr.Error("Enter a prompt to generate an image.")
        if seed is None:
            raise gr.Error("Enter a seed, such as 42.")

        rng = model.make_generator(seed)
        latents = model.sample_latents(rng) * temperature
        return pipe(
            prompt=prompt,
            num_inference_steps=int(steps),
            guidance_scale=0.0,
            generator=rng,
            latents=latents,
        ).images[0]

    max_seed = 2**53 - 1  # Largest integer the browser can represent exactly.
    demo = gr.Interface(
        fn=generate,
        inputs=[
            gr.Textbox(label="Prompt", lines=3, placeholder="Describe the image you want to generate."),
            gr.Number(label="Seed", value=42, precision=0, minimum=-max_seed, maximum=max_seed),
            gr.Slider(label="Denoising steps", value=1, minimum=1, maximum=10, step=1),
            gr.Slider(label="Temperature", value=1.0, minimum=0.0, maximum=2.0, step=0.01,
                      info="Scales the initial noise. Default is 1.0."),
        ],
        outputs=gr.Image(label="Generated image", format="png", buttons=["download", "fullscreen"]),
        title="Image Generator",
        submit_btn="Generate",
        flagging_mode="never",
    )
    demo.launch()


if __name__ == "__main__":
    main()
