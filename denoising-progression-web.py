import gradio as gr
import torch
from PIL import Image

from generator.ImageGenerator import ImageGenerator


def main() -> None:
    model = ImageGenerator.from_pretrained()
    pipe = model.pipe
    vae = pipe.vae
    if vae.config.force_upcast:
        vae.to(dtype=torch.float32)

    @torch.inference_mode()
    def generate(
        prompt: str | None,
        seed: int | None,
        steps: float,
        temperature: float,
        progress=gr.Progress(),
    ) -> list[tuple[Image.Image, str]]:
        prompt = (prompt or "").strip()
        if not prompt:
            raise gr.Error("Enter a prompt to generate an image.")
        if seed is None:
            raise gr.Error("Enter a seed, such as 42.")

        step_count = int(steps)
        work_total = 2 * step_count + 1
        rng = model.make_generator(seed)
        noise = model.sample_latents(rng) * temperature
        snapshots = []

        def capture_step(_pipeline, step, _timestep, tensors):
            snapshots.append(tensors["latents"].clone())
            progress((step + 1, work_total), desc="Denoising")
            return {}

        pipe(
            prompt=prompt,
            num_inference_steps=step_count,
            guidance_scale=0.0,
            generator=rng,
            latents=noise,
            output_type="latent",
            callback_on_step_end=capture_step,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        # Match the scheduler's scaling of the input noise before step one.
        initial_latents = noise * pipe.scheduler.init_noise_sigma
        captions = ["Step 0 · Initial noise", *(f"Step {n}" for n in range(1, step_count)),
                    f"Step {step_count} · Final image"]
        frames = []
        for index, (latents, caption) in enumerate(zip([initial_latents, *snapshots], captions, strict=True)):
            progress((step_count + index, work_total), desc="Rendering frames")
            # SDXL Turbo latents use the VAE's scaling factor.
            decoded = vae.decode(latents.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
            image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
            frames.append((image, caption))
        return frames

    max_seed = 2**53 - 1
    demo = gr.Interface(
        fn=generate,
        inputs=[
            gr.Textbox(label="Prompt", lines=3, placeholder="Describe the image you want to generate."),
            gr.Number(label="Seed", value=42, precision=0, minimum=-max_seed, maximum=max_seed),
            gr.Slider(label="Denoising steps", value=4, minimum=1, maximum=4, step=1),
            gr.Slider(label="Temperature", value=1.0, minimum=0.0, maximum=2.0, step=0.01,
                      info="Scales the initial noise. Default is 1.0."),
        ],
        outputs=gr.Gallery(label="Denoising frames", format="png", columns=3, object_fit="contain",
                           buttons=["download", "download_all", "fullscreen"]),
        title="Denoising Progression",
        description="Follow one SDXL Turbo run from initial noise to the final image.",
        submit_btn="Generate",
        flagging_mode="never",
    )
    demo.launch()


if __name__ == "__main__":
    main()
