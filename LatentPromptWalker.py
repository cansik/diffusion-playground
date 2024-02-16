from itertools import tee
from typing import List, Optional

from PIL import Image

from ImageGenerator import ImageGenerator
from models.text_embedding import InterpolationMethod, TextEncoderResult, interpolate_text_encodings_custom, \
    cat_text_encodings


class LatentPromptWalker:
    def __init__(self, generator: ImageGenerator):
        self.generator = generator

    def generate_walk(self, prompts: List[str],
                      interpolation_steps: int,
                      interpolation_method: Optional[InterpolationMethod] = None,
                      **generator_args) -> List[Image]:
        # generate prompt encodings
        encodings = [TextEncoderResult(*self.generator.pipe.encode_prompt(p, num_images_per_prompt=1))
                     for p in prompts]

        # create list (e0 to e1, e1 to e2 and so on)
        it1, it2 = tee(encodings)
        next(it2, None)  # Advance the second iterator by one position
        encoding_tuples = list(zip(it1, it2))

        # create interpolation between
        interpolated_embeddings = []
        for encodings_a, encodings_b in encoding_tuples:
            interpolated_step_encodings = interpolate_text_encodings_custom(encodings_a, encodings_b,
                                                                            interpolation_steps,
                                                                            interpolation_method=interpolation_method)
            interpolated_embeddings.append(interpolated_step_encodings)
        stacked_encodings = cat_text_encodings(interpolated_embeddings)

        # generate images
        images = self.generator.generate_images(stacked_encodings, **generator_args)
        return images
