from dataclasses import dataclass
from typing import Union, List, Callable, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt

from utils import torch_utils

InterpolationMethod = Callable[[float], float]


@dataclass
class TextEmbedding:
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    negative_pooled_prompt_embeds: torch.Tensor

    def split(self, split_size: Union[int, List[int]], dim: int = 0) -> List["TextEmbedding"]:
        prompt_embeds_list = torch.split(self.prompt_embeds, split_size, dim)
        negative_prompt_embeds_list = torch.split(self.negative_prompt_embeds, split_size, dim)
        pooled_prompt_embeds_list = torch.split(self.pooled_prompt_embeds, split_size, dim)
        negative_pooled_prompt_embeds_list = torch.split(self.negative_pooled_prompt_embeds, split_size, dim)

        results = []
        for i in range(len(prompt_embeds_list)):
            results.append(TextEmbedding(
                prompt_embeds_list[i],
                negative_prompt_embeds_list[i],
                pooled_prompt_embeds_list[i],
                negative_pooled_prompt_embeds_list[i]
            ))
        return results

    def expand(self, count: int) -> "TextEmbedding":
        return TextEmbedding(
            self.prompt_embeds.expand(count, -1, -1),
            self.negative_prompt_embeds.expand(count, -1, -1),
            self.pooled_prompt_embeds.expand(count, -1),
            self.negative_pooled_prompt_embeds.expand(count, -1)
        )

    @property
    def count(self) -> int:
        return int(self.prompt_embeds.shape[0])


def interpolate_text_encodings(a: TextEmbedding, b: TextEmbedding, steps: int) -> TextEmbedding:
    return TextEmbedding(
        torch_utils.linspace(a.prompt_embeds.squeeze(), b.prompt_embeds.squeeze(), num=steps),
        torch_utils.linspace(a.negative_prompt_embeds.squeeze(), b.negative_prompt_embeds.squeeze(), num=steps),
        torch_utils.linspace(a.pooled_prompt_embeds.squeeze(), b.pooled_prompt_embeds.squeeze(), num=steps),
        torch_utils.linspace(a.negative_pooled_prompt_embeds.squeeze(), b.negative_pooled_prompt_embeds.squeeze(),
                             num=steps),
    )


def interpolate_text_encodings_custom(a: TextEmbedding, b: TextEmbedding, steps: int,
                                      interpolation_method: Optional[InterpolationMethod] = None,
                                      show_plot: bool = False) -> TextEmbedding:
    sample_points = np.linspace(0, 1, steps, dtype=np.float32)

    if interpolation_method is not None:
        sample_points = [interpolation_method(s) for s in sample_points]

    # Generate x-values as indices
    if show_plot:
        x_values = range(len(sample_points))
        plt.plot(x_values, sample_points, marker='x', linestyle='-')
        plt.show()

    weights = torch.tensor(sample_points, device=a.prompt_embeds.device, dtype=torch_utils.get_dtype())

    return TextEmbedding(
        torch_utils.lerp_at(a.prompt_embeds.squeeze(), b.prompt_embeds.squeeze(), weights),
        torch_utils.lerp_at(a.negative_prompt_embeds.squeeze(), b.negative_prompt_embeds.squeeze(), weights),
        torch_utils.lerp_at(a.pooled_prompt_embeds.squeeze(), b.pooled_prompt_embeds.squeeze(), weights),
        torch_utils.lerp_at(a.negative_pooled_prompt_embeds.squeeze(), b.negative_pooled_prompt_embeds.squeeze(),
                            weights),
    )


def cat_text_encodings(encodings: List[TextEmbedding], dim: int = 0) -> TextEmbedding:
    return TextEmbedding(
        torch.cat([e.prompt_embeds for e in encodings], dim=dim),
        torch.cat([e.negative_prompt_embeds for e in encodings], dim=dim),
        torch.cat([e.pooled_prompt_embeds for e in encodings], dim=dim),
        torch.cat([e.negative_pooled_prompt_embeds for e in encodings], dim=dim)
    )
