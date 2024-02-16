import os
import sys
from typing import Optional

import torch


def is_macosx() -> bool:
    return sys.platform == "darwin"


def is_windows() -> bool:
    return os.name == "nt"


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def get_device_string() -> str:
    device = "cpu"

    if sys.platform.startswith("linux") or is_windows():
        if torch.cuda.is_available():
            device = "cuda"
    elif is_macosx():
        if torch.backends.mps.is_available():
            device = "mps"

    return device


def get_device() -> torch.device:
    return torch.device(get_device_string())


def get_variant() -> Optional[str]:
    if is_macosx():
        return None

    return "fp16"


def get_dtype() -> torch.dtype:
    if is_macosx():
        return torch.float32

    return torch.float16


@torch.jit.script
def linspace(start: torch.Tensor, stop: torch.Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps * (stop - start)[None]

    return out


@torch.jit.script
def lerp_at(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    results = [torch.lerp(a, b, lerp_val) for lerp_val in weights]
    return torch.stack(results)
