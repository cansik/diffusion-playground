import torch


def get_device_string() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def linspace(start: torch.Tensor, stop: torch.Tensor, num: int) -> torch.Tensor:
    """Return num evenly spaced samples with shape [num, *start.shape]."""
    weights = torch.linspace(0, 1, num, device=start.device, dtype=start.dtype)
    return lerp_at(start, stop, weights)


def lerp_at(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.lerp(a, b, weights.reshape(-1, *([1] * a.ndim)))
