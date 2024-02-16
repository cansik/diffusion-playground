from pathlib import Path
from typing import List

import ffmpegio
import numpy as np
from PIL import Image

from utils.path_utils import PathLike


def generate_video_from_pil_images(video_path: PathLike, images: List[Image], frame_rate: int = 30) -> Path:
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    numpy_images = np.array([np.asarray(i) for i in images])
    ffmpegio.video.write(str(video_path), frame_rate, numpy_images, overwrite=True, pix_fmt="yuv420p")

    return video_path
