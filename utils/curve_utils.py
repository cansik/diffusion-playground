from typing import Tuple

import bezier
import numpy as np


def ease_in_quad(f: float) -> float:
    return f * f


def ease_out_quad(f: float) -> float:
    return f * (2 - f)


def ease_in_cubic(f: float) -> float:
    return f * f * f


def ease_out_cubic(f: float) -> float:
    t = f - 1
    return t * t * t + 1


def cubic_bezier(p0: Tuple[float, float], p1: Tuple[float, float]) -> bezier.Curve:
    nodes = np.asfortranarray([
        [0.0, p0[0], p1[0], 1.0],
        [0.0, p0[1], p1[1], 1.0],
    ])
    curve = bezier.Curve(nodes, degree=3)
    return curve


def symmetric_cubic_bezier(dx: float, dy: float) -> bezier.Curve:
    return cubic_bezier(
        (dx, dy),
        (1.0 - dx, 1.0 - dy)
    )
