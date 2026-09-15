from __future__ import annotations

import numpy as np


def threshold_based_weighting(image: np.ndarray, t1: float, t2: float):
    """Separate water-like and bone-like components using linear ramps."""
    if t2 <= t1:
        raise ValueError("t2 must be greater than t1")
    image = np.asarray(image, dtype=np.float32)
    w_bone = np.clip((image - t1) / (t2 - t1), 0.0, 1.0)
    bone = w_bone * image
    w_water = np.clip((t2 - image) / (t2 - t1), 0.0, 1.0)
    water = w_water * image
    return water, bone
