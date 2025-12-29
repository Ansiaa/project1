from __future__ import annotations
import numpy as np

from src.preprocess.roi import roi_crop_largest_component
from src.preprocess.clahe import apply_clahe_lab


def preprocess_image(
    bgr: np.ndarray,
    roi_align: bool = True,
    clahe: bool = True,
    pad_ratio: float = 0.02,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    out = bgr
    if roi_align:
        out = roi_crop_largest_component(out, pad_ratio=pad_ratio)
    if clahe:
        out = apply_clahe_lab(out, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    return out
