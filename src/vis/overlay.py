from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2


def heat_to_colormap(heat01: np.ndarray) -> np.ndarray:
    h8 = np.clip(heat01 * 255.0, 0, 255).astype(np.uint8)
    cm = cv2.applyColorMap(h8, cv2.COLORMAP_JET)  # BGR
    return cm


def overlay_heat(rgb: np.ndarray, heat01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    rgb: uint8 HWC RGB
    heat01: float 0~1 HxW
    return: uint8 RGB
    """
    cm_bgr = heat_to_colormap(heat01)
    cm_rgb = cv2.cvtColor(cm_bgr, cv2.COLOR_BGR2RGB)
    out = (rgb.astype(np.float32) * (1 - alpha) + cm_rgb.astype(np.float32) * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_bboxes(rgb: np.ndarray, bboxes: List[Tuple[int,int,int,int]]) -> np.ndarray:
    out = rgb.copy()
    for (x,y,w,h) in bboxes:
        cv2.rectangle(out, (x,y), (x+w, y+h), (255, 255, 255), 2)  # 흰색 박스 (RGB지만 cv2는 그냥 값만 씀)
    return out


def save_grid(images_rgb: List[np.ndarray], out_path: str | Path, cols: int = 4, pad: int = 4) -> None:
    assert len(images_rgb) > 0
    h, w, _ = images_rgb[0].shape
    rows = (len(images_rgb) + cols - 1) // cols

    grid_h = rows * h + (rows - 1) * pad
    grid_w = cols * w + (cols - 1) * pad
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for i, img in enumerate(images_rgb):
        r = i // cols
        c = i % cols
        y0 = r * (h + pad)
        x0 = c * (w + pad)
        canvas[y0:y0+h, x0:x0+w] = img

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))