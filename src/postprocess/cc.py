from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2


@dataclass
class Component:
    area_px: int
    bbox: Tuple[int, int, int, int]  # x,y,w,h
    centroid: Tuple[float, float]


def components_from_mask(mask: np.ndarray, min_area: int = 20) -> List[Component]:
    """
    mask: uint8 (0/255) or bool/0-1
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps: List[Component] = []
    for i in range(1, num):  # 0=background
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        cx, cy = centroids[i]
        comps.append(Component(area_px=int(area), bbox=(int(x), int(y), int(w), int(h)), centroid=(float(cx), float(cy))))
    # 큰 것부터
    comps.sort(key=lambda c: c.area_px, reverse=True)
    return comps


def otsu_threshold(heat01: np.ndarray) -> float:
    """
    heat01: float 0~1
    return: threshold in 0~1
    """
    h8 = np.clip(heat01 * 255.0, 0, 255).astype(np.uint8)
    retval, _ = cv2.threshold(h8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(retval) / 255.0


def make_mask_from_heat(heat01: np.ndarray, thr: float) -> np.ndarray:
    h8 = np.clip(heat01 * 255.0, 0, 255).astype(np.uint8)
    m = (h8 >= int(thr * 255)).astype(np.uint8) * 255
    return m