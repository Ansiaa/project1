from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

IMG_EXITS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

def ensure_dir(p: str | Path) -> Path :
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def list_images(folder : str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    files = [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXITS]
    return sorted(files)

def imread(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img
def imwrite(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise IOError(f"Failed to write image: {path}")
