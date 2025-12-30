
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_img(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


@dataclass
class Sample:
    path: Path
    label: int          # good=0, anomaly=1
    cls: str            # pcb4/cashew
    split: str          # train/test (있으면), 없으면 unknown


class VisAProcessedIndex:
    """
    data/processed 아래를 스캔해서 good/anomaly 샘플 경로를 만든다.
    폴더 구조가 다를 수 있으니 규칙은 '폴더명에 good이 있으면 0, 아니면 anomaly=1' 같은 보수적 방식.
    """
    def __init__(self, root: str | Path, classes: List[str]):
        self.root = Path(root)
        self.classes = classes

    def build(self) -> List[Sample]:
        samples: List[Sample] = []
        for cls in self.classes:
            cls_root = self.root / cls
            if not cls_root.exists():
                # 폴더가 없으면 전체에서 cls 문자열이 들어간 경로를 찾는 fallback
                cand = [p for p in self.root.rglob("*") if p.is_dir() and p.name == cls]
                if cand:
                    cls_root = cand[0]

            if not cls_root.exists():
                raise FileNotFoundError(f"[datasets] class folder not found: {cls_root}")

            # 이미지 파일 전부 스캔
            for p in cls_root.rglob("*"):
                if not p.is_file() or not _is_img(p):
                    continue

                parts = {x.lower() for x in p.parts}
                # 라벨 규칙(우선순위): good/ok/normal이면 0, 아니면 1
                if "good" in parts or "ok" in parts or "normal" in parts:
                    label = 0
                else:
                    label = 1

                # split 추정
                split = "train" if "train" in parts else ("test" if "test" in parts else "unknown")

                samples.append(Sample(path=p, label=label, cls=cls, split=split))

        # 안정적으로 섞기 전, 통계 체크
        return samples


def imread_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cv2.imread failed: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_to(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def normalize_0_1(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


if __name__ == "__main__":
    # quick smoke test
    idx = VisAProcessedIndex("data/processed", ["pcb4", "cashew"]).build()
    print("[datasets] total:", len(idx))
    by = {}
    for s in idx:
        key = (s.cls, s.label, s.split)
        by[key] = by.get(key, 0) + 1
    for k, v in sorted(by.items()):
        print(k, v)

    # 샘플 1장 로딩 테스트
    if idx:
        x = imread_rgb(idx[0].path)
        x = resize_to(x, 256)
        x = normalize_0_1(x)
        print("[datasets] sample shape:", x.shape, x.dtype, x.min(), x.max())
