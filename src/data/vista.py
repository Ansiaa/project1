from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.io import list_images, ensure_dir


@dataclass
class VistaPaths:
    root: Path

    def class_dir(self, cls: str) -> Path:
        return self.root / cls

    def img_dir_normal(self, cls: str) -> Path:
        return self.class_dir(cls) / "Data" / "Images" / "Normal"

    def img_dir_anom(self, cls: str) -> Path:
        return self.class_dir(cls) / "Data" / "Images" / "Anomaly"

    def mask_dir_anom(self, cls: str) -> Path:
        return self.class_dir(cls) / "Data" / "Masks" / "Anomaly"


def _assert_structure(vp: VistaPaths, cls: str) -> None:
    must = [
        vp.img_dir_normal(cls),
        vp.img_dir_anom(cls),
        vp.mask_dir_anom(cls),
        vp.class_dir(cls) / "image_anno.csv",
    ]
    missing = [str(p) for p in must if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "VisA structure check failed.\n"
            f"class={cls}\nmissing:\n- " + "\n- ".join(missing)
        )


def build_index(root: Path, classes: list[str]) -> pd.DataFrame:
    vp = VistaPaths(root=root)
    rows: list[dict] = []

    for cls in classes:
        _assert_structure(vp, cls)

        normal_imgs = list_images(vp.img_dir_normal(cls))
        anom_imgs = list_images(vp.img_dir_anom(cls))
        anom_masks = {p.stem: p for p in list_images(vp.mask_dir_anom(cls))}

        for p in normal_imgs:
            rows.append(
                {
                    "object": cls,
                    "split": "normal",
                    "label": 0,
                    "img_path": str(p),
                    "mask_path": "",
                }
            )

        for p in anom_imgs:
            mp = anom_masks.get(p.stem, None)
            rows.append(
                {
                    "object": cls,
                    "split": "anomaly",
                    "label": 1,
                    "img_path": str(p),
                    "mask_path": str(mp) if mp else "",
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/VisA")
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    out = Path(args.out)
    ensure_dir(out)

    df = build_index(Path(args.root), args.classes)
    out_csv = out / f"index_{'_'.join(args.classes)}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] index saved: {out_csv} (rows={len(df)})")


if __name__ == "__main__":
    main()
