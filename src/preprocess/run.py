from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils.io import imread, imwrite, ensure_dir
from src.preprocess.pipeline import preprocess_image


def sample_per_object(df: pd.DataFrame, obj: str, n: int, seed: int = 42) -> pd.DataFrame:
    """각 object에서 n장 샘플링(가능하면 normal/anomaly 반반)."""
    sub = df[df["object"] == obj].copy()
    if len(sub) == 0:
        return sub

    normal = sub[sub["split"] == "normal"]
    anomaly = sub[sub["split"] == "anomaly"]

    n_n = n // 2
    n_a = n - n_n

    picked = []
    if len(normal) > 0:
        picked.append(normal.sample(n=min(n_n, len(normal)), random_state=seed))
    if len(anomaly) > 0:
        picked.append(anomaly.sample(n=min(n_a, len(anomaly)), random_state=seed))

    out = pd.concat(picked, axis=0) if picked else sub.head(0)

    # 부족하면 전체에서 추가로 채움
    if len(out) < n:
        remain = sub.drop(out.index, errors="ignore")
        if len(remain) > 0:
            extra = remain.sample(n=min(n - len(out), len(remain)), random_state=seed + 1)
            out = pd.concat([out, extra], axis=0)

    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def save_grid_pairs(pairs: list[tuple[np.ndarray, np.ndarray]], out_png: Path, max_rows: int = 8) -> None:
    """(원본, 전처리) 쌍을 2열 그리드로 저장."""
    pairs = pairs[:max_rows]
    rows = len(pairs)
    fig = plt.figure(figsize=(10, 3 * rows))

    for i, (before_bgr, after_bgr) in enumerate(pairs, start=1):
        ax1 = fig.add_subplot(rows, 2, (i - 1) * 2 + 1)
        ax2 = fig.add_subplot(rows, 2, (i - 1) * 2 + 2)

        ax1.imshow(cv2.cvtColor(before_bgr, cv2.COLOR_BGR2RGB))
        ax1.set_title("before", fontsize=10)
        ax1.axis("off")

        ax2.imshow(cv2.cvtColor(after_bgr, cv2.COLOR_BGR2RGB))
        ax2.set_title("after", fontsize=10)
        ax2.axis("off")

    ensure_dir(out_png.parent)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def resolve_grid_path(save_grid: Path, obj: str, objects: list[str]) -> Path:
    """
    - save_grid에 '{obj}' 템플릿이 있으면 치환: ablation_grid_{obj}.png
    - 템플릿이 없고 objects가 2개 이상이면 stem 뒤에 _{obj}를 붙임
    - objects가 1개면 원래 save_grid 그대로
    """
    s = str(save_grid)
    if "{obj}" in s:
        return Path(s.format(obj=obj))

    if len(objects) > 1:
        return save_grid.with_name(f"{save_grid.stem}_{obj}{save_grid.suffix}")

    return save_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, default="data/raw/index_pcb4_cashew.csv")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--save_grid", type=str, required=True)
    ap.add_argument("--objects", nargs="+", default=["pcb4", "cashew"])
    ap.add_argument("--n_per_object", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--roi_align", action="store_true")
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--pad_ratio", type=float, default=0.02)
    ap.add_argument("--clip_limit", type=float, default=2.0)
    ap.add_argument("--tile_grid_size", type=int, default=8)

    args = ap.parse_args()

    df = pd.read_csv(args.index)
    out_root = Path(args.out)
    grid_png = Path(args.save_grid)

    random.seed(args.seed)
    np.random.seed(args.seed)

    chosen = []
    for obj in args.objects:
        part = sample_per_object(df, obj, n=args.n_per_object, seed=args.seed)
        if len(part) == 0:
            raise ValueError(f"object '{obj}' not found in index: {args.index}")
        chosen.append(part)
    df_s = pd.concat(chosen, axis=0).reset_index(drop=True)

    pairs_for_grid_by_obj: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {obj: [] for obj in args.objects}
    saved = 0

    for _, r in tqdm(df_s.iterrows(), total=len(df_s)):
        img_path = Path(r["img_path"])
        before = imread(img_path)

        after = preprocess_image(
            before,
            roi_align=bool(args.roi_align),
            clahe=bool(args.clahe),
            pad_ratio=args.pad_ratio,
            clip_limit=args.clip_limit,
            tile_grid_size=args.tile_grid_size,
        )

        rel = Path(r["object"]) / r["split"] / img_path.name
        out_path = out_root / rel
        imwrite(out_path, after)
        saved += 1

        obj = str(r["object"])
        lst = pairs_for_grid_by_obj.get(obj)
        if lst is not None and len(lst) < 8:
            lst.append((before, after))

    for obj in args.objects:
        pairs = pairs_for_grid_by_obj.get(obj, [])
        if len(pairs) == 0:
            continue
        out_grid = resolve_grid_path(grid_png, obj=obj, objects=args.objects)
        save_grid_pairs(pairs, out_grid, max_rows=8)
        print(f"[OK] grid saved: {out_grid}")

    print(f"[OK] saved processed images: {saved} -> {out_root}")


if __name__ == "__main__":
    main()
