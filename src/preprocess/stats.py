from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mlflow
from tqdm import tqdm

from src.utils.io import imread, ensure_dir
from src.preprocess.pipeline import preprocess_image


def gray_mean_std(bgr: np.ndarray) -> tuple[float, float]:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(g.mean()), float(g.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, default="data/raw/index_pcb4_cashew.csv")
    ap.add_argument("--out_dir", type=str, default="artifacts/day01")
    ap.add_argument("--objects", nargs="+", default=["pcb4", "cashew"])
    ap.add_argument("--max_per_object", type=int, default=200)  # 너무 오래 걸리면 줄여

    ap.add_argument("--roi_align", action="store_true")
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--pad_ratio", type=float, default=0.02)
    ap.add_argument("--clip_limit", type=float, default=2.0)
    ap.add_argument("--tile_grid_size", type=int, default=8)

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.index)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # object별로 max_per_object 제한
    parts = []
    for obj in args.objects:
        sub = df[df["object"] == obj].copy()
        if len(sub) == 0:
            raise ValueError(f"object '{obj}' not found in index: {args.index}")
        sub = sub.sample(n=min(args.max_per_object, len(sub)), random_state=args.seed)
        parts.append(sub)
    df_s = pd.concat(parts, axis=0).reset_index(drop=True)

    rows = []
    for _, r in tqdm(df_s.iterrows(), total=len(df_s)):
        img = imread(Path(r["img_path"]))

        b_mean, b_std = gray_mean_std(img)

        proc = preprocess_image(
            img,
            roi_align=bool(args.roi_align),
            clahe=bool(args.clahe),
            pad_ratio=args.pad_ratio,
            clip_limit=args.clip_limit,
            tile_grid_size=args.tile_grid_size,
        )
        a_mean, a_std = gray_mean_std(proc)

        rows.append(
            {
                "object": r["object"],
                "split": r["split"],
                "img_path": r["img_path"],
                "before_mean": b_mean,
                "before_std": b_std,
                "after_mean": a_mean,
                "after_std": a_std,
            }
        )

    st = pd.DataFrame(rows)
    csv_path = out_dir / "day01_preprocess_stats.csv"
    st.to_csv(csv_path, index=False)

    # 집계(전체)
    metrics = {
        "before_mean_avg": float(st["before_mean"].mean()),
        "before_std_avg": float(st["before_std"].mean()),
        "after_mean_avg": float(st["after_mean"].mean()),
        "after_std_avg": float(st["after_std"].mean()),
    }

    # MLflow 기록
    mlflow.set_experiment("day01_preprocess")
    with mlflow.start_run(run_name="stats_on_off"):
        mlflow.log_param("roi_align", bool(args.roi_align))
        mlflow.log_param("clahe", bool(args.clahe))
        mlflow.log_param("pad_ratio", float(args.pad_ratio))
        mlflow.log_param("clip_limit", float(args.clip_limit))
        mlflow.log_param("tile_grid_size", int(args.tile_grid_size))
        mlflow.log_param("max_per_object", int(args.max_per_object))

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.log_artifact(str(csv_path))

    print(f"[OK] stats saved: {csv_path}")
    print("[OK] mlflow logged metrics:", metrics)


if __name__ == "__main__":
    main()
