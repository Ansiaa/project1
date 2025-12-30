from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import cv2

from src.data.datasets import VisAProcessedIndex, imread_rgb, resize_to, normalize_0_1
from src.postprocess.cc import components_from_mask, otsu_threshold, make_mask_from_heat
from src.vis.overlay import overlay_heat, draw_bboxes, save_grid


def heat_from_cv_baseline(x01: np.ndarray) -> np.ndarray:
    """
    torch 없이도 파이프라인 확인용.
    idea: 원본 - gaussian blur 차이(고주파/텍스처) -> heat
    x01: HWC float 0~1
    return: HxW float 0~1
    """
    x8 = np.clip(x01 * 255.0, 0, 255).astype(np.uint8)
    g = cv2.GaussianBlur(x8, (0,0), 2.0)
    diff = cv2.absdiff(x8, g).astype(np.float32) / 255.0
    heat = diff.mean(axis=2)
    # robust normalize
    lo, hi = np.percentile(heat, 5), np.percentile(heat, 99)
    heat = (heat - lo) / (hi - lo + 1e-6)
    return np.clip(heat, 0.0, 1.0)


def heat_from_ae_torch(x01: np.ndarray, model, device: str) -> np.ndarray:
    """
    x01: HWC float 0~1
    return: HxW float 0~1
    """
    import torch  # torch 필요
    xt = torch.from_numpy(np.transpose(x01, (2,0,1))).unsqueeze(0).float().to(device)  # 1,3,H,W
    with torch.no_grad():
        y = model(xt).clamp(0,1)
    x_np = xt.squeeze(0).detach().cpu().numpy()   # 3,H,W
    y_np = y.squeeze(0).detach().cpu().numpy()
    err = np.abs(x_np - y_np).mean(axis=0)        # H,W
    # robust normalize
    lo, hi = np.percentile(err, 5), np.percentile(err, 99)
    heat = (err - lo) / (hi - lo + 1e-6)
    return np.clip(heat, 0.0, 1.0)


def load_ae_model(ckpt_path: Path, device: str):
    import torch
    from src.models.ae import ConvAE

    map_loc = torch.device(device) if device == "cuda" else torch.device("cpu")
    ckpt = torch.load(str(ckpt_path), map_location=map_loc)

    model = ConvAE().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/processed")
    ap.add_argument("--classes", nargs="+", default=["pcb4", "cashew"])
    ap.add_argument("--ckpt_dir", type=str, default="artifacts/day02/models")
    ap.add_argument("--out", type=str, default="artifacts/day02/preds")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--mode", choices=["ae", "cv"], default="cv")  # torch 안 되면 cv로
    ap.add_argument("--min_area", type=int, default=20)
    ap.add_argument("--save_grid", type=str, default="artifacts/day02/ae_overlay_grid.png")
    ap.add_argument("--grid_n_per_class", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.45)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    import torch
    device = "cuda" if (args.mode == "ae" and torch.cuda.is_available()) else "cpu"
    print("[ae_infer] device:", device)
    model_by_cls = {}

    if args.mode == "ae":
        # torch 필요
        import torch  # noqa
        for cls in args.classes:
            ckpt = Path(args.ckpt_dir) / f"ae_{cls}.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"ckpt not found: {ckpt}")
            model_by_cls[cls] = load_ae_model(ckpt, device)

    # 전체 샘플 인덱스
    idx = VisAProcessedIndex(args.data, args.classes).build()

    grid_imgs: List[np.ndarray] = []
    grid_count: Dict[str, int] = {c: 0 for c in args.classes}

    for s in idx:
        # 이미지 로드/리사이즈/정규화
        rgb = imread_rgb(s.path)               # uint8 RGB
        rgb = resize_to(rgb, args.img_size)
        x01 = normalize_0_1(rgb)               # float 0~1 HWC

        # heatmap
        if args.mode == "ae":
            heat = heat_from_ae_torch(x01, model_by_cls[s.cls], device)
        else:
            heat = heat_from_cv_baseline(x01)

        score = float(np.max(heat))  # 간단 스코어

        # threshold/mask/components
        thr = otsu_threshold(heat)
        mask = make_mask_from_heat(heat, thr)
        comps = components_from_mask(mask, min_area=args.min_area)
        H, W = args.img_size, args.img_size
        comps = [c for c in comps if not (c.bbox[0] == 0 and c.bbox[1] == 0 and c.bbox[2] == W and c.bbox[3] == H)]
        # overlay + bbox
        ov = overlay_heat(rgb, heat, alpha=args.alpha)
        bboxes = [c.bbox for c in comps[:5]]
        ov2 = draw_bboxes(ov, bboxes)

        # 저장 경로
        label_name = "good" if s.label == 0 else "anomaly"
        rel_name = f"{label_name}_{s.path.stem}"

        save_dir = out_root / s.cls / label_name
        save_dir.mkdir(parents=True, exist_ok=True)

        heat_path = save_dir / f"{rel_name}_heat.png"
        ov_path = save_dir / f"{rel_name}_overlay.png"
        json_path = save_dir / f"{rel_name}.json"

        # heat 저장(0~255)
        h8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(heat_path), h8)
        cv2.imwrite(str(ov_path), cv2.cvtColor(ov2, cv2.COLOR_RGB2BGR))

        j: Dict[str, Any] = {
            "path": str(s.path),
            "class": s.cls,
            "label": int(s.label),
            "mode": args.mode,
            "img_size": int(args.img_size),
            "score_max_heat": score,
            "threshold_otsu": float(thr),
            "components": [
                {
                    "area_px": int(c.area_px),
                    "bbox_xywh": [int(v) for v in c.bbox],
                    "centroid_xy": [float(c.centroid[0]), float(c.centroid[1])],
                } for c in comps
            ],
        }
        json_path.write_text(json.dumps(j, ensure_ascii=False, indent=2), encoding="utf-8")

        # grid용 이미지 수집(클래스별 N장)
        if grid_count[s.cls] < args.grid_n_per_class:
            grid_imgs.append(ov2)
            grid_count[s.cls] += 1

    # grid 저장
    save_grid(grid_imgs, args.save_grid, cols=4)
    print("[ae_infer] saved grid:", args.save_grid)
    print("[ae_infer] saved preds to:", str(out_root))


if __name__ == "__main__":
    main()