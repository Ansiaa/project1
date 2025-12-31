import argparse, json, random
from pathlib import Path

import cv2
import numpy as np

from anomalib.deploy import TorchInferencer


def overlay_from_map(img_bgr, amap01, alpha=0.45):
    h, w = img_bgr.shape[:2]

    if amap01.shape[0] != h or amap01.shape[1] != w:
        amap01 = cv2.resize(amap01.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    hm = (np.clip(amap01, 0, 1) * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)

    ov = cv2.addWeighted(img_bgr, 1 - alpha, hm_color, alpha, 0)
    return hm, hm_color, ov, amap01


def cc_from_map(amap01, thr_mode="otsu", fixed=0.5, topk=3,
                min_area_px=2000, pad=0, morph=0):
    """
    min_area_px: connected component 최소 면적(작은 노이즈 제거)
    pad: 가장자리 n픽셀 무시(경계/프레임 과검출 줄이기)
    morph: 0이면 off, 3~7 추천(OPEN+CLOSE로 노이즈/구멍 정리)
    """
    u8 = (np.clip(amap01, 0, 1) * 255).astype(np.uint8)

    # (A) 경계 제외
    if pad and pad > 0:
        u8[:pad, :] = 0
        u8[-pad:, :] = 0
        u8[:, :pad] = 0
        u8[:, -pad:] = 0

    # threshold
    if thr_mode == "otsu":
        _, mask = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr_val = None
    else:
        thr_val = int(max(0, min(255, fixed * 255)))
        _, mask = cv2.threshold(u8, thr_val, 255, cv2.THRESH_BINARY)

    # (B) morphology 정리
    if morph and morph > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if min_area_px and area < min_area_px:
            continue
        blobs.append({"bbox": [int(x), int(y), int(w), int(h)], "area_px": int(area)})

    blobs.sort(key=lambda b: b["area_px"], reverse=True)
    return mask, blobs[:topk], thr_val


def to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_pt", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cls", default="cashew")
    ap.add_argument("--n_good", type=int, default=10)
    ap.add_argument("--n_defect", type=int, default=5)
    ap.add_argument("--out", default="artifacts/day03")
    ap.add_argument("--thr_mode", choices=["otsu", "fixed"], default="otsu")
    ap.add_argument("--fixed", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.45)

    # ★ 품질 튜닝 옵션
    ap.add_argument("--min_area", type=int, default=2000)
    ap.add_argument("--pad", type=int, default=15)
    ap.add_argument("--morph", type=int, default=5)

    args = ap.parse_args()
    random.seed(0)

    root = Path(args.root) / args.cls
    good_dir = root / "good_test"
    defect_dir = root / "defect"

    valid_ext = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
    good = sorted([p for p in good_dir.rglob("*") if p.suffix.lower() in valid_ext])
    defect = sorted([p for p in defect_dir.rglob("*") if p.suffix.lower() in valid_ext])

    picks = []
    if good:
        picks += random.sample(good, min(args.n_good, len(good)))
    if defect:
        picks += random.sample(defect, min(args.n_defect, len(defect)))

    out = Path(args.out)
    (out / "overlays" / args.cls).mkdir(parents=True, exist_ok=True)
    (out / "heatmaps" / args.cls).mkdir(parents=True, exist_ok=True)
    (out / "masks" / args.cls).mkdir(parents=True, exist_ok=True)
    (out / "preds" / args.cls).mkdir(parents=True, exist_ok=True)

    infer = TorchInferencer(path=args.model_pt, device="auto")

    tiles = []
    target_wh = None

    for idx, p in enumerate(picks):
        pred = infer.predict(str(p))

        amap = getattr(pred, "anomaly_map", None)
        score = getattr(pred, "pred_score", None)

        amap = to_numpy(amap)
        if amap.ndim == 3:
            amap = amap[0]
        amap01 = (amap - float(amap.min())) / (float(amap.max() - amap.min()) + 1e-8)

        score_val = None
        if score is not None:
            s = to_numpy(score)
            try:
                score_val = float(s[0])
            except Exception:
                try:
                    score_val = float(s)
                except Exception:
                    score_val = None

        img = cv2.imread(str(p))
        if img is None:
            print("WARN: failed to read image:", p)
            continue

        hm_u8, hm_color, ov, amap01r = overlay_from_map(img, amap01, alpha=args.alpha)

        # ★ 튜닝 파라미터 적용
        mask, blobs, thr_val = cc_from_map(
            amap01r,
            thr_mode=args.thr_mode,
            fixed=args.fixed,
            topk=3,
            min_area_px=args.min_area,
            pad=args.pad,
            morph=args.morph,
        )

        p_str = str(p).replace("\\", "/").lower()
        src_label = "good" if "/good_test/" in p_str or p_str.endswith("/good_test") else "defect"
        src_stem = Path(p).stem
        score_tag = "na" if score_val is None else f"{score_val:.3f}".replace(".", "p")

        stem = f"{idx:03d}_{args.cls}_{src_label}_{src_stem}_s{score_tag}"
        cv2.imwrite(str(out / "heatmaps" / args.cls / f"{stem}.png"), hm_color)
        cv2.imwrite(str(out / "overlays" / args.cls / f"{stem}.png"), ov)
        cv2.imwrite(str(out / "masks" / args.cls / f"{stem}.png"), mask)

        rec = {
            "image_path": str(p),
            "src_label": src_label,
            "src_stem": src_stem,
            "out_stem": stem,
            "pred_score": score_val,
            "threshold_mode": args.thr_mode,
            "threshold_fixed": args.fixed if args.thr_mode == "fixed" else None,
            "threshold_value_u8": thr_val,
            "post_min_area_px": args.min_area,
            "post_pad": args.pad,
            "post_morph": args.morph,
            "blobs": blobs,
        }
        with open(out / "preds" / args.cls / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

        if target_wh is None:
            target_wh = (ov.shape[1], ov.shape[0])
        if (ov.shape[1], ov.shape[0]) != target_wh:
            ov = cv2.resize(ov, target_wh, interpolation=cv2.INTER_AREA)
        tiles.append(ov)

    if tiles:
        cols = 4
        h, w = tiles[0].shape[:2]
        rows = int(np.ceil(len(tiles) / cols))
        grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for i, im in enumerate(tiles):
            r, c = divmod(i, cols)
            grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = im

        grid_path = out / f"patchcore_overlay_grid_{args.cls}.png"
        cv2.imwrite(str(grid_path), grid)
        print("GRID:", grid_path)
    else:
        print("WARN: no tiles generated (check picks / image read).")


if __name__ == "__main__":
    main()
