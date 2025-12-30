from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from src.models.ae import ConvAE
from src.data.datasets import VisAProcessedIndex, imread_rgb, resize_to, normalize_0_1


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ImgListDataset(Dataset):
    def __init__(self, paths, img_size: int):
        self.paths = paths
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        x = imread_rgb(p)
        x = resize_to(x, self.img_size)
        x = normalize_0_1(x)              # HWC float32 0~1
        x = np.transpose(x, (2, 0, 1))    # CHW
        return torch.from_numpy(x), str(p)


def try_mlflow():
    try:
        import mlflow  # noqa
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/processed")
    ap.add_argument("--classes", nargs="+", default=["pcb4", "cashew"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--out", type=str, default="artifacts/day02/models")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[ae_train] device:", device)

    use_mlflow = try_mlflow()
    if use_mlflow:
        import mlflow
        mlflow.set_experiment("visa_ae_baseline")

    for cls in args.classes:
        # index -> good(label=0)만
        idx = VisAProcessedIndex(args.data, [cls]).build()
        good_paths = [s.path for s in idx if s.cls == cls and s.label == 0]
        if len(good_paths) < 2:
            raise RuntimeError(f"[ae_train] not enough good samples for {cls}: {len(good_paths)}")

        # train/val split
        random.shuffle(good_paths)
        n_train = max(1, int(len(good_paths) * 0.9))
        tr_paths = good_paths[:n_train]
        va_paths = good_paths[n_train:] if len(good_paths) - n_train >= 1 else good_paths[:1]

        tr_ds = ImgListDataset(tr_paths, args.img_size)
        va_ds = ImgListDataset(va_paths, args.img_size)
        tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=0)
        va_ld = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=0)

        model = ConvAE().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        tr_losses, va_losses = [], []

        if use_mlflow:
            import mlflow
            run = mlflow.start_run(run_name=f"ae_{cls}")
            mlflow.log_params({
                "model": "ConvAE",
                "class": cls,
                "epochs": args.epochs,
                "batch": args.batch,
                "lr": args.lr,
                "img_size": args.img_size,
                "seed": args.seed,
                "n_good_total": len(good_paths),
                "n_good_train": len(tr_paths),
                "n_good_val": len(va_paths),
            })
        else:
            run = None

        for ep in range(1, args.epochs + 1):
            model.train()
            ep_tr = 0.0
            for x, _ in tr_ld:
                x = x.to(device, non_blocking=True).float()
                y = model(x)
                loss = F.mse_loss(y, x)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                ep_tr += loss.item() * x.size(0)
            ep_tr /= len(tr_ds)
            tr_losses.append(ep_tr)

            model.eval()
            ep_va = 0.0
            with torch.no_grad():
                for x, _ in va_ld:
                    x = x.to(device, non_blocking=True).float()
                    y = model(x)
                    loss = F.mse_loss(y, x)
                    ep_va += loss.item() * x.size(0)
            ep_va /= len(va_ds)
            va_losses.append(ep_va)

            print(f"[{cls}] epoch {ep:03d}/{args.epochs}  train_mse={ep_tr:.6f}  val_mse={ep_va:.6f}")
            if use_mlflow:
                import mlflow
                mlflow.log_metric("train_mse", ep_tr, step=ep)
                mlflow.log_metric("val_mse", ep_va, step=ep)

        # save ckpt
        ckpt_path = out_dir / f"ae_{cls}.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "class": cls,
            "img_size": args.img_size,
        }, ckpt_path)

        # save loss curve
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(tr_losses, label="train")
        plt.plot(va_losses, label="val")
        plt.xlabel("epoch")
        plt.ylabel("mse")
        plt.legend()
        fig_path = plot_dir / f"loss_curve_{cls}.png"
        plt.savefig(fig_path, dpi=160)
        plt.close()

        if use_mlflow:
            import mlflow
            mlflow.log_artifact(str(ckpt_path))
            mlflow.log_artifact(str(fig_path))
            mlflow.end_run()

        print("[ae_train] saved:", ckpt_path)
        print("[ae_train] saved:", fig_path)


if __name__ == "__main__":
    main()