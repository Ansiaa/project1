import argparse
from pathlib import Path
import inspect
import re

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore


def patch_anomalib_symlink_issue():
    """
    Windows에서 symlink(latest) 생성이 막혀 WinError 1 터지는 문제 우회.
    utils.path 와 engine.engine 양쪽의 create_versioned_dir 레퍼런스를 모두 패치한다.
    """
    def create_versioned_dir_no_symlink(root_dir):
        root_dir = Path(root_dir)
        root_dir.mkdir(parents=True, exist_ok=True)

        vers = []
        for p in root_dir.iterdir():
            if p.is_dir():
                m = re.fullmatch(r"v(\d+)", p.name)
                if m:
                    vers.append(int(m.group(1)))
        next_v = (max(vers) + 1) if vers else 0

        new_dir = root_dir / f"v{next_v}"
        new_dir.mkdir(parents=True, exist_ok=True)
        return new_dir  # latest symlink 생성 안 함

    import anomalib.utils.path as pathmod
    import anomalib.engine.engine as engmod

    # 1) 실제 구현 모듈
    pathmod.create_versioned_dir = create_versioned_dir_no_symlink
    # 2) Engine이 참조하는 레퍼런스(중요)
    engmod.create_versioned_dir = create_versioned_dir_no_symlink


def build_folder_datamodule(cls_root: Path, name: str, img_size: int, batch: int):
    """
    anomalib Folder는 버전마다 __init__ 인자가 다름.
    -> 현재 설치된 버전의 signature에 존재하는 인자만 필터링해서 생성한다.
    + 확장자 대/소문자(.jpg/.JPG 등) 모두 인식하도록 extensions 주입
    """
    sig = inspect.signature(Folder)
    allowed = set(sig.parameters.keys())

    # anomalib 기본 확장자 + 대문자 버전까지
    exts = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
    exts = tuple(sorted(set(exts + tuple(e.upper() for e in exts))))

    kwargs = dict(
        name=name,
        root=str(cls_root),
        normal_dir="good",
        abnormal_dir="defect",
        normal_test_dir="good_test",
        train_batch_size=batch,
        eval_batch_size=batch,
        num_workers=4,
        image_size=(img_size, img_size),  # 없으면 자동으로 빠짐
        task="classification",            # 없으면 자동으로 빠짐
        extensions=exts,                  # 있으면 넣고, 없으면 아래에서 속성으로 주입
    )

    # 버전별 batch 키 대응
    if "train_batch_size" not in allowed and "batch_size" in allowed:
        kwargs["batch_size"] = batch
        kwargs.pop("train_batch_size", None)
        kwargs.pop("eval_batch_size", None)

    # signature 기반 필터링
    filtered = {k: v for k, v in kwargs.items() if k in allowed}

    dm = Folder(**filtered)

    # __init__에서 extensions를 안 받는 버전이면, 속성으로라도 주입
    if "extensions" not in allowed and hasattr(dm, "extensions"):
        dm.extensions = exts

    if hasattr(dm, "setup"):
        dm.setup()

    return dm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", default="pcb4")
    ap.add_argument("--root", default="data/anomalib")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    # ★ symlink 패치는 Engine 사용 전에 해야 함
    patch_anomalib_symlink_issue()

    cls_root = Path(args.root) / args.cls
    assert cls_root.exists(), f"not found: {cls_root}"

    dm = build_folder_datamodule(
        cls_root=cls_root,
        name=f"visa_{args.cls}",
        img_size=args.img_size,
        batch=args.batch,
    )

    model = Patchcore(backbone=args.backbone)

    # out_dir은 그냥 cls까지만 두는 게 깔끔 (versioning은 패치가 처리)
    out_dir = Path("artifacts/day03/runs") / args.cls
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = Engine(
        default_root_dir=str(out_dir),
        accelerator="auto",
        devices=1,
        max_epochs=args.epochs,
    )

    engine.fit(model=model, datamodule=dm)
    print("BEST_CKPT:", engine.best_model_path)

    res = engine.test(model=model, datamodule=dm)
    print("TEST:", res)


if __name__ == "__main__":
    main()
