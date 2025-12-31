import argparse
from pathlib import Path
import re

from anomalib.engine import Engine
from anomalib.models import Patchcore


def patch_anomalib_symlink_issue():
    # 학습 때처럼 export 때도 혹시 workspace 만들면서 symlink 만들면 터질 수 있어서 안전빵으로 패치
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
        return new_dir

    import anomalib.utils.path as pathmod
    import anomalib.engine.engine as engmod
    pathmod.create_versioned_dir = create_versioned_dir_no_symlink
    engmod.create_versioned_dir = create_versioned_dir_no_symlink


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default="artifacts/day03/models")
    ap.add_argument("--name", default="patchcore_cashew")
    ap.add_argument("--backbone", default="resnet18")
    args = ap.parse_args()

    patch_anomalib_symlink_issue()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ExportType 경로가 버전별로 달라서 try 2번
    try:
        from anomalib.deploy import ExportType
    except Exception:
        from anomalib.deploy.export import ExportType

    engine = Engine()
    model = Patchcore(backbone=args.backbone)

    exported = engine.export(
        model=model,
        export_type=ExportType.TORCH,
        export_root=args.out_dir,
        model_file_name=args.name,
        ckpt_path=args.ckpt,
    )
    print("EXPORTED:", exported)


if __name__ == "__main__":
    main()
