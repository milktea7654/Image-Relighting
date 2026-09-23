from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from light_protocol import DIRECTION_TO_PAN, kelvin_to_rgb, light_record


COMPARE = Path(__file__).resolve().parents[1]
DATASETS = COMPARE / "datasets"
DIL = COMPARE / "sources" / "DeepIntrinsicRelighting"

EVAL_SIZE = 256
BASELINE_SIZE = 512


def safe_rmtree(path: Path) -> None:
    resolved = path.resolve()
    compare = COMPARE.resolve()
    if compare not in resolved.parents:
        raise RuntimeError(f"Refusing to delete outside compare/: {resolved}")
    shutil.rmtree(resolved)


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def copy_resized(src: Path, dst: Path, size: int) -> str:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS).save(dst)
    return str(dst)


def parse_isr_light(name: str) -> dict[str, Any]:
    parts = Path(name).stem.split("_")
    pan = float(parts[2])
    tilt = float(parts[3])
    kelvin = int(parts[4])
    return light_record(pan=pan, tilt=tilt, rgb=kelvin_to_rgb(kelvin), kelvin=kelvin)


def parse_rsr_light(name: str) -> dict[str, Any]:
    parts = Path(name).stem.split("_")
    rgb = [int(parts[4]) / 255.0, int(parts[5]) / 255.0, int(parts[6]) / 255.0]
    return light_record(
        pan=float(parts[2]),
        tilt=float(parts[3]),
        rgb=rgb,
        rgb_255=[int(parts[4]), int(parts[5]), int(parts[6])],
        scene_index=int(parts[7]),
        view_index=int(parts[9]),
        light_color_index=int(parts[9]),
        light_position_index=int(parts[10]),
    )


def parse_vidit_light(name: str) -> dict[str, Any]:
    stem = Path(name).stem
    parts = stem.split("_")
    direction = parts[-1]
    kelvin = int(parts[-2])
    scene = "_".join(parts[:-2])
    pan = DIRECTION_TO_PAN[direction]
    return light_record(
        pan=pan,
        tilt=45.0,
        rgb=kelvin_to_rgb(kelvin),
        kelvin=kelvin,
        direction=direction,
        scene_name=scene,
    )


def build_record(
    *,
    root: Path,
    sample_id: str,
    dataset: str,
    source_path: Path,
    target_path: Path,
    source_name: str,
    target_name: str,
    scene: str,
    source_light: dict[str, Any],
    target_light: dict[str, Any],
    annotation: str,
    annotation_index: int,
    include_512: bool,
) -> dict[str, Any]:
    sample_dir = root / "samples" / sample_id
    files: dict[str, str] = {}

    source_256 = sample_dir / "source_rgb.png"
    target_256 = sample_dir / "target_rgb.png"
    copy_resized(source_path, source_256, EVAL_SIZE)
    copy_resized(target_path, target_256, EVAL_SIZE)
    files["source_rgb"] = rel(source_256, root)
    files["target_rgb"] = rel(target_256, root)

    if include_512:
        source_512 = sample_dir / "source_rgb_512.png"
        target_512 = sample_dir / "target_rgb_512.png"
        copy_resized(source_path, source_512, BASELINE_SIZE)
        copy_resized(target_path, target_512, BASELINE_SIZE)
        files["source_rgb_512"] = rel(source_512, root)
        files["target_rgb_512"] = rel(target_512, root)

    record = {
        "id": sample_id,
        "dataset": dataset,
        "split": "test",
        "protocol": "deepintrinsicrelighting_official_quantitative",
        "official_annotation": annotation,
        "official_annotation_index": annotation_index,
        "stage4_id_prefix": f"{dataset}_paper_{sample_id}",
        "scene": scene,
        "source_name": source_name,
        "target_name": target_name,
        "source_original": str(source_path.resolve()),
        "target_original": str(target_path.resolve()),
        "source_light": source_light,
        "target_light": target_light,
        "stage4_target": {
            "point_light_position": target_light["stage4_point_light_position"],
            "point_light_rgb": target_light["stage4_point_light_rgb"],
            "env_rgb": target_light["rgb"],
            "coordinate_note": target_light["coordinate_note"],
        },
        "metric_protocol": {
            "name": "DeepIntrinsicRelighting util.metric",
            "image_range": "[0, 1]",
            "metrics": ["MSE", "SSIM", "LPIPS", "PSNR", "MPS"],
            "scoring_size": [EVAL_SIZE, EVAL_SIZE],
        },
        "files": files,
        "prediction_name": f"{sample_id}.png",
    }
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "metadata.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def prepare_isr(root: Path, limit: int, include_512: bool) -> list[dict[str, Any]]:
    annotation = DIL / "data" / "anno_ISR" / "test_quantitative_pairs_10x.txt"
    image_dir = DATASETS / "isr_image" / "Image"
    rows = [line.strip().split() for line in annotation.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]
    records = []
    for index, (source_name, target_name) in enumerate(rows):
        scene = "_".join(Path(source_name).stem.split("_")[:2])
        records.append(
            build_record(
                root=root,
                sample_id=f"{index:06d}",
                dataset="isr",
                source_path=image_dir / source_name,
                target_path=image_dir / target_name,
                source_name=source_name,
                target_name=target_name,
                scene=scene,
                source_light=parse_isr_light(source_name),
                target_light=parse_isr_light(target_name),
                annotation=str(annotation.relative_to(DIL).as_posix()),
                annotation_index=index,
                include_512=include_512,
            )
        )
    return records


def prepare_rsr(root: Path, limit: int, include_512: bool) -> list[dict[str, Any]]:
    annotation = DIL / "data" / "anno_RSR" / "AnyLighttest_pairs.txt"
    image_root = DATASETS / "rsr_256" / "RSR_256"
    rows = [line.strip().split() for line in annotation.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]
    records = []
    for index, (folder, source_name, target_name) in enumerate(rows):
        records.append(
            build_record(
                root=root,
                sample_id=f"{index:06d}",
                dataset="rsr",
                source_path=image_root / folder / source_name,
                target_path=image_root / folder / target_name,
                source_name=source_name,
                target_name=target_name,
                scene=folder,
                source_light=parse_rsr_light(source_name),
                target_light=parse_rsr_light(target_name),
                annotation=str(annotation.relative_to(DIL).as_posix()),
                annotation_index=index,
                include_512=include_512,
            )
        )
    return records


def prepare_vidit(root: Path, limit: int, include_512: bool) -> list[dict[str, Any]]:
    annotation = DIL / "data" / "anno_VIDIT" / "any2any" / "AnyLight_test_pairs.txt"
    image_root = DATASETS / "VIDIT_full"
    if not image_root.exists():
        raise FileNotFoundError(
            "Missing official VIDIT_full folder. Download VIDIT train/full images and place all "
            f"40-light images under: {image_root}"
        )
    rows = [line.strip().split() for line in annotation.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]
    records = []
    for index, (source_name, target_name) in enumerate(rows):
        source_light = parse_vidit_light(source_name)
        target_light = parse_vidit_light(target_name)
        records.append(
            build_record(
                root=root,
                sample_id=f"{index:06d}",
                dataset="vidit",
                source_path=image_root / source_name,
                target_path=image_root / target_name,
                source_name=source_name,
                target_name=target_name,
                scene=source_light["scene_name"],
                source_light=source_light,
                target_light=target_light,
                annotation=str(annotation.relative_to(DIL).as_posix()),
                annotation_index=index,
                include_512=include_512,
            )
        )
    return records


def write_root(root: Path, dataset: str, records: list[dict[str, Any]], include_512: bool) -> None:
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (root / "README.md").write_text(
        f"# {dataset.upper()} Paper Quantitative\n\n"
        "Protocol: DeepIntrinsicRelighting official quantitative annotations.\n\n"
        f"Samples: {len(records)}\n\n"
        f"Scoring size: {EVAL_SIZE} x {EVAL_SIZE}\n\n"
        f"Includes 512 copies for external baselines: {include_512}\n\n"
        "Target light information is stored in `target_light` and `stage4_target`.\n"
        "Use `target_light.stage4_point_light_position` for Stage4 point-light relighting.\n",
        encoding="utf-8",
    )


def prepare_dataset(dataset: str, args: argparse.Namespace) -> Path:
    root = COMPARE / f"{dataset}_paper_quantitative"
    if root.exists() and args.force:
        safe_rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    if dataset == "isr":
        records = prepare_isr(root, args.limit, not args.no_512)
    elif dataset == "rsr":
        records = prepare_rsr(root, args.limit, not args.no_512)
    elif dataset == "vidit":
        records = prepare_vidit(root, args.limit, not args.no_512)
    else:
        raise ValueError(dataset)
    write_root(root, dataset, records, not args.no_512)
    print(f"[done] {dataset}: {len(records)} samples -> {root}")
    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare paper-protocol quantitative roots.")
    parser.add_argument("dataset", choices=["isr", "rsr", "vidit", "all"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-512", action="store_true", help="Do not write 512x512 copies for external baselines.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = ["isr", "rsr", "vidit"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        prepare_dataset(dataset, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
