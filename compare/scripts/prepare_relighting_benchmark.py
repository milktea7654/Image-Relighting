from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


COMPARE = Path(__file__).resolve().parents[1]
WORKSPACE = COMPARE.parent
SOURCES = COMPARE / "sources" / "DeepIntrinsicRelighting"
DATASETS = COMPARE / "datasets"

EVAL_SIZE = 256
BASELINE_SIZE = 512


def kelvin_to_rgb(kelvin: int) -> list[float]:
    temp = max(1000.0, min(40000.0, float(kelvin))) / 100.0
    if temp <= 66.0:
        red = 255.0
        green = 99.4708025861 * math.log(temp) - 161.1195681661
        blue = 0.0 if temp <= 19.0 else 138.5177312231 * math.log(temp - 10.0) - 305.0447927307
    else:
        red = 329.698727446 * ((temp - 60.0) ** -0.1332047592)
        green = 288.1221695283 * ((temp - 60.0) ** -0.0755148492)
        blue = 255.0
    return [float(np.clip(v / 255.0, 0.0, 1.0)) for v in (red, green, blue)]


def light_gradient(size: int, pan_deg: float | None, color: list[float] | None, tilt_deg: float | None = None) -> Image.Image:
    color_arr = np.asarray(color or [1.0, 1.0, 1.0], dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    xx = (xx / max(1, size - 1)) * 2.0 - 1.0
    yy = (yy / max(1, size - 1)) * 2.0 - 1.0
    if pan_deg is None:
        field = np.ones((size, size), dtype=np.float32)
    else:
        rad = math.radians(float(pan_deg))
        # Bright side points toward the incoming light direction.
        field = 0.65 + 0.35 * (math.cos(rad) * xx - math.sin(rad) * yy)
    if tilt_deg is not None:
        field *= 0.75 + 0.25 * np.clip(float(tilt_deg) / 50.0, 0.0, 1.0)
    field = np.clip(field, 0.05, 1.35)[..., None]
    arr = np.clip(field * color_arr.reshape(1, 1, 3), 0.0, 1.0)
    return Image.fromarray((arr * 255.0).round().astype(np.uint8), "RGB")


def image_mean(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB").resize((64, 64), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    return arr.reshape(-1, 3).mean(axis=0).clip(1e-4, 4.0)


def write_resized(src: Path, dst: Path, size: int) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img.convert("RGB").resize((size, size), Image.Resampling.LANCZOS).save(dst)
    return str(dst.relative_to(dst.parents[2]).as_posix())


def copy_or_resize(src: Path, dst: Path, size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img.convert("RGB").resize((size, size), Image.Resampling.LANCZOS).save(dst)


def save_light_products(sample_dir: Path, source_path: Path, source_light: dict[str, Any], target_light: dict[str, Any]) -> dict[str, str]:
    source_irr_256 = sample_dir / "source_irradiance.png"
    target_irr_256 = sample_dir / "target_irradiance.png"
    source_irr_512 = sample_dir / "source_irradiance_512.png"
    target_irr_512 = sample_dir / "target_irradiance_512.png"
    light_gradient(EVAL_SIZE, source_light.get("pan"), source_light.get("rgb"), source_light.get("tilt")).save(source_irr_256)
    light_gradient(EVAL_SIZE, target_light.get("pan"), target_light.get("rgb"), target_light.get("tilt")).save(target_irr_256)
    light_gradient(BASELINE_SIZE, source_light.get("pan"), source_light.get("rgb"), source_light.get("tilt")).save(source_irr_512)
    light_gradient(BASELINE_SIZE, target_light.get("pan"), target_light.get("rgb"), target_light.get("tilt")).save(target_irr_512)

    with Image.open(source_path) as img:
        source = np.asarray(img.convert("RGB").resize((EVAL_SIZE, EVAL_SIZE), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    src_i = np.asarray(Image.open(source_irr_256).convert("RGB"), dtype=np.float32) / 255.0
    tgt_i = np.asarray(Image.open(target_irr_256).convert("RGB"), dtype=np.float32) / 255.0
    ratio = np.clip(tgt_i / np.maximum(src_i.mean(axis=(0, 1), keepdims=True), 0.08), 0.35, 2.8)
    render = np.clip(source * ratio, 0.0, 1.0)
    render_proxy = sample_dir / "target_render_proxy.png"
    Image.fromarray((render * 255.0).round().astype(np.uint8), "RGB").save(render_proxy)
    return {
        "source_irradiance": str(source_irr_256),
        "target_irradiance": str(target_irr_256),
        "source_irradiance_512": str(source_irr_512),
        "target_irradiance_512": str(target_irr_512),
        "target_render_proxy": str(render_proxy),
    }


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def parse_isr_light(name: str) -> dict[str, Any]:
    parts = Path(name).stem.split("_")
    return {
        "pan": float(parts[2]),
        "tilt": float(parts[3]),
        "kelvin": int(parts[4]),
        "rgb": kelvin_to_rgb(int(parts[4])),
    }


def parse_rsr_light(name: str) -> dict[str, Any]:
    parts = Path(name).stem.split("_")
    return {
        "pan": float(parts[2]),
        "tilt": float(parts[3]),
        "rgb": [int(parts[4]) / 255.0, int(parts[5]) / 255.0, int(parts[6]) / 255.0],
        "light_position_index": int(parts[10]),
        "light_color_index": int(parts[9]),
    }


def build_record(
    *,
    root: Path,
    sample_id: str,
    dataset: str,
    split: str,
    scene: str,
    source_path: Path,
    target_path: Path,
    reference_path: Path | None,
    source_name: str,
    target_name: str,
    source_light: dict[str, Any],
    target_light: dict[str, Any],
) -> dict[str, Any]:
    sample_dir = root / "samples" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for key, path, size in [
        ("source_rgb", sample_dir / "source_rgb.png", EVAL_SIZE),
        ("target_rgb", sample_dir / "target_rgb.png", EVAL_SIZE),
        ("source_rgb_512", sample_dir / "source_rgb_512.png", BASELINE_SIZE),
        ("target_rgb_512", sample_dir / "target_rgb_512.png", BASELINE_SIZE),
    ]:
        copy_or_resize(source_path if key.startswith("source") else target_path, path, size)
        files[key] = rel(path, root)

    if reference_path is not None:
        ref_512 = sample_dir / "reference_rgb_512.png"
        copy_or_resize(reference_path, ref_512, BASELINE_SIZE)
        files["reference_rgb_512"] = rel(ref_512, root)
        target_light["reference_rgb_mean"] = image_mean(ref_512).tolist()
    else:
        files["reference_rgb_512"] = ""

    light_files = save_light_products(sample_dir, source_path, source_light, target_light)
    for key, path in light_files.items():
        files[key] = rel(Path(path), root)
    if not files["reference_rgb_512"]:
        files["reference_rgb_512"] = files["target_irradiance_512"]

    record = {
        "id": sample_id,
        "dataset": dataset,
        "split": split,
        "scene": scene,
        "source_name": source_name,
        "target_name": target_name,
        "source_light": source_light,
        "target_light": target_light,
        "image_sizes": {"evaluation": [EVAL_SIZE, EVAL_SIZE], "baseline_square": [BASELINE_SIZE, BASELINE_SIZE]},
        "files": files,
        "prediction_name": f"{sample_id}.png",
    }
    (sample_dir / "metadata.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def prepare_vidit(root: Path, split: str, limit: int) -> list[dict[str, Any]]:
    if split != "val":
        raise ValueError("VIDIT downloader currently prepares the public NTIRE2021 track2 validation split.")
    input_dir = DATASETS / "vidit_track2_validation" / "validation" / "input"
    guide_dir = DATASETS / "vidit_track2_validation" / "validation" / "guide"
    target_dir = DATASETS / "vidit_track2_validation_gt" / "validation_gt"
    pairs = sorted(target_dir.glob("Pair*.png"))
    if limit:
        pairs = pairs[:limit]
    records = []
    for i, target_path in enumerate(pairs):
        stem = target_path.stem
        source_path = input_dir / f"{stem}.png"
        guide_path = guide_dir / f"{stem}.png"
        if not source_path.exists() or not guide_path.exists():
            raise FileNotFoundError(stem)
        source_rgb = image_mean(source_path).tolist()
        guide_rgb = image_mean(guide_path).tolist()
        records.append(
            build_record(
                root=root,
                sample_id=f"{i:06d}",
                dataset="vidit",
                split=split,
                scene=stem,
                source_path=source_path,
                target_path=target_path,
                reference_path=guide_path,
                source_name=source_path.name,
                target_name=target_path.name,
                source_light={"rgb": source_rgb, "pan": None, "tilt": None, "source": "input_image_mean"},
                target_light={"rgb": guide_rgb, "pan": None, "tilt": None, "source": "guide_image_mean"},
            )
        )
    return records


def prepare_isr(root: Path, split: str, limit: int) -> list[dict[str, Any]]:
    anno = SOURCES / "data" / "anno_ISR" / ("val_pairs.txt" if split == "val" else "test_quantitative_pairs_10x.txt")
    image_dir = DATASETS / "isr_image" / "Image"
    lines = [line.strip().split() for line in anno.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        lines = lines[:limit]
    records = []
    for i, (src_name, tgt_name) in enumerate(lines):
        source_path = image_dir / src_name
        target_path = image_dir / tgt_name
        if not source_path.exists() or not target_path.exists():
            raise FileNotFoundError(f"{src_name} / {tgt_name}")
        scene = "_".join(Path(src_name).stem.split("_")[:2])
        records.append(
            build_record(
                root=root,
                sample_id=f"{i:06d}",
                dataset="isr",
                split=split,
                scene=scene,
                source_path=source_path,
                target_path=target_path,
                reference_path=None,
                source_name=src_name,
                target_name=tgt_name,
                source_light=parse_isr_light(src_name),
                target_light=parse_isr_light(tgt_name),
            )
        )
    return records


def prepare_rsr(root: Path, split: str, limit: int) -> list[dict[str, Any]]:
    anno = SOURCES / "data" / "anno_RSR" / ("AnyLightval_pairs.txt" if split == "val" else "AnyLighttest_pairs.txt")
    image_root = DATASETS / "rsr_256" / "RSR_256"
    rows = [line.strip().split() for line in anno.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]
    records = []
    for i, (folder, src_name, tgt_name) in enumerate(rows):
        source_path = image_root / folder / src_name
        target_path = image_root / folder / tgt_name
        if not source_path.exists() or not target_path.exists():
            raise FileNotFoundError(f"{folder}: {src_name} / {tgt_name}")
        records.append(
            build_record(
                root=root,
                sample_id=f"{i:06d}",
                dataset="rsr",
                split=split,
                scene=folder,
                source_path=source_path,
                target_path=target_path,
                reference_path=None,
                source_name=src_name,
                target_name=tgt_name,
                source_light=parse_rsr_light(src_name),
                target_light=parse_rsr_light(tgt_name),
            )
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare paired relighting benchmark manifests for VIDIT/ISR/RSR.")
    parser.add_argument("dataset", choices=["vidit", "isr", "rsr"])
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = COMPARE / f"{args.dataset}_quantitative"
    if root.exists() and args.force:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    if args.dataset == "vidit":
        records = prepare_vidit(root, args.split, args.limit)
    elif args.dataset == "isr":
        records = prepare_isr(root, args.split, args.limit)
    else:
        records = prepare_rsr(root, args.split, args.limit)

    with (root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (root / "README.md").write_text(
        f"# {args.dataset.upper()} Quantitative\n\n"
        f"Prepared split: `{args.split}`\n\n"
        f"Samples: {len(records)}\n\n"
        "Predictions are written to `predictions/<method>/<sample_id>.png`.\n",
        encoding="utf-8",
    )
    print(f"[done] {args.dataset} {args.split}: {len(records)} samples -> {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
