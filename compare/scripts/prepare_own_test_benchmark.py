from __future__ import annotations

import argparse
import json
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


COMPARE = Path(__file__).resolve().parents[1]
WORKSPACE = COMPARE.parent
DEFAULT_HF_SNAPSHOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "datasets--eenvil--stage3-relighting-dataset"
    / "snapshots"
    / "50498757723fa4948aed335a02fb23491bb58c67"
)

EVAL_SIZE = 256
BASELINE_SIZE = 512
RGB_KEYS = {
    "reference",
    "composite_optimized",
    "render_optimized",
    "hybrid_albedo",
    "normal",
    "shading",
}
GRAY_KEYS = {"roughness", "metallic", "glass_mask_soft"}
IMAGE_KEYS = sorted(RGB_KEYS | GRAY_KEYS)
METADATA_KEYS = [
    "scene_id",
    "source_stem",
    "split",
    "quality_final_loss",
    "quality_best_loss",
    "quality_final_rgb_loss",
    "glass_mask_ratio",
    "original_width",
    "original_height",
]


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def pil_from_hf_image(value: Any) -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(BytesIO(value["bytes"]))
        if value.get("path"):
            return Image.open(value["path"])
    raise TypeError(f"Unsupported HF image value: {type(value)!r}")


def save_image(value: Any, path: Path, size: int, mode: str) -> bool:
    img = pil_from_hf_image(value)
    if img is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with img:
        if mode == "L":
            out = img.convert("L").resize((size, size), Image.Resampling.LANCZOS)
        else:
            out = img.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
        out.save(path)
    return True


def save_zero(path: Path, size: int, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "L":
        arr = np.zeros((size, size), dtype=np.uint8)
        Image.fromarray(arr, "L").save(path)
    else:
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        Image.fromarray(arr, "RGB").save(path)


def copy_json_if_exists(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.name


def load_test_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            scene_id = str(row.get("scene_id") or row.get("source_stem") or "")
            if scene_id:
                rows[scene_id] = row
    return rows


def build_dataset(parquet_files: list[Path]):
    from datasets import Image as HFImage
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files=[str(p) for p in parquet_files], split="train")
    for column in IMAGE_KEYS:
        if column in ds.column_names:
            ds = ds.cast_column(column, HFImage(decode=False))
    return ds


def image_mode(key: str) -> str:
    return "L" if key in GRAY_KEYS else "RGB"


def build_record(
    *,
    root: Path,
    row: dict[str, Any],
    sample_id: str,
    local_light_root: Path,
    test_row: dict[str, Any] | None,
) -> dict[str, Any]:
    scene_id = str(row.get("scene_id") or row.get("source_stem") or sample_id)
    sample_dir = root / "samples" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    optional_status: dict[str, str] = {}

    save_plan = [
        ("reference", "target_rgb", "target_rgb.png", EVAL_SIZE),
        ("reference", "target_rgb_512", "target_rgb_512.png", BASELINE_SIZE),
        ("composite_optimized", "source_rgb", "source_rgb.png", EVAL_SIZE),
        ("composite_optimized", "source_rgb_512", "source_rgb_512.png", BASELINE_SIZE),
        ("composite_optimized", "composite_optimized", "composite_optimized.png", EVAL_SIZE),
        ("render_optimized", "render_optimized", "render_optimized.png", EVAL_SIZE),
        ("render_optimized", "target_render_proxy", "target_render_proxy.png", EVAL_SIZE),
        ("render_optimized", "target_render_proxy_512", "target_render_proxy_512.png", BASELINE_SIZE),
        ("shading", "target_irradiance", "target_irradiance.png", EVAL_SIZE),
        ("shading", "target_irradiance_512", "target_irradiance_512.png", BASELINE_SIZE),
        ("shading", "reference_rgb_512", "reference_rgb_512.png", BASELINE_SIZE),
        ("hybrid_albedo", "hybrid_albedo", "hybrid_albedo.png", EVAL_SIZE),
        ("roughness", "roughness", "roughness.png", EVAL_SIZE),
        ("metallic", "metallic", "metallic.png", EVAL_SIZE),
        ("glass_mask_soft", "glass_mask_soft", "glass_mask_soft.png", EVAL_SIZE),
        ("normal", "normal", "normal.png", EVAL_SIZE),
        ("shading", "shading", "shading.png", EVAL_SIZE),
    ]

    for source_key, file_key, filename, size in save_plan:
        mode = image_mode(source_key)
        dst = sample_dir / filename
        ok = save_image(row.get(source_key), dst, size, mode)
        if not ok:
            save_zero(dst, size, mode)
            optional_status[file_key] = "zero_filled_missing_hf_image"
        else:
            optional_status[file_key] = "hf_image"
        files[file_key] = rel(dst, root)

    local_scene = local_light_root / scene_id
    light_dir = sample_dir / "light"
    light_files: dict[str, str | None] = {
        "optimized_light": copy_json_if_exists(local_scene / "light" / "optimized_light.json", light_dir / "optimized_light.json"),
        "optimized_point_lights": copy_json_if_exists(
            local_scene / "light" / "optimized_point_lights.json",
            light_dir / "optimized_point_lights.json",
        ),
        "initial_light": copy_json_if_exists(local_scene / "light" / "initial_light.json", light_dir / "initial_light.json"),
        "manifest": copy_json_if_exists(local_scene / "manifest.json", sample_dir / "stage2_manifest.json"),
    }
    light_files = {k: rel((light_dir if k != "manifest" else sample_dir) / v, root) if v else None for k, v in light_files.items()}

    metadata = {key: row[key] for key in METADATA_KEYS if key in row}
    metadata["test_jsonl"] = test_row or {}
    metadata["optional_status"] = optional_status

    record = {
        "id": sample_id,
        "dataset": "own_stage3_test",
        "split": "test",
        "scene": scene_id,
        "scene_id": scene_id,
        "source_name": f"{scene_id}:composite_optimized",
        "target_name": f"{scene_id}:reference",
        "image_sizes": {"evaluation": [EVAL_SIZE, EVAL_SIZE], "baseline_square": [BASELINE_SIZE, BASELINE_SIZE]},
        "files": files,
        "light_condition": light_files,
        "hf_metadata": metadata,
        "prediction_name": f"{sample_id}.png",
    }
    (sample_dir / "metadata.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the local Stage3 own-test benchmark root.")
    parser.add_argument("--hf-snapshot", type=Path, default=DEFAULT_HF_SNAPSHOT)
    parser.add_argument("--split", default="test", choices=["test", "validation", "train"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--root", type=Path, default=COMPARE / "own_test_quantitative")
    parser.add_argument("--test-jsonl", type=Path, default=WORKSPACE / "test.jsonl")
    parser.add_argument("--local-light-root", type=Path, default=WORKSPACE / "downloaded_data")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if root.exists() and args.force:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    data_dir = args.hf_snapshot / "data"
    parquet_files = sorted(data_dir.glob(f"{args.split}-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No {args.split}-*.parquet files under {data_dir}")

    test_rows = load_test_jsonl(args.test_jsonl)
    ds = build_dataset(parquet_files)
    total = len(ds)
    count = min(args.limit, total) if args.limit > 0 else total
    records = []
    for idx in range(count):
        row = ds[idx]
        scene_id = str(row.get("scene_id") or row.get("source_stem") or f"row_{idx:06d}")
        sample_id = f"{idx:06d}"
        records.append(
            build_record(
                root=root,
                row=row,
                sample_id=sample_id,
                local_light_root=args.local_light_root,
                test_row=test_rows.get(scene_id),
            )
        )
        if idx == 0 or (idx + 1) % 100 == 0:
            print(f"[prepare] {idx + 1:04d}/{count:04d} {scene_id}")

    with (root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    readme = (
        "# Own Test Quantitative\n\n"
        f"Split: `{args.split}`\n\n"
        f"Samples: {len(records)}\n\n"
        "Stage3 inputs use `composite_optimized`, `render_optimized`, `hybrid_albedo`, "
        "`roughness`, `metallic`, and `glass_mask_soft`. Missing HF image columns are "
        "written as zero images and recorded in each sample metadata.\n\n"
        "External RGBX/LumiNet/IC-Light runners use `source_rgb_512` and target `shading` "
        "as `target_irradiance_512`/`reference_rgb_512`; they do not consume the Stage2 "
        "point-light JSON directly.\n"
    )
    (root / "README.md").write_text(readme, encoding="utf-8")
    print(f"[done] prepared {len(records)} samples -> {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
