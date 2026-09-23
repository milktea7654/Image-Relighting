from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


COMPARE = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = COMPARE / "own_test_quantitative"


def load_records(root: Path) -> list[dict]:
    return [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def write_records(root: Path, records: list[dict]) -> None:
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_inputs(root: Path, force: bool) -> None:
    records = load_records(root)
    inputs = root / "_mvinverse_inputs"
    image_dir = inputs / "images"
    if force and inputs.exists():
        shutil.rmtree(inputs)
    image_dir.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    for index, record in enumerate(records, 1):
        sample_id = record["id"]
        src = root / record["files"]["target_rgb_512"]
        dst = image_dir / f"{sample_id}.png"
        if not src.exists():
            raise FileNotFoundError(src)
        hardlink_or_copy(src, dst)
        paths.append(str(dst.resolve()))
        if index == 1 or index % 100 == 0:
            print(f"[prepare-inputs] {index:04d}/{len(records):04d} {sample_id}")

    list_path = inputs / "image_list.txt"
    list_path.write_text("\n".join(paths) + "\n", encoding="utf-8")
    print("[done]", list_path.resolve())


def save_black_mask(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((size[1], size[0]), dtype=np.uint8), "L").save(path)


def apply_outputs(root: Path, mvinverse_root: Path) -> None:
    records = load_records(root)
    missing: list[str] = []
    for index, record in enumerate(records, 1):
        sample_id = record["id"]
        sample_dir = root / "samples" / sample_id
        src_albedo = mvinverse_root / sample_id / "000_albedo.png"
        if not src_albedo.exists():
            missing.append(sample_id)
            continue

        hybrid_path = sample_dir / "hybrid_albedo.png"
        glass_soft_path = sample_dir / "glass_mask_soft.png"

        with Image.open(src_albedo) as image:
            image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS).save(hybrid_path)
        save_black_mask(glass_soft_path, (256, 256))

        files = record.setdefault("files", {})
        files["hybrid_albedo"] = hybrid_path.relative_to(root).as_posix()
        files["glass_mask_soft"] = glass_soft_path.relative_to(root).as_posix()

        metadata = record.setdefault("hf_metadata", {})
        optional = metadata.setdefault("optional_status", {})
        optional["hybrid_albedo"] = "mvinverse_regenerated_from_target_rgb_512"
        optional["glass_mask_soft"] = "black_mask_assumed_no_glass"
        metadata["albedo_fix"] = {
            "source": "Stage12_remote_run/single_image_batch_inference.py",
            "input": files.get("target_rgb_512"),
            "raw_output": str(src_albedo.relative_to(root).as_posix()) if src_albedo.is_relative_to(root) else str(src_albedo),
            "note": "HF test parquet omits hybrid_albedo; regenerated MVInverse albedo from target/reference image. Glass mask set to black.",
        }

        sample_meta = sample_dir / "metadata.json"
        if sample_meta.exists():
            sample_meta.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        if index == 1 or index % 100 == 0:
            print(f"[apply] {index:04d}/{len(records):04d} {sample_id}")

    if missing:
        raise RuntimeError(f"Missing MVInverse albedo for {len(missing)} samples, first: {missing[:10]}")
    write_records(root, records)
    print("[done] updated manifest and sample metadata")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate own-test albedo with MVInverse and set black glass masks.")
    parser.add_argument("mode", choices=["prepare-inputs", "apply"])
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--mvinverse-root", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.benchmark_root.resolve()
    if args.mode == "prepare-inputs":
        prepare_inputs(root, args.force)
    else:
        mvinverse_root = args.mvinverse_root or (root / "_mvinverse_outputs")
        apply_outputs(root, mvinverse_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
