from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_URL = "https://data.csail.mit.edu/multilum/multi_illumination_test_mip2_jpg.zip"
ARCHIVE = ROOT / "assets" / "downloads" / "miiw_test_jpg.zip"
EXTRACTED = ROOT / "assets" / "miiw_test_jpg"
SAMPLES = ROOT / "samples"
MANIFEST = ROOT / "manifest.jsonl"
EVAL_SIZE = 256
BASELINE_SIZE = 512


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_remove(path: Path) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    allowed = ROOT.resolve()
    if not str(resolved).lower().startswith(str(allowed).lower()):
        raise RuntimeError(f"Refusing to remove outside {allowed}: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)
    else:
        resolved.unlink()


def extract_if_needed() -> None:
    if not ARCHIVE.exists():
        raise FileNotFoundError(
            f"Missing {ARCHIVE}. Download it from {ARCHIVE_URL} first."
        )
    if EXTRACTED.exists() and any(EXTRACTED.iterdir()):
        return
    ensure_dir(EXTRACTED)
    with zipfile.ZipFile(ARCHIVE, "r") as archive:
        archive.extractall(EXTRACTED)


def scene_dirs() -> list[Path]:
    scenes = []
    for path in sorted(EXTRACTED.iterdir()):
        if not path.is_dir():
            continue
        dirs = sorted(path.glob("dir_*_mip2.jpg"))
        if len(dirs) >= 25:
            scenes.append(path)
    return scenes


def center_crop_resize(src: Path, dst: Path, size: int) -> str:
    ensure_dir(dst.parent)
    image = Image.open(src).convert("RGB")
    width, height = image.size
    crop = min(width, height)
    left = (width - crop) // 2
    top = (height - crop) // 2
    image = image.crop((left, top, left + crop, top + crop)).resize((size, size), Image.Resampling.LANCZOS)
    image.save(dst)
    return str(dst.relative_to(ROOT)).replace("\\", "/")


def copy_pair_file(src: Path, dst: Path) -> str:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return str(dst.relative_to(ROOT)).replace("\\", "/")


def build_pairs(count: int, force: bool) -> list[dict]:
    extract_if_needed()
    scenes = scene_dirs()
    if not scenes:
        raise RuntimeError(f"No MIIW scenes found under {EXTRACTED}")
    pairs_per_scene = count // len(scenes)
    remainder = count % len(scenes)
    if pairs_per_scene < 1:
        raise ValueError(f"Count {count} is too small for {len(scenes)} scenes")

    if force:
        safe_remove(SAMPLES)
        safe_remove(MANIFEST)
    ensure_dir(SAMPLES)

    records = []
    sample_idx = 0
    for scene_idx, scene in enumerate(scenes):
        n_pairs = pairs_per_scene + (1 if scene_idx < remainder else 0)
        for local_idx in range(n_pairs):
            if sample_idx >= count:
                break
            target_dir = (sample_idx * 7 + scene_idx * 3 + local_idx) % 25
            source_dir = (target_dir + 11 + local_idx % 5) % 25
            if source_dir == target_dir:
                source_dir = (source_dir + 1) % 25

            source = scene / f"dir_{source_dir}_mip2.jpg"
            target = scene / f"dir_{target_dir}_mip2.jpg"
            probes = scene / "probes"
            required = [
                source,
                target,
                probes / f"dir_{source_dir}_gray256.jpg",
                probes / f"dir_{source_dir}_chrome256.jpg",
                probes / f"dir_{target_dir}_gray256.jpg",
                probes / f"dir_{target_dir}_chrome256.jpg",
            ]
            if not all(path.exists() for path in required):
                continue

            sample_id = f"{sample_idx:06d}"
            out = SAMPLES / sample_id
            files = {
                "source_rgb": center_crop_resize(source, out / "source_rgb.png", EVAL_SIZE),
                "target_rgb": center_crop_resize(target, out / "target_rgb.png", EVAL_SIZE),
                "source_rgb_512": center_crop_resize(source, out / "source_rgb_512.png", BASELINE_SIZE),
                "target_rgb_512": center_crop_resize(target, out / "target_rgb_512.png", BASELINE_SIZE),
                "source_rgb_full": copy_pair_file(source, out / "source_rgb_full.jpg"),
                "target_rgb_full": copy_pair_file(target, out / "target_rgb_full.jpg"),
                "source_probe_gray": copy_pair_file(
                    probes / f"dir_{source_dir}_gray256.jpg",
                    out / "source_probe_gray.jpg",
                ),
                "source_probe_chrome": copy_pair_file(
                    probes / f"dir_{source_dir}_chrome256.jpg",
                    out / "source_probe_chrome.jpg",
                ),
                "target_probe_gray": copy_pair_file(
                    probes / f"dir_{target_dir}_gray256.jpg",
                    out / "target_probe_gray.jpg",
                ),
                "target_probe_chrome": copy_pair_file(
                    probes / f"dir_{target_dir}_chrome256.jpg",
                    out / "target_probe_chrome.jpg",
                ),
                "metadata": f"samples/{sample_id}/metadata.json",
            }

            record = {
                "id": sample_id,
                "dataset": "miiw",
                "scene": scene.name,
                "source_light_dir": source_dir,
                "target_light_dir": target_dir,
                "image_sizes": {
                    "evaluation": [EVAL_SIZE, EVAL_SIZE],
                    "baseline_square": [BASELINE_SIZE, BASELINE_SIZE],
                    "original": list(Image.open(source).size),
                },
                "preprocess": {
                    "evaluation": "center_crop_square_then_resize_lanczos",
                    "baseline_square": "center_crop_square_then_resize_lanczos",
                    "original": "copied_miiw_mip2_jpg",
                },
                "files": files,
                "method_inputs": {
                    "ours-best": {
                        "source_rgb": files["source_rgb"],
                        "target_rgb_for_scoring": files["target_rgb"],
                        "size": [EVAL_SIZE, EVAL_SIZE],
                    },
                    "luminet": {
                        "input_image": files["source_rgb_512"],
                        "reference_image": files["target_rgb_512"],
                        "size": [BASELINE_SIZE, BASELINE_SIZE],
                    },
                    "ic-light": {
                        "input_image": files["source_rgb_512"],
                        "target_reference": files["target_rgb_512"],
                        "recommended_width": BASELINE_SIZE,
                        "recommended_height": BASELINE_SIZE,
                    },
                    "rgbx": {
                        "rgb2x_input": files["source_rgb_512"],
                        "rgb2x_recommended_max_side": BASELINE_SIZE,
                        "x2rgb_aov_size": [BASELINE_SIZE, BASELINE_SIZE],
                    },
                },
                "prediction_name": f"{sample_id}.png",
            }
            with (out / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)
            records.append(record)
            sample_idx += 1

    if len(records) != count:
        raise RuntimeError(f"Built {len(records)} pairs, requested {count}")
    with MANIFEST.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    for method in ["rgbx", "luminet", "ic-light", "ours-best"]:
        ensure_dir(ROOT / "predictions" / method)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare 300 MIIW real paired samples.")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    records = build_pairs(args.count, args.force)
    print(f"[done] wrote {len(records)} MIIW pairs to {SAMPLES}")
    print(f"[done] manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
