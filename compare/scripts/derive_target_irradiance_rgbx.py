from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import diffusers.utils as diffusers_utils
import torch
import torchvision
from diffusers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image


COMPARE = Path(__file__).resolve().parents[1]
RGBX = COMPARE / "rgbx"
RGB2X_DIR = RGBX / "rgb2x"

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

if not hasattr(diffusers_utils, "randn_tensor"):
    diffusers_utils.randn_tensor = randn_tensor

if str(RGB2X_DIR) not in sys.path:
    sys.path.insert(0, str(RGB2X_DIR))
from load_image import load_ldr_image as rgb2x_load_ldr_image  # type: ignore  # noqa: E402
from pipeline_rgb2x import StableDiffusionAOVMatEstPipeline  # type: ignore  # noqa: E402


PROMPT = "Irradiance (diffuse lighting)"
DEFAULT_KEY_256 = "target_irradiance_from_target_rgb"
DEFAULT_KEY_512 = "target_irradiance_from_target_rgb_512"


def load_manifest(root: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return records


def write_manifest(root: Path, records: list[dict[str, Any]]) -> None:
    path = root / "manifest.jsonl"
    backup = root / "manifest.before_target_rgb_irradiance.jsonl"
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_rgb2x_pipeline(device: str):
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(
        "zheng95z/rgb-to-x",
        torch_dtype=dtype,
        cache_dir=str(RGB2X_DIR / "model_cache"),
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing",
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def save_resized(image: Image.Image, path_512: Path, path_256: Path) -> None:
    path_512.parent.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGB")
    image_512 = image.resize((512, 512), Image.Resampling.BICUBIC)
    image_512.save(path_512)
    image_512.resize((256, 256), Image.Resampling.BICUBIC).save(path_256)


def update_record(
    root: Path,
    record: dict[str, Any],
    *,
    key_256: str,
    key_512: str,
    out_256: Path,
    out_512: Path,
    source_key: str,
    steps: int,
) -> None:
    record.setdefault("files", {})[key_256] = rel(out_256, root)
    record.setdefault("files", {})[key_512] = rel(out_512, root)
    record.setdefault("derived_target_irradiance", {})[key_512] = {
        "source_key": source_key,
        "source_path": record["files"][source_key],
        "model": "zheng95z/rgb-to-x",
        "prompt": PROMPT,
        "steps": steps,
        "note": "Target irradiance estimated directly from target/reference RGB for external relighting baselines.",
    }

    metadata_path = root / "samples" / record["id"] / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.setdefault("files", {})[key_256] = rel(out_256, root)
        metadata.setdefault("files", {})[key_512] = rel(out_512, root)
        metadata.setdefault("derived_target_irradiance", {})[key_512] = record["derived_target_irradiance"][key_512]
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


@torch.inference_mode()
def derive_one(
    *,
    pipe,
    root: Path,
    record: dict[str, Any],
    source_key: str,
    out_key_256: str,
    out_key_512: str,
    device: str,
    steps: int,
    seed: int,
    overwrite: bool,
) -> float:
    sample_dir = root / "samples" / record["id"]
    out_256 = sample_dir / f"{out_key_256}.png"
    out_512 = sample_dir / f"{out_key_512}.png"
    if out_256.exists() and out_512.exists() and not overwrite:
        update_record(root, record, key_256=out_key_256, key_512=out_key_512, out_256=out_256, out_512=out_512, source_key=source_key, steps=steps)
        return 0.0

    photo = rgb2x_load_ldr_image(str(root / record["files"][source_key]), from_srgb=True).to(device)
    generator = torch.Generator(device=device).manual_seed(seed + int(record["id"]))
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    generated = pipe(
        prompt=PROMPT,
        photo=photo,
        num_inference_steps=steps,
        height=512,
        width=512,
        generator=generator,
        required_aovs=["irradiance"],
    ).images[0][0]
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    generated = torchvision.transforms.Resize((512, 512))(generated)
    save_resized(generated, out_512, out_256)
    update_record(root, record, key_256=out_key_256, key_512=out_key_512, out_256=out_256, out_512=out_512, source_key=source_key, steps=steps)
    return elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive target irradiance from target/reference RGB with RGB<->X RGB->X.")
    parser.add_argument("--benchmark-root", type=Path, default=COMPARE / "own_test_quantitative")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--source-key", default="target_rgb_512")
    parser.add_argument("--out-key-256", default=DEFAULT_KEY_256)
    parser.add_argument("--out-key-512", default=DEFAULT_KEY_512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.benchmark_root.resolve()
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    records = load_manifest(root)
    selected = records[: args.limit] if args.limit > 0 else records
    print("[info] root:", root)
    print("[info] device:", device)
    print("[info] samples:", len(selected), "of", len(records))
    print(f"[info] source: {args.source_key} -> {args.out_key_512}")
    pipe = build_rgb2x_pipeline(device)
    runtimes: dict[str, float] = {}
    for index, record in enumerate(selected, 1):
        elapsed = derive_one(
            pipe=pipe,
            root=root,
            record=record,
            source_key=args.source_key,
            out_key_256=args.out_key_256,
            out_key_512=args.out_key_512,
            device=device,
            steps=args.steps,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        if elapsed:
            runtimes[record["id"]] = elapsed
        if index == 1 or index % 25 == 0:
            print(f"  [derive {index:04d}/{len(selected):04d}] {record['id']} {elapsed:.3f}s")
    write_manifest(root, records)
    out_runtime = root / "predictions" / "rgbx_target_irradiance_from_target_rgb_runtime.json"
    out_runtime.parent.mkdir(parents=True, exist_ok=True)
    out_runtime.write_text(json.dumps(runtimes, indent=2), encoding="utf-8")
    print(f"[done] manifest updated with {args.out_key_512}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
