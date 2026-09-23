from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision
from diffusers import DDIMScheduler


ROOT = Path(__file__).resolve().parent
RGB2X_DIR = ROOT / "rgbx" / "rgb2x"
if str(RGB2X_DIR) not in sys.path:
    sys.path.insert(0, str(RGB2X_DIR))

from load_image import load_exr_image, load_ldr_image  # type: ignore  # noqa: E402
from pipeline_rgb2x import StableDiffusionAOVMatEstPipeline  # type: ignore  # noqa: E402


PROMPTS = {
    "albedo": "Albedo (diffuse basecolor)",
    "normal": "Camera-space Normal",
    "roughness": "Roughness",
    "metallic": "Metallicness",
    "irradiance": "Irradiance (diffuse lighting)",
}
REQUIRED_AOVS = ["albedo", "normal", "roughness", "metallic", "irradiance"]
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".exr"}


def compute_size(height: int, width: int, max_side: int) -> tuple[int, int]:
    new_height = height
    new_width = width
    ratio = height / width
    if height > width:
        new_height = max_side
        new_width = int(new_height / ratio)
    else:
        new_width = max_side
        new_height = int(new_width * ratio)

    if new_width % 8 != 0:
        new_width = new_width // 8 * 8
    if new_height % 8 != 0:
        new_height = new_height // 8 * 8
    return new_height, new_width


def load_photo(image_path: Path, device: str) -> torch.Tensor:
    suffix = image_path.suffix.lower()
    if suffix == ".exr":
        return load_exr_image(str(image_path), tonemapping=True, clamp=True).to(device)
    return load_ldr_image(str(image_path), from_srgb=True).to(device)


def build_pipeline(device: str) -> StableDiffusionAOVMatEstPipeline:
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


def gather_images(input_dir: Path, matches: list[str]) -> list[Path]:
    images = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    )
    if not matches:
        return images
    lowered = [item.lower() for item in matches]
    return [path for path in images if path.name.lower() in lowered or path.stem.lower() in lowered]


def save_metadata(output_dir: Path, source: Path, seed: int, inference_steps: int) -> None:
    metadata = {
        "source": str(source),
        "seed": seed,
        "inference_steps": inference_steps,
        "aovs": REQUIRED_AOVS,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_single(
    pipe: StableDiffusionAOVMatEstPipeline,
    image_path: Path,
    output_root: Path,
    device: str,
    seed: int,
    inference_steps: int,
    max_side: int,
    overwrite: bool,
) -> None:
    output_dir = output_root / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = [output_dir / f"{aov}.png" for aov in REQUIRED_AOVS]
    if not overwrite and all(path.exists() for path in expected):
        print(f"[skip] {image_path.name}")
        return

    photo = load_photo(image_path, device)
    old_height = int(photo.shape[1])
    old_width = int(photo.shape[2])
    new_height, new_width = compute_size(old_height, old_width, max_side)
    photo = torchvision.transforms.Resize((new_height, new_width))(photo)
    generator = torch.Generator(device=device).manual_seed(seed)

    print(f"[run] {image_path.name} -> {output_dir}")
    for aov_name in REQUIRED_AOVS:
        generated_image = pipe(
            prompt=PROMPTS[aov_name],
            photo=photo,
            num_inference_steps=inference_steps,
            height=new_height,
            width=new_width,
            generator=generator,
            required_aovs=[aov_name],
        ).images[0][0]
        generated_image = torchvision.transforms.Resize((old_height, old_width))(generated_image)
        generated_image.save(output_dir / f"{aov_name}.png")

    save_metadata(output_dir, image_path, seed, inference_steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch RGB->X inference for a folder of images.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing input RGB images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder to save predicted channels.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--max-side", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=0, help="Optional image limit for smoke tests.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--match",
        action="append",
        default=[],
        help="Optional exact filename or stem to process. Can be provided more than once.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; rerun with --device cpu if you want a CPU attempt.", file=sys.stderr)
        return 2
    if not args.input_dir.is_dir():
        print(f"Missing input directory: {args.input_dir}", file=sys.stderr)
        return 2

    images = gather_images(args.input_dir, args.match)
    if not images:
        print(f"No supported images found in: {args.input_dir}", file=sys.stderr)
        return 2
    if args.limit > 0:
        images = images[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipe = build_pipeline(args.device)
    for image_path in images:
        run_single(
            pipe=pipe,
            image_path=image_path,
            output_root=args.output_dir,
            device=args.device,
            seed=args.seed,
            inference_steps=args.inference_steps,
            max_side=args.max_side,
            overwrite=args.overwrite,
        )
    print(f"[done] processed {len(images)} image(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())