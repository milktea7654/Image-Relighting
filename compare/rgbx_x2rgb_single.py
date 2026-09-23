from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from diffusers import DDIMScheduler


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

ROOT = Path(__file__).resolve().parent
X2RGB_DIR = ROOT / "rgbx" / "x2rgb"
if str(X2RGB_DIR) not in sys.path:
    sys.path.insert(0, str(X2RGB_DIR))

from load_image import load_exr_image, load_ldr_image  # type: ignore  # noqa: E402
from pipeline_x2rgb import StableDiffusionAOVDropoutPipeline  # type: ignore  # noqa: E402


REQUIRED_AOVS = ["albedo", "normal", "roughness", "metallic", "irradiance"]


def load_aov(path: Path, kind: str, device: str) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix == ".exr":
        if kind == "normal":
            return load_exr_image(str(path), normalize=True).to(device)
        if kind == "irradiance":
            return load_exr_image(str(path), tonemapping=True, clamp=True).to(device)
        return load_exr_image(str(path), clamp=True).to(device)

    if kind == "normal":
        return load_ldr_image(str(path), normalize=True).to(device)
    if kind == "irradiance":
        return load_ldr_image(str(path), from_srgb=True, clamp=True).to(device)
    if kind == "albedo":
        return load_ldr_image(str(path), from_srgb=True).to(device)
    return load_ldr_image(str(path), clamp=True).to(device)


def build_pipeline(device: str) -> StableDiffusionAOVDropoutPipeline:
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionAOVDropoutPipeline.from_pretrained(
        "zheng95z/x-to-rgb",
        torch_dtype=dtype,
        cache_dir=str(X2RGB_DIR / "model_cache"),
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing",
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RGBX X->RGB from partial intrinsic inputs.")
    parser.add_argument("--albedo", type=Path, required=True)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--roughness", type=Path, default=None)
    parser.add_argument("--metallic", type=Path, default=None)
    parser.add_argument("--irradiance", type=Path, default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; rerun with --device cpu if you want a CPU attempt.", file=sys.stderr)
        return 2

    device = args.device
    pipe = build_pipeline(device)

    albedo = load_aov(args.albedo, "albedo", device)
    normal = load_aov(args.normal, "normal", device)
    roughness = load_aov(args.roughness, "roughness", device) if args.roughness else None
    metallic = load_aov(args.metallic, "metallic", device) if args.metallic else None
    irradiance = load_aov(args.irradiance, "irradiance", device) if args.irradiance else None

    height = int(albedo.shape[1])
    width = int(albedo.shape[2])
    generator = torch.Generator(device=device).manual_seed(args.seed)

    image = pipe(
        prompt=args.prompt,
        albedo=albedo,
        normal=normal,
        roughness=roughness,
        metallic=metallic,
        irradiance=irradiance,
        num_inference_steps=args.inference_steps,
        height=height,
        width=width,
        generator=generator,
        required_aovs=REQUIRED_AOVS,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        guidance_rescale=0.7,
        output_type="pil",
    ).images[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    metadata = {
        "prompt": args.prompt,
        "seed": args.seed,
        "inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "image_guidance_scale": args.image_guidance_scale,
        "albedo": str(args.albedo),
        "normal": str(args.normal),
        "roughness": str(args.roughness) if args.roughness else None,
        "metallic": str(args.metallic) if args.metallic else None,
        "irradiance": str(args.irradiance) if args.irradiance else None,
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[done] saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())