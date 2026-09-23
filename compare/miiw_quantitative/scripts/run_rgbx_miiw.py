from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from diffusers import DDIMScheduler
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
COMPARE = Path(__file__).resolve().parents[2]
RGBX = COMPARE / "rgbx"
RGB2X_DIR = RGBX / "rgb2x"
X2RGB_DIR = RGBX / "x2rgb"
MANIFEST = ROOT / "manifest.jsonl"
PREDICTIONS = ROOT / "predictions" / "rgbx"
INTERMEDIATES = ROOT / "predictions" / "rgbx_intermediates"

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

if str(RGB2X_DIR) not in sys.path:
    sys.path.insert(0, str(RGB2X_DIR))
from load_image import load_ldr_image as rgb2x_load_ldr_image  # type: ignore  # noqa: E402
from pipeline_rgb2x import StableDiffusionAOVMatEstPipeline  # type: ignore  # noqa: E402

sys.path = [p for p in sys.path if p != str(RGB2X_DIR)]
if str(X2RGB_DIR) not in sys.path:
    sys.path.insert(0, str(X2RGB_DIR))
from load_image import load_ldr_image as x2rgb_load_ldr_image  # type: ignore  # noqa: E402
from pipeline_x2rgb import StableDiffusionAOVDropoutPipeline  # type: ignore  # noqa: E402


SOURCE_AOVS = ["albedo", "normal", "roughness", "metallic"]
PROMPTS = {
    "albedo": "Albedo (diffuse basecolor)",
    "normal": "Camera-space Normal",
    "roughness": "Roughness",
    "metallic": "Metallicness",
    "irradiance": "Irradiance (diffuse lighting)",
}
REQUIRED_AOVS = ["albedo", "normal", "roughness", "metallic", "irradiance"]


def load_manifest(limit: int = 0) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    return records[:limit] if limit > 0 else records


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


def build_x2rgb_pipeline(device: str):
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


def save_probe_irradiance(record: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(ROOT / record["files"]["target_probe_gray"]) as probe:
        probe = probe.convert("RGB").resize((512, 512), Image.BICUBIC)
        arr = np.asarray(probe, dtype=np.float32) / 255.0
    intensity = arr.mean(axis=2, keepdims=True)
    mask = intensity > max(0.03, float(np.percentile(intensity, 70)) * 0.25)
    color = arr[mask[..., 0]].mean(axis=0) if np.any(mask) else arr.reshape(-1, 3).mean(axis=0)
    gradient = np.tile(color.reshape(1, 1, 3), (512, 512, 1))
    # Keep a little structure from the probe so the conditioning is not just a flat color.
    mixed = 0.65 * gradient + 0.35 * arr
    Image.fromarray((mixed.clip(0, 1) * 255.0).round().astype(np.uint8)).save(out_path)


def run_rgb2x_stage(
    records: list[dict[str, Any]],
    device: str,
    steps: int,
    seed: int,
    overwrite: bool,
) -> dict[str, float]:
    pipe = build_rgb2x_pipeline(device)
    runtimes: dict[str, float] = {}
    for index, record in enumerate(records, 1):
        sample_id = record["id"]
        sample_dir = INTERMEDIATES / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        expected = [sample_dir / f"{name}.png" for name in SOURCE_AOVS] + [sample_dir / "irradiance.png"]
        if all(path.exists() for path in expected) and not overwrite:
            continue

        photo = rgb2x_load_ldr_image(str(ROOT / record["files"]["source_rgb_512"]), from_srgb=True).to(device)
        generator = torch.Generator(device=device).manual_seed(seed + int(sample_id))
        start = time.perf_counter()
        for aov_name in SOURCE_AOVS:
            out_path = sample_dir / f"{aov_name}.png"
            if out_path.exists() and not overwrite:
                continue
            if device == "cuda":
                torch.cuda.synchronize()
            call_start = time.perf_counter()
            generated = pipe(
                prompt=PROMPTS[aov_name],
                photo=photo,
                num_inference_steps=steps,
                height=512,
                width=512,
                generator=generator,
                required_aovs=[aov_name],
            ).images[0][0]
            if device == "cuda":
                torch.cuda.synchronize()
            generated = torchvision.transforms.Resize((512, 512))(generated)
            generated.save(out_path)
            runtimes[f"{sample_id}:{aov_name}"] = time.perf_counter() - call_start
        irr_path = sample_dir / "irradiance.png"
        if not irr_path.exists() or overwrite:
            save_probe_irradiance(record, irr_path)
        runtimes[f"{sample_id}:rgb2x_total"] = time.perf_counter() - start
        if index == 1 or index % 10 == 0:
            print(f"  [rgb2x {index:03d}/{len(records):03d}] {sample_id}")
    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()
    return runtimes


def load_aov(path: Path, kind: str, device: str) -> torch.Tensor:
    if kind == "normal":
        return x2rgb_load_ldr_image(str(path), normalize=True).to(device)
    if kind == "irradiance":
        return x2rgb_load_ldr_image(str(path), from_srgb=True, clamp=True).to(device)
    if kind == "albedo":
        return x2rgb_load_ldr_image(str(path), from_srgb=True).to(device)
    return x2rgb_load_ldr_image(str(path), clamp=True).to(device)


def run_x2rgb_stage(
    records: list[dict[str, Any]],
    device: str,
    steps: int,
    seed: int,
    overwrite: bool,
) -> dict[str, float]:
    pipe = build_x2rgb_pipeline(device)
    runtimes: dict[str, float] = {}
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records, 1):
        sample_id = record["id"]
        out_path = PREDICTIONS / record["prediction_name"]
        if out_path.exists() and not overwrite:
            continue
        sample_dir = INTERMEDIATES / sample_id
        aovs = {name: sample_dir / f"{name}.png" for name in REQUIRED_AOVS}
        missing = [str(path) for path in aovs.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing RGBX AOVs for {sample_id}: {missing}")

        albedo = load_aov(aovs["albedo"], "albedo", device)
        normal = load_aov(aovs["normal"], "normal", device)
        roughness = load_aov(aovs["roughness"], "roughness", device)
        metallic = load_aov(aovs["metallic"], "metallic", device)
        irradiance = load_aov(aovs["irradiance"], "irradiance", device)
        generator = torch.Generator(device=device).manual_seed(seed + int(sample_id))

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        image = pipe(
            prompt="a realistic indoor scene under the target illumination",
            albedo=albedo,
            normal=normal,
            roughness=roughness,
            metallic=metallic,
            irradiance=irradiance,
            num_inference_steps=steps,
            height=512,
            width=512,
            generator=generator,
            required_aovs=REQUIRED_AOVS,
            guidance_scale=7.5,
            image_guidance_scale=1.5,
            guidance_rescale=0.7,
            output_type="pil",
        ).images[0]
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        image.save(out_path)
        runtimes[sample_id] = elapsed
        if index == 1 or index % 10 == 0:
            print(f"  [x2rgb {index:03d}/{len(records):03d}] {sample_id} {elapsed:.3f}s")
    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()
    return runtimes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RGB↔X on MIIW quantitative split.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rgb2x-steps", type=int, default=20)
    parser.add_argument("--x2rgb-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-rgb2x", action="store_true")
    parser.add_argument("--skip-x2rgb", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    records = load_manifest(args.limit)
    INTERMEDIATES.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    print("[info] device:", device)
    print("[info] samples:", len(records))
    print("[info] RGBX source AOVs from source_rgb_512; target irradiance proxy from target_probe_gray")

    stage_runtime: dict[str, Any] = {}
    if not args.skip_rgb2x:
        stage_runtime["rgb2x"] = run_rgb2x_stage(records, device, args.rgb2x_steps, args.seed, args.overwrite)
    if not args.skip_x2rgb:
        stage_runtime["x2rgb"] = run_x2rgb_stage(records, device, args.x2rgb_steps, args.seed, args.overwrite)

    runtime_path = PREDICTIONS / "runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {}
    rgb2x_runtime = stage_runtime.get("rgb2x", {})
    x2rgb_runtime = stage_runtime.get("x2rgb", {})
    for record in records:
        sample_id = record["id"]
        parts = [value for key, value in rgb2x_runtime.items() if key.startswith(f"{sample_id}:")]
        if sample_id in x2rgb_runtime:
            parts.append(x2rgb_runtime[sample_id])
        if parts:
            runtime[sample_id] = float(sum(parts))
    runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    (PREDICTIONS / "run_config.json").write_text(
        json.dumps(
            {
                "method": "RGB↔X",
                "source_aovs": "RGB→X from source_rgb_512",
                "target_irradiance": "proxy from target_probe_gray",
                "input_size": [512, 512],
                "rgb2x_steps": args.rgb2x_steps,
                "x2rgb_steps": args.x2rgb_steps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
