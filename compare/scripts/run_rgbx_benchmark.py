from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torchvision
import diffusers.utils as diffusers_utils
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DDIMScheduler
from PIL import Image


COMPARE = Path(__file__).resolve().parents[1]
RGBX = COMPARE / "rgbx"
RGB2X_DIR = RGBX / "rgb2x"
X2RGB_DIR = RGBX / "x2rgb"

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

if not hasattr(diffusers_utils, "randn_tensor"):
    diffusers_utils.randn_tensor = randn_tensor

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
REQUIRED_AOVS = ["albedo", "normal", "roughness", "metallic", "irradiance"]
PROMPTS = {
    "albedo": "Albedo (diffuse basecolor)",
    "normal": "Camera-space Normal",
    "roughness": "Roughness",
    "metallic": "Metallicness",
    "irradiance": "Irradiance (diffuse lighting)",
}


def load_manifest(root: Path, limit: int = 0) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
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


def copy_target_irradiance(root: Path, record: dict[str, Any], out_path: Path, target_irradiance_key: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if target_irradiance_key not in record["files"]:
        raise KeyError(f"{record['id']} is missing files.{target_irradiance_key}")
    with Image.open(root / record["files"][target_irradiance_key]) as image:
        image.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC).save(out_path)


def run_rgb2x_stage(
    *,
    root: Path,
    records: list[dict[str, Any]],
    intermediates: Path,
    device: str,
    steps: int,
    seed: int,
    overwrite: bool,
    target_irradiance_key: str,
    refresh_target_irradiance: bool,
) -> dict[str, float]:
    pipe = None
    runtimes: dict[str, float] = {}
    for index, record in enumerate(records, 1):
        sample_id = record["id"]
        sample_dir = intermediates / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        source_expected = [sample_dir / f"{name}.png" for name in SOURCE_AOVS]
        irr_path = sample_dir / "irradiance.png"
        expected = source_expected + [irr_path]
        if all(path.exists() for path in expected) and not overwrite and not refresh_target_irradiance:
            continue

        generator = torch.Generator(device=device).manual_seed(seed + int(sample_id))
        start = time.perf_counter()
        for aov_name in SOURCE_AOVS:
            out_path = sample_dir / f"{aov_name}.png"
            if out_path.exists() and not overwrite:
                continue
            if pipe is None:
                pipe = build_rgb2x_pipeline(device)
            photo = rgb2x_load_ldr_image(str(root / record["files"]["source_rgb_512"]), from_srgb=True).to(device)
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

        if not irr_path.exists() or overwrite or refresh_target_irradiance:
            copy_target_irradiance(root, record, irr_path, target_irradiance_key)
        runtimes[f"{sample_id}:rgb2x_total"] = time.perf_counter() - start
        if index == 1 or index % 10 == 0:
            print(f"  [rgb2x {index:04d}/{len(records):04d}] {sample_id}")
    if pipe is not None:
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
    *,
    root: Path,
    records: list[dict[str, Any]],
    predictions: Path,
    intermediates: Path,
    device: str,
    steps: int,
    seed: int,
    overwrite: bool,
) -> dict[str, float]:
    pipe = build_x2rgb_pipeline(device)
    runtimes: dict[str, float] = {}
    predictions.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records, 1):
        sample_id = record["id"]
        out_path = predictions / record["prediction_name"]
        if out_path.exists() and not overwrite:
            continue

        sample_dir = intermediates / sample_id
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
            print(f"  [x2rgb {index:04d}/{len(records):04d}] {sample_id} {elapsed:.3f}s")
    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()
    return runtimes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RGB<->X on a prepared VIDIT/ISR/RSR benchmark root.")
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rgb2x-steps", type=int, default=20)
    parser.add_argument("--x2rgb-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--method-id", default="rgbx", help="Prediction directory name.")
    parser.add_argument("--intermediate-id", default="rgbx_intermediates", help="Intermediate AOV directory name.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-rgb2x", action="store_true", help="Regenerate source RGB->X AOVs.")
    parser.add_argument("--overwrite-x2rgb", action="store_true", help="Regenerate final RGBX predictions.")
    parser.add_argument(
        "--refresh-target-irradiance",
        action="store_true",
        help="Refresh intermediate irradiance.png from --target-irradiance-key without regenerating existing source AOVs.",
    )
    parser.add_argument("--skip-rgb2x", action="store_true")
    parser.add_argument("--skip-x2rgb", action="store_true")
    parser.add_argument(
        "--target-irradiance-key",
        default="target_irradiance_512",
        help="Manifest files key used as the target irradiance AOV for X->RGB.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.benchmark_root.resolve()
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    records = load_manifest(root, args.limit)
    predictions = root / "predictions" / args.method_id
    intermediates = root / "predictions" / args.intermediate_id
    predictions.mkdir(parents=True, exist_ok=True)
    intermediates.mkdir(parents=True, exist_ok=True)

    print("[info] root:", root)
    print("[info] device:", device)
    print("[info] samples:", len(records))
    print("[info] predictions:", predictions)
    print("[info] intermediates:", intermediates)
    print(f"[info] RGBX source AOVs from source_rgb_512; target irradiance from {args.target_irradiance_key}")

    stage_runtime: dict[str, Any] = {}
    if not args.skip_rgb2x:
        stage_runtime["rgb2x"] = run_rgb2x_stage(
            root=root,
            records=records,
            intermediates=intermediates,
            device=device,
            steps=args.rgb2x_steps,
            seed=args.seed,
            overwrite=args.overwrite or args.overwrite_rgb2x,
            target_irradiance_key=args.target_irradiance_key,
            refresh_target_irradiance=args.refresh_target_irradiance,
        )
    if not args.skip_x2rgb:
        stage_runtime["x2rgb"] = run_x2rgb_stage(
            root=root,
            records=records,
            predictions=predictions,
            intermediates=intermediates,
            device=device,
            steps=args.x2rgb_steps,
            seed=args.seed,
            overwrite=args.overwrite or args.overwrite_x2rgb,
        )

    runtime_path = predictions / "runtime.json"
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
    (predictions / "run_config.json").write_text(
        json.dumps(
            {
                "method": "RGB<->X",
                "method_id": args.method_id,
                "intermediate_id": args.intermediate_id,
                "source_aovs": "RGB->X from source_rgb_512",
                "target_irradiance": args.target_irradiance_key,
                "input_size": [512, 512],
                "rgb2x_steps": args.rgb2x_steps,
                "x2rgb_steps": args.x2rgb_steps,
                "refresh_target_irradiance": args.refresh_target_irradiance,
                "overwrite_rgb2x": args.overwrite or args.overwrite_rgb2x,
                "overwrite_x2rgb": args.overwrite or args.overwrite_x2rgb,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
