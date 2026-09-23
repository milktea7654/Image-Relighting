from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from diffusers import OmniGenPipeline
from PIL import Image


PROMPT = (
    "Relight the indoor scene in <img><|image_1|></img>. Keep the camera, "
    "geometry, objects, materials, and composition from the first image. "
    "Use <img><|image_2|></img> only as the target illumination and shading "
    "reference. Generate the first scene under that lighting."
)


def load_manifest(root: Path, limit: int = 0) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return records[:limit] if limit > 0 else records


def load_rgb(path: Path, size: int) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def dtype_from_name(name: str, device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return torch.bfloat16


def load_pipeline(args: argparse.Namespace, device: str) -> OmniGenPipeline:
    dtype = dtype_from_name(args.dtype, device)
    pipe = OmniGenPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    if device == "cuda":
        if args.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
    else:
        pipe = pipe.to("cpu")
    return pipe


def run_one(
    *,
    pipe: OmniGenPipeline,
    root: Path,
    record: dict[str, Any],
    out_path: Path,
    device: str,
    args: argparse.Namespace,
) -> float:
    source = load_rgb(root / record["files"]["source_rgb_512"], args.input_size)
    irradiance = load_rgb(root / record["files"]["target_irradiance_512"], args.input_size)
    generator = torch.Generator(device=device if device == "cuda" and not args.cpu_offload else "cpu").manual_seed(
        args.seed + int(record["id"])
    )
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    output = pipe(
        prompt=PROMPT,
        input_images=[source, irradiance],
        height=args.height,
        width=args.width,
        max_input_image_size=args.input_size,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        img_guidance_scale=args.img_guidance_scale,
        generator=generator,
        output_type="pil",
    ).images[0]
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.convert("RGB").save(out_path)
    return elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OmniGen on a prepared benchmark root.")
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--model-id", default="Shitao/OmniGen-v1-diffusers")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--img-guidance-scale", type=float, default=1.6)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--cpu-offload", action="store_true", default=True)
    parser.add_argument("--no-cpu-offload", action="store_false", dest="cpu_offload")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.benchmark_root.resolve()
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    records = load_manifest(root, args.limit)
    predictions = root / "predictions" / "omnigen"
    predictions.mkdir(parents=True, exist_ok=True)
    runtime_path = predictions / "runtime.json"
    runtimes = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {}

    print("[info] root:", root)
    print("[info] device:", device)
    print("[info] samples:", len(records))
    print("[info] conditioning: source_rgb_512 + target_irradiance_512")
    print("[info] model:", args.model_id)
    print("[info] cpu_offload:", args.cpu_offload)
    pipe = load_pipeline(args, device)

    new_runtimes: dict[str, float] = {}
    for index, record in enumerate(records, 1):
        sample_id = record["id"]
        out_path = predictions / record["prediction_name"]
        if out_path.exists() and not args.overwrite:
            continue
        elapsed = run_one(pipe=pipe, root=root, record=record, out_path=out_path, device=device, args=args)
        runtimes[sample_id] = elapsed
        new_runtimes[sample_id] = elapsed
        runtime_path.write_text(json.dumps(runtimes, indent=2), encoding="utf-8")
        if index == 1 or index % 10 == 0:
            print(f"  [{index:04d}/{len(records):04d}] {sample_id} {elapsed:.3f}s")

    runtime_path.write_text(json.dumps(runtimes, indent=2), encoding="utf-8")
    (predictions / "run_config.json").write_text(
        json.dumps(
            {
                "method": "OmniGen",
                "model_id": args.model_id,
                "conditioning": "source_rgb_512_plus_target_irradiance_512",
                "prompt": PROMPT,
                "input_size": [args.input_size, args.input_size],
                "output_size": [args.width, args.height],
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "img_guidance_scale": args.img_guidance_scale,
                "dtype": args.dtype,
                "cpu_offload": args.cpu_offload,
                "new_predictions": len(new_runtimes),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if device == "cuda":
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
