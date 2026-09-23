from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[3]
STAGE3 = WORKSPACE / "Stage3"
RESULT = WORKSPACE / "result"
PREDICTIONS = ROOT / "predictions"
MANIFEST = ROOT / "manifest.jsonl"

if str(STAGE3) not in sys.path:
    sys.path.insert(0, str(STAGE3))

from stage3_unet import build_stage3_model, resolve_stage3_hparams  # type: ignore  # noqa: E402


GRAY_INPUTS = {"roughness", "metallic", "glass_mask"}
DEFAULT_MODELS = [
    "dcpt-promptir-FT",
    "hair_FT",
    "promptir_12ch",
    "restormer",
    "xrestormer-FT",
]


def load_manifest(limit: int = 0) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    return records[:limit] if limit > 0 else records


def find_checkpoint(model_dir: Path) -> Path:
    for name in ["checkpoint_final.pt", "checkpoint_last.pt"]:
        path = model_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No checkpoint_final.pt/checkpoint_last.pt in {model_dir}")


def load_rgb_tensor(path: Path, size: int = 256) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((size, size), Image.BICUBIC)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def save_rgb_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor.detach().float().cpu().clamp(0, 1)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[0] == 1:
        arr = arr.repeat(3, 1, 1)
    image = (arr.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(image).save(path)


def probe_mean(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    intensity = arr.mean(axis=2)
    mask = intensity > max(0.03, float(np.percentile(intensity, 70)) * 0.25)
    if not np.any(mask):
        mask = intensity > 0.0
    if not np.any(mask):
        return np.ones(3, dtype=np.float32)
    return arr[mask].mean(axis=0).astype(np.float32).clip(1e-4, 4.0)


def pseudo_target_render(record: dict[str, Any], source: torch.Tensor) -> torch.Tensor:
    files = record["files"]
    source_probe = probe_mean(ROOT / files["source_probe_gray"])
    target_probe = probe_mean(ROOT / files["target_probe_gray"])
    scale = np.clip(target_probe / np.maximum(source_probe, 1e-4), 0.35, 2.8).astype(np.float32)
    scale_t = torch.from_numpy(scale).view(3, 1, 1)
    return (source * scale_t).clamp(0, 1)


def build_input(record: dict[str, Any], input_names: list[str]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    files = record["files"]
    source = load_rgb_tensor(ROOT / files["source_rgb"], 256)
    render = pseudo_target_render(record, source)

    tensors: dict[str, torch.Tensor] = {
        "composite": source,
        "render": render,
        "albedo": source,
        "roughness": torch.zeros(1, 256, 256, dtype=torch.float32),
        "metallic": torch.zeros(1, 256, 256, dtype=torch.float32),
        "glass_mask": torch.zeros(1, 256, 256, dtype=torch.float32),
        "normal": torch.zeros(3, 256, 256, dtype=torch.float32),
        "shading": render,
    }

    x = torch.cat([tensors[name] for name in input_names], dim=0).unsqueeze(0)
    meta = {
        "conditioning": "miiw_probe_ratio_pseudo_render",
        "composite": files["source_rgb"],
        "render": "source_rgb_scaled_by_target_over_source_gray_probe_mean",
        "albedo": "source_rgb",
        "roughness": "zeros",
        "metallic": "zeros",
        "glass_mask": "zeros",
    }
    return x, source.unsqueeze(0), meta


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    input_names = list(ckpt.get("input_names") or ckpt.get("inputs") or [])
    if not input_names:
        raise RuntimeError(f"Checkpoint missing input_names: {ckpt_path}")
    channels = int(ckpt.get("channels", 0)) or sum(1 if name in GRAY_INPUTS else 3 for name in input_names)
    backbone = str(ckpt.get("backbone", "unet"))
    base_channels, residual_scale = resolve_stage3_hparams(
        backbone,
        int(ckpt["base_channels"]) if ckpt.get("base_channels") is not None else None,
        float(ckpt["residual_scale"]) if ckpt.get("residual_scale") is not None else None,
    )
    model = build_stage3_model(
        backbone=backbone,
        input_channels=channels,
        base_channels=base_channels,
        residual_scale=residual_scale,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, input_names, {
        "checkpoint": str(ckpt_path),
        "backbone": backbone,
        "channels": channels,
        "base_channels": base_channels,
        "residual_scale": residual_scale,
        "input_names": input_names,
    }


def run_model(model_name: str, records: list[dict[str, Any]], device: torch.device, overwrite: bool) -> None:
    model_dir = RESULT / model_name
    ckpt_path = find_checkpoint(model_dir)
    output_dir = PREDICTIONS / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[model] {model_name}")
    model, input_names, model_meta = load_model(ckpt_path, device)
    runtimes: dict[str, float] = {}
    skipped = 0
    input_meta: dict[str, Any] | None = None

    for index, record in enumerate(records, 1):
        sample_id = record["id"]
        out_path = output_dir / f"{sample_id}.png"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        x, composite, meta = build_input(record, input_names)
        input_meta = meta
        x = x.to(device, non_blocking=True)
        composite = composite.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            pred = model(x, composite)
        if device.type == "cuda":
            torch.cuda.synchronize()
        runtimes[sample_id] = time.perf_counter() - start
        save_rgb_tensor(out_path, pred)

        if index == 1 or index % 25 == 0:
            print(f"  [{index:03d}/{len(records):03d}] {sample_id}")

    if skipped:
        print(f"  [skip] existing predictions: {skipped}")

    existing_runtime = {}
    runtime_path = output_dir / "runtime.json"
    if runtime_path.exists():
        existing_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    existing_runtime.update(runtimes)
    runtime_path.write_text(json.dumps(existing_runtime, indent=2), encoding="utf-8")

    run_meta = {
        **model_meta,
        "method_id": model_name,
        "miiw_manifest": str(MANIFEST),
        "image_size": [256, 256],
        "input_conditioning": input_meta,
        "samples": len(records),
        "new_predictions": len(runtimes),
        "skipped_existing": skipped,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run result/* Stage3 models on MIIW quantitative split.")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    records = load_manifest(args.limit)
    print("[info] device:", device)
    print("[info] samples:", len(records))
    for model_name in args.models:
        run_model(model_name, records, device, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
