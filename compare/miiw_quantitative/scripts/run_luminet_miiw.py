from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import einops
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
COMPARE = Path(__file__).resolve().parents[2]
LUMINET = COMPARE / "luminet"
MANIFEST = ROOT / "manifest.jsonl"
PREDICTIONS = ROOT / "predictions" / "luminet"

if str(LUMINET) not in sys.path:
    sys.path.insert(0, str(LUMINET))

os.chdir(LUMINET)
os.environ.setdefault("LUMINET_DISABLE_XFORMERS", "1")
os.environ.setdefault("ATTN_PRECISION", "fp16")

from cldm.ddim_hacked import DDIMSampler  # type: ignore  # noqa: E402
from cldm.model import create_model, load_state_dict  # type: ignore  # noqa: E402


INF_SIZE = 512


def load_manifest(limit: int = 0) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    return records[:limit] if limit > 0 else records


def image_np(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def tensor_from_np(img_np: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(img_np.copy()).float()
    t = einops.rearrange(t, "h w c -> 1 c h w")
    return t.to(device)


def save_np(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((arr.clip(0, 1) * 255.0).round().astype(np.uint8)).save(path)


def load_model(device: torch.device):
    model = create_model(str(LUMINET / "models" / "cldm_v21_LumiNet.yaml")).cpu()
    model.add_new_layers()
    model.concat = False
    state = load_state_dict(str(LUMINET / "LumiNet.ckpt"), location="cpu")
    model.load_state_dict(state)
    model.parameterization = "v"
    model = model.to(device).eval()
    return model, DDIMSampler(model)


@torch.inference_mode()
def run_one(
    model,
    sampler: DDIMSampler,
    source_path: Path,
    reference_path: Path,
    out_path: Path,
    device: torch.device,
    seed: int,
    steps: int,
) -> float:
    input_np = cv2.resize(image_np(source_path), (INF_SIZE, INF_SIZE), interpolation=cv2.INTER_LANCZOS4)
    ref_np = cv2.resize(image_np(reference_path), (INF_SIZE, INF_SIZE), interpolation=cv2.INTER_LANCZOS4)
    control_feat = np.concatenate((input_np, ref_np), axis=2).astype(np.float32)
    control = tensor_from_np(control_feat, device)
    input_tensor = tensor_from_np(input_np, device)

    c_cat = control
    c = model.get_unconditional_conditioning(1)
    uc_cross = model.get_unconditional_conditioning(1)
    uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
    cond = {"c_concat": [c_cat], "c_crossattn": [c]}
    shape = (4, INF_SIZE // 8, INF_SIZE // 8)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.synchronize()
    random.seed(seed)
    start = time.perf_counter()
    samples, _ = sampler.sample(
        S=steps,
        batch_size=1,
        shape=shape,
        conditioning=cond,
        verbose=False,
        eta=0.0,
        unconditional_guidance_scale=9.0,
        unconditional_conditioning=uc_full,
    )
    x = model.decode_first_stage(samples)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    x = (x.squeeze(0) + 1.0) / 2.0
    x = x.clamp(0, 1)
    arr = einops.rearrange(x, "c h w -> h w c").detach().cpu().numpy()
    save_np(out_path, arr)
    del input_tensor
    return elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LumiNet on MIIW quantitative split.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    records = load_manifest(args.limit)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    print("[info] device:", device)
    print("[info] samples:", len(records))
    print("[info] conditioning: target_probe_chrome as 512x512 reference image")
    model, sampler = load_model(device)
    runtimes: dict[str, float] = {}
    existing_runtime = {}
    runtime_path = PREDICTIONS / "runtime.json"
    if runtime_path.exists():
        existing_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))

    for index, record in enumerate(records, 1):
        sample_id = record["id"]
        out_path = PREDICTIONS / record["prediction_name"]
        if out_path.exists() and not args.overwrite:
            continue
        elapsed = run_one(
            model=model,
            sampler=sampler,
            source_path=ROOT / record["files"]["source_rgb_512"],
            reference_path=ROOT / record["files"]["target_probe_chrome"],
            out_path=out_path,
            device=device,
            seed=1000 + int(sample_id),
            steps=args.steps,
        )
        runtimes[sample_id] = elapsed
        if index == 1 or index % 25 == 0:
            print(f"  [{index:03d}/{len(records):03d}] {sample_id} {elapsed:.3f}s")

    existing_runtime.update(runtimes)
    runtime_path.write_text(json.dumps(existing_runtime, indent=2), encoding="utf-8")
    (PREDICTIONS / "run_config.json").write_text(
        json.dumps(
            {
                "method": "LumiNet",
                "checkpoint": str(LUMINET / "LumiNet.ckpt"),
                "conditioning": "target_probe_chrome_reference",
                "input_size": [512, 512],
                "steps": args.steps,
                "new_predictions": len(runtimes),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
