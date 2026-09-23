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


COMPARE = Path(__file__).resolve().parents[1]
LUMINET = COMPARE / "luminet"
INF_SIZE = 512
ORIGINAL_CWD = Path.cwd()

if str(LUMINET) not in sys.path:
    sys.path.insert(0, str(LUMINET))

os.chdir(LUMINET)
os.environ.setdefault("LUMINET_DISABLE_XFORMERS", "1")
os.environ.setdefault("ATTN_PRECISION", "fp16")

from cldm.ddim_hacked import DDIMSampler  # type: ignore  # noqa: E402
from cldm.model import create_model, load_state_dict  # type: ignore  # noqa: E402


def load_manifest(root: Path, limit: int = 0) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return records[:limit] if limit > 0 else records


def image_np(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def tensor_from_np(img_np: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(img_np.copy()).float()
    tensor = einops.rearrange(tensor, "h w c -> 1 c h w")
    return tensor.to(device)


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
    *,
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
    return elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LumiNet on a prepared VIDIT/ISR/RSR benchmark root.")
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--method-id", default="luminet", help="Prediction directory name.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reference-key",
        default="reference_rgb_512",
        help="Manifest files key used as LumiNet's reference lighting image.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.benchmark_root if args.benchmark_root.is_absolute() else ORIGINAL_CWD / args.benchmark_root
    root = root.resolve()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    records = load_manifest(root, args.limit)
    predictions = root / "predictions" / args.method_id
    predictions.mkdir(parents=True, exist_ok=True)
    runtime_path = predictions / "runtime.json"
    runtimes = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {}

    print("[info] root:", root)
    print("[info] device:", device)
    print("[info] samples:", len(records))
    print("[info] predictions:", predictions)
    print(f"[info] conditioning: source_rgb_512 + {args.reference_key}")
    model, sampler = load_model(device)

    new_runtimes: dict[str, float] = {}
    for index, record in enumerate(records, 1):
        sample_id = record["id"]
        out_path = predictions / record["prediction_name"]
        if out_path.exists() and not args.overwrite:
            continue
        elapsed = run_one(
            model=model,
            sampler=sampler,
            source_path=root / record["files"]["source_rgb_512"],
            reference_path=root / record["files"][args.reference_key],
            out_path=out_path,
            device=device,
            seed=1000 + int(sample_id),
            steps=args.steps,
        )
        runtimes[sample_id] = elapsed
        new_runtimes[sample_id] = elapsed
        if index == 1 or index % 25 == 0:
            print(f"  [{index:04d}/{len(records):04d}] {sample_id} {elapsed:.3f}s")

    runtime_path.write_text(json.dumps(runtimes, indent=2), encoding="utf-8")
    (predictions / "run_config.json").write_text(
        json.dumps(
            {
                "method": "LumiNet",
                "method_id": args.method_id,
                "checkpoint": str(LUMINET / "LumiNet.ckpt"),
                "conditioning": f"source_rgb_512_plus_{args.reference_key}",
                "input_size": [INF_SIZE, INF_SIZE],
                "steps": args.steps,
                "new_predictions": len(new_runtimes),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
