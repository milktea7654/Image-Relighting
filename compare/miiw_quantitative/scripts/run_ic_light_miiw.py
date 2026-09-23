from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
COMPARE = Path(__file__).resolve().parents[2]
ICLIGHT = COMPARE / "ic-light"
MANIFEST = ROOT / "manifest.jsonl"
PREDICTIONS = ROOT / "predictions" / "ic-light"

if str(ICLIGHT) not in sys.path:
    sys.path.insert(0, str(ICLIGHT))

os.chdir(ICLIGHT)

import gradio as gr  # noqa: E402


def _no_launch(self, *args, **kwargs):
    print("[info] gradio launch suppressed for headless IC-Light batch")
    return None


gr.Blocks.launch = _no_launch  # type: ignore[assignment]

import gradio_demo_bg as iclight  # type: ignore  # noqa: E402


def load_manifest(limit: int = 0) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    return records[:limit] if limit > 0 else records


def load_np(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def save_np(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IC-Light on MIIW quantitative split.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--highres-scale", type=float, default=1.0)
    parser.add_argument("--highres-denoise", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_manifest(args.limit)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    runtime_path = PREDICTIONS / "runtime.json"
    runtimes = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {}
    print("[info] samples:", len(records))
    print("[info] conditioning: target_probe_chrome as uploaded background")

    for index, record in enumerate(records, 1):
        sample_id = record["id"]
        out_path = PREDICTIONS / record["prediction_name"]
        if out_path.exists() and not args.overwrite:
            continue

        input_fg = load_np(ROOT / record["files"]["source_rgb_512"])
        input_bg = load_np(ROOT / record["files"]["target_probe_chrome"])

        if hasattr(iclight.torch, "cuda") and iclight.torch.cuda.is_available():
            iclight.torch.cuda.synchronize()
        start = time.perf_counter()
        results, _extra = iclight.process(
            input_fg=input_fg,
            input_bg=input_bg,
            prompt="realistic indoor scene relighting",
            image_width=args.width,
            image_height=args.height,
            num_samples=1,
            seed=args.seed + int(sample_id),
            steps=args.steps,
            a_prompt="best quality",
            n_prompt="lowres, bad anatomy, bad hands, cropped, worst quality",
            cfg=args.cfg,
            highres_scale=args.highres_scale,
            highres_denoise=args.highres_denoise,
            bg_source=iclight.BGSource.UPLOAD.value,
        )
        if hasattr(iclight.torch, "cuda") and iclight.torch.cuda.is_available():
            iclight.torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        pred = (results[0] * 255.0).clip(0, 255).astype(np.uint8)
        save_np(out_path, pred)
        runtimes[sample_id] = elapsed
        if index == 1 or index % 25 == 0:
            print(f"  [{index:03d}/{len(records):03d}] {sample_id} {elapsed:.3f}s")

    runtime_path.write_text(json.dumps(runtimes, indent=2), encoding="utf-8")
    (PREDICTIONS / "run_config.json").write_text(
        json.dumps(
            {
                "method": "IC-Light background-conditioned",
                "conditioning": "target_probe_chrome_upload_background",
                "input_size": [args.width, args.height],
                "steps": args.steps,
                "cfg": args.cfg,
                "highres_scale": args.highres_scale,
                "highres_denoise": args.highres_denoise,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
