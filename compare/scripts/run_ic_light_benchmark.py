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


COMPARE = Path(__file__).resolve().parents[1]
ICLIGHT = COMPARE / "ic-light"
ORIGINAL_CWD = Path.cwd()

if str(ICLIGHT) not in sys.path:
    sys.path.insert(0, str(ICLIGHT))

os.chdir(ICLIGHT)

import gradio as gr  # noqa: E402


def _no_launch(self, *args, **kwargs):
    print("[info] gradio launch suppressed for headless IC-Light batch")
    return None


gr.Blocks.launch = _no_launch  # type: ignore[assignment]

import gradio_demo_bg as iclight  # type: ignore  # noqa: E402


def load_manifest(root: Path, limit: int = 0) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return records[:limit] if limit > 0 else records


def load_np(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def save_np(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IC-Light on a prepared VIDIT/ISR/RSR benchmark root.")
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--method-id", default="ic-light", help="Prediction directory name.")
    parser.add_argument("--source-key", default="source_rgb_512", help="Manifest files key used as IC-Light foreground/source RGB.")
    parser.add_argument("--background-key", default="reference_rgb_512", help="Manifest files key used as IC-Light uploaded background RGB.")
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--highres-scale", type=float, default=1.0)
    parser.add_argument("--highres-denoise", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.benchmark_root if args.benchmark_root.is_absolute() else ORIGINAL_CWD / args.benchmark_root
    root = root.resolve()
    records = load_manifest(root, args.limit)
    predictions = root / "predictions" / args.method_id
    predictions.mkdir(parents=True, exist_ok=True)
    runtime_path = predictions / "runtime.json"
    runtimes = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {}

    print("[info] root:", root)
    print("[info] samples:", len(records))
    print("[info] predictions:", predictions)
    print(f"[info] conditioning: {args.source_key} foreground + {args.background_key} uploaded background")

    for index, record in enumerate(records, 1):
        sample_id = record["id"]
        out_path = predictions / record["prediction_name"]
        if out_path.exists() and not args.overwrite:
            continue

        input_fg = load_np(root / record["files"][args.source_key])
        input_bg = load_np(root / record["files"][args.background_key])

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
            print(f"  [{index:04d}/{len(records):04d}] {sample_id} {elapsed:.3f}s")

    runtime_path.write_text(json.dumps(runtimes, indent=2), encoding="utf-8")
    (predictions / "run_config.json").write_text(
        json.dumps(
            {
                "method": "IC-Light background-conditioned",
                "method_id": args.method_id,
                "conditioning": f"{args.source_key}_foreground_plus_{args.background_key}_upload_background",
                "source_key": args.source_key,
                "background_key": args.background_key,
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
