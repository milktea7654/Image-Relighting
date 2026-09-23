from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_msssim import ssim


METHOD_LABELS = {
    "rgbx": "RGB<->X",
    "luminet": "LumiNet",
    "ic-light": "IC-Light",
    "stage4": "Ours Stage4",
    "stage4-dcpt-promptir-FT": "Stage4 DCPT-PromptIR-FT",
    "stage4-hair_FT": "Stage4 HAIR-FT",
    "stage4-promptir_12ch": "Stage4 PromptIR-12ch",
    "stage4-restormer": "Stage4 Restormer",
    "stage4-xrestormer-FT": "Stage4 X-Restormer-FT",
    "ours-best": "Ours best",
}

DEFAULT_METHODS = [
    "rgbx",
    "luminet",
    "ic-light",
    "stage4",
    "stage4-dcpt-promptir-FT",
    "stage4-hair_FT",
    "stage4-promptir_12ch",
    "stage4-restormer",
    "stage4-xrestormer-FT",
    "ours-best",
]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_image_tensor(path: Path, size: tuple[int, int], device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def try_lpips(device: torch.device):
    try:
        import lpips
    except Exception:
        return None
    model = lpips.LPIPS(net="alex").to(device).eval()
    return model


def load_runtime(method_dir: Path) -> dict[str, float]:
    path = method_dir / "runtime.json"
    if not path.exists():
        return {}
    return {str(k): float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


@torch.inference_mode()
def image_metrics(pred: torch.Tensor, target: torch.Tensor, lpips_model) -> dict[str, float]:
    mse = torch.nn.functional.mse_loss(target, pred)
    psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-12))
    ssim_value = ssim(target, pred, data_range=1.0, size_average=True)
    if lpips_model is not None:
        lpips_value = lpips_model(target * 2.0 - 1.0, pred * 2.0 - 1.0).mean()
    else:
        lpips_value = torch.tensor(float("nan"), device=target.device)
    mps = (1.0 - lpips_value + ssim_value) / 2.0
    return {
        "mse": float(mse.item()),
        "rmse": float(torch.sqrt(mse).item()),
        "psnr": float(psnr.item()),
        "ssim": float(ssim_value.item()),
        "lpips": float(lpips_value.item()),
        "mps": float(mps.item()),
    }


def evaluate_method(root: Path, records: list[dict], method_id: str, device: torch.device, lpips_model) -> dict:
    method_dir = root / "predictions" / method_id
    runtimes = load_runtime(method_dir)
    rows = []
    missing = []
    resized = 0
    for record in records:
        pred_path = method_dir / record["prediction_name"]
        target_path = root / record["files"]["target_rgb"]
        if not pred_path.exists():
            missing.append(record["id"])
            continue
        with Image.open(target_path) as target_image:
            size = target_image.size
        with Image.open(pred_path) as pred_image:
            if pred_image.size != size:
                resized += 1
        pred = load_image_tensor(pred_path, size, device)
        target = load_image_tensor(target_path, size, device)
        item = {"id": record["id"], **image_metrics(pred, target, lpips_model)}
        rows.append(item)
    return {
        "method": METHOD_LABELS.get(method_id, method_id),
        "method_id": method_id,
        "count": len(rows),
        "missing": len(missing),
        "resized_predictions": resized,
        "mse": float(np.mean([r["mse"] for r in rows])) if rows else None,
        "rmse": float(np.mean([r["rmse"] for r in rows])) if rows else None,
        "ssim": float(np.mean([r["ssim"] for r in rows])) if rows else None,
        "lpips": float(np.mean([r["lpips"] for r in rows])) if rows else None,
        "psnr": float(np.mean([r["psnr"] for r in rows])) if rows else None,
        "mps": float(np.mean([r["mps"] for r in rows])) if rows else None,
        "runtime": float(np.mean([runtimes[r["id"]] for r in rows if r["id"] in runtimes])) if runtimes else None,
        "missing_ids": missing,
    }


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"


def write_outputs(root: Path, metrics: list[dict]) -> None:
    results = root / "results_paper"
    results.mkdir(parents=True, exist_ok=True)
    (results / "table_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (results / "table_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["method", "count", "missing", "mps", "ssim", "lpips", "psnr", "mse", "rmse", "runtime", "resized_predictions"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow({key: row.get(key) for key in fieldnames})
    lines = [
        f"# {root.name} Paper Quantitative",
        "",
        "| Method | Count | MPS ↑ | SSIM ↑ | LPIPS ↓ | PSNR ↑ | MSE ↓ | RMSE ↓ | Runtime ↓ | Missing |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics:
        lines.append(
            f"| {row['method']} | {row['count']} | {fmt(row['mps'])} | {fmt(row['ssim'])} | {fmt(row['lpips'])} | "
            f"{fmt(row['psnr'])} | {fmt(row['mse'])} | {fmt(row['rmse'])} | {fmt(row['runtime'])} | {row['missing']} |"
        )
    (results / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate paper-protocol benchmark predictions.")
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.benchmark_root.resolve()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        records = records[: args.limit]
    lpips_model = try_lpips(device)
    metrics = [evaluate_method(root, records, method_id, device, lpips_model) for method_id in args.methods]
    write_outputs(root, metrics)
    print((root / "results_paper" / "table.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
