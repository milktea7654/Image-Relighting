from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from pytorch_msssim import ssim


METHOD_LABELS = {
    "source-copy": "Source copy",
    "rgbx": "RGB<->X",
    "luminet": "LumiNet",
    "ic-light": "IC-Light",
    "stage4-miiw-xyz": "Ours Stage4 MIIW-XYZ",
    "stage4": "Ours Stage4",
    "ours-best": "Ours best",
}

DEFAULT_METHODS = ["source-copy", "rgbx", "luminet", "ic-light", "stage4-miiw-xyz", "ours-best"]
LUMA = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32)

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
    return lpips.LPIPS(net="alex").to(device).eval()


def load_runtime(method_dir: Path) -> dict[str, float]:
    path = method_dir / "runtime.json"
    if not path.exists():
        return {}
    return {str(k): float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-12))


@torch.inference_mode()
def image_metrics(
    pred: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    lpips_model,
    changed_threshold: float,
) -> dict[str, float]:
    mse = torch.nn.functional.mse_loss(pred, target)
    psnr = psnr_from_mse(mse)
    source_mse = torch.nn.functional.mse_loss(source, target)
    source_psnr = psnr_from_mse(source_mse)
    ssim_value = ssim(target, pred, data_range=1.0, size_average=True)
    if lpips_model is not None:
        lpips_value = lpips_model(target * 2.0 - 1.0, pred * 2.0 - 1.0).mean()
    else:
        lpips_value = torch.tensor(float("nan"), device=target.device)

    luma = LUMA.to(target.device).view(1, 3, 1, 1)
    source_y = (source * luma).sum(dim=1, keepdim=True)
    target_y = (target * luma).sum(dim=1, keepdim=True)
    changed = (target_y - source_y).abs() > float(changed_threshold)
    changed_ratio = changed.float().mean()
    if bool(changed.any()):
        changed_mask = changed.expand_as(target)
        changed_mse = ((pred - target) ** 2)[changed_mask].mean()
        source_changed_mse = ((source - target) ** 2)[changed_mask].mean()
        changed_psnr = psnr_from_mse(changed_mse)
        source_changed_psnr = psnr_from_mse(source_changed_mse)
    else:
        changed_mse = mse
        source_changed_mse = source_mse
        changed_psnr = psnr
        source_changed_psnr = source_psnr

    return {
        "mse": float(mse.item()),
        "rmse": float(torch.sqrt(mse).item()),
        "psnr": float(psnr.item()),
        "source_psnr": float(source_psnr.item()),
        "psnr_gain_over_source": float((psnr - source_psnr).item()),
        "changed_mse": float(changed_mse.item()),
        "changed_rmse": float(torch.sqrt(changed_mse).item()),
        "changed_psnr": float(changed_psnr.item()),
        "source_changed_psnr": float(source_changed_psnr.item()),
        "changed_psnr_gain_over_source": float((changed_psnr - source_changed_psnr).item()),
        "changed_ratio": float(changed_ratio.item()),
        "ssim": float(ssim_value.item()),
        "lpips": float(lpips_value.item()),
    }


def prediction_path(root: Path, record: dict[str, Any], method_id: str) -> Path:
    if method_id == "source-copy":
        return root / record["files"]["source_rgb"]
    return root / "predictions" / method_id / record["prediction_name"]


def evaluate_method(root: Path, records: list[dict[str, Any]], method_id: str, device: torch.device, lpips_model, threshold: float) -> dict:
    method_dir = root / "predictions" / method_id
    runtimes = load_runtime(method_dir)
    rows = []
    missing = []
    resized = 0
    for record in records:
        pred_path = prediction_path(root, record, method_id)
        target_path = root / record["files"]["target_rgb"]
        source_path = root / record["files"]["source_rgb"]
        if not pred_path.exists():
            missing.append(record["id"])
            continue
        with Image.open(target_path) as target_image:
            size = target_image.size
        with Image.open(pred_path) as pred_image:
            if pred_image.size != size:
                resized += 1
        pred = load_image_tensor(pred_path, size, device)
        source = load_image_tensor(source_path, size, device)
        target = load_image_tensor(target_path, size, device)
        rows.append({"id": record["id"], **image_metrics(pred, source, target, lpips_model, threshold)})

    def mean(key: str) -> float | None:
        values = [row[key] for row in rows]
        return float(np.mean(values)) if values else None

    return {
        "method": METHOD_LABELS.get(method_id, method_id),
        "method_id": method_id,
        "count": len(rows),
        "missing": len(missing),
        "resized_predictions": resized,
        "mse": mean("mse"),
        "rmse": mean("rmse"),
        "psnr": mean("psnr"),
        "source_psnr": mean("source_psnr"),
        "psnr_gain_over_source": mean("psnr_gain_over_source"),
        "changed_mse": mean("changed_mse"),
        "changed_rmse": mean("changed_rmse"),
        "changed_psnr": mean("changed_psnr"),
        "source_changed_psnr": mean("source_changed_psnr"),
        "changed_psnr_gain_over_source": mean("changed_psnr_gain_over_source"),
        "changed_ratio": mean("changed_ratio"),
        "ssim": mean("ssim"),
        "lpips": mean("lpips"),
        "runtime": float(np.mean([runtimes[row["id"]] for row in rows if row["id"] in runtimes])) if runtimes else None,
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


def write_outputs(root: Path, metrics: list[dict[str, Any]], threshold: float) -> None:
    results = root / "results_miiw_xyz"
    results.mkdir(parents=True, exist_ok=True)
    (results / "table_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fieldnames = [
        "method",
        "count",
        "missing",
        "psnr",
        "changed_psnr",
        "psnr_gain_over_source",
        "changed_psnr_gain_over_source",
        "ssim",
        "lpips",
        "rmse",
        "changed_rmse",
        "changed_ratio",
        "runtime",
        "resized_predictions",
    ]
    with (results / "table_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow({key: row.get(key) for key in fieldnames})
    lines = [
        "# MIIW Estimated-XYZ Quantitative",
        "",
        f"Changed-region mask: `abs(luminance(target)-luminance(source)) > {threshold}`.",
        "",
        "| Method | Count | Full PSNR ↑ | Changed PSNR ↑ | Full Gain ↑ | Changed Gain ↑ | SSIM ↑ | LPIPS ↓ | RMSE ↓ | Runtime ↓ | Missing |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics:
        lines.append(
            f"| {row['method']} | {row['count']} | {fmt(row['psnr'])} | {fmt(row['changed_psnr'])} | "
            f"{fmt(row['psnr_gain_over_source'])} | {fmt(row['changed_psnr_gain_over_source'])} | "
            f"{fmt(row['ssim'])} | {fmt(row['lpips'])} | {fmt(row['rmse'])} | {fmt(row['runtime'])} | {row['missing']} |"
        )
    (results / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the MIIW estimated-XYZ benchmark.")
    parser.add_argument("--benchmark-root", type=Path, default=Path("compare/miiw_xyz_quantitative"))
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--changed-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.benchmark_root.resolve()
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit > 0:
        records = records[: args.limit]
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    lpips_model = try_lpips(device)
    metrics = [evaluate_method(root, records, method_id, device, lpips_model, args.changed_threshold) for method_id in args.methods]
    write_outputs(root, metrics, args.changed_threshold)
    print((root / "results_miiw_xyz" / "table.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
