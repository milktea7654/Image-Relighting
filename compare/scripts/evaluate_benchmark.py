from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


METHODS = [
    ("rgbx", "RGB<->X"),
    ("rgbx-last", "RGB<->X last"),
    ("luminet", "LumiNet"),
    ("luminet-last", "LumiNet last"),
    ("ic-light", "IC-Light"),
    ("ic-light-last", "IC-Light last"),
    ("omnigen", "OmniGen"),
    ("stage4", "Ours Stage4"),
    ("stage4-dcpt-promptir-FT", "Stage4 DCPT-PromptIR-FT"),
    ("stage4-hair_FT", "Stage4 HAIR-FT"),
    ("stage4-promptir_12ch", "Stage4 PromptIR-12ch"),
    ("stage4-restormer", "Stage4 Restormer"),
    ("stage4-xrestormer-FT", "Stage4 X-Restormer-FT"),
    ("dcpt-promptir-FT", "DCPT-PromptIR-FT"),
    ("hair_FT", "HAIR-FT"),
    ("promptir_12ch", "PromptIR-12ch"),
    ("restormer", "Restormer"),
    ("xrestormer-FT", "X-Restormer-FT"),
    ("ours-best", "Ours best"),
]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def image_to_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def try_lpips():
    try:
        import lpips
        import torch
    except Exception:
        return None, None
    model = lpips.LPIPS(net="alex")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.lpips_device = device
    model.eval()
    return model, torch


def lpips_distance(model, torch, pred: np.ndarray, target: np.ndarray) -> float:
    device = getattr(model, "lpips_device", torch.device("cpu"))
    pred_t = (torch.from_numpy(pred.transpose(2, 0, 1)).unsqueeze(0) * 2.0 - 1.0).to(device)
    target_t = (torch.from_numpy(target.transpose(2, 0, 1)).unsqueeze(0) * 2.0 - 1.0).to(device)
    with torch.no_grad():
        return float(model(pred_t, target_t).item())


def load_runtime(method_dir: Path) -> dict[str, float]:
    path = method_dir / "runtime.json"
    if not path.exists():
        return {}
    return {str(k): float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def evaluate_method(root: Path, records: list[dict], method_id: str, label: str, lpips_model, torch_mod) -> dict:
    method_dir = root / "predictions" / method_id
    runtimes = load_runtime(method_dir)
    rows = []
    missing = []
    resized_predictions = 0
    for record in records:
        pred_path = method_dir / record["prediction_name"]
        target_path = root / record["files"]["target_rgb"]
        if not pred_path.exists():
            missing.append(record["id"])
            continue
        target_image = Image.open(target_path).convert("RGB")
        pred_image = Image.open(pred_path).convert("RGB")
        if pred_image.size != target_image.size:
            pred_image = pred_image.resize(target_image.size, Image.Resampling.LANCZOS)
            resized_predictions += 1
        pred = image_to_float(pred_image)
        target = image_to_float(target_image)
        item = {
            "id": record["id"],
            "rmse": rmse(pred, target),
            "psnr": psnr(pred, target),
            "ssim": float(structural_similarity(target, pred, channel_axis=2, data_range=1.0)),
        }
        if lpips_model is not None:
            item["lpips"] = lpips_distance(lpips_model, torch_mod, pred, target)
        rows.append(item)
    return {
        "method": label,
        "method_id": method_id,
        "count": len(rows),
        "missing": len(missing),
        "rmse": float(np.mean([r["rmse"] for r in rows])) if rows else None,
        "psnr": float(np.mean([r["psnr"] for r in rows])) if rows else None,
        "ssim": float(np.mean([r["ssim"] for r in rows])) if rows else None,
        "lpips": float(np.mean([r["lpips"] for r in rows])) if rows and "lpips" in rows[0] else None,
        "runtime": float(np.mean([runtimes[r["id"]] for r in rows if r["id"] in runtimes])) if runtimes else None,
        "resized_predictions": resized_predictions,
        "missing_ids": missing,
    }


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    if math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"


def write_outputs(root: Path, metrics: list[dict]) -> None:
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "table_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (results / "table_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "count", "missing", "resized_predictions", "rmse", "psnr", "ssim", "lpips", "runtime"],
        )
        writer.writeheader()
        for row in metrics:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
    dataset = metrics[0].get("dataset", root.name) if metrics else root.name
    lines = [
        f"# {root.name} Quantitative",
        "",
        "| Method | Count | RMSE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Runtime ↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics:
        lines.append(
            f"| {row['method']} | {row['count']} | {fmt(row['rmse'])} | {fmt(row['psnr'])} | {fmt(row['ssim'])} | {fmt(row['lpips'])} | {fmt(row['runtime'])} |"
        )
    (results / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predictions for a prepared benchmark root.")
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.benchmark_root.resolve()
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        records = records[: args.limit]
    lpips_model, torch_mod = try_lpips()
    metrics = [evaluate_method(root, records, method_id, label, lpips_model, torch_mod) for method_id, label in METHODS]
    write_outputs(root, metrics)
    print((root / "results" / "table.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
