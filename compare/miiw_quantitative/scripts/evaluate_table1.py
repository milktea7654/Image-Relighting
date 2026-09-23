from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifest.jsonl"
PREDICTIONS = ROOT / "predictions"
RESULTS = ROOT / "results"

METHODS = [
    ("rgbx", "RGB↔X"),
    ("luminet", "LumiNet"),
    ("ic-light", "IC-Light"),
    ("dcpt-promptir-FT", "DCPT-PromptIR-FT"),
    ("hair_FT", "HAIR-FT"),
    ("promptir_12ch", "PromptIR-12ch"),
    ("restormer", "Restormer"),
    ("xrestormer-FT", "X-Restormer-FT"),
    ("ours-best", "Ours best"),
]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_manifest() -> list[dict]:
    with MANIFEST.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def image_to_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB")).astype(np.float32) / 255.0


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    value = np.mean((pred - target) ** 2)
    if value <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10(1.0 / value))


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
    except Exception as exc:
        raise RuntimeError("SSIM requires scikit-image. Install requirements.txt.") from exc
    return float(structural_similarity(target, pred, channel_axis=2, data_range=1.0))


def try_lpips():
    try:
        import lpips
        import torch
    except Exception:
        return None, None
    model = lpips.LPIPS(net="alex")
    model.eval()
    return model, torch


def lpips_distance(model, torch, pred: np.ndarray, target: np.ndarray) -> float:
    pred_t = torch.from_numpy(pred.transpose(2, 0, 1)).unsqueeze(0) * 2.0 - 1.0
    target_t = torch.from_numpy(target.transpose(2, 0, 1)).unsqueeze(0) * 2.0 - 1.0
    with torch.no_grad():
        return float(model(pred_t, target_t).item())


def load_runtime(method_dir: Path) -> dict[str, float]:
    path = method_dir / "runtime.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(k): float(v) for k, v in data.items()}


def evaluate_method(method_id: str, label: str, records: list[dict], lpips_model, torch_mod) -> dict:
    method_dir = PREDICTIONS / method_id
    runtimes = load_runtime(method_dir)
    rows = []
    missing = []
    resized_predictions = 0
    for record in records:
        pred_path = method_dir / record["prediction_name"]
        target_path = ROOT / record["files"]["target_rgb"]
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
            "ssim": ssim(pred, target),
        }
        if lpips_model is not None:
            item["lpips"] = lpips_distance(lpips_model, torch_mod, pred, target)
        rows.append(item)

    aggregate = {
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
    return aggregate


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    if math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"


def write_outputs(metrics: list[dict]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "table1_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (RESULTS / "table1_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "count",
                "missing",
                "resized_predictions",
                "rmse",
                "psnr",
                "ssim",
                "lpips",
                "runtime",
            ],
        )
        writer.writeheader()
        for row in metrics:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    lines = [
        "# Table 1: MIIW Real Paired Quantitative",
        "",
        "| Method | RMSE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Runtime ↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics:
        lines.append(
            f"| {row['method']} | {fmt(row['rmse'])} | {fmt(row['psnr'])} | {fmt(row['ssim'])} | {fmt(row['lpips'])} | {fmt(row['runtime'])} |"
        )
    (RESULTS / "table1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing {MANIFEST}. Run prepare_miiw_pairs.py first.")
    records = load_manifest()
    lpips_model, torch_mod = try_lpips()
    metrics = [evaluate_method(method_id, label, records, lpips_model, torch_mod) for method_id, label in METHODS]
    write_outputs(metrics)
    print((RESULTS / "table1.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
