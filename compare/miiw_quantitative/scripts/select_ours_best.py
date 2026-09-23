from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "predictions"
RESULTS = ROOT / "results"
MANIFEST = ROOT / "manifest.jsonl"
OURS_METHODS = [
    "dcpt-promptir-FT",
    "hair_FT",
    "promptir_12ch",
    "restormer",
    "xrestormer-FT",
]


def main() -> int:
    metrics_path = RESULTS / "table1_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Run evaluate_table1.py first: {metrics_path}")
    rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    candidates = [
        row for row in rows
        if row.get("method_id") in OURS_METHODS and row.get("count", 0) > 0 and row.get("rmse") is not None
    ]
    if not candidates:
        raise RuntimeError("No completed own-model metrics found.")
    best = min(candidates, key=lambda row: row["rmse"])
    best_id = best["method_id"]
    dst = PREDICTIONS / "ours-best"
    dst.mkdir(parents=True, exist_ok=True)
    for old in dst.glob("*.png"):
        old.unlink()
    records = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = []
    for record in records:
        name = record["prediction_name"]
        src = PREDICTIONS / best_id / name
        if not src.exists():
            missing.append(name)
            continue
        shutil.copy2(src, dst / name)
    meta = {
        "selected_method_id": best_id,
        "selected_method": best["method"],
        "selection_metric": "aggregate_rmse",
        "rmse": best["rmse"],
        "psnr": best["psnr"],
        "ssim": best["ssim"],
        "lpips": best["lpips"],
        "copied": len(records) - len(missing),
        "missing": missing,
    }
    (dst / "selection.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    runtime_src = PREDICTIONS / best_id / "runtime.json"
    if runtime_src.exists():
        shutil.copy2(runtime_src, dst / "runtime.json")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
