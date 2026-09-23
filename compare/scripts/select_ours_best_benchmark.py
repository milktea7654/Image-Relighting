from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


OWN_METHODS = ["dcpt-promptir-FT", "hair_FT", "promptir_12ch", "restormer", "xrestormer-FT"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the best local Stage3 model for a benchmark root.")
    parser.add_argument("--benchmark-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.benchmark_root.resolve()
    metrics_path = root / "results" / "table_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Run evaluate_benchmark.py first: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    candidates = [row for row in metrics if row["method_id"] in OWN_METHODS and row.get("rmse") is not None]
    if not candidates:
        raise RuntimeError("No own-model metrics available.")
    best = min(candidates, key=lambda row: row["rmse"])
    src = root / "predictions" / best["method_id"]
    dst = root / "predictions" / "ours-best"
    dst.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    copied = 0
    for record in records:
        src_file = src / record["prediction_name"]
        if src_file.exists():
            shutil.copy2(src_file, dst / record["prediction_name"])
            copied += 1
    runtime_src = src / "runtime.json"
    if runtime_src.exists():
        shutil.copy2(runtime_src, dst / "runtime.json")
    (dst / "selection.json").write_text(
        json.dumps({"selected_method_id": best["method_id"], "criterion": "lowest_rmse", "metrics": best, "copied": copied}, indent=2),
        encoding="utf-8",
    )
    print(f"[done] selected {best['method_id']} as ours-best ({copied} predictions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
