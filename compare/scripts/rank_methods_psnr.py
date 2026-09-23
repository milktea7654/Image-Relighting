from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def image_to_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def load_records(root: Path, limit: int = 0) -> list[dict]:
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return records[:limit] if limit else records


def rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def evaluate_method(root: Path, records: list[dict], method_id: str, target_key: str, source_key: str) -> list[dict]:
    rows: list[dict] = []
    method_dir = root / "predictions" / method_id
    for record in records:
        pred_path = method_dir / record["prediction_name"]
        if not pred_path.exists():
            continue
        target_path = root / record["files"][target_key]
        source_path = root / record["files"][source_key]
        target_image = Image.open(target_path).convert("RGB")
        pred_image = Image.open(pred_path).convert("RGB")
        resized_prediction = pred_image.size != target_image.size
        if resized_prediction:
            pred_image = pred_image.resize(target_image.size, Image.Resampling.LANCZOS)
        rows.append(
            {
                "id": record["id"],
                "scene_id": record.get("scene_id") or record.get("scene") or "",
                "psnr": psnr(image_to_float(pred_image), image_to_float(target_image)),
                "resized_prediction": resized_prediction,
                "source_path": source_path,
                "target_path": target_path,
                "prediction_path": pred_path,
                "source_name": record.get("source_name", ""),
                "target_name": record.get("target_name", ""),
            }
        )
    return rows


def open_thumb(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image


def write_contact_sheet(root: Path, out_path: Path, title: str, rows: list[dict], thumb_size: int) -> None:
    if not rows:
        return
    font = ImageFont.load_default()
    gap = 8
    label_h = 34
    header_h = 34
    cols = 3
    width = cols * thumb_size + (cols + 1) * gap
    height = header_h + len(rows) * (thumb_size + label_h + gap) + gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, gap), title, fill="black", font=font)
    y = header_h + gap
    for rank, row in enumerate(rows, start=1):
        paths = [root / row["source_path"], root / row["target_path"], root / row["prediction_path"]]
        labels = [f"{rank}. {row['id']} source", f"target PSNR {row['psnr']:.3f}", "prediction"]
        for col, (path, label) in enumerate(zip(paths, labels)):
            x = gap + col * (thumb_size + gap)
            sheet.paste(open_thumb(path, thumb_size), (x, y))
            draw.text((x, y + thumb_size + 2), label, fill="black", font=font)
        y += thumb_size + label_h + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def copy_images(root: Path, out_dir: Path, group: str, rows: list[dict]) -> None:
    group_dir = out_dir / "selected_images" / group
    group_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(rows, start=1):
        prefix = f"{rank:02d}_{row['id']}"
        for key, suffix in [("source_path", "source"), ("target_path", "target"), ("prediction_path", "prediction")]:
            src = root / row[key]
            shutil.copy2(src, group_dir / f"{prefix}_{suffix}{src.suffix.lower()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank method predictions by per-image PSNR.")
    parser.add_argument("--benchmark-root", type=Path, default=Path("compare/own_test_quantitative"))
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--order", choices=["best", "worst"], default="best")
    parser.add_argument("--target-key", default="target_rgb")
    parser.add_argument("--source-key", default="source_rgb")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--thumb-size", type=int, default=160)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.benchmark_root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else root / "results" / "method_psnr_rankings"
    labels = dict(zip(args.methods, args.labels)) if args.labels else {}
    records = load_records(root, args.limit)
    reverse = args.order == "best"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    md = [
        "# Method PSNR Rankings",
        "",
        f"Benchmark: `{root.name}`",
        f"Target key: `{args.target_key}`",
        f"Order: `{args.order}`",
        "",
    ]
    for method_id in args.methods:
        label = labels.get(method_id, method_id)
        rows = evaluate_method(root, records, method_id, args.target_key, args.source_key)
        selected = sorted(rows, key=lambda row: row["psnr"], reverse=reverse)[: args.top_k]
        group = f"{method_id}_{args.order}{args.top_k}".replace("<", "").replace(">", "").replace(":", "")
        sheet = f"{group}_contact_sheet.png"
        write_contact_sheet(root, out_dir / sheet, f"{label} {args.order} PSNR {args.top_k}", selected, args.thumb_size)
        copy_images(root, out_dir, group, selected)
        md.extend(
            [
                f"## {label}",
                "",
                f"Predictions found: {len(rows)}",
                f"Contact sheet: [{sheet}]({sheet})",
                "",
                "| Rank | Sample | Scene | PSNR | Source | Target | Prediction |",
                "| ---: | --- | --- | ---: | --- | --- | --- |",
            ]
        )
        for rank, row in enumerate(selected, start=1):
            all_rows.append({"method_id": method_id, "method": label, "rank": rank, **row})
            md.append(
                f"| {rank} | {row['id']} | {row['scene_id']} | {row['psnr']:.3f} | "
                f"[source]({rel(root / row['source_path'], out_dir)}) | "
                f"[target]({rel(root / row['target_path'], out_dir)}) | "
                f"[prediction]({rel(root / row['prediction_path'], out_dir)}) |"
            )
        md.append("")

    with (out_dir / "rankings.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "method_id",
            "method",
            "rank",
            "id",
            "scene_id",
            "psnr",
            "resized_prediction",
            "source_path",
            "target_path",
            "prediction_path",
            "source_name",
            "target_name",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(
                {
                    **row,
                    "source_path": rel(root / row["source_path"], root),
                    "target_path": rel(root / row["target_path"], root),
                    "prediction_path": rel(root / row["prediction_path"], root),
                }
            )
    (out_dir / "rankings.json").write_text(json.dumps(all_rows, indent=2, default=str), encoding="utf-8")
    (out_dir / "rankings.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[done] wrote {out_dir / 'rankings.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
