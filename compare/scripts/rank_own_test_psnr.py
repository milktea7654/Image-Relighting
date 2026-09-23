from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


RANK_SPECS = [
    {
        "name": "hair_worst10",
        "title": "HAIR-FT worst PSNR 10",
        "method_id": "hair_FT",
        "label": "HAIR-FT",
        "order": "asc",
    },
    {
        "name": "luminet_best10",
        "title": "LumiNet best PSNR 10",
        "method_id": "luminet",
        "label": "LumiNet",
        "order": "desc",
    },
    {
        "name": "rgbx_best10",
        "title": "RGB<->X best PSNR 10",
        "method_id": "rgbx",
        "label": "RGB<->X",
        "order": "desc",
    },
]


def image_to_float(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def load_records(root: Path, limit: int = 0) -> list[dict]:
    path = root / "manifest.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        return records[:limit]
    return records


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
        target_path = root / record["files"][target_key]
        source_path = root / record["files"].get(source_key, record["files"].get("source_rgb", ""))
        if not pred_path.exists() or not target_path.exists():
            continue

        target_image = Image.open(target_path).convert("RGB")
        pred_image = Image.open(pred_path).convert("RGB")
        resized_prediction = pred_image.size != target_image.size
        if resized_prediction:
            pred_image = pred_image.resize(target_image.size, Image.Resampling.LANCZOS)

        pred = np.asarray(pred_image, dtype=np.float32) / 255.0
        target = np.asarray(target_image, dtype=np.float32) / 255.0
        rows.append(
            {
                "id": record["id"],
                "scene_id": record.get("scene_id") or record.get("scene") or "",
                "source_name": record.get("source_name", ""),
                "target_name": record.get("target_name", ""),
                "psnr": psnr(pred, target),
                "resized_prediction": resized_prediction,
                "source_path": source_path,
                "target_path": target_path,
                "prediction_path": pred_path,
            }
        )
    return rows


def sort_rows(rows: list[dict], order: str, top_k: int) -> list[dict]:
    reverse = order == "desc"
    return sorted(rows, key=lambda row: row["psnr"], reverse=reverse)[:top_k]


def open_for_sheet(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def write_contact_sheet(root: Path, out_path: Path, title: str, rows: list[dict], thumb_size: int = 160) -> None:
    if not rows:
        return
    font = ImageFont.load_default()
    label_h = 34
    row_h = thumb_size + label_h
    header_h = 34
    gap = 8
    cols = 3
    width = cols * thumb_size + (cols + 1) * gap
    height = header_h + len(rows) * row_h + (len(rows) + 1) * gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, gap), title, fill="black", font=font)

    y = header_h + gap
    for rank, row in enumerate(rows, start=1):
        source = open_for_sheet(root / row["source_path"], (thumb_size, thumb_size))
        target = open_for_sheet(root / row["target_path"], (thumb_size, thumb_size))
        pred = open_for_sheet(root / row["prediction_path"], (thumb_size, thumb_size))
        images = [source, target, pred]
        labels = [
            f"{rank}. {row['id']} source",
            f"target PSNR {row['psnr']:.3f}",
            "prediction",
        ]
        for col, image in enumerate(images):
            x = gap + col * (thumb_size + gap)
            sheet.paste(image, (x, y))
            draw.text((x, y + thumb_size + 2), labels[col], fill="black", font=font)
        y += row_h + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def copy_selected_images(root: Path, out_dir: Path, group_name: str, rows: list[dict]) -> None:
    group_dir = out_dir / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(rows, start=1):
        prefix = f"{rank:02d}_{row['id']}"
        for key, suffix in [
            ("source_path", "source"),
            ("target_path", "target"),
            ("prediction_path", "prediction"),
        ]:
            src = root / row[key]
            dst = group_dir / f"{prefix}_{suffix}{src.suffix.lower()}"
            shutil.copy2(src, dst)


def write_outputs(root: Path, out_dir: Path, selections: list[tuple[dict, list[dict]]], copy_images: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "rankings.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "group",
            "rank",
            "method_id",
            "method",
            "sample_id",
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
        for spec, rows in selections:
            for rank, row in enumerate(rows, start=1):
                writer.writerow(
                    {
                        "group": spec["name"],
                        "rank": rank,
                        "method_id": spec["method_id"],
                        "method": spec["label"],
                        "sample_id": row["id"],
                        "scene_id": row["scene_id"],
                        "psnr": row["psnr"],
                        "resized_prediction": row["resized_prediction"],
                        "source_path": rel(root / row["source_path"], root),
                        "target_path": rel(root / row["target_path"], root),
                        "prediction_path": rel(root / row["prediction_path"], root),
                        "source_name": row["source_name"],
                        "target_name": row["target_name"],
                    }
                )

    json_rows = []
    for spec, rows in selections:
        json_rows.append(
            {
                "group": spec["name"],
                "method_id": spec["method_id"],
                "method": spec["label"],
                "rows": [
                    {
                        **{k: v for k, v in row.items() if k not in {"source_path", "target_path", "prediction_path"}},
                        "source_path": rel(root / row["source_path"], root),
                        "target_path": rel(root / row["target_path"], root),
                        "prediction_path": rel(root / row["prediction_path"], root),
                    }
                    for row in rows
                ],
            }
        )
    (out_dir / "rankings.json").write_text(json.dumps(json_rows, indent=2), encoding="utf-8")

    md_lines = [
        "# own_test_quantitative PSNR rankings",
        "",
        "Target is `target_rgb` and predictions are resized to the target size before PSNR, matching `evaluate_benchmark.py`.",
        "",
    ]
    for spec, rows in selections:
        sheet_name = f"{spec['name']}_contact_sheet.png"
        write_contact_sheet(root, out_dir / sheet_name, spec["title"], rows)
        if copy_images:
            copy_selected_images(root, out_dir / "selected_images", spec["name"], rows)
        md_lines.extend(
            [
                f"## {spec['title']}",
                "",
                f"Contact sheet: [{sheet_name}]({sheet_name})",
                "",
                "| Rank | Sample | Scene | PSNR | Source | Target | Prediction |",
                "| ---: | --- | --- | ---: | --- | --- | --- |",
            ]
        )
        for rank, row in enumerate(rows, start=1):
            source_rel = rel(root / row["source_path"], out_dir)
            target_rel = rel(root / row["target_path"], out_dir)
            pred_rel = rel(root / row["prediction_path"], out_dir)
            md_lines.append(
                f"| {rank} | {row['id']} | {row['scene_id']} | {row['psnr']:.3f} | "
                f"[source]({source_rel}) | [target]({target_rel}) | [prediction]({pred_rel}) |"
            )
        md_lines.append("")
    (out_dir / "rankings.md").write_text("\n".join(md_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank selected own_test_quantitative images by per-image PSNR.")
    parser.add_argument("--benchmark-root", type=Path, default=Path("compare/own_test_quantitative"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--target-key", default="target_rgb")
    parser.add_argument("--source-key", default="source_rgb")
    parser.add_argument("--no-copy-images", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.benchmark_root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else root / "results" / "psnr_rankings"
    records = load_records(root, args.limit)

    selections = []
    for spec in RANK_SPECS:
        rows = evaluate_method(root, records, spec["method_id"], args.target_key, args.source_key)
        selected = sort_rows(rows, spec["order"], args.top_k)
        selections.append((spec, selected))

    write_outputs(root, out_dir, selections, copy_images=not args.no_copy_images)
    print(f"[done] wrote {out_dir / 'rankings.md'}")
    print(f"[done] wrote {out_dir / 'rankings.csv'}")
    print(f"[done] wrote contact sheets and selected image copies under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
