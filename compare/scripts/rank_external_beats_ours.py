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


def load_records(root: Path) -> list[dict]:
    return [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def per_image_psnr(root: Path, records: list[dict], method_id: str, target_key: str) -> dict[str, dict]:
    method_dir = root / "predictions" / method_id
    rows: dict[str, dict] = {}
    for record in records:
        pred_path = method_dir / record["prediction_name"]
        if not pred_path.exists():
            continue
        target_path = root / record["files"][target_key]
        target_image = Image.open(target_path).convert("RGB")
        pred_image = Image.open(pred_path).convert("RGB")
        resized = pred_image.size != target_image.size
        if resized:
            pred_image = pred_image.resize(target_image.size, Image.Resampling.LANCZOS)
        rows[record["id"]] = {
            "psnr": psnr(image_to_float(pred_image), image_to_float(target_image)),
            "prediction_path": pred_path,
            "resized_prediction": resized,
        }
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
    label_h = 38
    header_h = 34
    cols = 4
    width = cols * thumb_size + (cols + 1) * gap
    height = header_h + len(rows) * (thumb_size + label_h + gap) + gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, gap), title, fill="black", font=font)
    y = header_h + gap
    for rank, row in enumerate(rows, start=1):
        images = [
            root / row["source_path"],
            root / row["target_path"],
            root / row["ours_prediction_path"],
            root / row["external_prediction_path"],
        ]
        labels = [
            f"{rank}. {row['id']} source",
            "target",
            f"ours {row['ours_psnr']:.2f}",
            f"{row['external_id']} {row['external_psnr']:.2f} gap {row['gap']:.2f}",
        ]
        for col, (path, label) in enumerate(zip(images, labels)):
            x = gap + col * (thumb_size + gap)
            sheet.paste(open_thumb(path, thumb_size), (x, y))
            draw.text((x, y + thumb_size + 2), label, fill="black", font=font)
        y += thumb_size + label_h + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def copy_selected(root: Path, out_dir: Path, group: str, rows: list[dict]) -> None:
    group_dir = out_dir / "selected_images" / group
    group_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(rows, start=1):
        prefix = f"{rank:02d}_{row['id']}"
        for key, suffix in [
            ("source_path", "source"),
            ("target_path", "target"),
            ("ours_prediction_path", "ours"),
            ("external_prediction_path", row["external_id"]),
        ]:
            src = root / row[key]
            shutil.copy2(src, group_dir / f"{prefix}_{suffix}{src.suffix.lower()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find examples where external methods beat ours by PSNR.")
    parser.add_argument("--benchmark-root", type=Path, default=Path("compare/own_test_quantitative"))
    parser.add_argument("--ours-method", default="ours-best")
    parser.add_argument("--external-methods", nargs="+", default=["ic-light", "luminet", "rgbx"])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--target-key", default="target_rgb")
    parser.add_argument("--source-key", default="source_rgb")
    parser.add_argument("--thumb-size", type=int, default=150)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.benchmark_root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else root / "results" / "external_beats_ours"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(root)
    by_id = {record["id"]: record for record in records}
    ours = per_image_psnr(root, records, args.ours_method, args.target_key)

    all_selected: list[dict] = []
    md = [
        "# External Beats Ours",
        "",
        f"Benchmark: `{root.name}`",
        f"Ours method: `{args.ours_method}`",
        "Metric: per-image PSNR against `target_rgb`; positive gap means external PSNR is higher.",
        "",
    ]

    for external_id in args.external_methods:
        external = per_image_psnr(root, records, external_id, args.target_key)
        rows = []
        for sample_id, ext_row in external.items():
            if sample_id not in ours:
                continue
            record = by_id[sample_id]
            gap = ext_row["psnr"] - ours[sample_id]["psnr"]
            if gap <= 0:
                continue
            rows.append(
                {
                    "id": sample_id,
                    "scene_id": record.get("scene_id") or record.get("scene") or "",
                    "external_id": external_id,
                    "gap": gap,
                    "external_psnr": ext_row["psnr"],
                    "ours_psnr": ours[sample_id]["psnr"],
                    "source_path": record["files"][args.source_key],
                    "target_path": record["files"][args.target_key],
                    "ours_prediction_path": rel(ours[sample_id]["prediction_path"], root),
                    "external_prediction_path": rel(ext_row["prediction_path"], root),
                    "external_resized_prediction": ext_row["resized_prediction"],
                    "ours_resized_prediction": ours[sample_id]["resized_prediction"],
                }
            )
        selected = sorted(rows, key=lambda row: row["gap"], reverse=True)[: args.top_k]
        all_selected.extend(selected)
        group = f"{external_id}_beats_{args.ours_method}_top{args.top_k}"
        sheet = f"{group}_contact_sheet.png"
        write_contact_sheet(root, out_dir / sheet, f"{external_id} beats {args.ours_method}", selected, args.thumb_size)
        copy_selected(root, out_dir, group, selected)
        md.extend(
            [
                f"## {external_id}",
                "",
                f"Positive-gap examples: {len(rows)}",
                f"Contact sheet: [{sheet}]({sheet})",
                "",
                "| Rank | Sample | Scene | Gap | External PSNR | Ours PSNR | Source | Target | Ours | External |",
                "| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |",
            ]
        )
        for rank, row in enumerate(selected, start=1):
            md.append(
                f"| {rank} | {row['id']} | {row['scene_id']} | {row['gap']:.3f} | "
                f"{row['external_psnr']:.3f} | {row['ours_psnr']:.3f} | "
                f"[source]({rel(root / row['source_path'], out_dir)}) | "
                f"[target]({rel(root / row['target_path'], out_dir)}) | "
                f"[ours]({rel(root / row['ours_prediction_path'], out_dir)}) | "
                f"[external]({rel(root / row['external_prediction_path'], out_dir)}) |"
            )
        md.append("")

    with (out_dir / "external_beats_ours.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "external_id",
            "id",
            "scene_id",
            "gap",
            "external_psnr",
            "ours_psnr",
            "source_path",
            "target_path",
            "ours_prediction_path",
            "external_prediction_path",
            "external_resized_prediction",
            "ours_resized_prediction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_selected:
            writer.writerow(row)
    (out_dir / "external_beats_ours.json").write_text(json.dumps(all_selected, indent=2), encoding="utf-8")
    (out_dir / "external_beats_ours.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[done] wrote {out_dir / 'external_beats_ours.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
