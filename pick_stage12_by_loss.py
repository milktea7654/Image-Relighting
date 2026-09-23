#!/usr/bin/env python3
"""
Pick Stage12 examples by optimization loss and export preview folders + mosaics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

IMAGE_KEYS = {
    "reference": ["reference", "reference_path", "ref", "ref_path", "original", "original_path", "input", "input_path", "image", "image_path"],
    "render_optimized": ["render_optimized", "render_optimized_path", "render_optimize", "render_optimize_path", "optimized_render", "optimized_render_path"],
    "composite_optimized": ["composite_optimized", "composite_optimized_path", "optimized_composite", "optimized_composite_path"],
    "residual": ["residual", "residual_path", "residual_png", "residual_image"],
}

LOG_KEYS = ["optimization_log", "optimization_log_path", "log", "log_path"]
LOSS_KEYS = ["final_loss", "loss_final", "best_loss", "last_loss", "rgb_loss", "final_rgb_loss", "loss", "residual_rms", "rms"]
ID_KEYS = ["sample_id", "id", "scene_id", "name", "scene", "scene_name"]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    obj["_jsonl_line"] = line_no
                    rows.append(obj)
            except Exception as e:
                print(f"[WARN] bad JSONL line {line_no}: {e}")
    return rows


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def coerce_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x) if math.isfinite(float(x)) else None
    s = str(x).strip()
    if not s or s.lower() in {"none", "nan", "inf", "-inf", "null"}:
        return None
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def sample_id(row: Dict[str, Any], fallback: str = "") -> str:
    for k in ID_KEYS:
        v = row.get(k)
        if v not in (None, ""):
            return str(v)
    for k in ["scene_dir", "scene_path", "folder", "path"]:
        v = row.get(k)
        if v not in (None, ""):
            return Path(str(v)).name
    return fallback


def resolve_path(value: Any, dataset_root: Path, scene_dir: Optional[Path] = None) -> Optional[Path]:
    if value in (None, ""):
        return None
    s = str(value).strip().strip('"')
    if not s:
        return None
    p = Path(s)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(dataset_root / p)
        if scene_dir is not None:
            candidates.append(scene_dir / p)
        if p.parts and p.parts[0].lower() == dataset_root.name.lower():
            candidates.append(dataset_root.parent / p)
    for c in candidates:
        if c.exists():
            return c
    return candidates[0] if candidates else None


def find_first_path(row: Dict[str, Any], keys: Iterable[str], dataset_root: Path, scene_dir: Optional[Path]) -> Optional[Path]:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return resolve_path(row[k], dataset_root, scene_dir)
    return None


def guess_scene_dir(row: Dict[str, Any], dataset_root: Path) -> Optional[Path]:
    for k in ["scene_dir", "scene_path", "folder", "dir", "sample_dir"]:
        if k in row and row[k] not in (None, ""):
            return resolve_path(row[k], dataset_root)
    for keys in IMAGE_KEYS.values():
        p = find_first_path(row, keys, dataset_root, None)
        if p is not None:
            return p.parent
    return None


def get_loss_from_row(row: Dict[str, Any], preferred_key: str = "auto") -> Optional[Tuple[float, str]]:
    if preferred_key != "auto":
        v = coerce_float(row.get(preferred_key))
        if v is not None:
            return v, preferred_key
    for k in LOSS_KEYS:
        v = coerce_float(row.get(k))
        if v is not None:
            return v, k
    for k, value in row.items():
        kl = k.lower()
        if "loss" in kl or "rms" in kl or "residual" in kl:
            v = coerce_float(value)
            if v is not None:
                return v, k
    return None


def recursive_find_loss(obj: Any) -> Optional[Tuple[float, str]]:
    if isinstance(obj, dict):
        for k in LOSS_KEYS:
            if k in obj:
                v = coerce_float(obj[k])
                if v is not None:
                    return v, k
        for k in ["history", "losses", "iters", "iterations", "records"]:
            if k in obj and isinstance(obj[k], list) and obj[k]:
                for item in reversed(obj[k]):
                    found = recursive_find_loss(item)
                    if found is not None:
                        return found
        for k, v in obj.items():
            if "loss" in k.lower():
                fv = coerce_float(v)
                if fv is not None:
                    return fv, k
        for v in obj.values():
            found = recursive_find_loss(v)
            if found is not None:
                return found
    elif isinstance(obj, list) and obj:
        for item in reversed(obj):
            found = recursive_find_loss(item)
            if found is not None:
                return found
    return None


def get_loss_from_log(log_path: Optional[Path]) -> Optional[Tuple[float, str]]:
    if log_path is None or not log_path.exists():
        return None
    try:
        with log_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        found = recursive_find_loss(obj)
        if found is not None:
            return found[0], f"optimization_log:{found[1]}"
    except Exception as e:
        print(f"[WARN] cannot parse log {log_path}: {e}")
    return None


def merge_rows(index_rows: List[Dict[str, Any]], metric_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not index_rows:
        return metric_rows
    if not metric_rows:
        return index_rows
    metric_by_id: Dict[str, Dict[str, Any]] = {}
    for i, m in enumerate(metric_rows):
        sid = sample_id(m, fallback=f"metric_{i:06d}")
        metric_by_id[sid] = m
    merged = []
    for i, r in enumerate(index_rows):
        sid = sample_id(r, fallback=f"index_{i:06d}")
        out = dict(r)
        if sid in metric_by_id:
            for k, v in metric_by_id[sid].items():
                if k not in out or ("loss" in k.lower() or "rms" in k.lower() or "residual" in k.lower()):
                    out[k] = v
        out["_sample_id"] = sid
        merged.append(out)
    index_ids = {sample_id(r, fallback=f"index_{i:06d}") for i, r in enumerate(index_rows)}
    for i, m in enumerate(metric_rows):
        sid = sample_id(m, fallback=f"metric_{i:06d}")
        if sid not in index_ids:
            out = dict(m)
            out["_sample_id"] = sid
            merged.append(out)
    return merged


def build_samples(rows: List[Dict[str, Any]], dataset_root: Path, loss_field: str) -> List[Dict[str, Any]]:
    samples = []
    for i, row in enumerate(rows):
        sid = row.get("_sample_id") or sample_id(row, fallback=f"sample_{i:06d}")
        scene_dir = guess_scene_dir(row, dataset_root)
        paths = {name: find_first_path(row, keys, dataset_root, scene_dir) for name, keys in IMAGE_KEYS.items()}
        log_path = find_first_path(row, LOG_KEYS, dataset_root, scene_dir)
        row_loss = get_loss_from_row(row, loss_field) or get_loss_from_log(log_path)
        if row_loss is None:
            continue
        loss, loss_source = row_loss
        samples.append({
            "sample_id": sid,
            "loss": float(loss),
            "loss_source": loss_source,
            "scene_dir": str(scene_dir) if scene_dir else "",
            "reference": str(paths["reference"]) if paths["reference"] else "",
            "render_optimized": str(paths["render_optimized"]) if paths["render_optimized"] else "",
            "composite_optimized": str(paths["composite_optimized"]) if paths["composite_optimized"] else "",
            "residual": str(paths["residual"]) if paths["residual"] else "",
            "optimization_log": str(log_path) if log_path else "",
        })
    return samples


def safe_name(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in s)[:120]


def select_samples(samples: List[Dict[str, Any]], n_best: int, n_mid: int, n_worst: int, n_random: int, seed: int) -> Dict[str, List[Dict[str, Any]]]:
    ordered = sorted(samples, key=lambda x: x["loss"])
    out = {
        "best": ordered[:n_best],
        "worst": list(reversed(ordered[-n_worst:])) if n_worst > 0 else [],
        "mid": [],
        "random": [],
    }
    if n_mid > 0 and ordered:
        mid = len(ordered) // 2
        half = n_mid // 2
        start = max(0, mid - half)
        end = min(len(ordered), start + n_mid)
        start = max(0, end - n_mid)
        out["mid"] = ordered[start:end]
    if n_random > 0 and ordered:
        rng = random.Random(seed)
        pool = ordered[:]
        rng.shuffle(pool)
        out["random"] = pool[:min(n_random, len(pool))]
    return out


def copy_selected(selected: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> List[Dict[str, Any]]:
    out_rows = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for category, rows in selected.items():
        cat_dir = out_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        for rank, row in enumerate(rows, 1):
            sid = safe_name(row["sample_id"])
            sample_dir = cat_dir / f"{rank:03d}_loss_{row['loss']:.6g}_{sid}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            copied = {}
            for key in ["reference", "render_optimized", "composite_optimized", "residual"]:
                src_s = row.get(key, "")
                src = Path(src_s) if src_s else None
                if src is not None and src.exists():
                    ext = src.suffix.lower() or ".png"
                    dst = sample_dir / f"{key}{ext}"
                    shutil.copy2(src, dst)
                    copied[key] = str(dst)
                else:
                    copied[key] = ""
            meta = dict(row)
            meta["category"] = category
            meta["rank"] = rank
            meta.update({f"copied_{k}": v for k, v in copied.items()})
            with (sample_dir / "meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            out_rows.append(meta)
    return out_rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    keys = [
        "category", "rank", "sample_id", "loss", "loss_source", "scene_dir",
        "reference", "render_optimized", "composite_optimized", "residual", "optimization_log",
        "copied_reference", "copied_render_optimized", "copied_composite_optimized", "copied_residual",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def make_mosaics(selected: Dict[str, List[Dict[str, Any]]], out_dir: Path, thumb_w: int = 220) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        print("[WARN] Pillow is not installed; skipping mosaics. Install with: pip install pillow")
        return
    mosaic_dir = out_dir / "mosaics"
    mosaic_dir.mkdir(parents=True, exist_ok=True)
    cols = ["reference", "render_optimized", "composite_optimized", "residual"]
    header_h, label_h, gap = 28, 42, 8
    for category, rows in selected.items():
        if not rows:
            continue
        thumbs_by_row, row_heights = [], []
        for row in rows:
            thumbs, max_h = [], 1
            for key in cols:
                p = Path(row.get(key, ""))
                if p.exists():
                    try:
                        img = Image.open(p).convert("RGB")
                        w, h = img.size
                        new_h = max(1, int(h * (thumb_w / max(1, w))))
                        img = img.resize((thumb_w, new_h), Image.LANCZOS)
                    except Exception:
                        img = Image.new("RGB", (thumb_w, thumb_w), (30, 30, 30))
                else:
                    img = Image.new("RGB", (thumb_w, thumb_w), (30, 30, 30))
                thumbs.append(img)
                max_h = max(max_h, img.height)
            thumbs_by_row.append(thumbs)
            row_heights.append(max_h + label_h)
        width = len(cols) * thumb_w + (len(cols) + 1) * gap
        height = header_h + gap + sum(row_heights) + (len(rows) + 1) * gap
        canvas = Image.new("RGB", (width, height), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        x = gap
        for col in cols:
            draw.text((x, 6), col, fill=(0, 0, 0))
            x += thumb_w + gap
        y = header_h + gap
        for row, thumbs, row_h in zip(rows, thumbs_by_row, row_heights):
            draw.text((gap, y), f"{row['sample_id']} | loss={row['loss']:.6g}", fill=(0, 0, 0))
            img_y = y + label_h
            x = gap
            for img in thumbs:
                canvas.paste(img, (x, img_y))
                x += thumb_w + gap
            y += row_h + gap
        canvas.save(mosaic_dir / f"{category}_mosaic.jpg", quality=92)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--index", type=Path, default=None, help="dataset_index.jsonl")
    ap.add_argument("--metrics-csv", type=Path, default=None, help="stage12_quality_metrics.csv")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--loss-field", default="auto", help="Column/key to use as loss, or auto")
    ap.add_argument("--n-best", type=int, default=8)
    ap.add_argument("--n-mid", type=int, default=8)
    ap.add_argument("--n-worst", type=int, default=8)
    ap.add_argument("--n-random", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-mosaic", action="store_true")
    args = ap.parse_args()

    index_path = args.index or (args.dataset_root / "dataset_index.jsonl")
    metrics_path = args.metrics_csv or (args.dataset_root / "splits" / "stage12_quality_metrics.csv")

    print(f"[INFO] dataset_root = {args.dataset_root}")
    print(f"[INFO] index        = {index_path}")
    print(f"[INFO] metrics_csv  = {metrics_path}")
    print(f"[INFO] out_dir      = {args.out_dir}")

    index_rows = read_jsonl(index_path)
    metric_rows = read_csv(metrics_path)
    print(f"[INFO] index rows   = {len(index_rows)}")
    print(f"[INFO] metric rows  = {len(metric_rows)}")

    rows = merge_rows(index_rows, metric_rows)
    samples = build_samples(rows, args.dataset_root, args.loss_field)
    samples = [s for s in samples if math.isfinite(s["loss"])]
    if not samples:
        raise SystemExit("[ERROR] No samples with valid loss found. Check --loss-field, metrics CSV, or optimization_log.json paths.")

    missing_ref = sum(1 for s in samples if not s["reference"] or not Path(s["reference"]).exists())
    missing_render = sum(1 for s in samples if not s["render_optimized"] or not Path(s["render_optimized"]).exists())
    missing_comp = sum(1 for s in samples if not s["composite_optimized"] or not Path(s["composite_optimized"]).exists())

    print(f"[INFO] usable samples with loss = {len(samples)}")
    print(f"[INFO] loss range = {min(s['loss'] for s in samples):.6g} .. {max(s['loss'] for s in samples):.6g}")
    print(f"[INFO] missing reference/render/composite = {missing_ref}/{missing_render}/{missing_comp}")

    selected = select_samples(samples, args.n_best, args.n_mid, args.n_worst, args.n_random, args.seed)
    out_rows = copy_selected(selected, args.out_dir)
    write_csv(out_rows, args.out_dir / "selected_samples.csv")
    if not args.no_mosaic:
        make_mosaics(selected, args.out_dir)

    print("[DONE] Wrote:")
    print(f"  {args.out_dir / 'selected_samples.csv'}")
    print(f"  {args.out_dir / 'best'}")
    print(f"  {args.out_dir / 'mid'}")
    print(f"  {args.out_dir / 'worst'}")
    print(f"  {args.out_dir / 'random'}")
    if not args.no_mosaic:
        print(f"  {args.out_dir / 'mosaics'}")


if __name__ == "__main__":
    main()
