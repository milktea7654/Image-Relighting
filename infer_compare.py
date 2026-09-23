#!/usr/bin/env python3
"""Multi-version inference + side-by-side comparison grid.

Loads checkpoints from any of v0/v1/v2/v3/v5 and produces a comparison
PNG per scene with: cg | reference | pred_v0 | pred_v1 | ... | pred_vX.

Usage:
    python infer_compare.py \
        --dataset d:\coding\relighting\Dataset \
        --scenes scene_000050 scene_000123 scene_000888 ... \
        --v1 d:\coding\relighting\Stage3_ver1\asset\runs\full_30k_v1\checkpoints\last.pt \
        --v3 d:\coding\relighting\Stage3_ver3\asset\runs\full_30k_v3\checkpoints\last.pt \
        --out compare_grid.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent
V1_DIR = str(REPO_ROOT / "Stage3_ver1")
V3_DIR = str(REPO_ROOT / "Stage3_ver3")


def load_rgb(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def load_mask(path: Path | None, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if path is None or not path.exists():
        valid = torch.ones((1, size, size))
    else:
        img = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        valid = torch.from_numpy((arr > 0.5).astype(np.float32))[None]
    return valid, 1.0 - valid


def to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray((arr * 255).astype(np.uint8))


def label(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    d.rectangle([0, 0, out.width, 22], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 255), font=font)
    return out


def build_v1_model(ckpt_path: Path, device: torch.device):
    # Ensure Stage3_ver1 is first so 'train' resolves to ver1
    if V1_DIR not in sys.path:
        sys.path.insert(0, V1_DIR)
    # Remove any cached ver3 train module
    for k in list(sys.modules.keys()):
        if k == "train" or k.startswith("train."):
            del sys.modules[k]
    from train import MiDaSRelightNet  # type: ignore
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    if a.get("no_mask_channel"):
        in_ch = 3
    elif a.get("no_pbr"):
        in_ch = 4
    else:
        in_ch = 9
    model = MiDaSRelightNet(in_channels=in_ch, pretrained=False).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return model, in_ch


def build_v3_model(ckpt_path: Path, device: torch.device):
    import importlib
    # Remove cached ver1 train module, put ver3 first
    for k in list(sys.modules.keys()):
        if k == "train" or k.startswith("train."):
            del sys.modules[k]
    if V3_DIR not in sys.path:
        sys.path.insert(0, V3_DIR)
    # Ensure ver1 is NOT before ver3 for this load
    while V1_DIR in sys.path:
        sys.path.remove(V1_DIR)
    v3 = importlib.import_module("train")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = v3.V3RelightNet(in_channels=4, light_dim=96, pretrained=False).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return model, v3.parse_light_vector


@torch.no_grad()
def run_v1(model, in_ch: int, scene_dir: Path, size: int, device) -> torch.Tensor:
    # prefer render_optimized (matches training); fallback to composite_optimized
    cg_path = scene_dir / "render_optimized.png"
    if not cg_path.exists():
        cg_path = scene_dir / "composite_optimized.png"
    cg = load_rgb(cg_path, size)
    mask_path = _find_mask(scene_dir)
    no_mask = (mask_path is None or not mask_path.exists())
    if in_ch == 3 or no_mask:
        x = cg[None].to(device)
    elif in_ch == 4:
        _, invalid = load_mask(mask_path, size)
        x = torch.cat([cg, invalid], 0)[None].to(device)
    else:
        zeros3 = torch.zeros((3, size, size))
        zeros1 = torch.zeros((1, size, size))
        x = torch.cat([cg, zeros3, torch.zeros(1, size, size), zeros1, zeros1], 0)[None].to(device)
    return model(x)[0].cpu()


@torch.no_grad()
def run_v3(model, parse_light, scene_dir: Path, size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    cg_path = scene_dir / "render_optimized.png"
    if not cg_path.exists():
        cg_path = scene_dir / "composite_optimized.png"
    cg = load_rgb(cg_path, size)
    x = cg[None].to(device)  # 3ch (no mask files in dataset)
    light = parse_light(scene_dir)[None].to(device)
    final, residual = model(x, light)
    return final[0].cpu(), residual[0].cpu()


def _find_mask(scene_dir: Path) -> Path | None:
    for n in ["geometry_mask.png", "valid_mask.png", "mask.png", "invalid_mask.png"]:
        p = scene_dir / n
        if p.exists():
            return p
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--v0", default=None, help="Stage3 (9ch) checkpoint")
    p.add_argument("--v1", default=None, help="Stage3_ver1 (4ch) checkpoint")
    p.add_argument("--v3", default=None, help="Stage3_ver3 (4ch + light FiLM) checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    dataset = Path(args.dataset)

    columns: list[tuple[str, callable]] = []  # (label, fn(scene_dir) -> tensor (3,H,W))

    columns.append(("CG (input)", lambda d: load_rgb(d / "composite_optimized.png", args.size)))
    columns.append(("reference",  lambda d: load_rgb(d / "reference.png",          args.size)))

    if args.v0:
        m0, ch0 = build_v1_model(Path(args.v0), device)
        columns.append((f"v0 (9ch)", lambda d, m=m0, c=ch0: run_v1(m, c, d, args.size, device)))
    if args.v1:
        m1, ch1 = build_v1_model(Path(args.v1), device)
        columns.append((f"v1 (4ch)", lambda d, m=m1, c=ch1: run_v1(m, c, d, args.size, device)))
    if args.v3:
        m3, parse_light = build_v3_model(Path(args.v3), device)
        columns.append((f"v3 (4ch+light)", lambda d, m=m3, pl=parse_light: run_v3(m, pl, d, args.size, device)[0]))

    rows = []
    for name in args.scenes:
        scene_dir = dataset / name
        if not scene_dir.is_dir():
            print(f"[skip] {name} not found")
            continue
        cells = []
        for col_name, fn in columns:
            try:
                img = fn(scene_dir)
            except Exception as e:
                print(f"[err] {name} / {col_name}: {e}")
                img = torch.zeros((3, args.size, args.size))
            cells.append(label(to_pil(img), col_name))
        rows.append((name, cells))

    if not rows:
        raise SystemExit("No scenes rendered.")

    n_cols = len(rows[0][1])
    cell_w = args.size
    cell_h = args.size
    name_w = 100
    grid = Image.new("RGB", (name_w + n_cols * cell_w, len(rows) * cell_h), (30, 30, 30))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for r, (name, cells) in enumerate(rows):
        draw.text((6, r * cell_h + 8), name, fill=(255, 255, 255), font=font)
        for c, cell in enumerate(cells):
            grid.paste(cell, (name_w + c * cell_w, r * cell_h))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(f"Saved {out_path}  ({grid.size})")


if __name__ == "__main__":
    main()
