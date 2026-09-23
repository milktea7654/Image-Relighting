from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def rgb_to_luma(x: np.ndarray) -> np.ndarray:
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def pil_morphology(mask: np.ndarray, dilate: int, erode: int, blur_radius: float) -> np.ndarray:
    img = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8), mode="L")
    for _ in range(max(0, dilate)):
        img = img.filter(ImageFilter.MaxFilter(3))
    for _ in range(max(0, erode)):
        img = img.filter(ImageFilter.MinFilter(3))
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return np.asarray(img).astype(np.float32) / 255.0


def save_rgb(path: Path, x: np.ndarray) -> None:
    Image.fromarray((np.clip(x, 0, 1) * 255).round().astype(np.uint8), mode="RGB").save(path)


def save_l(path: Path, x: np.ndarray) -> None:
    Image.fromarray((np.clip(x, 0, 1) * 255).round().astype(np.uint8), mode="L").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-image", required=True, type=Path)
    ap.add_argument("--albedo-image", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--dark-thresh", type=float, default=0.10)
    ap.add_argument("--bright-thresh", type=float, default=0.20)
    ap.add_argument("--diff-thresh", type=float, default=0.12)
    ap.add_argument("--ratio-thresh", type=float, default=2.0)
    ap.add_argument("--hard-thresh", type=float, default=0.12)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    src_img = Image.open(args.source_image).convert("RGB")
    alb_img = Image.open(args.albedo_image).convert("RGB")
    if alb_img.size != src_img.size:
        alb_img = alb_img.resize(src_img.size, Image.BICUBIC)

    src = np.asarray(src_img).astype(np.float32) / 255.0
    alb = np.asarray(alb_img).astype(np.float32) / 255.0

    src_l = rgb_to_luma(src)
    alb_l = rgb_to_luma(alb)

    dark_score = np.clip((args.dark_thresh - alb_l) / max(args.dark_thresh, 1e-6), 0.0, 1.0)
    bright_score = np.clip((src_l - args.bright_thresh) / max(1.0 - args.bright_thresh, 1e-6), 0.0, 1.0)
    diff_score = np.clip((src_l - alb_l - args.diff_thresh) / max(1.0 - args.diff_thresh, 1e-6), 0.0, 1.0)
    ratio = src_l / (alb_l + 1e-4)
    ratio_score = np.clip((ratio - args.ratio_thresh) / max(args.ratio_thresh * 2.0, 1e-6), 0.0, 1.0)

    soft = np.clip(dark_score * bright_score * diff_score * ratio_score * 2.0, 0.0, 1.0)
    soft = pil_morphology(soft, dilate=2, erode=1, blur_radius=1.0)
    hard = (soft > args.hard_thresh).astype(np.float32)
    blend = pil_morphology(hard, dilate=2, erode=0, blur_radius=3.0)

    hybrid = (1.0 - blend[..., None]) * alb + blend[..., None] * src
    hybrid = np.clip(hybrid, 0.0, 1.0)

    save_rgb(args.out_dir / "hybrid_albedo.png", hybrid)
    save_l(args.out_dir / "glass_mask.png", hard)
    save_l(args.out_dir / "glass_mask_soft.png", blend)

    print("[DONE]", args.out_dir)


if __name__ == "__main__":
    main()
