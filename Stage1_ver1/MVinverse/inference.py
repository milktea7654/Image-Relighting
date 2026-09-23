import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from mvinverse.models.mvinverse import MVInverse


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_model(ckpt_path_or_repo_id, device):
    if os.path.exists(ckpt_path_or_repo_id):
        print(f"Loading model from local checkpoint: {ckpt_path_or_repo_id}")
        model = MVInverse().to(device).eval()
        weight = torch.load(ckpt_path_or_repo_id, map_location=device, weights_only=False)
        weight = weight["model"] if "model" in weight else weight
        missing, unused = model.load_state_dict(weight, strict=False)
        if len(missing) != 0:
            print(f"Warning: Missing keys found: {missing}")
        if len(unused) != 0:
            print(f"Got unexpected keys: {unused}")
    else:
        print(f"Checkpoint not found locally. Attempting to download from Hugging Face Hub: {ckpt_path_or_repo_id}")
        model = MVInverse.from_pretrained(ckpt_path_or_repo_id)
        model = model.to(device).eval()
    return model


def load_single_image(image_path: Path) -> torch.Tensor:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        width, height = img.size

        # Match upstream MVInverse inference.py behavior.
        if max(width, height) > 1024:
            scaling_factor = 1024 / max(width, height)
            new_width, new_height = int(width * scaling_factor), int(height * scaling_factor)
        else:
            new_width, new_height = width, height

        # MVInverse requires dimensions divisible by 14.
        new_w = max(14, new_width // 14 * 14)
        new_h = max(14, new_height // 14 * 14)
        img = img.resize((new_w, new_h))

    return transforms.ToTensor()(img)


def collect_images_from_data_path(data_path: Path, num_frames: int) -> list[Path]:
    if data_path.is_file():
        image_paths = [data_path]
    else:
        image_paths = sorted(p for p in data_path.iterdir() if p.suffix.lower() in IMG_EXTS)

    if num_frames > 0:
        image_paths = image_paths[:num_frames]

    return image_paths


def save_one_result(res, save_dir: Path, index: int = 0):
    save_dir.mkdir(parents=True, exist_ok=True)

    albedo = res["albedo"][0, 0]
    metallic = res["metallic"][0, 0]
    roughness = res["roughness"][0, 0]
    normal = res["normal"][0, 0]
    shading = res["shading"][0, 0]
    prefix = f"{index:03d}"

    Image.fromarray((albedo.cpu().numpy() * 255).astype(np.uint8)).save(save_dir / f"{prefix}_albedo.png")
    Image.fromarray((metallic.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)).save(save_dir / f"{prefix}_metallic.png")
    Image.fromarray((roughness.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)).save(save_dir / f"{prefix}_roughness.png")
    Image.fromarray(((normal * 0.5 + 0.5).cpu().numpy() * 255).astype(np.uint8)).save(save_dir / f"{prefix}_normal.png")
    Image.fromarray((shading.cpu().numpy() * 255).astype(np.uint8)).save(save_dir / f"{prefix}_shading.png")


def run_single_image_jobs(image_jobs: list[tuple[Path, Path, int]], args):
    if not image_jobs:
        raise RuntimeError("No input images found")

    print("Loading model once...")
    device = torch.device(args.device)
    model = load_model(args.ckpt, device)

    use_cuda_amp = device.type == "cuda"

    for idx, (image_path, out_dir, out_index) in enumerate(image_jobs, start=1):
        print("-" * 70)
        print(f"[{idx}/{len(image_jobs)}] Single-image MVInverse: {image_path.name}")
        img = load_single_image(image_path).unsqueeze(0).to(device)   # [1, 3, H, W]
        print(f"Loaded 1 image, size is {tuple(img.shape[-2:])}")

        with torch.no_grad():
            if use_cuda_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    res = model(img[None])  # [B=1, N=1, C, H, W]
            else:
                res = model(img[None])

        save_one_result(res, out_dir, out_index)
        print(f"Results saved to {out_dir}")

    print("Done. MVInverse model was loaded once for all unrelated images.")


def main():
    parser = argparse.ArgumentParser(
        description="MVInverse single-image runner: load model once, process images independently."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image_list", help="Text file containing one absolute image path per line.")
    input_group.add_argument("--data_path", help="Image file or directory of images. Kept for compatibility with upstream MVInverse CLI.")
    parser.add_argument("--ckpt", default="maddog241/mvinverse")
    parser.add_argument("--save_path", default="outputs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_frames", type=int, default=-1, help="With --data_path, limit the number of sorted input images. -1 means all.")
    args = parser.parse_args()

    save_root = Path(args.save_path)

    if args.image_list:
        with open(args.image_list, "r", encoding="utf-8") as f:
            image_paths = [Path(line.strip()) for line in f if line.strip()]
        if not image_paths:
            raise RuntimeError(f"No images listed in {args.image_list}")
        image_jobs = [(image_path, save_root / image_path.stem, 0) for image_path in image_paths]
    else:
        data_path = Path(args.data_path)
        if not data_path.exists():
            raise RuntimeError(f"Input path does not exist: {data_path}")
        image_paths = collect_images_from_data_path(data_path, args.num_frames)
        if not image_paths:
            raise RuntimeError(f"No images found in {data_path}")
        out_dir = save_root / data_path.stem
        image_jobs = [(image_path, out_dir, idx) for idx, image_path in enumerate(image_paths)]

    run_single_image_jobs(image_jobs, args)


if __name__ == "__main__":
    main()