from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


GRAY_INPUTS = {"roughness", "metallic", "glass_mask"}


class Timer:
    def __init__(self) -> None:
        self.rows: list[tuple[str, float]] = []

    def step(self, name: str, start: float) -> None:
        self.rows.append((name, time.perf_counter() - start))

    def print(self, frames: int = 1) -> None:
        print()
        print("[TIMING]")
        for name, seconds in self.rows:
            fps = frames / seconds if seconds > 0 else float("inf")
            print(f"{name:24s} {seconds * 1000.0:9.2f} ms   {fps:8.2f} fps")
        total = sum(x for _, x in self.rows)
        print(f"{'total':24s} {total * 1000.0:9.2f} ms   {(frames / total if total > 0 else float('inf')):8.2f} fps")


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def read_rgb(path: Path, size: int | None = None) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if size is not None:
            img = img.resize((size, size), Image.BICUBIC)
        return np.asarray(img, dtype=np.float32) / 255.0


def read_gray(path: Path, size: int | None = None) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("L")
        if size is not None:
            img = img.resize((size, size), Image.BICUBIC)
        return np.asarray(img, dtype=np.float32)[..., None] / 255.0


def write_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255.0).round().astype(np.uint8), "RGB").save(path)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def rgb_tensor(path: Path, size: int) -> torch.Tensor:
    arr = read_rgb(path, size=size)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def gray_tensor(path: Path, size: int) -> torch.Tensor:
    arr = read_gray(path, size=size)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def save_tensor(path: Path, tensor: torch.Tensor) -> None:
    x = tensor.detach().float().cpu().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, "RGB").save(path)


def make_black_mask(path: Path, like_path: Path) -> Path:
    if path.exists():
        return path
    with Image.open(like_path) as img:
        size = img.size
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, 0).save(path)
    return path


def find_render_backend(scene_dir: Path, scene_meta: dict[str, Any]) -> Path:
    candidates = []
    backend = scene_meta.get("shape", {}).get("filename")
    if backend:
        candidates.append(scene_dir / backend)
    candidates.extend([
        scene_dir / "scene_render_backend.obj",
        scene_dir / "scene_render_backend.ply",
        scene_dir / "scene_uv.obj",
        scene_dir / "scene.ply",
    ])
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No Mitsuba render backend mesh found. Re-run Stage12 with "
        "'--worker-extra-args --keep-render-backend' so scene_render_backend.obj/ply is kept."
    )


def make_sensor(mi, scene_meta: dict[str, Any], spp: int, width: int | None, height: int | None) -> dict[str, Any]:
    sensor = dict(scene_meta["sensor"])
    sensor["type"] = "perspective"
    sensor["to_world"] = mi.ScalarTransform4f.look_at(origin=[0, 0, 0], target=[0, 0, -1], up=[0, 1, 0])
    sensor["sampler"] = {"type": "independent", "sample_count": int(spp)}
    res = sensor.pop("resolution", {})
    fallback_w = res.get("width") if isinstance(res, dict) else None
    fallback_h = res.get("height") if isinstance(res, dict) else None
    sensor["film"] = {
        "type": "hdrfilm",
        "width": int(width or sensor.get("width") or fallback_w or 1024),
        "height": int(height or sensor.get("height") or fallback_h or 768),
        "pixel_format": "rgb",
        "rfilter": {"type": "box"},
    }
    for key in ["width", "height", "look_at", "camera_metadata", "intrinsics", "coordinate_note", "fov_x_degrees"]:
        sensor.pop(key, None)
    return sensor


def make_bsdf(scene_dir: Path, scene_meta: dict[str, Any]) -> dict[str, Any]:
    shape_meta = scene_meta.get("shape", {}).get("bsdf", {})
    model = shape_meta.get("model", "principled")
    if model == "diffuse":
        return {
            "type": "twosided",
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "bitmap", "filename": str(scene_dir / "albedo.png")},
            },
        }
    return {
        "type": "twosided",
        "bsdf": {
            "type": "principled",
            "base_color": {"type": "bitmap", "filename": str(scene_dir / "albedo.png")},
            "roughness": {"type": "bitmap", "filename": str(scene_dir / "roughness.png")},
            "metallic": {"type": "bitmap", "filename": str(scene_dir / "metallic.png")},
        },
    }


def render_mitsuba(args: argparse.Namespace, scene_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    import mitsuba as mi

    mi.set_variant(args.mitsuba_variant)
    scene_meta = json.loads((scene_dir / "scene_mitsuba.json").read_text(encoding="utf-8"))
    mesh_path = find_render_backend(scene_dir, scene_meta)

    scene_dict: dict[str, Any] = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 4},
        "sensor": make_sensor(mi, scene_meta, args.spp, args.render_width, args.render_height),
        "shape": {
            "type": "obj" if mesh_path.suffix.lower() == ".obj" else "ply",
            "filename": str(mesh_path),
            "bsdf": make_bsdf(scene_dir, scene_meta),
        },
        "env_light": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": list(args.env_rgb)},
        },
    }

    if args.point_light_position is not None:
        scene_dict["target_point"] = {
            "type": "point",
            "position": list(args.point_light_position),
            "intensity": {"type": "rgb", "value": list(args.point_light_rgb)},
        }

    scene = mi.load_dict(scene_dict)
    img = mi.render(scene, spp=args.spp, seed=args.seed)
    arr = np.array(img, dtype=np.float32)
    arr = linear_to_srgb(arr)

    render_path = out_dir / "render_in.png"
    write_rgb(render_path, arr)

    composite = arr
    bg_path = scene_dir / "background.png"
    mask_path = scene_dir / "geometry_mask.png"
    if bg_path.exists() and mask_path.exists():
        bg = read_rgb(bg_path)
        mask = read_gray(mask_path)
        if bg.shape[:2] != arr.shape[:2]:
            bg = np.asarray(Image.fromarray((bg * 255).astype(np.uint8)).resize((arr.shape[1], arr.shape[0]), Image.BICUBIC), dtype=np.float32) / 255.0
        if mask.shape[:2] != arr.shape[:2]:
            mask_img = Image.fromarray((mask[..., 0] * 255).astype(np.uint8)).resize((arr.shape[1], arr.shape[0]), Image.BICUBIC)
            mask = np.asarray(mask_img, dtype=np.float32)[..., None] / 255.0
        composite = arr * mask + bg * (1.0 - mask)

    composite_path = out_dir / "composite_in.png"
    write_rgb(composite_path, composite)
    return render_path, composite_path


def load_efficientvit(checkpoint: Path, stage3_root: Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location="cpu")
    input_names = list(ckpt.get("input_names") or ckpt.get("inputs") or [])
    if not input_names:
        raise RuntimeError("Checkpoint is missing input_names.")
    channels = int(ckpt.get("channels", 0))
    if channels <= 0:
        channels = sum(1 if name in GRAY_INPUTS else 3 for name in input_names)

    backbone = str(ckpt.get("backbone", "efficientvit")).lower().replace("_", "-")
    if backbone in {"efficientvit", "stage3-efficientvit"}:
        module = load_module(stage3_root / "stage3_efficientvit.py", "stage3_efficientvit_online")
        EfficientViTRelighter = module.EfficientViTRelighter
        cfg = ckpt.get("model_config") or {}
        model = EfficientViTRelighter(
            input_channels=channels,
            channels=cfg.get("channels", [32, 64, 128, 192]),
            depths=cfg.get("depths", [1, 1, 2, 2]),
            heads=cfg.get("heads", [2, 4, 4, 6]),
            residual_scale=float(cfg.get("residual_scale", 0.25)),
        )
    else:
        if str(stage3_root) not in sys.path:
            sys.path.insert(0, str(stage3_root))
        stage3_unet = load_module(stage3_root / "stage3_unet.py", "stage3_unet")
        base_channels = ckpt.get("base_channels")
        residual_scale = ckpt.get("residual_scale")
        model = stage3_unet.build_stage3_model(
            backbone=backbone,
            input_channels=channels,
            base_channels=int(base_channels) if base_channels is not None else None,
            residual_scale=float(residual_scale) if residual_scale is not None else None,
        )

    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, input_names, channels


def build_model_input(
    input_names: list[str],
    image_size: int,
    composite_path: Path,
    render_path: Path,
    scene_dir: Path,
    glass_mask_path: Path | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    paths = {
        "composite": composite_path,
        "render": render_path,
        "albedo": scene_dir / "albedo.png",
        "roughness": scene_dir / "roughness.png",
        "metallic": scene_dir / "metallic.png",
        "normal": scene_dir / "normal.png",
        "shading": scene_dir / "irradiance.png",
        "glass_mask": glass_mask_path,
    }

    tensors = []
    composite = rgb_tensor(composite_path, image_size)
    for name in input_names:
        path = paths.get(name)
        if name == "composite":
            tensors.append(composite)
        elif name == "render":
            tensors.append(rgb_tensor(render_path, image_size))
        elif path is not None and path.exists():
            tensors.append(gray_tensor(path, image_size) if name in GRAY_INPUTS else rgb_tensor(path, image_size))
        else:
            channels = 1 if name in GRAY_INPUTS else 3
            tensors.append(torch.zeros(channels, image_size, image_size, dtype=torch.float32))

    x = torch.cat(tensors, dim=0)[None, ...]
    comp = composite[None, ...]
    return x, comp


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, type=Path, help="Stage12/Stage2 output folder containing scene_mitsuba.json and material maps.")
    ap.add_argument("--checkpoint", type=Path, default=Path(r"D:\relighting\Stage3\runs\efficientvit_glassaware_256\checkpoint_last.pt"))
    ap.add_argument("--stage3-root", type=Path, default=Path(r"D:\relighting\Stage3"))
    ap.add_argument("--out-dir", type=Path, default=Path("work/online_efficientvit"))
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mitsuba-variant", default="cuda_ad_rgb")
    ap.add_argument("--spp", type=int, default=32)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--render-width", type=int, default=None)
    ap.add_argument("--render-height", type=int, default=None)
    ap.add_argument("--env-rgb", type=float, nargs=3, default=[0.2, 0.2, 0.2])
    ap.add_argument("--point-light-position", type=float, nargs=3, default=None)
    ap.add_argument("--point-light-rgb", type=float, nargs=3, default=[35.0, 32.0, 26.0])
    ap.add_argument("--glass-mask", type=Path, default=None)
    ap.add_argument("--black-glass-mask", action="store_true")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--model-iters", type=int, default=30)
    args = ap.parse_args()

    timer = Timer()
    args.scene_dir = args.scene_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    render_path, composite_path = render_mitsuba(args, args.scene_dir, args.out_dir)
    timer.step("mitsuba_render", t0)

    t0 = time.perf_counter()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, input_names, channels = load_efficientvit(args.checkpoint.resolve(), args.stage3_root.resolve(), device)
    timer.step("load_efficientvit", t0)

    glass_mask = args.glass_mask
    if args.black_glass_mask:
        glass_mask = make_black_mask(args.out_dir / "black_glass_mask.png", composite_path)

    t0 = time.perf_counter()
    x, comp = build_model_input(input_names, args.image_size, composite_path, render_path, args.scene_dir, glass_mask)
    x = x.to(device, non_blocking=True)
    comp = comp.to(device, non_blocking=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    timer.step("construct_model_input", t0)

    for _ in range(max(0, args.warmup)):
        _ = model(x, comp)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    pred = None
    for _ in range(max(1, args.model_iters)):
        pred = model(x, comp)
    if device.type == "cuda":
        torch.cuda.synchronize()
    model_seconds = time.perf_counter() - t0
    timer.rows.append((f"efficientvit_forward_x{args.model_iters}", model_seconds))

    assert pred is not None
    save_tensor(args.out_dir / "efficientvit_prediction.png", pred[0])

    meta = {
        "scene_dir": str(args.scene_dir),
        "checkpoint": str(args.checkpoint),
        "input_names": input_names,
        "channels": channels,
        "render": str(render_path),
        "composite": str(composite_path),
        "prediction": str(args.out_dir / "efficientvit_prediction.png"),
        "env_rgb": list(args.env_rgb),
        "point_light_position": list(args.point_light_position) if args.point_light_position is not None else None,
        "point_light_rgb": list(args.point_light_rgb),
        "timing_seconds": {name: seconds for name, seconds in timer.rows},
        "efficientvit_single_forward_fps": args.model_iters / model_seconds if model_seconds > 0 else math.inf,
    }
    (args.out_dir / "online_metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[INFO] checkpoint   :", args.checkpoint)
    print("[INFO] input_names  :", input_names)
    print("[INFO] channels     :", channels)
    print("[INFO] device       :", device)
    print("[OUT] render        :", render_path)
    print("[OUT] composite     :", composite_path)
    print("[OUT] prediction    :", args.out_dir / "efficientvit_prediction.png")
    timer.print(frames=1)
    print(f"[FPS] mitsuba       : {1.0 / timer.rows[0][1]:.2f}")
    print(f"[FPS] efficientvit  : {args.model_iters / model_seconds:.2f}")


if __name__ == "__main__":
    main()
