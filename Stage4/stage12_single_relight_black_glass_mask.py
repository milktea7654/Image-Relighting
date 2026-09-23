from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

def run(cmd, cwd=None):
    print("[RUN]", " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def stage12_python_cmd(stage12_root: Path) -> list[str]:
    venv_python = stage12_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return [str(venv_python)]

    uv = shutil.which("uv")
    if uv:
        return [uv, "run", "python"]

    return [sys.executable]


def save_black_mask(path: Path, like_image: Path) -> None:
    from PIL import Image

    with Image.open(like_image) as img:
        size = img.size
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, 0).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True, type=Path, help="JSON request file")
    ap.add_argument("--stage4-root", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()

    stage4_root = args.stage4_root.resolve()
    req = json.loads(args.request.read_text(encoding="utf-8"))

    input_image = Path(req["input_image"])
    scene_id = req.get("scene_id", input_image.stem)
    light_mode = req.get("light_mode", "env")
    output_name = req.get("output_name", f"{scene_id}_black_glass_mask")

    env_rgb = req.get("env_rgb", [1.0, 1.0, 1.0])
    point_light_position = req.get("point_light_position")
    point_light_rgb = req.get("point_light_rgb")
    spp = int(req.get("spp", 32))
    final_spp = int(req.get("final_spp", 128))
    iters = int(req.get("iters", 80))
    mesh_stride = int(req.get("mesh_stride", 2))
    device = str(req.get("device", "cuda"))
    mitsuba_variant = str(req.get("mitsuba_variant", "cuda_ad_rgb"))
    stage3_checkpoint = Path(req.get("stage3_checkpoint", stage4_root / "checkpoints" / "stage3" / "checkpoint_final.pt"))

    work_root = ensure_dir(stage4_root / "work" / f"{scene_id}_black_glass_mask")
    scene_root = ensure_dir(work_root / "scene")
    stage2_root = ensure_dir(work_root / "stage2")
    glass_root = ensure_dir(work_root / "glassaware")
    stage3_root = ensure_dir(work_root / "stage3")
    out_root = ensure_dir(stage4_root / "outputs" / output_name)

    stage12_root = stage4_root / "external" / "Stage12_remote_run"

    if not input_image.exists():
        raise FileNotFoundError(input_image)
    if not stage12_root.exists():
        raise FileNotFoundError(f"Missing Stage12_remote_run at {stage12_root}")
    if not stage3_checkpoint.exists():
        raise FileNotFoundError(f"Missing Stage3 checkpoint at {stage3_checkpoint}")

    stage12_single = stage12_root / "stage12_single_relight.py"
    if not stage12_single.exists():
        raise FileNotFoundError(f"Expected Stage12 wrapper not found: {stage12_single}")

    stage12_cmd = [
        *stage12_python_cmd(stage12_root), str(stage12_single),
        "--input-image", str(input_image),
        "--scene-id", scene_id,
        "--scene-out", str(scene_root),
        "--stage2-out", str(stage2_root),
        "--light-mode", light_mode,
        "--env-rgb", *map(str, env_rgb),
        "--spp", str(spp),
        "--final-spp", str(final_spp),
        "--iters", str(iters),
        "--mesh-stride", str(mesh_stride),
        "--device", device,
        "--mitsuba-variant", mitsuba_variant,
    ] + (
        ["--point-light-position", *map(str, point_light_position)]
        if point_light_position is not None
        else []
    ) + (
        ["--point-light-rgb", *map(str, point_light_rgb)]
        if point_light_rgb is not None
        else []
    )
    stage12_cmd.extend(["--worker-extra-args", "--background-percentile", "1.0", "--keep-render-backend"])
    run(stage12_cmd, cwd=stage12_root)

    render_png = stage2_root / "render_optimized.png"
    composite_png = stage2_root / "composite_optimized.png"
    albedo_png = scene_root / "albedo.png"
    roughness_png = scene_root / "roughness.png"
    metallic_png = scene_root / "metallic.png"

    for p in [render_png, composite_png, albedo_png, roughness_png, metallic_png]:
        if not p.exists():
            raise FileNotFoundError(f"Expected Stage12 output not found: {p}")

    build_glassaware = stage4_root / "build_glassaware_single.py"
    run([
        sys.executable, str(build_glassaware),
        "--source-image", str(input_image),
        "--albedo-image", str(albedo_png),
        "--out-dir", str(glass_root),
    ], cwd=stage4_root)

    hybrid_albedo = glass_root / "hybrid_albedo.png"
    black_mask = glass_root / "glass_mask_black.png"
    black_mask_soft = glass_root / "glass_mask_soft_black.png"
    save_black_mask(black_mask, input_image)
    save_black_mask(black_mask_soft, input_image)
    print("[DONE] black glass masks:", black_mask, black_mask_soft)

    make_jsonl = stage4_root / "make_stage3_input_single.py"
    stage3_jsonl = stage3_root / "stage3_input.jsonl"
    run([
        sys.executable, str(make_jsonl),
        "--scene-id", scene_id,
        "--composite", str(composite_png),
        "--render", str(render_png),
        "--albedo", str(hybrid_albedo),
        "--roughness", str(roughness_png),
        "--metallic", str(metallic_png),
        "--glass-mask-soft", str(black_mask_soft),
        "--glass-mask", str(black_mask),
        "--output-jsonl", str(stage3_jsonl),
    ], cwd=stage4_root)

    infer_stage3 = stage4_root / "tools" / "infer_stage3_relight_unlabeled.py"
    run([
        sys.executable, str(infer_stage3),
        "--checkpoint", str(stage3_checkpoint),
        "--jsonl", str(stage3_jsonl),
        "--out-dir", str(stage3_root / "prediction"),
        "--image-size", "256",
        "--batch-size", "1",
        "--device", "cuda",
        "--save-inputs",
    ], cwd=stage4_root)

    pred = stage3_root / "prediction" / scene_id / "stage3_prediction.png"
    if not pred.exists():
        raise FileNotFoundError(f"Expected Stage3 output not found: {pred}")

    shutil.copy2(pred, out_root / "final_relit.png")
    contact = stage3_root / "prediction" / scene_id / "contact_sheet.jpg"
    if contact.exists():
        shutil.copy2(contact, out_root / "contact_sheet.jpg")

    print()
    print("[DONE] Final relit image:")
    print(out_root / "final_relit.png")


if __name__ == "__main__":
    main()
