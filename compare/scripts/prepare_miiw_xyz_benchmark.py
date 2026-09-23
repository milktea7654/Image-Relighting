from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


COMPARE = Path(__file__).resolve().parents[1]
DEFAULT_MIIW_SOURCE = COMPARE / "miiw_quantitative" / "assets" / "miiw_test_jpg"
DEFAULT_OUTPUT = COMPARE / "miiw_xyz_quantitative"
FRONTAL_DIRECTIONS = [2, 3, 19, 20, 21, 22, 24]
NOFRONTAL_DIRECTIONS = [i for i in range(25) if i not in FRONTAL_DIRECTIONS]
EVAL_SIZE = 256
BASELINE_SIZE = 512
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass(frozen=True)
class Candidate:
    scene: str
    scene_path: Path
    source_dir: int
    target_dir: int
    psnr: float
    band: str = ""


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_remove(path: Path, allowed_root: Path) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    allowed = allowed_root.resolve()
    if not str(resolved).lower().startswith(str(allowed).lower()):
        raise RuntimeError(f"Refusing to remove outside {allowed}: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)
    else:
        resolved.unlink()


def clean_float(value: float, eps: float = 1e-8, digits: int = 8) -> float:
    if abs(float(value)) < eps:
        return 0.0
    return float(f"{float(value):.{digits}f}")


def load_rgb(path: Path, size: int | None = None) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size is not None:
            width, height = image.size
            crop = min(width, height)
            left = (width - crop) // 2
            top = (height - crop) // 2
            image = image.crop((left, top, left + crop, top + crop)).resize((size, size), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0


def save_crop_resize(src: Path, dst: Path, size: int) -> None:
    ensure_dir(dst.parent)
    with Image.open(src) as image:
        image = image.convert("RGB")
        width, height = image.size
        crop = min(width, height)
        left = (width - crop) // 2
        top = (height - crop) // 2
        image = image.crop((left, top, left + crop, top + crop)).resize((size, size), Image.Resampling.LANCZOS)
        image.save(dst)


def copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def output_rel(path: Path, output_root: Path) -> str:
    return str(path.relative_to(output_root)).replace("\\", "/")


def image_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def scene_dirs(source_root: Path) -> list[Path]:
    scenes = []
    for path in sorted(source_root.iterdir()):
        if not path.is_dir():
            continue
        dirs = sorted(path.glob("dir_*_mip2.jpg"))
        probes = path / "probes"
        if len(dirs) >= 25 and probes.exists():
            scenes.append(path)
    return scenes


def available_dirs(scene: Path, exclude_dirs: set[int]) -> list[int]:
    dirs = []
    for path in scene.glob("dir_*_mip2.jpg"):
        try:
            idx = int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        probes = scene / "probes"
        required = [
            path,
            probes / f"dir_{idx}_gray256.jpg",
            probes / f"dir_{idx}_chrome256.jpg",
        ]
        if idx not in exclude_dirs and all(item.exists() for item in required):
            dirs.append(idx)
    return sorted(dirs)


def build_candidates(source_root: Path, exclude_dirs: set[int]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for scene in scene_dirs(source_root):
        dirs = available_dirs(scene, exclude_dirs)
        cached = {idx: load_rgb(scene / f"dir_{idx}_mip2.jpg", EVAL_SIZE) for idx in dirs}
        for source_dir in dirs:
            for target_dir in dirs:
                if source_dir == target_dir:
                    continue
                candidates.append(
                    Candidate(
                        scene=scene.name,
                        scene_path=scene,
                        source_dir=source_dir,
                        target_dir=target_dir,
                        psnr=image_psnr(cached[source_dir], cached[target_dir]),
                    )
                )
    if not candidates:
        raise RuntimeError(f"No MIIW pair candidates found under {source_root}")
    return candidates


def pick_evenly(items: list[Candidate], count: int, used: set[tuple[str, int, int]]) -> list[Candidate]:
    if count <= 0:
        return []
    if len(items) < count:
        raise ValueError(f"Need {count} candidates but only {len(items)} are available")
    selected: list[Candidate] = []
    positions = np.linspace(0, len(items) - 1, count * 3)
    for pos in positions:
        item = items[int(round(float(pos)))]
        key = (item.scene, item.source_dir, item.target_dir)
        if key in used:
            continue
        selected.append(item)
        used.add(key)
        if len(selected) == count:
            return selected
    for item in items:
        key = (item.scene, item.source_dir, item.target_dir)
        if key in used:
            continue
        selected.append(item)
        used.add(key)
        if len(selected) == count:
            return selected
    raise RuntimeError(f"Could only select {len(selected)} of {count} requested candidates")


def stratified_select(candidates: list[Candidate], hard: int, medium: int, easy: int) -> list[Candidate]:
    ordered = sorted(candidates, key=lambda item: item.psnr)
    n = len(ordered)
    hard_band = [Candidate(**{**item.__dict__, "band": "hard"}) for item in ordered[: n // 3]]
    medium_band = [Candidate(**{**item.__dict__, "band": "medium"}) for item in ordered[n // 3 : 2 * n // 3]]
    easy_band = [Candidate(**{**item.__dict__, "band": "easy"}) for item in ordered[2 * n // 3 :]]
    used: set[tuple[str, int, int]] = set()
    selected = []
    selected.extend(pick_evenly(hard_band, hard, used))
    selected.extend(pick_evenly(medium_band, medium, used))
    selected.extend(pick_evenly(easy_band, easy, used))
    return selected


def luminance(rgb: np.ndarray) -> np.ndarray:
    return np.tensordot(rgb[..., :3], LUMA, axes=([-1], [0]))


def sphere_geometry(gray: np.ndarray, threshold: float) -> tuple[np.ndarray, float, float, float]:
    lum = luminance(gray)
    mask = lum > threshold
    if int(mask.sum()) < 1000:
        mask = lum > max(0.01, float(np.percentile(lum, 20)))
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        height, width = lum.shape
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        radius = min(width, height) * 0.48
        yy, xx = np.indices(lum.shape)
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2, cx, cy, radius
    cx = (float(xs.min()) + float(xs.max())) / 2.0
    cy = (float(ys.min()) + float(ys.max())) / 2.0
    radius = (float(xs.max() - xs.min()) + float(ys.max() - ys.min())) / 4.0
    yy, xx = np.indices(lum.shape)
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius * 0.96) ** 2
    return disk, cx, cy, radius


def estimate_light_from_probes(
    chrome_path: Path,
    gray_path: Path,
    *,
    model_radius: float,
    point_scale: float,
    env_scale: float,
    env_floor: float,
    point_ceiling: float,
    highlight_percentile: float,
    gray_subtract_weight: float,
    sphere_threshold: float,
    flip_z_for_stage4: bool,
    stage4_front_hemisphere: bool,
    stage4_z_depth: float,
    stage4_xy_limit: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    chrome = load_rgb(chrome_path)
    gray = load_rgb(gray_path)
    disk, cx, cy, radius = sphere_geometry(gray, sphere_threshold)
    chrome_luma = luminance(chrome)
    gray_luma = luminance(gray)
    spec = np.clip(chrome_luma - gray_luma * gray_subtract_weight, 0.0, 1.0)
    valid_values = spec[disk]
    threshold = float(np.percentile(valid_values, highlight_percentile))
    highlight = disk & (spec >= threshold)
    if int(highlight.sum()) < 8:
        threshold = float(np.percentile(chrome_luma[disk], highlight_percentile))
        spec = chrome_luma
        highlight = disk & (chrome_luma >= threshold)

    yy, xx = np.indices(spec.shape)
    weights = np.clip(spec - threshold, 0.0, None) * highlight
    if float(weights.sum()) <= 1e-8:
        weights = highlight.astype(np.float32)
    u = float((weights * xx).sum() / max(float(weights.sum()), 1e-8))
    v = float((weights * yy).sum() / max(float(weights.sum()), 1e-8))

    sx = (u - cx) / max(radius, 1e-8)
    sy = -(v - cy) / max(radius, 1e-8)
    norm2 = sx * sx + sy * sy
    if norm2 >= 0.999:
        scale = 0.999 / math.sqrt(norm2)
        sx *= scale
        sy *= scale
        norm2 = sx * sx + sy * sy
    sz = math.sqrt(max(0.0, 1.0 - norm2))
    probe_light_dir = np.array([2.0 * sz * sx, 2.0 * sz * sy, 2.0 * sz * sz - 1.0], dtype=np.float32)
    probe_light_dir /= max(float(np.linalg.norm(probe_light_dir)), 1e-8)
    light_dir = probe_light_dir.copy()
    if flip_z_for_stage4:
        light_dir[2] *= -1.0
    if stage4_front_hemisphere:
        light_dir[2] = -abs(float(light_dir[2]))
    light_dir /= max(float(np.linalg.norm(light_dir)), 1e-8)
    probe_xyz = probe_light_dir * float(model_radius)
    direction_xyz = light_dir * float(model_radius)
    z_depth = abs(float(stage4_z_depth))
    z_abs = max(abs(float(light_dir[2])), 1e-4)
    xyz = light_dir * (z_depth / z_abs)
    xyz[2] = -z_depth
    xy_abs = max(abs(float(xyz[0])), abs(float(xyz[1])))
    if xy_abs > float(stage4_xy_limit):
        xy_scale = float(stage4_xy_limit) / max(xy_abs, 1e-8)
        xyz[0] *= xy_scale
        xyz[1] *= xy_scale

    top_rgb = chrome[highlight].mean(axis=0) if int(highlight.sum()) else chrome[disk].mean(axis=0)
    gray_mean = gray[disk].mean(axis=0)
    rgb_color = top_rgb / max(float(top_rgb.max()), 1e-6)
    env_rgb = np.maximum(gray_mean * float(env_scale), float(env_floor))
    point_rgb = np.clip(rgb_color * float(point_scale), 0.0, float(point_ceiling))

    estimate = {
        "method": "chrome_gray_probe_highlight_centroid",
        "probe_direction_unit": [clean_float(v) for v in probe_light_dir.tolist()],
        "raw_probe_xyz": [clean_float(v) for v in probe_xyz.tolist()],
        "direction_unit": [clean_float(v) for v in light_dir.tolist()],
        "direction_xyz_model_radius": [clean_float(v) for v in direction_xyz.tolist()],
        "xyz": [clean_float(v) for v in xyz.tolist()],
        "rgb": [clean_float(v) for v in rgb_color.tolist()],
        "raw_highlight_rgb": [clean_float(v) for v in top_rgb.tolist()],
        "raw_gray_env_rgb": [clean_float(v) for v in gray_mean.tolist()],
        "stage4_point_light_position": [clean_float(v) for v in xyz.tolist()],
        "stage4_point_light_rgb": [clean_float(v) for v in point_rgb.tolist()],
        "stage4_env_rgb": [clean_float(v) for v in env_rgb.tolist()],
        "model_radius": float(model_radius),
        "highlight_uv": [clean_float(u), clean_float(v)],
        "sphere_center_uv": [clean_float(cx), clean_float(cy)],
        "sphere_radius_px": clean_float(radius),
        "highlight_percentile": float(highlight_percentile),
        "highlight_pixel_count": int(highlight.sum()),
        "gray_subtract_weight": float(gray_subtract_weight),
        "flip_z_for_stage4": bool(flip_z_for_stage4),
        "stage4_front_hemisphere": bool(stage4_front_hemisphere),
        "stage4_position_policy": "project_direction_to_near_camera_plane",
        "stage4_z_depth": float(z_depth),
        "stage4_xy_limit": float(stage4_xy_limit),
        "coordinate_note": (
            "Estimated from MIIW target chrome/gray probes. XYZ is camera-space: "
            "x=image-right, y=image-up. Stage4 active xyz is the estimated direction projected to "
            "a near-camera plane because Stage4 source point grids are placed near z=-0.2."
        ),
    }
    debug = {"disk": disk, "highlight": highlight, "weights": weights}
    return estimate, debug


def estimate_gray_probe_env(gray_path: Path, *, env_scale: float, env_floor: float, sphere_threshold: float) -> dict[str, list[float]]:
    gray = load_rgb(gray_path)
    disk, _cx, _cy, _radius = sphere_geometry(gray, sphere_threshold)
    gray_mean = gray[disk].mean(axis=0)
    stage4_env_rgb = np.maximum(gray_mean * float(env_scale), float(env_floor))
    return {
        "raw_gray_env_rgb": [clean_float(v) for v in gray_mean.tolist()],
        "stage4_env_rgb": [clean_float(v) for v in stage4_env_rgb.tolist()],
    }


def write_light_npz(path: Path, estimate: dict[str, Any]) -> str:
    ensure_dir(path.parent)
    np.savez(
        path,
        xyz=np.array(estimate["xyz"], dtype=np.float32),
        direction_unit=np.array(estimate["direction_unit"], dtype=np.float32),
        probe_direction_unit=np.array(estimate["probe_direction_unit"], dtype=np.float32),
        raw_probe_xyz=np.array(estimate["raw_probe_xyz"], dtype=np.float32),
        direction_xyz_model_radius=np.array(estimate["direction_xyz_model_radius"], dtype=np.float32),
        rgb=np.array(estimate["rgb"], dtype=np.float32),
        raw_highlight_rgb=np.array(estimate["raw_highlight_rgb"], dtype=np.float32),
        raw_gray_env_rgb=np.array(estimate["raw_gray_env_rgb"], dtype=np.float32),
        source_raw_gray_env_rgb=np.array(estimate["source_raw_gray_env_rgb"], dtype=np.float32),
        target_raw_gray_env_rgb=np.array(estimate["target_raw_gray_env_rgb"], dtype=np.float32),
        stage4_point_light_rgb=np.array(estimate["stage4_point_light_rgb"], dtype=np.float32),
        stage4_env_rgb=np.array(estimate["stage4_env_rgb"], dtype=np.float32),
    )
    return path.name


def write_readme(output_root: Path, records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    band_counts: dict[str, int] = {}
    for record in records:
        band_counts[record["difficulty_band"]] = band_counts.get(record["difficulty_band"], 0) + 1
    lines = [
        "# MIIW Estimated-XYZ Relighting Benchmark",
        "",
        "This root contains paired MIIW samples prepared from source and target captures.",
        "The target light is estimated from the target chrome/gray probe as a dominant incident light direction in camera coordinates.",
        "It is not an exact physical point-light annotation.",
        "",
        "Each sample stores:",
        "",
        "- source RGB and target RGB for scoring",
        "- target chrome/gray probes",
        "- `target_light.xyz`, `target_light.rgb`, and fixed source-probe `target_light.stage4_env_rgb`",
        "- `target_light.npz` for model loaders that prefer arrays",
        "",
        "The default split uses MIIW SDK-style non-frontal directions only:",
        "",
        f"- excluded frontal directions: {FRONTAL_DIRECTIONS}",
        f"- allowed directions: {NOFRONTAL_DIRECTIONS}",
        "",
        "Pair selection is stratified by source-vs-target PSNR:",
        "",
        f"- hard: {band_counts.get('hard', 0)}",
        f"- medium: {band_counts.get('medium', 0)}",
        f"- easy: {band_counts.get('easy', 0)}",
        "",
        "Recommended Stage4 run:",
        "",
        "```powershell",
        "python compare/scripts/run_stage4_benchmark.py --benchmark-root compare/miiw_xyz_quantitative --method-id stage4-miiw-xyz --use-point-light",
        "```",
        "",
        "The default keeps ambient fixed from the source gray probe and changes only the estimated target point light.",
        "Use `--point-only-target` only for a point-light-only ablation with zero ambient.",
        "",
        "Builder settings:",
        "",
        "```json",
        json.dumps(vars(args), indent=2, default=str),
        "```",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_records(selected: list[Candidate], output_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    samples = output_root / "samples"
    ensure_dir(samples)
    for index, item in enumerate(selected):
        sample_id = f"{index:06d}"
        sample_dir = samples / sample_id
        ensure_dir(sample_dir)
        probes = item.scene_path / "probes"
        source = item.scene_path / f"dir_{item.source_dir}_mip2.jpg"
        target = item.scene_path / f"dir_{item.target_dir}_mip2.jpg"
        target_gray = probes / f"dir_{item.target_dir}_gray256.jpg"
        target_chrome = probes / f"dir_{item.target_dir}_chrome256.jpg"
        source_gray = probes / f"dir_{item.source_dir}_gray256.jpg"
        source_chrome = probes / f"dir_{item.source_dir}_chrome256.jpg"

        save_crop_resize(source, sample_dir / "source_rgb.png", EVAL_SIZE)
        save_crop_resize(target, sample_dir / "target_rgb.png", EVAL_SIZE)
        save_crop_resize(source, sample_dir / "source_rgb_512.png", BASELINE_SIZE)
        save_crop_resize(target, sample_dir / "target_rgb_512.png", BASELINE_SIZE)
        copy_file(source, sample_dir / "source_rgb_full.jpg")
        copy_file(target, sample_dir / "target_rgb_full.jpg")
        copy_file(source_gray, sample_dir / "source_probe_gray.jpg")
        copy_file(source_chrome, sample_dir / "source_probe_chrome.jpg")
        copy_file(target_gray, sample_dir / "target_probe_gray.jpg")
        copy_file(target_chrome, sample_dir / "target_probe_chrome.jpg")

        estimate, _debug = estimate_light_from_probes(
            target_chrome,
            target_gray,
            model_radius=args.model_radius,
            point_scale=args.point_scale,
            env_scale=args.env_scale,
            env_floor=args.env_floor,
            point_ceiling=args.point_ceiling,
            highlight_percentile=args.highlight_percentile,
            gray_subtract_weight=args.gray_subtract_weight,
            sphere_threshold=args.sphere_threshold,
            flip_z_for_stage4=not args.no_flip_z_for_stage4,
            stage4_front_hemisphere=not args.allow_stage4_positive_z,
            stage4_z_depth=args.stage4_z_depth,
            stage4_xy_limit=args.stage4_xy_limit,
        )
        source_env = estimate_gray_probe_env(
            source_gray,
            env_scale=args.env_scale,
            env_floor=args.env_floor,
            sphere_threshold=args.sphere_threshold,
        )
        target_env = {
            "raw_gray_env_rgb": estimate["raw_gray_env_rgb"],
            "stage4_env_rgb": estimate["stage4_env_rgb"],
        }
        estimate["target_raw_gray_env_rgb"] = target_env["raw_gray_env_rgb"]
        estimate["target_stage4_env_rgb"] = target_env["stage4_env_rgb"]
        estimate["source_raw_gray_env_rgb"] = source_env["raw_gray_env_rgb"]
        estimate["source_stage4_env_rgb"] = source_env["stage4_env_rgb"]
        estimate["raw_gray_env_rgb"] = source_env["raw_gray_env_rgb"]
        estimate["stage4_env_rgb"] = source_env["stage4_env_rgb"]
        estimate["env_policy"] = "fixed_from_source_gray_probe"

        files = {
            "source_rgb": output_rel(sample_dir / "source_rgb.png", output_root),
            "target_rgb": output_rel(sample_dir / "target_rgb.png", output_root),
            "source_rgb_512": output_rel(sample_dir / "source_rgb_512.png", output_root),
            "target_rgb_512": output_rel(sample_dir / "target_rgb_512.png", output_root),
            "source_rgb_full": output_rel(sample_dir / "source_rgb_full.jpg", output_root),
            "target_rgb_full": output_rel(sample_dir / "target_rgb_full.jpg", output_root),
            "source_probe_gray": output_rel(sample_dir / "source_probe_gray.jpg", output_root),
            "source_probe_chrome": output_rel(sample_dir / "source_probe_chrome.jpg", output_root),
            "target_probe_gray": output_rel(sample_dir / "target_probe_gray.jpg", output_root),
            "target_probe_chrome": output_rel(sample_dir / "target_probe_chrome.jpg", output_root),
            "target_light_npz": output_rel(sample_dir / "target_light.npz", output_root),
            "metadata": output_rel(sample_dir / "metadata.json", output_root),
        }
        write_light_npz(sample_dir / "target_light.npz", estimate)

        record = {
            "id": sample_id,
            "dataset": "miiw_estimated_xyz",
            "benchmark_name": "MIIW estimated-xyz relighting benchmark",
            "scene": item.scene,
            "source_light_dir": item.source_dir,
            "target_light_dir": item.target_dir,
            "difficulty_band": item.band,
            "source_target_psnr": clean_float(item.psnr),
            "image_sizes": {
                "evaluation": [EVAL_SIZE, EVAL_SIZE],
                "baseline_square": [BASELINE_SIZE, BASELINE_SIZE],
                "original": list(Image.open(source).size),
            },
            "preprocess": {
                "evaluation": "center_crop_square_then_resize_lanczos",
                "baseline_square": "center_crop_square_then_resize_lanczos",
                "original": "copied_miiw_mip2_jpg",
            },
            "files": files,
            "target_light": {
                **estimate,
                "source": "target_chrome_for_xyz_rgb_source_gray_for_fixed_env",
            },
            "model_inputs": {
                "source_rgb": files["source_rgb"],
                "target_xyz": estimate["xyz"],
                "target_rgb": estimate["rgb"],
                "target_env_rgb": estimate["source_raw_gray_env_rgb"],
                "env_policy": "fixed_from_source_gray_probe",
                "target_light_npz": files["target_light_npz"],
            },
            "stage4_id_prefix": f"miiw_xyz_{sample_id}",
            "prediction_name": f"{sample_id}.png",
            "metric_targets": {
                "target_rgb": files["target_rgb"],
                "changed_region_mask": "abs(luminance(target)-luminance(source)) > threshold",
                "changed_region_threshold": args.changed_threshold,
            },
        }
        (sample_dir / "metadata.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        records.append(record)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MIIW estimated-XYZ paired relighting benchmark.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_MIIW_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--hard", type=int, default=100)
    parser.add_argument("--medium", type=int, default=100)
    parser.add_argument("--easy", type=int, default=100)
    parser.add_argument(
        "--exclude-dirs",
        nargs="*",
        type=int,
        default=FRONTAL_DIRECTIONS,
        help="MIIW direction ids to exclude. Defaults to SDK frontal directions.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--model-radius", type=float, default=2.0)
    parser.add_argument("--stage4-z-depth", type=float, default=0.2)
    parser.add_argument("--stage4-xy-limit", type=float, default=0.75)
    parser.add_argument("--point-scale", type=float, default=40.0)
    parser.add_argument("--point-ceiling", type=float, default=120.0)
    parser.add_argument("--env-scale", type=float, default=0.2)
    parser.add_argument("--env-floor", type=float, default=0.02)
    parser.add_argument("--highlight-percentile", type=float, default=99.5)
    parser.add_argument("--gray-subtract-weight", type=float, default=0.8)
    parser.add_argument("--sphere-threshold", type=float, default=0.03)
    parser.add_argument("--changed-threshold", type=float, default=0.05)
    parser.add_argument("--no-flip-z-for-stage4", action="store_true")
    parser.add_argument(
        "--allow-stage4-positive-z",
        action="store_true",
        help="Allow Stage4 point lights behind the camera. Disabled by default; active xyz is projected to z<=0.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.hard + args.medium + args.easy != args.count:
        raise ValueError("--hard + --medium + --easy must equal --count")
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if args.force:
        safe_remove(output_root / "samples", output_root)
        safe_remove(output_root / "manifest.jsonl", output_root)
        safe_remove(output_root / "README.md", output_root)
    ensure_dir(output_root)

    candidates = build_candidates(source_root, set(args.exclude_dirs))
    selected = stratified_select(candidates, args.hard, args.medium, args.easy)
    records = create_records(selected, output_root, args)
    manifest = output_root / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    for method in ["source-copy", "rgbx", "luminet", "ic-light", "stage4-miiw-xyz", "ours-best"]:
        ensure_dir(output_root / "predictions" / method)
    write_readme(output_root, records, args)
    print(f"[done] candidates: {len(candidates)}")
    print(f"[done] records: {len(records)}")
    print(f"[done] manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
