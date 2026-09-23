from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from light_protocol import stage4_point_light_position


COMPARE = Path(__file__).resolve().parents[1]
WORKSPACE = COMPARE.parent
STAGE4 = WORKSPACE / "Stage4"


def load_manifest(root: Path, limit: int = 0) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return records[:limit] if limit > 0 else records


def clamp_rgb(values: list[float], scale: float, floor: float, ceiling: float) -> list[float]:
    return [float(max(floor, min(ceiling, v * scale))) for v in values]


def pan_tilt_to_position(pan_deg: float, tilt_deg: float, radius: float) -> list[float]:
    return stage4_point_light_position(float(pan_deg), float(tilt_deg), float(radius))


def build_request(args: argparse.Namespace, root: Path, record: dict[str, Any]) -> dict[str, Any]:
    target_light = record.get("target_light", {})
    rgb = [float(v) for v in target_light.get("rgb", [1.0, 1.0, 1.0])]
    explicit_env_rgb = target_light.get("stage4_env_rgb")
    env_rgb = [float(v) for v in explicit_env_rgb] if explicit_env_rgb is not None else clamp_rgb(rgb, args.env_scale, args.env_floor, args.env_ceiling)
    if args.point_only_target and args.use_point_light:
        env_rgb = [0.0, 0.0, 0.0]
    stage4_id_prefix = str(record.get("stage4_id_prefix") or f"{record['dataset']}_{record['id']}")
    stage4_scene_id = f"{stage4_id_prefix}_{args.method_id}"

    request: dict[str, Any] = {
        "input_image": str((root / record["files"][args.source_key]).resolve()),
        "scene_id": stage4_scene_id,
        "output_name": stage4_scene_id,
        "light_mode": "env",
        "env_rgb": env_rgb,
        "spp": args.spp,
        "final_spp": args.final_spp,
        "iters": args.iters,
        "mesh_stride": args.mesh_stride,
        "device": args.device,
        "mitsuba_variant": args.mitsuba_variant,
        "stage3_checkpoint": str(args.stage3_checkpoint.resolve()),
    }
    pan = target_light.get("pan")
    tilt = target_light.get("tilt")
    point_position = target_light.get("stage4_point_light_position")
    point_rgb = target_light.get("stage4_point_light_rgb")
    if args.use_point_light and (point_position is not None or (pan is not None and tilt is not None)):
        request["point_light_position"] = point_position or pan_tilt_to_position(float(pan), float(tilt), args.point_radius)
        request["point_light_rgb"] = point_rgb or clamp_rgb(rgb, args.point_scale, 0.0, args.point_ceiling)

    return request


def copy_reference_images(args: argparse.Namespace, root: Path, record: dict[str, Any], predictions: Path) -> None:
    if not args.copy_reference_images:
        return
    sample_id = record["id"]
    files = record.get("files", {})
    refs = {
        "source": files.get(args.source_key) or files.get("source_rgb"),
        "target": files.get("target_rgb"),
    }
    for suffix, rel_path in refs.items():
        if not rel_path:
            continue
        src = root / rel_path
        if not src.exists():
            continue
        dst = predictions / f"{sample_id}_{suffix}{src.suffix.lower()}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def run_one(args: argparse.Namespace, root: Path, record: dict[str, Any], predictions: Path, requests_dir: Path, artifacts_dir: Path) -> float:
    sample_id = record["id"]
    out_path = predictions / record["prediction_name"]
    if out_path.exists() and not args.overwrite:
        copy_reference_images(args, root, record, predictions)
        return 0.0

    request = build_request(args, root, record)
    request_path = requests_dir / f"{sample_id}.json"
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")

    if args.dry_run:
        return 0.0

    wrapper = STAGE4 / ("stage12_single_relight_black_glass_mask.py" if args.black_glass_mask else "stage4_relight_single.py")
    cmd = [
        str(args.stage4_python.resolve()),
        str(wrapper.resolve()),
        "--request",
        str(request_path.resolve()),
        "--stage4-root",
        str(STAGE4.resolve()),
    ]
    log_dir = artifacts_dir / sample_id
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "stage4.stdout.log"
    stderr_path = log_dir / "stage4.stderr.log"

    start = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        subprocess.run(cmd, cwd=STAGE4, check=True, stdout=stdout, stderr=stderr)
    elapsed = time.perf_counter() - start

    final_relit = STAGE4 / "outputs" / request["output_name"] / "final_relit.png"
    if not final_relit.exists():
        raise FileNotFoundError(f"Stage4 did not produce final_relit.png: {final_relit}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_relit, out_path)
    copy_reference_images(args, root, record, predictions)

    contact = STAGE4 / "outputs" / request["output_name"] / "contact_sheet.jpg"
    if contact.exists():
        shutil.copy2(contact, log_dir / "contact_sheet.jpg")
    shutil.copy2(request_path, log_dir / "request.json")
    return elapsed


def append_failure(path: Path, record: dict[str, Any], error: BaseException) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": record.get("id"),
        "dataset": record.get("dataset"),
        "source_name": record.get("source_name"),
        "target_name": record.get("target_name"),
        "error_type": type(error).__name__,
        "error": str(error),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Stage4 relighting pipeline on a prepared benchmark root.")
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--method-id", default="stage4")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0, help="Skip manifest rows before this zero-based index.")
    parser.add_argument("--max-new", type=int, default=0, help="Stop after this many non-skipped samples have been attempted.")
    parser.add_argument("--max-consecutive-failures", type=int, default=0, help="Stop after this many consecutive failures. Zero disables the guard.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--black-glass-mask", action="store_true")
    parser.add_argument("--source-key", default="source_rgb_512")
    parser.add_argument("--copy-reference-images", action="store_true", help="Copy source/target images next to each prediction.")

    parser.add_argument("--stage4-python", type=Path, default=STAGE4 / ".venv" / "Scripts" / "python.exe")
    parser.add_argument("--stage3-checkpoint", type=Path, default=STAGE4 / "checkpoints" / "glass" / "checkpoint_final.pt")

    parser.add_argument("--spp", type=int, default=32)
    parser.add_argument("--final-spp", type=int, default=128)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--mesh-stride", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mitsuba-variant", default="cuda_ad_rgb")

    parser.add_argument("--env-scale", type=float, default=0.2)
    parser.add_argument("--env-floor", type=float, default=0.02)
    parser.add_argument("--env-ceiling", type=float, default=2.0)
    parser.add_argument("--use-point-light", action="store_true")
    parser.add_argument("--point-only-target", action="store_true", help="Use only the target point light, with zero target environment.")
    parser.add_argument("--point-radius", type=float, default=2.0)
    parser.add_argument("--point-scale", type=float, default=40.0)
    parser.add_argument("--point-ceiling", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.benchmark_root.resolve()
    records = load_manifest(root, 0)
    if args.start_index > 0:
        records = records[args.start_index :]
    if args.limit > 0:
        records = records[: args.limit]
    predictions = root / "predictions" / args.method_id
    requests_dir = root / "stage4_requests" / args.method_id
    artifacts_dir = root / "stage4_artifacts" / args.method_id
    predictions.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not args.stage4_python.exists():
        raise FileNotFoundError(args.stage4_python)
    if not args.stage3_checkpoint.exists():
        raise FileNotFoundError(args.stage3_checkpoint)

    print("[info] root:", root)
    print("[info] samples:", len(records))
    print("[info] method:", args.method_id)
    print("[info] checkpoint:", args.stage3_checkpoint)
    print("[info] requests:", requests_dir)
    print("[info] predictions:", predictions)

    runtime_path = predictions / "runtime.json"
    runtimes = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {}
    new_count = 0
    attempted_count = 0
    failures = 0
    consecutive_failures = 0
    failure_path = artifacts_dir / "failures.jsonl"
    for index, record in enumerate(records, 1):
        sample_id = record["id"]
        out_path = predictions / record["prediction_name"]
        if out_path.exists() and not args.overwrite:
            continue
        attempted_count += 1
        try:
            elapsed = run_one(args, root, record, predictions, requests_dir, artifacts_dir)
        except Exception as error:
            failures += 1
            consecutive_failures += 1
            append_failure(failure_path, record, error)
            print(f"  [{index:04d}/{len(records):04d}] {sample_id} FAILED: {error}")
            if args.max_consecutive_failures > 0 and consecutive_failures >= args.max_consecutive_failures:
                raise RuntimeError(
                    f"stopping after {consecutive_failures} consecutive failures; "
                    "rerun with --max-consecutive-failures 0 to disable this guard"
                ) from error
            if not args.continue_on_error:
                raise
            if args.max_new > 0 and attempted_count >= args.max_new:
                break
            continue
        if not args.dry_run:
            runtimes[sample_id] = elapsed
            runtime_path.write_text(json.dumps(runtimes, indent=2), encoding="utf-8")
        new_count += 1
        consecutive_failures = 0
        if index == 1 or index % 10 == 0:
            print(f"  [{index:04d}/{len(records):04d}] {sample_id} {elapsed:.2f}s")
        if args.max_new > 0 and attempted_count >= args.max_new:
            break

    (predictions / "run_config.json").write_text(
        json.dumps(
            {
                "method": "Stage4 full pipeline",
                "method_id": args.method_id,
                "benchmark_root": str(root),
                "stage4_root": str(STAGE4),
                "stage3_checkpoint": str(args.stage3_checkpoint),
                "source_key": args.source_key,
                "spp": args.spp,
                "final_spp": args.final_spp,
                "iters": args.iters,
                "mesh_stride": args.mesh_stride,
                "env_scale": args.env_scale,
                "env_floor": args.env_floor,
                "use_point_light": args.use_point_light,
                "point_radius": args.point_radius,
                "point_scale": args.point_scale,
                "black_glass_mask": args.black_glass_mask,
                "copy_reference_images": args.copy_reference_images,
                "dry_run": args.dry_run,
                "attempted_predictions": attempted_count,
                "new_predictions": new_count,
                "failures": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
