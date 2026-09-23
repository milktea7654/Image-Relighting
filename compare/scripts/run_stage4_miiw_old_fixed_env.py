from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


COMPARE = Path(__file__).resolve().parents[1]
WORKSPACE = COMPARE.parent
STAGE4 = WORKSPACE / "Stage4"


def load_manifest(root: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit > 0 else rows


def clamp_rgb(values: list[float], scale: float, ceiling: float) -> list[float]:
    return [float(max(0.0, min(ceiling, float(v) * scale))) for v in values]


def build_request(args: argparse.Namespace, root: Path, record: dict[str, Any]) -> dict[str, Any]:
    target_light = record["target_light"]
    sample_id = record["id"]
    stage4_id_prefix = str(record.get("stage4_id_prefix") or f"{record['dataset']}_{sample_id}")
    scene_id = f"{stage4_id_prefix}_{args.method_id}"
    source = (root / record["files"][args.source_key]).resolve()

    return {
        "input_image": str(source),
        "scene_id": scene_id,
        "output_name": scene_id,
        "light_mode": "env",
        "env_rgb": [float(v) for v in target_light["stage4_env_rgb"]],
        "spp": args.spp,
        "final_spp": args.final_spp,
        "iters": args.iters,
        "mesh_stride": args.mesh_stride,
        "device": args.device,
        "mitsuba_variant": args.mitsuba_variant,
        # Old MIIW behavior: use the full estimated camera-space direction vector,
        # not the later near-plane projected stage4_point_light_position.
        "point_light_position": [float(v) for v in target_light["direction_xyz_model_radius"]],
        "point_light_rgb": clamp_rgb([float(v) for v in target_light["rgb"]], args.point_scale, args.point_ceiling),
    }


def copy_reference_images(root: Path, record: dict[str, Any], predictions: Path) -> None:
    files = record.get("files", {})
    for suffix, key in [("source", "source_rgb"), ("target", "target_rgb")]:
        src = root / files[key]
        if src.exists():
            shutil.copy2(src, predictions / f"{record['id']}_{suffix}{src.suffix.lower()}")


def run_one(args: argparse.Namespace, root: Path, record: dict[str, Any], predictions: Path, requests: Path, artifacts: Path) -> float:
    sample_id = record["id"]
    out_path = predictions / record["prediction_name"]
    copy_reference_images(root, record, predictions)
    if out_path.exists() and not args.overwrite:
        return 0.0

    request = build_request(args, root, record)
    request_path = requests / f"{sample_id}.json"
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    if args.dry_run:
        return 0.0

    log_dir = artifacts / sample_id
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.stage4_python.resolve()),
        str((STAGE4 / "stage4_relight_single.py").resolve()),
        "--request",
        str(request_path.resolve()),
        "--stage4-root",
        str(STAGE4.resolve()),
    ]
    start = time.perf_counter()
    with (log_dir / "stage4.stdout.log").open("w", encoding="utf-8") as stdout, (log_dir / "stage4.stderr.log").open("w", encoding="utf-8") as stderr:
        subprocess.run(cmd, cwd=STAGE4, check=True, stdout=stdout, stderr=stderr)
    elapsed = time.perf_counter() - start

    final_relit = STAGE4 / "outputs" / request["output_name"] / "final_relit.png"
    if not final_relit.exists():
        raise FileNotFoundError(final_relit)
    predictions.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_relit, out_path)
    contact = STAGE4 / "outputs" / request["output_name"] / "contact_sheet.jpg"
    if contact.exists():
        shutil.copy2(contact, log_dir / "contact_sheet.jpg")
    shutil.copy2(request_path, log_dir / "request.json")
    return elapsed


def append_failure(path: Path, record: dict[str, Any], error: BaseException) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"id": record.get("id"), "error_type": type(error).__name__, "error": str(error)}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run old MIIW fixed-env Stage4 behavior: full estimated xyz and rgb*scale point light.")
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--method-id", default="stage4-miiw-xyz-fixed-env")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-new", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--max-consecutive-failures", type=int, default=0)
    parser.add_argument("--source-key", default="source_rgb")
    parser.add_argument("--stage4-python", type=Path, default=STAGE4 / ".venv" / "Scripts" / "python.exe")
    parser.add_argument("--spp", type=int, default=32)
    parser.add_argument("--final-spp", type=int, default=128)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--mesh-stride", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mitsuba-variant", default="cuda_ad_rgb")
    parser.add_argument("--point-scale", type=float, default=40.0)
    parser.add_argument("--point-ceiling", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.benchmark_root.resolve()
    records = load_manifest(root, args.limit)
    if args.start_index:
        records = records[args.start_index :]

    predictions = root / "predictions" / args.method_id
    requests = root / "stage4_requests" / args.method_id
    artifacts = root / "stage4_artifacts" / args.method_id
    predictions.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    runtime_path = predictions / "runtime.json"
    runtimes = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {}
    failures_path = artifacts / "failures.jsonl"
    attempted = 0
    new_count = 0
    failures = 0
    consecutive_failures = 0

    print("[info] root:", root)
    print("[info] samples:", len(records))
    print("[info] method:", args.method_id)
    print("[info] old point: direction_xyz_model_radius, rgb *", args.point_scale)

    for index, record in enumerate(records, 1):
        sample_id = record["id"]
        if (predictions / record["prediction_name"]).exists() and not args.overwrite:
            copy_reference_images(root, record, predictions)
            continue
        attempted += 1
        try:
            elapsed = run_one(args, root, record, predictions, requests, artifacts)
        except Exception as error:
            failures += 1
            consecutive_failures += 1
            append_failure(failures_path, record, error)
            print(f"  [{index:04d}/{len(records):04d}] {sample_id} FAILED: {error}")
            if args.max_consecutive_failures and consecutive_failures >= args.max_consecutive_failures:
                raise
            if not args.continue_on_error:
                raise
            continue
        runtimes[sample_id] = elapsed
        runtime_path.write_text(json.dumps(runtimes, indent=2), encoding="utf-8")
        new_count += 1
        consecutive_failures = 0
        if index == 1 or index % 10 == 0:
            print(f"  [{index:04d}/{len(records):04d}] {sample_id} {elapsed:.2f}s")
        if args.max_new and attempted >= args.max_new:
            break

    (predictions / "run_config.json").write_text(
        json.dumps(
            {
                "method": "Stage4 old MIIW fixed-env",
                "method_id": args.method_id,
                "benchmark_root": str(root),
                "source_key": args.source_key,
                "point_position": "target_light.direction_xyz_model_radius",
                "point_rgb": f"target_light.rgb * {args.point_scale}",
                "env_rgb": "target_light.stage4_env_rgb",
                "attempted_predictions": attempted,
                "new_predictions": new_count,
                "failures": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[done] new:", new_count, "failures:", failures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
