from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from stage3_dataset import Stage3RelightDataset
from stage3_hf_dataset import HF_IMAGE_KEYS, Stage3HFRelightDataset
from stage3_unet import build_stage3_model, count_parameters, resolve_stage3_hparams


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().float().cpu().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    x = (x * 255.0).round().byte()
    x = x.permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def abs_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).abs().mean(dim=0, keepdim=True).repeat(3, 1, 1).clamp(0, 1)


def save_visual_grid(batch: Dict, pred: torch.Tensor, out_path: Path, input_names: List[str], max_items: int = 4) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ref = batch["reference"][:max_items]
    comp = batch["composite"][:max_items]
    render = batch["render"][:max_items]
    pred = pred[:max_items]

    cols = [
        ("reference", ref),
        ("composite_in", comp),
        ("render_in", render),
    ]

    for name in ["albedo", "normal", "roughness", "metallic", "shading", "glass_mask"]:
        if name in input_names and name in batch:
            cols.append((f"{name}_in", batch[name][:max_items]))

    before_err = torch.stack([abs_error(comp[i], ref[i]) for i in range(len(ref))])
    after_err = torch.stack([abs_error(pred[i], ref[i]) for i in range(len(ref))])

    cols.extend([
        ("prediction", pred),
        ("err_before", before_err),
        ("err_after", after_err),
    ])

    imgs = [[tensor_to_pil(tensor[i]) for _, tensor in cols] for i in range(len(ref))]

    w, h = imgs[0][0].size
    label_h = 24
    gap = 6
    canvas = Image.new("RGB", (len(cols) * w + (len(cols) + 1) * gap, len(imgs) * (h + label_h) + (len(imgs) + 1) * gap), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    y = gap
    for row_imgs in imgs:
        x = gap
        for (label, _), im in zip(cols, row_imgs):
            draw.text((x, y), label, fill=(0, 0, 0))
            canvas.paste(im, (x, y + label_h))
            x += w + gap
        y += h + label_h + gap

    canvas.save(out_path, quality=92)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


@torch.no_grad()
def evaluate(model: torch.nn.Module, dl: DataLoader, device: torch.device, max_batches: int = 50) -> Dict[str, float]:
    model.eval()
    total_l1 = 0.0
    total_mse = 0.0
    total_n = 0

    for i, batch in enumerate(dl):
        if i >= max_batches:
            break

        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        comp = batch["composite"].to(device, non_blocking=True)

        pred = model(x, comp)
        l1 = F.l1_loss(pred, y, reduction="sum")
        mse = F.mse_loss(pred, y, reduction="sum")

        total_l1 += float(l1.item())
        total_mse += float(mse.item())
        total_n += y.numel()

    mean_l1 = total_l1 / max(1, total_n)
    mean_mse = total_mse / max(1, total_n)
    psnr = -10.0 * math.log10(max(mean_mse, 1e-12))

    return {
        "val_l1": mean_l1,
        "val_mse": mean_mse,
        "val_psnr": psnr,
    }


def parse_input_names(args) -> List[str]:
    inputs = ["composite", "render"]
    if args.use_albedo:
        inputs.append("albedo")
    if args.use_normal:
        inputs.append("normal")
    if args.use_roughness:
        inputs.append("roughness")
    if args.use_metallic:
        inputs.append("metallic")
    if args.use_shading:
        inputs.append("shading")
    if args.use_glass_mask:
        inputs.append("glass_mask")
    return inputs


def maybe_select_limit(ds, limit: int):
    if limit is not None and limit > 0:
        return ds.select(range(min(limit, len(ds))))
    return ds


def dataset_len_text(ds) -> str:
    try:
        return str(len(ds))
    except TypeError:
        return "unknown"


HF_METADATA_COLUMNS = {
    "scene_id",
    "source_stem",
    "split",
    "quality_final_loss",
    "quality_best_loss",
    "quality_final_rgb_loss",
    "glass_mask_ratio",
    "original_width",
    "original_height",
}


def prepare_hf_dataset(ds, input_names: List[str]):
    from datasets import Image as HFImage

    logical_images = {"reference", "composite", "render", *input_names}
    image_columns = {HF_IMAGE_KEYS.get(name, name) for name in logical_images}
    keep_columns = image_columns | HF_METADATA_COLUMNS

    column_names = list(getattr(ds, "column_names", None) or [])
    if not column_names and getattr(ds, "features", None):
        column_names = list(ds.features.keys())

    remove_columns = [name for name in column_names if name not in keep_columns]
    if remove_columns:
        ds = ds.remove_columns(remove_columns)
        column_names = [name for name in column_names if name not in remove_columns]

    for column in image_columns:
        if column in column_names:
            ds = ds.cast_column(column, HFImage(decode=False))
    return ds


def load_hf_split(args, split: str, data_files: List[str] | None = None):
    if args.hf_cache_dir:
        cache_root = args.hf_cache_dir.resolve()
        os.environ.setdefault("HF_HOME", str(cache_root))
        os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
        os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))
        cache_root.mkdir(parents=True, exist_ok=True)
        (cache_root / "hub").mkdir(parents=True, exist_ok=True)
        (cache_root / "datasets").mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    cache_dir = str(args.hf_cache_dir) if args.hf_cache_dir else None
    if data_files:
        return load_dataset("parquet", data_files=data_files, split="train", cache_dir=cache_dir)
    return load_dataset(args.hf_dataset, split=split, cache_dir=cache_dir)


def build_datasets(args, input_names: List[str]):
    if args.hf_dataset or args.hf_train_files or args.hf_val_files:
        train_hf = load_hf_split(args, args.hf_train_split, args.hf_train_files)
        val_hf = load_hf_split(args, args.hf_val_split, args.hf_val_files)

        train_hf = prepare_hf_dataset(train_hf, input_names)
        val_hf = prepare_hf_dataset(val_hf, input_names)

        if args.shuffle_train:
            train_hf = train_hf.shuffle(seed=123)

        train_hf = maybe_select_limit(train_hf, args.train_limit)
        val_hf = maybe_select_limit(val_hf, args.val_limit)
        return (
            Stage3HFRelightDataset(train_hf, image_size=args.image_size, input_names=input_names),
            Stage3HFRelightDataset(val_hf, image_size=args.image_size, input_names=input_names),
        )

    if args.train_jsonl is None or args.val_jsonl is None:
        raise ValueError("Provide --hf-dataset or both --train-jsonl and --val-jsonl.")

    train_ds = Stage3RelightDataset(
        jsonl_path=args.train_jsonl,
        dataset_root=args.dataset_root,
        image_size=args.image_size,
        input_names=input_names,
        target_name="reference",
        missing_optional="zeros",
        require_all_inputs=False,
        limit=args.train_limit,
        shuffle=True,
        seed=123,
    )
    val_ds = Stage3RelightDataset(
        jsonl_path=args.val_jsonl,
        dataset_root=args.dataset_root,
        image_size=args.image_size,
        input_names=input_names,
        target_name="reference",
        missing_optional="zeros",
        require_all_inputs=False,
        limit=args.val_limit,
        shuffle=False,
    )
    return train_ds, val_ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", type=Path, default=None)
    ap.add_argument("--val-jsonl", type=Path, default=None)
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--hf-dataset", default=None)
    ap.add_argument("--hf-train-files", nargs="+", default=None)
    ap.add_argument("--hf-val-files", nargs="+", default=None)
    ap.add_argument("--hf-train-split", default="train")
    ap.add_argument("--hf-val-split", default="validation")
    ap.add_argument("--hf-cache-dir", type=Path, default=None)
    ap.add_argument("--augment", choices=["none", "hflip", "vflip", "flip", "dihedral"], default="none")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument(
        "--backbone",
        default="unet",
        choices=[
            "unet",
            "cidnet",
            "hvi-cidnet",
            "saigformer",
            "mocformer",
            "mambairv2",
            "mambairv2-l",
            "restormer",
            "promptir",
            "nafnet",
            "retinexformer",
            "maxim",
            "adair",
            "bioir",
            "darkir",
            "hair",
            "hogformer",
            "onerestore",
            "anyir",
            "xrestormer",
            "x-restormer",
            "instructir",
            "dcpt",
            "dcpt-promptir",
            "perceiveir",
            "perceive-ir",
        ],
    )
    ap.add_argument("--base-channels", type=int, default=None)
    ap.add_argument("--residual-scale", type=float, default=None)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--train-limit", type=int, default=-1)
    ap.add_argument("--val-limit", type=int, default=512)
    ap.add_argument("--grad-loss-weight", type=float, default=0.1)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--shuffle-train", action="store_true", default=True)
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--profile-timing", action="store_true")
    ap.add_argument("--eval-max-batches", type=int, default=50)
    ap.add_argument("--no-visuals", action="store_true")

    ap.add_argument("--use-albedo", action="store_true")
    ap.add_argument("--use-normal", action="store_true")
    ap.add_argument("--use-roughness", action="store_true")
    ap.add_argument("--use-metallic", action="store_true")
    ap.add_argument("--use-shading", action="store_true")
    ap.add_argument("--use-glass-mask", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_names = parse_input_names(args)
    base_channels, residual_scale = resolve_stage3_hparams(args.backbone, args.base_channels, args.residual_scale)

    save_json(args.out_dir / "config.json", {
        **vars(args),
        "train_jsonl": str(args.train_jsonl) if args.train_jsonl else None,
        "val_jsonl": str(args.val_jsonl) if args.val_jsonl else None,
        "dataset_root": str(args.dataset_root) if args.dataset_root else None,
        "hf_cache_dir": str(args.hf_cache_dir) if args.hf_cache_dir else None,
        "out_dir": str(args.out_dir),
        "input_names": input_names,
        "resolved_base_channels": base_channels,
        "resolved_residual_scale": residual_scale,
    })

    train_ds, val_ds = build_datasets(args, input_names)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=args.shuffle_train,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_stage3_model(
        backbone=args.backbone,
        input_channels=train_ds.channels,
        base_channels=base_channels,
        residual_scale=residual_scale,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    print("[INFO] device       :", device, flush=True)
    print("[INFO] backbone     :", args.backbone, flush=True)
    print("[INFO] inputs       :", input_names, flush=True)
    print("[INFO] channels     :", train_ds.channels, flush=True)
    print("[INFO] base channels:", base_channels, flush=True)
    print("[INFO] residual     :", residual_scale, flush=True)
    print("[INFO] augment      :", args.augment, flush=True)
    print("[INFO] train rows   :", dataset_len_text(train_ds), "skipped", train_ds.skipped, flush=True)
    print("[INFO] val rows     :", dataset_len_text(val_ds), "skipped", val_ds.skipped, flush=True)
    print("[INFO] params       :", count_parameters(model), flush=True)

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        running_l1 = 0.0
        running_grad = 0.0
        n_steps = 0
        epoch_rows = 0
        timing_data_s = 0.0
        timing_h2d_s = 0.0
        timing_compute_s = 0.0
        timing_steps = 0
        last_step_end = time.perf_counter()

        for batch in train_dl:
            if args.profile_timing:
                batch_ready = time.perf_counter()
                timing_data_s += batch_ready - last_step_end
                if device.type == "cuda":
                    torch.cuda.synchronize()
                h2d_start = time.perf_counter()

            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            comp = batch["composite"].to(device, non_blocking=True)

            if args.profile_timing:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                h2d_done = time.perf_counter()
                timing_h2d_s += h2d_done - h2d_start
                compute_start = h2d_done

            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
                pred = model(x, comp)
                l1 = F.l1_loss(pred, y)
                gl = gradient_loss(pred, y)
                loss = l1 + args.grad_loss_weight * gl

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch} step {n_steps + 1}: "
                    f"loss={float(loss.detach().cpu())} "
                    f"l1={float(l1.detach().cpu())} "
                    f"grad={float(gl.detach().cpu())}. "
                    "Try disabling --amp or lowering --lr."
                )

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.item())
            running_l1 += float(l1.item())
            running_grad += float(gl.item())
            n_steps += 1
            epoch_rows += int(y.shape[0])

            if args.profile_timing:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                last_step_end = time.perf_counter()
                timing_compute_s += last_step_end - compute_start
                timing_steps += 1

            if args.log_every > 0 and n_steps % args.log_every == 0:
                elapsed_train = time.time() - t0
                rows_per_sec = epoch_rows / max(elapsed_train, 1e-9)
                timing_text = ""
                if args.profile_timing and timing_steps > 0:
                    timing_total = max(timing_data_s + timing_h2d_s + timing_compute_s, 1e-9)
                    timing_text = (
                        f" data={timing_data_s / timing_steps:.3f}s"
                        f" h2d={timing_h2d_s / timing_steps:.3f}s"
                        f" compute={timing_compute_s / timing_steps:.3f}s"
                        f" data%={100.0 * timing_data_s / timing_total:.1f}"
                    )
                print(
                    f"[TRAIN {epoch:03d}] "
                    f"step={n_steps} rows={epoch_rows} "
                    f"l1={running_l1 / max(1, n_steps):.6f} "
                    f"rows/s={rows_per_sec:.2f} "
                    f"elapsed={elapsed_train:.1f}s"
                    f"{timing_text}",
                    flush=True,
                )
                timing_data_s = 0.0
                timing_h2d_s = 0.0
                timing_compute_s = 0.0
                timing_steps = 0

        print(f"[EPOCH {epoch:03d}] train loop done, evaluating...", flush=True)
        metrics = evaluate(model, val_dl, device, max_batches=args.eval_max_batches)
        elapsed = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": running / max(1, n_steps),
            "train_l1": running_l1 / max(1, n_steps),
            "train_grad": running_grad / max(1, n_steps),
            **metrics,
            "seconds": elapsed,
        }
        history.append(record)
        save_json(args.out_dir / "history.json", history)

        print(
            f"[EPOCH {epoch:03d}] "
            f"train={record['train_loss']:.6f} "
            f"l1={record['train_l1']:.6f} "
            f"val_l1={record['val_l1']:.6f} "
            f"psnr={record['val_psnr']:.2f} "
            f"time={elapsed:.1f}s",
            flush=True,
        )

        if not args.no_visuals:
            print(f"[EPOCH {epoch:03d}] saving visual grid...", flush=True)
            batch = next(iter(val_dl))
            x = batch["x"].to(device)
            comp = batch["composite"].to(device)
            with torch.no_grad():
                pred = model(x, comp).cpu()
            save_visual_grid(batch, pred, args.out_dir / "visuals" / f"epoch_{epoch:03d}.jpg", input_names=input_names)

        if epoch % args.save_every == 0:
            print(f"[EPOCH {epoch:03d}] saving checkpoint...", flush=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "input_names": input_names,
                    "channels": train_ds.channels,
                    "backbone": args.backbone,
                    "base_channels": base_channels,
                    "residual_scale": residual_scale,
                    "history": history,
                },
                args.out_dir / "checkpoint_last.pt",
            )
            print(f"[EPOCH {epoch:03d}] checkpoint saved.", flush=True)

    torch.save(
        {
            "model": model.state_dict(),
            "epoch": args.epochs,
            "input_names": input_names,
            "channels": train_ds.channels,
            "backbone": args.backbone,
            "base_channels": base_channels,
            "residual_scale": residual_scale,
            "history": history,
        },
        args.out_dir / "checkpoint_final.pt",
    )
    print("[DONE] Training complete:", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
