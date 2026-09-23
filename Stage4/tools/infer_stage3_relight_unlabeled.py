from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader

from stage3_unet import ResidualUNet


KEYS = {
    "composite": ["composite_optimized", "optimized_composite", "composite", "composite_path"],
    "render": ["render_optimized", "render_optimize", "optimized_render", "render", "render_path"],
    "albedo": ["hybrid_albedo", "albedo", "mvinverse_albedo"],
    "normal": ["normal", "mvinverse_normal"],
    "roughness": ["roughness", "mvinverse_roughness"],
    "metallic": ["metallic", "mvinverse_metallic"],
    "shading": ["shading", "mvinverse_shading"],
    "glass_mask": ["glass_mask_soft", "glass_mask"],
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Bad JSONL at {path}:{line_no}: {e}") from e
            if isinstance(row, dict):
                rows.append(row)
    return rows


def resolve_path(v: Any, dataset_root: Optional[Path]) -> Optional[Path]:
    if v is None or v == "":
        return None
    p = Path(str(v))
    if p.is_absolute():
        return p
    if dataset_root is not None:
        return dataset_root / p
    return p


def find_path(row: Dict[str, Any], logical_name: str, dataset_root: Optional[Path], required: bool = True) -> Optional[Path]:
    keys = KEYS.get(logical_name, [logical_name])
    for k in keys:
        if row.get(k):
            p = resolve_path(row[k], dataset_root)
            if p is not None and p.exists():
                return p
    if required:
        raise FileNotFoundError(f"Missing required input '{logical_name}' in row: {row.get('scene_id', row)}")
    return None


def scene_id(row: Dict[str, Any], fallback: Optional[Path] = None) -> str:
    for k in ["scene_id", "source_stem", "sample_id", "id"]:
        if row.get(k):
            return str(row[k])
    if fallback is not None:
        return fallback.stem
    return "scene"


def load_rgb(path: Path, size: int) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB").resize((size, size), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def load_gray(path: Path, size: int) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("L").resize((size, size), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...].contiguous()


class UnlabeledRelightDataset(Dataset):
    def __init__(
        self,
        jsonl: Path,
        dataset_root: Optional[Path],
        input_names: List[str],
        image_size: int,
        missing_optional: str = "zeros",
    ) -> None:
        self.rows = read_jsonl(jsonl)
        self.dataset_root = dataset_root
        self.input_names = input_names
        self.image_size = image_size
        self.missing_optional = missing_optional

        usable = []
        skipped = 0
        for row in self.rows:
            try:
                find_path(row, "composite", dataset_root, required=True)
                find_path(row, "render", dataset_root, required=True)
                usable.append(row)
            except Exception:
                skipped += 1
        self.rows = usable
        self.skipped = skipped
        if not self.rows:
            raise RuntimeError(f"No usable rows in {jsonl}")

    def __len__(self) -> int:
        return len(self.rows)

    def _load_optional(self, row: Dict[str, Any], name: str) -> torch.Tensor:
        is_gray = name in {"roughness", "metallic", "glass_mask"}
        p = find_path(row, name, self.dataset_root, required=False)
        if p is not None:
            return load_gray(p, self.image_size) if is_gray else load_rgb(p, self.image_size)

        if self.missing_optional == "zeros":
            c = 1 if is_gray else 3
            return torch.zeros(c, self.image_size, self.image_size, dtype=torch.float32)
        raise FileNotFoundError(f"Missing input {name} for {scene_id(row)}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        comp_p = find_path(row, "composite", self.dataset_root, required=True)
        render_p = find_path(row, "render", self.dataset_root, required=True)
        assert comp_p is not None and render_p is not None

        composite = load_rgb(comp_p, self.image_size)
        render = load_rgb(render_p, self.image_size)

        tensors = {
            "composite": composite,
            "render": render,
        }

        xs = []
        for name in self.input_names:
            if name == "composite":
                t = composite
            elif name == "render":
                t = render
            else:
                t = self._load_optional(row, name)
            tensors[name] = t
            xs.append(t)

        return {
            "x": torch.cat(xs, dim=0),
            "composite": composite,
            "render": render,
            "scene_id": scene_id(row, comp_p),
            "row": row,
            **{k: v for k, v in tensors.items() if k not in {"composite", "render"}},
        }


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().float().cpu().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    x = (x * 255.0).round().byte()
    x = x.permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def save_tensor(path: Path, x: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(x).save(path)


def make_contact_sheet(path: Path, scene: str, batch: Dict[str, Any], pred: torch.Tensor, input_names: List[str], i: int) -> None:
    cols = [
        ("composite_in", batch["composite"][i]),
        ("render_in", batch["render"][i]),
    ]
    for name in ["albedo", "normal", "roughness", "metallic", "shading", "glass_mask"]:
        if name in input_names and name in batch:
            cols.append((f"{name}_in", batch[name][i]))

    cols.append(("stage3_prediction", pred))

    imgs = [(label, tensor_to_pil(t)) for label, t in cols]
    w, h = imgs[0][1].size
    gap = 6
    title_h = 28
    label_h = 24

    canvas = Image.new(
        "RGB",
        (len(imgs) * w + (len(imgs) + 1) * gap, h + title_h + label_h + 3 * gap),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, gap), scene, fill=(0, 0, 0))

    x = gap
    y = gap + title_h
    for label, img in imgs:
        draw.text((x, y), label, fill=(0, 0, 0))
        canvas.paste(img, (x, y + label_h))
        x += w + gap

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def find_checkpoint(run_dir: Optional[Path], checkpoint: Optional[Path]) -> Path:
    if checkpoint is not None:
        return checkpoint
    if run_dir is None:
        raise ValueError("Need --run-dir or --checkpoint")
    for name in ["checkpoint_final.pt", "checkpoint_last.pt"]:
        p = run_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


def infer_channels(input_names: List[str]) -> int:
    return sum(1 if n in {"roughness", "metallic", "glass_mask"} else 3 for n in input_names)


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    input_names = list(ckpt.get("input_names") or ckpt.get("inputs") or [])
    if not input_names:
        raise RuntimeError("Checkpoint missing input_names.")
    channels = int(ckpt.get("channels", 0)) or infer_channels(input_names)
    base_channels = int(ckpt.get("base_channels", 32))

    model = ResidualUNet(input_channels=channels, base_channels=base_channels)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, input_names, channels, base_channels


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-inputs", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt_path = find_checkpoint(args.run_dir, args.checkpoint)
    model, input_names, channels, base_channels = load_model(ckpt_path, device)

    print("[INFO] checkpoint   :", ckpt_path)
    print("[INFO] input_names  :", input_names)
    print("[INFO] channels     :", channels)
    print("[INFO] base_channels:", base_channels)
    print("[INFO] device       :", device)

    ds = UnlabeledRelightDataset(
        jsonl=args.jsonl,
        dataset_root=args.dataset_root,
        input_names=input_names,
        image_size=args.image_size,
    )
    print("[INFO] rows         :", len(ds), "skipped", ds.skipped)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for batch in dl:
        x = batch["x"].to(device, non_blocking=True)
        comp = batch["composite"].to(device, non_blocking=True)
        pred = model(x, comp).cpu()

        for i in range(pred.shape[0]):
            sid = str(batch["scene_id"][i])
            scene_dir = args.out_dir / sid
            scene_dir.mkdir(parents=True, exist_ok=True)

            save_tensor(scene_dir / "stage3_prediction.png", pred[i])
            make_contact_sheet(scene_dir / "contact_sheet.jpg", sid, batch, pred[i], input_names, i)

            if args.save_inputs:
                save_tensor(scene_dir / "composite_in.png", batch["composite"][i])
                save_tensor(scene_dir / "render_in.png", batch["render"][i])
                for name in ["albedo", "normal", "roughness", "metallic", "shading", "glass_mask"]:
                    if name in input_names and name in batch:
                        save_tensor(scene_dir / f"{name}_in.png", batch[name][i])

            records.append({
                "scene_id": sid,
                "prediction": str(scene_dir / "stage3_prediction.png"),
                "contact_sheet": str(scene_dir / "contact_sheet.jpg"),
            })
            print("[DONE]", sid)

    with (args.out_dir / "inference_outputs.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene_id", "prediction", "contact_sheet"])
        w.writeheader()
        for r in records:
            w.writerow(r)

    with (args.out_dir / "inference_outputs.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print("[OUT]", args.out_dir)


if __name__ == "__main__":
    main()
