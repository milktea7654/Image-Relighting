from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


HF_IMAGE_KEYS = {
    "reference": "reference",
    "composite": "composite_optimized",
    "render": "render_optimized",
    "albedo": "hybrid_albedo",
    "normal": "normal",
    "roughness": "roughness",
    "metallic": "metallic",
    "shading": "shading",
    "glass_mask": "glass_mask_soft",
}


def _pil_from_hf_image(value: Any) -> Optional[Image.Image]:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(BytesIO(value["bytes"]))
        if value.get("path"):
            return Image.open(value["path"])
    if isinstance(value, (str, Path)):
        return Image.open(value)
    raise TypeError(f"Unsupported HF image value: {type(value)!r}")


def _load_rgb(value: Any, size: Tuple[int, int]) -> torch.Tensor:
    img = _pil_from_hf_image(value)
    if img is None:
        return torch.zeros(3, size[0], size[1], dtype=torch.float32)
    with img:
        img = img.convert("RGB")
        img = img.resize((size[1], size[0]), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _load_gray(value: Any, size: Tuple[int, int]) -> torch.Tensor:
    img = _pil_from_hf_image(value)
    if img is None:
        return torch.zeros(1, size[0], size[1], dtype=torch.float32)
    with img:
        img = img.convert("L")
        img = img.resize((size[1], size[0]), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...].contiguous()


def _safe_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    keep = [
        "scene_id",
        "source_stem",
        "split",
        "quality_final_loss",
        "quality_best_loss",
        "quality_final_rgb_loss",
        "glass_mask_ratio",
        "original_width",
        "original_height",
    ]
    return {key: row[key] for key in keep if key in row}


def _compute_channels(input_names: Sequence[str]) -> int:
    channels = 0
    for name in input_names:
        channels += 1 if name in {"roughness", "metallic", "glass_mask"} else 3
    return channels


def _load_logical(row: Dict[str, Any], name: str, size: Tuple[int, int]) -> torch.Tensor:
    key = HF_IMAGE_KEYS.get(name, name)
    value = row.get(key)
    if name in {"roughness", "metallic", "glass_mask"}:
        return _load_gray(value, size)
    return _load_rgb(value, size)


def _make_sample(
    row: Dict[str, Any],
    idx: int,
    size: Tuple[int, int],
    input_names: Sequence[str],
    target_name: str,
) -> Dict[str, Any]:
    reference = _load_logical(row, "reference", size)
    composite = _load_logical(row, "composite", size)
    render = _load_logical(row, "render", size)

    tensors: Dict[str, torch.Tensor] = {
        "reference": reference,
        "composite": composite,
        "render": render,
    }

    for name in ["albedo", "normal", "roughness", "metallic", "shading", "glass_mask"]:
        if name in input_names:
            tensors[name] = _load_logical(row, name, size)

    x = torch.cat([tensors[name] for name in input_names], dim=0)
    scene_id = row.get("scene_id") or row.get("source_stem") or f"row_{idx:06d}"

    out = {
        "x": x,
        "y": tensors[target_name],
        "reference": reference,
        "composite": composite,
        "render": render,
        "scene_id": str(scene_id),
        "row": _safe_metadata(row),
        "paths": {},
    }

    for key in ["albedo", "normal", "roughness", "metallic", "shading", "glass_mask"]:
        if key in tensors:
            out[key] = tensors[key]

    return out


class Stage3HFRelightDataset(Dataset):
    """
    Stage 3 wrapper for the Hugging Face Parquet dataset.

    The HF dataset stores images as embedded Image features. This wrapper returns
    the same batch keys as Stage3RelightDataset so existing training/eval code can
    share the model and visualization paths.
    """

    def __init__(
        self,
        hf_dataset,
        image_size: int | Tuple[int, int] = 256,
        input_names: Sequence[str] = ("composite", "render", "albedo", "roughness", "metallic", "glass_mask"),
        target_name: str = "reference",
    ) -> None:
        if isinstance(image_size, int):
            self.size = (image_size, image_size)
        else:
            self.size = (int(image_size[0]), int(image_size[1]))

        self.ds = hf_dataset
        self.input_names = tuple(input_names)
        self.target_name = target_name
        self.skipped = 0
        self.channels = self._compute_channels()

    def _compute_channels(self) -> int:
        return _compute_channels(self.input_names)

    def __len__(self) -> int:
        return len(self.ds)

    def _load_logical(self, row: Dict[str, Any], name: str) -> torch.Tensor:
        return _load_logical(row, name, self.size)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.ds[idx]
        return _make_sample(row, idx, self.size, self.input_names, self.target_name)
