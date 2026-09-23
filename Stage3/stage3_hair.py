from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_hair_class():
    root = Path(__file__).resolve().parent / "external" / "HAIR"
    arch_file = root / "net" / "HAIR.py"
    if not arch_file.exists():
        raise FileNotFoundError(
            f"HAIR arch was not found at {arch_file}. "
            "Clone https://github.com/jin-cao-tma/HAIR into Stage3/external/HAIR first."
        )

    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("_stage3_external_hair_arch", arch_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load HAIR arch from {arch_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HAIR


class Stage3HAIR(nn.Module):
    """Stage3 adapter around HAIR / Res-HAIR.

    HAIR is Restormer-like but dynamically generates restoration kernels from
    the input image. We keep the 12-channel tensor intact through the backbone
    and learn a small RGB residual head for the Stage3 objective.
    """

    def __init__(self, input_channels: int = 12, base_channels: int = 64, residual_scale: float = 1.0) -> None:
        super().__init__()
        HAIR = _load_hair_class()
        self.residual_scale = residual_scale
        self.pad_multiple = 8
        self.backbone = HAIR(inp_channels=input_channels, out_channels=input_channels, dim=base_channels)
        expected_level3_in = int(base_channels * 2) + int(base_channels * 4)
        expected_level3_out = int(base_channels * 4)
        if self.backbone.reduce_chan_level3.in_channels != expected_level3_in:
            self.backbone.reduce_chan_level3 = nn.Conv2d(expected_level3_in, expected_level3_out, kernel_size=1, bias=False)
        self.out = nn.Conv2d(input_channels, 3, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, composite: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h = (self.pad_multiple - h % self.pad_multiple) % self.pad_multiple
        pad_w = (self.pad_multiple - w % self.pad_multiple) % self.pad_multiple
        if pad_h or pad_w:
            x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_in = x

        features = self.backbone(x_in)[..., :h, :w]
        residual = torch.tanh(self.out(features)) * self.residual_scale
        return torch.clamp(composite + residual, 0.0, 1.0)
