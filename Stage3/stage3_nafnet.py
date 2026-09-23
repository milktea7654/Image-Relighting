from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn


def _load_nafnet_class():
    root = Path(__file__).resolve().parent / "external" / "NAFNet"
    arch_file = root / "basicsr" / "models" / "archs" / "NAFNet_arch.py"
    if not arch_file.exists():
        raise FileNotFoundError(
            f"NAFNet official arch was not found at {arch_file}. "
            "Clone https://github.com/megvii-research/NAFNet into Stage3/external/NAFNet first."
        )

    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("_stage3_external_nafnet_arch", arch_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load NAFNet arch from {arch_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NAFNet


class Stage3NAFNet(nn.Module):
    """Stage3 adapter around NAFNet-width64."""

    def __init__(self, input_channels: int = 12, base_channels: int = 64, residual_scale: float = 1.0) -> None:
        super().__init__()
        NAFNet = _load_nafnet_class()
        self.residual_scale = residual_scale
        self.backbone = NAFNet(
            img_channel=input_channels,
            width=base_channels,
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1, 28],
            dec_blk_nums=[1, 1, 1, 1],
        )
        self.out = nn.Conv2d(input_channels, 3, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, composite: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        residual = torch.tanh(self.out(features)) * self.residual_scale
        return torch.clamp(composite + residual, 0.0, 1.0)
