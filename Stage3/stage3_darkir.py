from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn


def _load_darkir_class():
    root = Path(__file__).resolve().parent / "external" / "DarkIR"
    arch_file = root / "archs" / "DarkIR.py"
    if not arch_file.exists():
        raise FileNotFoundError(
            f"DarkIR arch was not found at {arch_file}. "
            "Clone https://github.com/cidautai/DarkIR into Stage3/external/DarkIR first."
        )

    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "archs"))
    spec = importlib.util.spec_from_file_location("_stage3_external_darkir_arch", arch_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load DarkIR arch from {arch_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DarkIR


class Stage3DarkIR(nn.Module):
    """Stage3 adapter around DarkIR, a CVPR 2025 low-light restoration model."""

    def __init__(self, input_channels: int = 12, base_channels: int = 128, residual_scale: float = 1.0) -> None:
        super().__init__()
        DarkIR = _load_darkir_class()
        self.residual_scale = residual_scale
        self.backbone = DarkIR(
            img_channel=input_channels,
            width=base_channels,
            middle_blk_num_enc=2,
            middle_blk_num_dec=2,
            enc_blk_nums=[1, 2, 3],
            dec_blk_nums=[3, 1, 1],
            dilations=[1, 4, 9],
            extra_depth_wise=True,
        )
        self.out = nn.Conv2d(input_channels, 3, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, composite: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        residual = torch.tanh(self.out(features)) * self.residual_scale
        return torch.clamp(composite + residual, 0.0, 1.0)
