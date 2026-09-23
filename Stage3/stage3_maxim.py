from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_maxim_class():
    arch_file = (
        Path(__file__).resolve().parent
        / "external"
        / "MAXIM-PyTorch"
        / "maxim_pytorch"
        / "maxim_torch.py"
    )
    if not arch_file.exists():
        raise FileNotFoundError(
            f"MAXIM PyTorch arch was not found at {arch_file}. "
            "Clone https://github.com/vztu/maxim-pytorch into Stage3/external/MAXIM-PyTorch first."
        )

    spec = importlib.util.spec_from_file_location("_stage3_external_maxim_arch", arch_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load MAXIM arch from {arch_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MAXIM_dns_3s


class Stage3MAXIM(nn.Module):
    """Stage3 adapter around the PyTorch MAXIM-3S implementation."""

    def __init__(self, input_channels: int = 12, base_channels: int = 32, residual_scale: float = 1.0) -> None:
        super().__init__()
        MAXIM = _load_maxim_class()
        self.residual_scale = residual_scale
        self.pad_multiple = 16
        self.stem = nn.Conv2d(input_channels, 3, kernel_size=1)
        self.backbone = MAXIM(features=base_channels, num_outputs=3)
        self.out = nn.Conv2d(3, 3, kernel_size=1)
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

        outputs = self.backbone(self.stem(x_in))
        features = outputs[-1][-1][..., :h, :w]
        residual = torch.tanh(self.out(features)) * self.residual_scale
        return torch.clamp(composite + residual, 0.0, 1.0)
