from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_bioir_class():
    arch_file = Path(__file__).resolve().parent / "external" / "BioIR" / "All_in_One" / "net" / "model.py"
    if not arch_file.exists():
        raise FileNotFoundError(
            f"BioIR arch was not found at {arch_file}. "
            "Clone https://github.com/c-yn/BioIR into Stage3/external/BioIR first."
        )

    spec = importlib.util.spec_from_file_location("_stage3_external_bioir_model", arch_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load BioIR arch from {arch_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BioIR


class Stage3BioIR(nn.Module):
    """Stage3 adapter around BioIR, a NeurIPS 2025 universal restoration model."""

    def __init__(self, input_channels: int = 12, base_channels: int = 80, residual_scale: float = 1.0) -> None:
        super().__init__()
        BioIR = _load_bioir_class()
        self.residual_scale = residual_scale
        self.pad_multiple = 4
        self.backbone = BioIR(
            inp_channels=input_channels,
            out_channels=input_channels,
            dim=base_channels,
            num_blocks=[6, 6, 14],
            num_refinement_blocks=4,
            ffn_expansion_factor=3,
            bias=False,
            LayerNorm_type="WithBias",
        )
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
