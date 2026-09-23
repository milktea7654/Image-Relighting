from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_adair_class():
    arch_file = Path(__file__).resolve().parent / "external" / "AdaIR" / "net" / "model.py"
    if not arch_file.exists():
        raise FileNotFoundError(
            f"AdaIR arch was not found at {arch_file}. "
            "Clone https://github.com/c-yn/AdaIR into Stage3/external/AdaIR first."
        )

    spec = importlib.util.spec_from_file_location("_stage3_external_adair_model", arch_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load AdaIR arch from {arch_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AdaIR


class Stage3AdaIR(nn.Module):
    """Stage3 adapter around AdaIR, an ICLR 2025 all-in-one restoration model."""

    def __init__(self, input_channels: int = 12, base_channels: int = 56, residual_scale: float = 1.0) -> None:
        super().__init__()
        AdaIR = _load_adair_class()
        self.residual_scale = residual_scale
        self.pad_multiple = 8
        self.backbone = AdaIR(
            inp_channels=input_channels,
            out_channels=input_channels,
            dim=base_channels,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias",
            decoder=True,
        )
        self._adapt_frequency_inputs(input_channels, base_channels)
        self.out = nn.Conv2d(input_channels, 3, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _adapt_frequency_inputs(self, input_channels: int, base_channels: int) -> None:
        for module, channels in [
            (self.backbone.fre1, base_channels * 8),
            (self.backbone.fre2, base_channels * 4),
            (self.backbone.fre3, base_channels * 2),
        ]:
            module.conv = nn.Conv2d(input_channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
            module.conv1 = nn.Conv2d(input_channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

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
