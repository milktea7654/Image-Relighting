from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_hogformer_module():
    arch_file = (
        Path(__file__).resolve().parent
        / "external"
        / "HOGformer"
        / "settingIII&IV"
        / "net"
        / "model.py"
    )
    if not arch_file.exists():
        raise FileNotFoundError(
            f"HOGformer arch was not found at {arch_file}. "
            "Clone https://github.com/Fire-friend/HOGformer into Stage3/external/HOGformer first."
        )

    spec = importlib.util.spec_from_file_location("_stage3_external_hogformer_arch", arch_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load HOGformer arch from {arch_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stage3HOGFormer(nn.Module):
    """Stage3 adapter around HOGformer.

    The first skip branch in the released model is hard-coded for RGB input.
    We replace that branch so the transformer can consume all Stage3 channels
    directly while keeping HOGformer's internal 3-channel skip hints.
    """

    def __init__(self, input_channels: int = 12, base_channels: int = 54, residual_scale: float = 1.0) -> None:
        super().__init__()
        module = _load_hogformer_module()
        self.residual_scale = residual_scale
        self.pad_multiple = 8
        self.backbone = module.HOGformer(
            inp_channels=input_channels,
            out_channels=input_channels,
            dim=base_channels,
        )
        self.backbone.skip_patch_embed1 = module.SkipPatchEmbed(input_channels, 3)
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
