from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn


def _load_instructir_class():
    root = Path(__file__).resolve().parent / "external" / "InstructIR"
    arch_file = root / "models" / "instructir.py"
    if not arch_file.exists():
        raise FileNotFoundError(
            f"InstructIR arch was not found at {arch_file}. "
            "Clone https://github.com/mv-lab/InstructIR into Stage3/external/InstructIR first."
        )

    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("_stage3_external_instructir_arch", arch_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load InstructIR arch from {arch_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.InstructIR


class Stage3InstructIR(nn.Module):
    """Stage3 adapter around InstructIR with learned Stage3 instructions."""

    def __init__(self, input_channels: int = 12, base_channels: int = 48, residual_scale: float = 1.0) -> None:
        super().__init__()
        InstructIR = _load_instructir_class()
        self.residual_scale = residual_scale
        self.backbone = InstructIR(
            img_channel=input_channels,
            width=base_channels,
            middle_blk_num=12,
            enc_blk_nums=[2, 2, 4, 8],
            dec_blk_nums=[2, 2, 2, 2],
            txtdim=768,
        )
        cond_width = max(64, base_channels)
        self.condition = nn.Sequential(
            nn.Conv2d(input_channels, cond_width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(cond_width, cond_width, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cond_width, 768),
            nn.LayerNorm(768),
        )
        self.out = nn.Conv2d(input_channels, 3, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, composite: torch.Tensor) -> torch.Tensor:
        text_embedding = self.condition(x)
        features = self.backbone(x, text_embedding)
        residual = torch.tanh(self.out(features)) * self.residual_scale
        return torch.clamp(composite + residual, 0.0, 1.0)
