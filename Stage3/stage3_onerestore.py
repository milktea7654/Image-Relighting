from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn


def _load_onerestore_class():
    arch_file = Path(__file__).resolve().parent / "external" / "OneRestore" / "model" / "OneRestore.py"
    if not arch_file.exists():
        raise FileNotFoundError(
            f"OneRestore arch was not found at {arch_file}. "
            "Clone https://github.com/gy65896/OneRestore into Stage3/external/OneRestore first."
        )

    if "thop" not in sys.modules and importlib.util.find_spec("thop") is None:
        thop_stub = types.ModuleType("thop")

        def _missing_profile(*_args, **_kwargs):
            raise ImportError("thop is only needed for OneRestore's standalone __main__ profiling block.")

        thop_stub.profile = _missing_profile
        sys.modules["thop"] = thop_stub

    spec = importlib.util.spec_from_file_location("_stage3_external_onerestore_arch", arch_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load OneRestore arch from {arch_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OneRestore


class Stage3OneRestore(nn.Module):
    """Stage3 adapter around a widened OneRestore.

    OneRestore is naturally condition-driven: it restores an RGB image using a
    324-d descriptor. Here the composite image is the RGB restoration target
    input, while the full 12-channel Stage3 tensor is encoded into that
    descriptor so render/material/mask information can steer the restoration.
    """

    def __init__(self, input_channels: int = 12, base_channels: int = 80, residual_scale: float = 1.0) -> None:
        super().__init__()
        OneRestore = _load_onerestore_class()
        self.residual_scale = residual_scale
        self.backbone = OneRestore(channel=base_channels)
        cond_width = max(64, base_channels)
        self.condition = nn.Sequential(
            nn.Conv2d(input_channels, cond_width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(cond_width, cond_width, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cond_width, 324),
            nn.LayerNorm(324),
        )
        self.out = nn.Conv2d(3, 3, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, composite: torch.Tensor) -> torch.Tensor:
        embedding = self.condition(x)
        features = self.backbone(composite, embedding)
        residual = torch.tanh(self.out(features)) * self.residual_scale
        return torch.clamp(composite + residual, 0.0, 1.0)
