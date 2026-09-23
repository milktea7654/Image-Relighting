from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _install_basic_stubs() -> None:
    if "basicsr.utils.registry" not in sys.modules:
        basicsr = sys.modules.setdefault("basicsr", types.ModuleType("basicsr"))
        basicsr_utils = sys.modules.setdefault("basicsr.utils", types.ModuleType("basicsr.utils"))
        registry_module = types.ModuleType("basicsr.utils.registry")

        class _Registry:
            def register(self, obj=None, **_kwargs):
                if obj is not None:
                    return obj

                def _decorator(cls):
                    return cls

                return _decorator

        registry_module.ARCH_REGISTRY = _Registry()
        basicsr.utils = basicsr_utils
        basicsr_utils.registry = registry_module
        sys.modules["basicsr.utils.registry"] = registry_module

    if importlib.util.find_spec("torchstat") is None and "torchstat" not in sys.modules:
        torchstat = types.ModuleType("torchstat")

        def _missing_stat(*_args, **_kwargs):
            raise ImportError("torchstat is only needed for X-Restormer's standalone profiling block.")

        torchstat.stat = _missing_stat
        sys.modules["torchstat"] = torchstat


def _load_xrestormer_class():
    arch_file = (
        Path(__file__).resolve().parent
        / "external"
        / "X-Restormer"
        / "xrestormer"
        / "archs"
        / "xrestormer_arch.py"
    )
    if not arch_file.exists():
        raise FileNotFoundError(
            f"X-Restormer arch was not found at {arch_file}. "
            "Clone https://github.com/Andrew0613/X-Restormer into Stage3/external/X-Restormer first."
        )

    _install_basic_stubs()
    spec = importlib.util.spec_from_file_location("_stage3_external_xrestormer_arch", arch_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load X-Restormer arch from {arch_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.XRestormer


class Stage3XRestormer(nn.Module):
    """Stage3 adapter around X-Restormer."""

    def __init__(self, input_channels: int = 12, base_channels: int = 48, residual_scale: float = 1.0) -> None:
        super().__init__()
        XRestormer = _load_xrestormer_class()
        self.residual_scale = residual_scale
        self.pad_multiple = 64
        self.backbone = XRestormer(
            inp_channels=input_channels,
            out_channels=input_channels,
            dim=base_channels,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            channel_heads=[1, 2, 4, 8],
            spatial_heads=[2, 2, 3, 4],
            overlap_ratio=[0.5, 0.5, 0.5, 0.5],
            window_size=8,
            spatial_dim_head=16,
            bias=False,
            ffn_expansion_factor=2.66,
            LayerNorm_type="WithBias",
            scale=1,
        )
        self.out = nn.Conv2d(input_channels, 3, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, composite: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h = (self.pad_multiple - h % self.pad_multiple) % self.pad_multiple
        pad_w = (self.pad_multiple - w % self.pad_multiple) % self.pad_multiple
        if pad_h or pad_w:
            pad_mode = "replicate" if pad_h >= h or pad_w >= w else "reflect"
            x_in = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
        else:
            x_in = x

        features = self.backbone(x_in)[..., :h, :w]
        residual = torch.tanh(self.out(features)) * self.residual_scale
        return torch.clamp(composite + residual, 0.0, 1.0)
