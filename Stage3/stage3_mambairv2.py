from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_mambairv2_class():
    repo = Path(__file__).resolve().parent / "external" / "MambaIR"
    arch_file = repo / "basicsr" / "archs" / "mambairv2_arch.py"
    if not arch_file.exists():
        raise FileNotFoundError(
            f"MambaIRv2 official arch was not found at {arch_file}. "
            "Clone https://github.com/csguoh/MambaIR into Stage3/external/MambaIR first."
        )

    spec = importlib.util.spec_from_file_location("_stage3_external_mambairv2_arch", arch_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load MambaIRv2 arch from {arch_file}")
    module = importlib.util.module_from_spec(spec)

    saved_modules = {name: sys.modules.get(name) for name in (
        "basicsr",
        "basicsr.archs",
        "basicsr.archs.arch_util",
        "basicsr.utils",
        "basicsr.utils.registry",
    )}

    def _to_2tuple(x):
        return x if isinstance(x, tuple) else (x, x)

    class _NoopRegistry:
        def register(self, obj=None):
            if obj is not None:
                return obj

            def deco(item):
                return item

            return deco

    basicsr = types.ModuleType("basicsr")
    basicsr_archs = types.ModuleType("basicsr.archs")
    arch_util = types.ModuleType("basicsr.archs.arch_util")
    arch_util.to_2tuple = _to_2tuple
    arch_util.trunc_normal_ = nn.init.trunc_normal_
    basicsr_utils = types.ModuleType("basicsr.utils")
    registry = types.ModuleType("basicsr.utils.registry")
    registry.ARCH_REGISTRY = _NoopRegistry()

    sys.modules["basicsr"] = basicsr
    sys.modules["basicsr.archs"] = basicsr_archs
    sys.modules["basicsr.archs.arch_util"] = arch_util
    sys.modules["basicsr.utils"] = basicsr_utils
    sys.modules["basicsr.utils.registry"] = registry

    try:
        spec.loader.exec_module(module)
    finally:
        for name, saved in saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved
    return module.MambaIRv2


class Stage3MambaIRv2(nn.Module):
    """Stage3 adapter around the official MambaIRv2 restoration backbone.

    The official MambaIRv2 restores same-channel images. For Stage3 we keep the
    full 12-channel conditioning tensor as the restoration input, then learn a
    small projection from restored features to an RGB residual over composite.
    """

    def __init__(
        self,
        input_channels: int = 12,
        base_channels: int = 174,
        residual_scale: float = 1.0,
        variant: str = "large",
    ) -> None:
        super().__init__()
        if variant.lower() not in {"large", "l"}:
            raise ValueError("Stage3MambaIRv2 currently supports the large/L setting only.")

        MambaIRv2 = _load_mambairv2_class()
        self.residual_scale = residual_scale
        self.window_size = 16
        self.backbone = MambaIRv2(
            img_size=64,
            patch_size=1,
            in_chans=input_channels,
            embed_dim=base_channels,
            d_state=16,
            depths=(6, 6, 6, 6, 6, 6, 6, 6, 6),
            num_heads=(6, 6, 6, 6, 6, 6, 6, 6, 6),
            window_size=self.window_size,
            inner_rank=64,
            num_tokens=128,
            convffn_kernel_size=5,
            mlp_ratio=2.0,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False,
            upscale=1,
            img_range=1.0,
            upsampler="",
            resi_connection="1conv",
        )
        self.out = nn.Conv2d(input_channels, 3, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, composite: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h or pad_w:
            x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_in = x

        features = self.backbone(x_in)
        features = features[..., :h, :w]
        residual = torch.tanh(self.out(features)) * self.residual_scale
        return torch.clamp(composite + residual, 0.0, 1.0)
