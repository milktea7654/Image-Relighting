from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _install_basic_registry_stub() -> None:
    if "basicsr.utils.registry" in sys.modules:
        return

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


def _load_dcpt_promptir_class():
    arch_file = Path(__file__).resolve().parent / "external" / "DCPT" / "basicsr" / "archs" / "promptir_arch.py"
    if not arch_file.exists():
        raise FileNotFoundError(
            f"DCPT PromptIR arch was not found at {arch_file}. "
            "Clone https://github.com/MILab-PKU/dcpt into Stage3/external/DCPT first."
        )

    _install_basic_registry_stub()
    spec = importlib.util.spec_from_file_location("_stage3_external_dcpt_promptir_arch", arch_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load DCPT PromptIR arch from {arch_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PromptIR


class Stage3DCPTPromptIR(nn.Module):
    """Stage3 adapter around the DCPT PromptIR implementation."""

    def __init__(self, input_channels: int = 12, base_channels: int = 48, residual_scale: float = 1.0) -> None:
        super().__init__()
        if base_channels != 48:
            raise ValueError("DCPT PromptIR prompt dimensions assume base_channels=48.")

        PromptIR = _load_dcpt_promptir_class()
        self.residual_scale = residual_scale
        self.pad_multiple = 8
        self.backbone = PromptIR(
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
