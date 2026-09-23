from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _install_optional_stubs() -> None:
    if importlib.util.find_spec("fvcore") is None and "fvcore" not in sys.modules:
        fvcore = types.ModuleType("fvcore")
        fvcore_nn = types.ModuleType("fvcore.nn")

        class _MissingFlopCountAnalysis:
            def __init__(self, *_args, **_kwargs) -> None:
                raise ImportError("fvcore is only needed for AnyIR's standalone profiling block.")

        def _missing_flop_count_table(*_args, **_kwargs) -> str:
            raise ImportError("fvcore is only needed for AnyIR's standalone profiling block.")

        fvcore_nn.FlopCountAnalysis = _MissingFlopCountAnalysis
        fvcore_nn.flop_count_table = _missing_flop_count_table
        sys.modules["fvcore"] = fvcore
        sys.modules["fvcore.nn"] = fvcore_nn

    if importlib.util.find_spec("timm") is None and "timm" not in sys.modules:
        timm = types.ModuleType("timm")
        timm_models = types.ModuleType("timm.models")
        timm_layers = types.ModuleType("timm.models.layers")

        class DropPath(nn.Identity):
            def __init__(self, drop_prob: float = 0.0) -> None:
                super().__init__()
                self.drop_prob = drop_prob

        timm_layers.DropPath = DropPath
        sys.modules["timm"] = timm
        sys.modules["timm.models"] = timm_models
        sys.modules["timm.models.layers"] = timm_layers


def _load_anyir_class():
    root = Path(__file__).resolve().parent / "external" / "AnyIR"
    arch_file = root / "net" / "anyir.py"
    if not arch_file.exists():
        raise FileNotFoundError(
            f"AnyIR arch was not found at {arch_file}. "
            "Clone https://github.com/Amazingren/AnyIR into Stage3/external/AnyIR first."
        )

    _install_optional_stubs()
    spec = importlib.util.spec_from_file_location("_stage3_external_anyir_arch", arch_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load AnyIR arch from {arch_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AnyIR


class Stage3AnyIR(nn.Module):
    """Stage3 adapter around AnyIR."""

    def __init__(self, input_channels: int = 12, base_channels: int = 48, residual_scale: float = 1.0) -> None:
        super().__init__()
        AnyIR = _load_anyir_class()
        self.residual_scale = residual_scale
        self.pad_multiple = 8
        self.backbone = AnyIR(
            inp_channels=input_channels,
            out_channels=input_channels,
            dim=base_channels,
            num_blocks=[3, 5, 5, 7],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2,
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
