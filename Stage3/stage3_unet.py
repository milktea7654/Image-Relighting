from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8) -> None:
        super().__init__()
        g1 = min(groups, out_ch)
        while out_ch % g1 != 0 and g1 > 1:
            g1 -= 1

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g1, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g1, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AvgPool2d(2),
            ConvBlock(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResidualUNet(nn.Module):
    """
    Small residual U-Net.

    Input:
      x: [B, input_channels, H, W]
      composite: [B, 3, H, W]

    Output:
      pred = clamp(composite + residual, 0, 1)
    """

    def __init__(self, input_channels: int = 9, base_channels: int = 32, residual_scale: float = 0.25) -> None:
        super().__init__()
        c = base_channels
        self.residual_scale = residual_scale

        self.inc = ConvBlock(input_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)

        self.mid = ConvBlock(c * 8, c * 8)

        self.up3 = Up(c * 8, c * 4, c * 4)
        self.up2 = Up(c * 4, c * 2, c * 2)
        self.up1 = Up(c * 2, c, c)

        self.out = nn.Conv2d(c, 3, kernel_size=1)

        # Start close to identity: composite + tiny correction.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, composite: torch.Tensor) -> torch.Tensor:
        s1 = self.inc(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        z = self.down3(s3)

        z = self.mid(z)

        z = self.up3(z, s3)
        z = self.up2(z, s2)
        z = self.up1(z, s1)

        residual = torch.tanh(self.out(z)) * self.residual_scale
        pred = torch.clamp(composite + residual, 0.0, 1.0)
        return pred


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_stage3_model(
    backbone: str,
    input_channels: int,
    base_channels: int | None = None,
    residual_scale: float | None = None,
) -> nn.Module:
    base_channels, residual_scale = resolve_stage3_hparams(backbone, base_channels, residual_scale)
    name = backbone.lower().replace("_", "-")
    if name in {"unet", "residual-unet", "baseline"}:
        return ResidualUNet(
            input_channels=input_channels,
            base_channels=base_channels,
            residual_scale=residual_scale,
        )
    if name in {"cidnet", "hvi-cidnet", "stage3-cidnet"}:
        from stage3_cidnet import Stage3CIDNet

        return Stage3CIDNet(
            input_channels=input_channels,
            base_channels=base_channels,
            residual_scale=residual_scale,
        )
    if name in {"saigformer", "stage3-saigformer"}:
        from stage3_saigformer import Stage3SAIGFormer

        return Stage3SAIGFormer(
            input_channels=input_channels,
            base_channels=base_channels,
            residual_scale=residual_scale,
        )
    if name in {"mocformer", "stage3-mocformer"}:
        from stage3_mocformer import Stage3MOCFormer

        return Stage3MOCFormer(
            input_channels=input_channels,
            base_channels=base_channels,
            residual_scale=residual_scale,
        )
    if name in {"mambairv2", "mambairv2-l", "stage3-mambairv2"}:
        from stage3_mambairv2 import Stage3MambaIRv2

        return Stage3MambaIRv2(
            input_channels=input_channels,
            base_channels=base_channels,
            residual_scale=residual_scale,
        )
    if name in {"restormer", "stage3-restormer"}:
        from stage3_restormer import Stage3Restormer

        return Stage3Restormer(
            input_channels=input_channels,
            base_channels=base_channels,
            residual_scale=residual_scale,
        )
    if name in {"promptir", "stage3-promptir"}:
        from stage3_promptir import Stage3PromptIR

        return Stage3PromptIR(
            input_channels=input_channels,
            base_channels=base_channels,
            residual_scale=residual_scale,
        )
    if name in {"nafnet", "stage3-nafnet"}:
        from stage3_nafnet import Stage3NAFNet

        return Stage3NAFNet(
            input_channels=input_channels,
            base_channels=base_channels,
            residual_scale=residual_scale,
        )
    if name in {"retinexformer", "stage3-retinexformer"}:
        from stage3_retinexformer import Stage3RetinexFormer

        return Stage3RetinexFormer(
            input_channels=input_channels,
            base_channels=base_channels,
            residual_scale=residual_scale,
        )
    if name in {"maxim", "stage3-maxim"}:
        from stage3_maxim import Stage3MAXIM

        return Stage3MAXIM(
            input_channels=input_channels,
            base_channels=base_channels,
            residual_scale=residual_scale,
        )
    if name in {"adair", "stage3-adair"}:
        from stage3_adair import Stage3AdaIR

        return Stage3AdaIR(
            input_channels=input_channels,
            base_channels=base_channels,
            residual_scale=residual_scale,
        )
    if name in {"bioir", "stage3-bioir"}:
        from stage3_bioir import Stage3BioIR

        return Stage3BioIR(
            input_channels=input_channels,
            base_channels=base_channels,
            residual_scale=residual_scale,
        )
    if name in {"darkir", "stage3-darkir"}:
        from stage3_darkir import Stage3DarkIR

        return Stage3DarkIR(
            input_channels=input_channels,
            base_channels=base_channels,
            residual_scale=residual_scale,
        )
    if name in {"hair", "res-hair", "stage3-hair"}:
        from stage3_hair import Stage3HAIR

        return Stage3HAIR(
            input_channels=input_channels,
            base_channels=base_channels,
            residual_scale=residual_scale,
        )
    if name in {"hogformer", "stage3-hogformer"}:
        from stage3_hogformer import Stage3HOGFormer

        return Stage3HOGFormer(
            input_channels=input_channels,
            base_channels=base_channels,
            residual_scale=residual_scale,
        )
    if name in {"onerestore", "onerestore-l", "stage3-onerestore"}:
        from stage3_onerestore import Stage3OneRestore

        return Stage3OneRestore(
            input_channels=input_channels,
            base_channels=base_channels,
            residual_scale=residual_scale,
        )
    if name in {"anyir", "stage3-anyir"}:
        from stage3_anyir import Stage3AnyIR

        return Stage3AnyIR(
            input_channels=input_channels,
            base_channels=base_channels,
            residual_scale=residual_scale,
        )
    if name in {"xrestormer", "x-restormer", "stage3-xrestormer"}:
        from stage3_xrestormer import Stage3XRestormer

        return Stage3XRestormer(
            input_channels=input_channels,
            base_channels=base_channels,
            residual_scale=residual_scale,
        )
    if name in {"instructir", "stage3-instructir"}:
        from stage3_instructir import Stage3InstructIR

        return Stage3InstructIR(
            input_channels=input_channels,
            base_channels=base_channels,
            residual_scale=residual_scale,
        )
    if name in {"dcpt-promptir", "dcpt", "stage3-dcpt-promptir"}:
        from stage3_dcpt import Stage3DCPTPromptIR

        return Stage3DCPTPromptIR(
            input_channels=input_channels,
            base_channels=base_channels,
            residual_scale=residual_scale,
        )
    if name in {"perceiveir", "perceive-ir", "stage3-perceiveir"}:
        from stage3_perceiveir import Stage3PerceiveIR

        return Stage3PerceiveIR(
            input_channels=input_channels,
            base_channels=base_channels,
            residual_scale=residual_scale,
        )
    raise ValueError(f"Unknown Stage3 backbone: {backbone}")


def resolve_stage3_hparams(
    backbone: str,
    base_channels: int | None = None,
    residual_scale: float | None = None,
) -> tuple[int, float]:
    name = backbone.lower().replace("_", "-")
    if name in {"cidnet", "hvi-cidnet", "stage3-cidnet"}:
        return (
            36 if base_channels is None else base_channels,
            1.0 if residual_scale is None else residual_scale,
        )
    if name in {"saigformer", "stage3-saigformer"}:
        return (
            32 if base_channels is None else base_channels,
            1.0 if residual_scale is None else residual_scale,
        )
    if name in {"mocformer", "stage3-mocformer"}:
        return (
            16 if base_channels is None else base_channels,
            1.0 if residual_scale is None else residual_scale,
        )
    if name in {"mambairv2", "mambairv2-l", "stage3-mambairv2"}:
        return (
            174 if base_channels is None else base_channels,
            1.0 if residual_scale is None else residual_scale,
        )
    if name in {"restormer", "stage3-restormer"}:
        return (
            48 if base_channels is None else base_channels,
            1.0 if residual_scale is None else residual_scale,
        )
    if name in {"promptir", "stage3-promptir"}:
        return (
            48 if base_channels is None else base_channels,
            1.0 if residual_scale is None else residual_scale,
        )
    if name in {"nafnet", "stage3-nafnet"}:
        return (
            64 if base_channels is None else base_channels,
            1.0 if residual_scale is None else residual_scale,
        )
    if name in {"retinexformer", "stage3-retinexformer"}:
        return (
            40 if base_channels is None else base_channels,
            1.0 if residual_scale is None else residual_scale,
        )
    if name in {"maxim", "stage3-maxim"}:
        return (
            32 if base_channels is None else base_channels,
            1.0 if residual_scale is None else residual_scale,
        )
    if name in {"adair", "stage3-adair"}:
        return (
            56 if base_channels is None else base_channels,
            1.0 if residual_scale is None else residual_scale,
        )
    if name in {"bioir", "stage3-bioir"}:
        return (
            80 if base_channels is None else base_channels,
            1.0 if residual_scale is None else residual_scale,
        )
    if name in {"darkir", "stage3-darkir"}:
        return (
            128 if base_channels is None else base_channels,
            1.0 if residual_scale is None else residual_scale,
        )
    if name in {"hair", "res-hair", "stage3-hair"}:
        return (
            64 if base_channels is None else base_channels,
            1.0 if residual_scale is None else residual_scale,
        )
    if name in {"hogformer", "stage3-hogformer"}:
        return (
            54 if base_channels is None else base_channels,
            1.0 if residual_scale is None else residual_scale,
        )
    if name in {"onerestore", "onerestore-l", "stage3-onerestore"}:
        return (
            80 if base_channels is None else base_channels,
            1.0 if residual_scale is None else residual_scale,
        )
    if name in {"anyir", "stage3-anyir"}:
        return (
            68 if base_channels is None else base_channels,
            1.0 if residual_scale is None else residual_scale,
        )
    if name in {"xrestormer", "x-restormer", "stage3-xrestormer"}:
        return (
            48 if base_channels is None else base_channels,
            1.0 if residual_scale is None else residual_scale,
        )
    if name in {"instructir", "stage3-instructir"}:
        return (
            48 if base_channels is None else base_channels,
            1.0 if residual_scale is None else residual_scale,
        )
    if name in {"dcpt-promptir", "dcpt", "stage3-dcpt-promptir"}:
        return (
            48 if base_channels is None else base_channels,
            1.0 if residual_scale is None else residual_scale,
        )
    if name in {"perceiveir", "perceive-ir", "stage3-perceiveir"}:
        return (
            48 if base_channels is None else base_channels,
            1.0 if residual_scale is None else residual_scale,
        )
    if name in {"unet", "residual-unet", "baseline"}:
        return (
            32 if base_channels is None else base_channels,
            0.25 if residual_scale is None else residual_scale,
        )
    raise ValueError(f"Unknown Stage3 backbone: {backbone}")
