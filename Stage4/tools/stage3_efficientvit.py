from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_group_count(channels: int, groups: int = 8) -> int:
    groups = min(groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.GroupNorm(_make_group_count(out_ch), out_ch),
            nn.SiLU(inplace=True) if act else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MBConv(nn.Module):
    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            ConvNormAct(channels, hidden, kernel_size=1),
            ConvNormAct(hidden, hidden, kernel_size=3, groups=hidden),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_make_group_count(channels), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class EfficientAttention(nn.Module):
    """
    EfficientViT-style lightweight attention.

    Attention is computed at the current feature resolution, which is already
    downsampled by the encoder. Keys and values can be additionally pooled to
    keep memory use predictable for Stage3 image sizes.
    """

    def __init__(self, channels: int, heads: int = 4, kv_stride: int = 2) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by heads ({heads})")
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim**-0.5
        self.kv_stride = kv_stride

        self.q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.kv = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_make_group_count(channels), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        kv_source = x
        if self.kv_stride > 1 and min(h, w) >= self.kv_stride:
            kv_source = F.avg_pool2d(x, kernel_size=self.kv_stride, stride=self.kv_stride, ceil_mode=True)

        q = self.q(x).reshape(b, self.heads, self.head_dim, h * w).transpose(-2, -1)
        kv = self.kv(kv_source)
        k, v = kv.chunk(2, dim=1)
        k = k.reshape(b, self.heads, self.head_dim, -1)
        v = v.reshape(b, self.heads, self.head_dim, -1).transpose(-2, -1)

        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(-2, -1).reshape(b, c, h, w)
        return x + self.proj(out)


class EfficientViTBlock(nn.Module):
    def __init__(self, channels: int, heads: int, kv_stride: int = 2, expansion: int = 4) -> None:
        super().__init__()
        self.local = MBConv(channels, expansion=expansion)
        self.attn = EfficientAttention(channels, heads=heads, kv_stride=kv_stride)
        self.ffn = MBConv(channels, expansion=expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local(x)
        x = self.attn(x)
        x = self.ffn(x)
        return x


class DownStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, depth: int, heads: int, use_attention: bool) -> None:
        super().__init__()
        blocks: list[nn.Module] = [ConvNormAct(in_ch, out_ch, stride=2)]
        for _ in range(depth):
            if use_attention:
                blocks.append(EfficientViTBlock(out_ch, heads=heads))
            else:
                blocks.append(MBConv(out_ch))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            ConvNormAct(in_ch + skip_ch, out_ch),
            MBConv(out_ch, expansion=3),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class EfficientViTRelighter(nn.Module):
    """
    EfficientViT-style image-to-image relighting network.

    The module intentionally matches the existing Stage3 model signature:
      x: [B, input_channels, H, W]
      composite: [B, 3, H, W]

    By default it predicts a bounded residual around the input composite, just
    like the current U-Net baseline, so old data and metrics remain comparable.
    Set residual_scale <= 0 to output a directly predicted RGB image instead.
    """

    def __init__(
        self,
        input_channels: int,
        channels: Sequence[int] = (32, 64, 128, 192),
        depths: Sequence[int] = (1, 1, 2, 2),
        heads: Sequence[int] = (2, 4, 4, 6),
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if not (len(channels) == len(depths) == len(heads) == 4):
            raise ValueError("channels, depths, and heads must each contain four entries")

        c1, c2, c3, c4 = channels
        self.residual_scale = residual_scale

        self.stem = nn.Sequential(
            ConvNormAct(input_channels, c1),
            MBConv(c1, expansion=3),
        )
        self.down1 = DownStage(c1, c2, depth=depths[1], heads=heads[1], use_attention=False)
        self.down2 = DownStage(c2, c3, depth=depths[2], heads=heads[2], use_attention=True)
        self.down3 = DownStage(c3, c4, depth=depths[3], heads=heads[3], use_attention=True)

        self.mid = nn.Sequential(
            EfficientViTBlock(c4, heads=heads[3], kv_stride=1),
            MBConv(c4),
        )

        self.up3 = UpStage(c4, c3, c3)
        self.up2 = UpStage(c3, c2, c2)
        self.up1 = UpStage(c2, c1, c1)
        self.out = nn.Conv2d(c1, 3, kernel_size=1)

        if self.residual_scale > 0:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, composite: torch.Tensor) -> torch.Tensor:
        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        z = self.down3(s3)

        z = self.mid(z)
        z = self.up3(z, s3)
        z = self.up2(z, s2)
        z = self.up1(z, s1)

        rgb = self.out(z)
        if self.residual_scale > 0:
            residual = torch.tanh(rgb) * self.residual_scale
            return torch.clamp(composite + residual, 0.0, 1.0)
        return F.relu(rgb)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
