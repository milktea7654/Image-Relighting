from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_norm: bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.act = nn.PReLU()
        self.norm = LayerNorm2d(out_ch) if use_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = max(1, x.shape[-2] // 2)
        w = max(1, x.shape[-1] // 2)
        x = self.conv(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        x = self.act(x)
        return self.norm(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_norm: bool = False) -> None:
        super().__init__()
        self.up_conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.fuse = nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=1, bias=False)
        self.act = nn.PReLU()
        self.norm = LayerNorm2d(out_ch) if use_norm else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(x)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.act(x)
        return self.norm(x)


class ChannelCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool = False) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=bias)
        self.out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        head_ch = c // self.num_heads

        q = self.q_dw(self.q(x))
        k, v = self.kv_dw(self.kv(context)).chunk(2, dim=1)

        q = q.reshape(b, self.num_heads, head_ch, h * w)
        k = k.reshape(b, self.num_heads, head_ch, h * w)
        v = v.reshape(b, self.num_heads, head_ch, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.reshape(b, c, h, w)
        return self.out(out)


class IntensityEnhancementLayer(nn.Module):
    def __init__(self, dim: int, expansion: float = 2.66, bias: bool = False) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1, groups=hidden * 2, bias=bias)
        self.dw1 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=bias)
        self.dw2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dw(x).chunk(2, dim=1)
        x1 = torch.tanh(self.dw1(x1)) + x1
        x2 = torch.tanh(self.dw2(x2)) + x2
        return self.project_out(x1 * x2)


class HVLightenCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool = False) -> None:
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.attn = ChannelCrossAttention(dim, num_heads, bias=bias)
        self.ffn = IntensityEnhancementLayer(dim, bias=bias)

    def forward(self, x: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm(x), self.norm(intensity))
        return self.ffn(self.norm(x))


class ILightenCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool = False) -> None:
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.attn = ChannelCrossAttention(dim, num_heads, bias=bias)
        self.ffn = IntensityEnhancementLayer(dim, bias=bias)

    def forward(self, x: torch.Tensor, hv: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm(x), self.norm(hv))
        return x + self.ffn(self.norm(x))


class RGBHVITransform(nn.Module):
    def __init__(self, density_k: float = 0.2) -> None:
        super().__init__()
        self.density_k = nn.Parameter(torch.tensor([density_k], dtype=torch.float32))
        self._last_inverse_k: torch.Tensor | None = None

    def _k(self, x: torch.Tensor) -> torch.Tensor:
        return self.density_k.to(device=x.device, dtype=x.dtype).view(1, 1, 1, 1)

    def rgb_to_hvi(self, img: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        img = img.clamp(0.0, 1.0)
        r, g, b = img[:, 0], img[:, 1], img[:, 2]

        value, max_idx = img.max(dim=1)
        img_min = img.min(dim=1).values
        delta = value - img_min

        hue_r = ((g - b) / (delta + eps)) % 6.0
        hue_g = ((b - r) / (delta + eps)) + 2.0
        hue_b = ((r - g) / (delta + eps)) + 4.0
        hue = torch.where(max_idx == 0, hue_r, torch.where(max_idx == 1, hue_g, hue_b))
        hue = torch.where(delta > eps, hue / 6.0, torch.zeros_like(hue))

        saturation = torch.where(value > eps, delta / (value + eps), torch.zeros_like(value))
        k = self._k(img)
        self._last_inverse_k = k.detach()
        color_sensitive = ((value[:, None] * 0.5 * math.pi).sin() + eps).pow(k)

        hue = hue[:, None]
        saturation = saturation[:, None]
        h = color_sensitive * saturation * torch.cos(2.0 * math.pi * hue)
        v = color_sensitive * saturation * torch.sin(2.0 * math.pi * hue)
        i = value[:, None]
        return torch.cat([h, v, i], dim=1)

    def hvi_to_rgb(self, hvi: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        h_coord = hvi[:, 0:1].clamp(-1.0, 1.0)
        v_coord = hvi[:, 1:2].clamp(-1.0, 1.0)
        intensity = hvi[:, 2:3].clamp(0.0, 1.0)

        k = self._last_inverse_k
        if k is None or k.device != hvi.device or k.dtype != hvi.dtype:
            k = self._k(hvi).detach()
        color_sensitive = ((intensity * 0.5 * math.pi).sin() + eps).pow(k)
        h_norm = (h_coord / (color_sensitive + eps)).clamp(-1.0, 1.0)
        v_norm = (v_coord / (color_sensitive + eps)).clamp(-1.0, 1.0)

        hue = torch.atan2(v_norm + eps, h_norm + eps) / (2.0 * math.pi)
        hue = hue % 1.0
        saturation = torch.sqrt(h_norm.pow(2) + v_norm.pow(2) + eps).clamp(0.0, 1.0)
        value = intensity

        h6 = hue * 6.0
        hi = torch.floor(h6).to(torch.int64) % 6
        f = h6 - hi.to(dtype=h6.dtype)
        p = value * (1.0 - saturation)
        q = value * (1.0 - f * saturation)
        t = value * (1.0 - (1.0 - f) * saturation)

        r = torch.where(
            (hi == 0) | (hi == 5),
            value,
            torch.where((hi == 1), q, torch.where((hi == 4), t, p)),
        )
        g = torch.where(
            (hi == 1) | (hi == 2),
            value,
            torch.where((hi == 0), t, torch.where((hi == 3), q, p)),
        )
        b = torch.where(
            (hi == 3) | (hi == 4),
            value,
            torch.where((hi == 2), t, torch.where((hi == 5), q, p)),
        )
        return torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)


def _stem(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ReplicationPad2d(1),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, bias=False),
        nn.PReLU(),
    )


def _pick_heads(dim: int, preferred: int) -> int:
    for heads in range(preferred, 0, -1):
        if dim % heads == 0:
            return heads
    return 1


class Stage3CIDNet(nn.Module):
    """
    HVI-CIDNet-inspired backbone for Stage 3 relighting.

    The composite RGB image is transformed into HVI space and processed by
    separate HV and intensity branches. The full Stage 3 input tensor is encoded
    as conditioning at each scale, so render/material buffers can guide the HVI
    residual without changing the training loop signature.
    """

    def __init__(
        self,
        input_channels: int = 12,
        base_channels: int = 36,
        residual_scale: float = 1.0,
        norm: bool = False,
        channels: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels or (
            base_channels,
            base_channels,
            base_channels * 2,
            base_channels * 4,
        )
        h2 = _pick_heads(c2, 2)
        h3 = _pick_heads(c3, 4)
        h4 = _pick_heads(c4, 8)

        self.residual_scale = residual_scale
        self.hvi = RGBHVITransform()

        self.cond0 = _stem(input_channels, c1)
        self.cond1 = DownsampleBlock(c1, c2, use_norm=norm)
        self.cond2 = DownsampleBlock(c2, c3, use_norm=norm)
        self.cond3 = DownsampleBlock(c3, c4, use_norm=norm)

        self.hv_cond = nn.ModuleList([nn.Conv2d(c, c, kernel_size=1, bias=False) for c in (c1, c2, c3, c4)])
        self.i_cond = nn.ModuleList([nn.Conv2d(c, c, kernel_size=1, bias=False) for c in (c1, c2, c3, c4)])
        for proj in [*self.hv_cond, *self.i_cond]:
            nn.init.zeros_(proj.weight)

        self.hv_enc0 = _stem(3, c1)
        self.hv_down1 = DownsampleBlock(c1, c2, use_norm=norm)
        self.hv_down2 = DownsampleBlock(c2, c3, use_norm=norm)
        self.hv_down3 = DownsampleBlock(c3, c4, use_norm=norm)
        self.hv_up3 = UpsampleBlock(c4, c3, c3, use_norm=norm)
        self.hv_up2 = UpsampleBlock(c3, c2, c2, use_norm=norm)
        self.hv_up1 = UpsampleBlock(c2, c1, c1, use_norm=norm)
        self.hv_out = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(c1, 2, kernel_size=3, bias=False))

        self.i_enc0 = _stem(1, c1)
        self.i_down1 = DownsampleBlock(c1, c2, use_norm=norm)
        self.i_down2 = DownsampleBlock(c2, c3, use_norm=norm)
        self.i_down3 = DownsampleBlock(c3, c4, use_norm=norm)
        self.i_up3 = UpsampleBlock(c4, c3, c3, use_norm=norm)
        self.i_up2 = UpsampleBlock(c3, c2, c2, use_norm=norm)
        self.i_up1 = UpsampleBlock(c2, c1, c1, use_norm=norm)
        self.i_out = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(c1, 1, kernel_size=3, bias=False))

        self.hv_lca1 = HVLightenCrossAttention(c2, h2)
        self.hv_lca2 = HVLightenCrossAttention(c3, h3)
        self.hv_lca3 = HVLightenCrossAttention(c4, h4)
        self.hv_lca4 = HVLightenCrossAttention(c4, h4)
        self.hv_lca5 = HVLightenCrossAttention(c3, h3)
        self.hv_lca6 = HVLightenCrossAttention(c2, h2)

        self.i_lca1 = ILightenCrossAttention(c2, h2)
        self.i_lca2 = ILightenCrossAttention(c3, h3)
        self.i_lca3 = ILightenCrossAttention(c4, h4)
        self.i_lca4 = ILightenCrossAttention(c4, h4)
        self.i_lca5 = ILightenCrossAttention(c3, h3)
        self.i_lca6 = ILightenCrossAttention(c2, h2)

        nn.init.zeros_(self.hv_out[-1].weight)
        nn.init.zeros_(self.i_out[-1].weight)

    def _fuse_condition(
        self,
        hv: torch.Tensor,
        intensity: torch.Tensor,
        cond: torch.Tensor,
        level: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cond.shape[-2:] != hv.shape[-2:]:
            cond = F.interpolate(cond, size=hv.shape[-2:], mode="bilinear", align_corners=False)
        return hv + self.hv_cond[level](cond), intensity + self.i_cond[level](cond)

    def forward(self, x: torch.Tensor, composite: torch.Tensor) -> torch.Tensor:
        hvi = self.hvi.rgb_to_hvi(composite)
        intensity = hvi[:, 2:3]

        cond0 = self.cond0(x)
        cond1 = self.cond1(cond0)
        cond2 = self.cond2(cond1)
        cond3 = self.cond3(cond2)

        i0 = self.i_enc0(intensity)
        hv0 = self.hv_enc0(hvi)
        hv0, i0 = self._fuse_condition(hv0, i0, cond0, level=0)
        hv_skip0, i_skip0 = hv0, i0

        i1 = self.i_down1(i0)
        hv1 = self.hv_down1(hv0)
        hv1, i1 = self._fuse_condition(hv1, i1, cond1, level=1)

        i2 = self.i_lca1(i1, hv1)
        hv2 = self.hv_lca1(hv1, i1)
        i_skip1, hv_skip1 = i2, hv2

        i2_down = self.i_down2(i2)
        hv2_down = self.hv_down2(hv2)
        hv2_down, i2_down = self._fuse_condition(hv2_down, i2_down, cond2, level=2)

        i3 = self.i_lca2(i2_down, hv2_down)
        hv3 = self.hv_lca2(hv2_down, i2_down)
        i_skip2, hv_skip2 = i3, hv3

        # Keep the deeper path identical to the official CIDNet graph: the
        # third downsample consumes the pre-LCA2 tensors, while skip2 stores
        # the post-LCA2 tensors.
        i3_down = self.i_down3(i2_down)
        hv3_down = self.hv_down3(hv2_down)
        hv3_down, i3_down = self._fuse_condition(hv3_down, i3_down, cond3, level=3)

        i4 = self.i_lca3(i3_down, hv3_down)
        hv4 = self.hv_lca3(hv3_down, i3_down)

        i4_in, hv4_in = i4, hv4
        i4 = self.i_lca4(i4_in, hv4_in)
        hv4 = self.hv_lca4(hv4_in, i4_in)

        hv2 = self.hv_up3(hv4, hv_skip2)
        i2 = self.i_up3(i4, i_skip2)
        _ = self.i_lca5(i2, hv2)
        hv2 = self.hv_lca5(hv2, i2)

        hv1 = self.hv_up2(hv2, hv_skip1)
        i1 = self.i_up2(i2, i_skip1)
        i1_in, hv1_in = i1, hv1
        i1 = self.i_lca6(i1_in, hv1_in)
        hv1 = self.hv_lca6(hv1_in, i1_in)

        hv0 = self.hv_up1(hv1, hv_skip0)
        i0 = self.i_up1(i1, i_skip0)

        residual_hv = self.hv_out(hv0)
        residual_i = self.i_out(i0)
        hvi_residual = torch.cat([residual_hv, residual_i], dim=1) * self.residual_scale
        pred_hvi = hvi + hvi_residual
        return self.hvi.hvi_to_rgb(pred_hvi)
