from __future__ import annotations

import numbers
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dBN(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bn_weight_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias=False),
        )
        self.add_module("bn", nn.BatchNorm2d(out_ch))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)


def to_3d(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, c)


def to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    b, _, c = x.shape
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("LayerNorm expects a single channel dimension.")
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("LayerNorm expects a single channel dimension.")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim: int, layer_norm_type: str = "WithBias") -> None:
        super().__init__()
        self.body = BiasFreeLayerNorm(dim) if layer_norm_type == "BiasFree" else WithBiasLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class CromaImplM(nn.Module):
    def __init__(self, channels: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.channels = channels
        self.num_heads = num_heads
        self.q1 = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.q2 = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=bias)
        self.q3 = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=bias)
        self.fac = nn.Parameter(torch.ones(1))
        self.fin = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        head_ch = c // self.num_heads
        x = x.reshape(n, self.num_heads, head_ch, h, w)
        x = x.permute(0, 1, 4, 3, 2).contiguous()
        return x.reshape(n * self.num_heads * w, h, head_ch)

    def _unreshape(self, x: torch.Tensor, n: int, h: int, w: int) -> torch.Tensor:
        head_ch = self.channels // self.num_heads
        x = x.reshape(n, self.num_heads, w, h, head_ch)
        x = x.permute(0, 1, 4, 3, 2).contiguous()
        return x.reshape(n, self.channels, h, w)

    def forward(self, x: torch.Tensor, ill_map: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        n, _, h, w = x.shape
        qkv = self.q3(self.q2(self.q1(x)))
        q, k, v = [self._reshape(t) for t in qkv.chunk(3, dim=1)]
        ill = self._reshape(ill_map)
        v = v * ill
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.fac

        if mask is not None:
            mask_r = self._reshape(mask)
            mask_r = torch.matmul(mask_r, mask_r.transpose(-2, -1)) * self.fac
            attn = attn.masked_fill(mask_r == 0, -1e9)

        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = self._unreshape(out, n, h, w)
        return self.fin(out)


class CromaM(nn.Module):
    def __init__(self, channels: int, num_heads: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.height_att = CromaImplM(channels, num_heads, bias)
        self.width_att = CromaImplM(channels, num_heads, bias)

    def forward(
        self,
        x: torch.Tensor,
        ill_map_height: torch.Tensor,
        ill_map_width: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.height_att(x, ill_map_width, mask=mask)
        x = x.transpose(-2, -1)
        mask_t = mask.transpose(-2, -1) if mask is not None else None
        x = self.width_att(x, ill_map_height.transpose(-2, -1), mask=mask_t)
        return x.transpose(-2, -1)


class FFN(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool) -> None:
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.rep_conv1 = Conv2dBN(hidden, hidden, 3, 1, 1, groups=hidden)
        self.rep_conv2 = Conv2dBN(hidden, hidden, 1, 1, 0, groups=hidden)
        self.project_in = nn.Conv2d(dim, hidden, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, groups=hidden, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.project_in(x)
        x1 = x + self.rep_conv1(x) + self.rep_conv2(x)
        x2 = self.dwconv(x)
        x = F.gelu(x2) * x1 + F.gelu(x1) * x2
        x = self.project_out(x)
        return x + identity


class Croma(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        ffn_expansion_factor: float = 2.66,
        bias: bool = True,
        layer_norm_type: str = "WithBias",
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.attn = CromaM(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.ffn = FFN(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor, ill_h: torch.Tensor, ill_w: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), ill_h, ill_w)
        return x + self.ffn(self.norm2(x))


class Brima(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        ffn_expansion_factor: float = 2.66,
        bias: bool = True,
        layer_norm_type: str = "WithBias",
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.attn = CromaM(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.ffn = FFN(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor, ill_h: torch.Tensor, ill_w: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), ill_h, ill_w, mask=mask)
        return x + self.ffn(self.norm2(x))


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class ChannelConcat(nn.Module):
    def __init__(self, in_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.channel_in = in_dim
        self.temperature = nn.Parameter(torch.ones(1))
        self.qkv = nn.Conv2d(in_dim, in_dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(in_dim * 3, in_dim * 3, kernel_size=3, stride=1, padding=1, groups=in_dim * 3, bias=bias)
        self.project_out = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_feats, channels, height, width = x.shape
        x_input = x.reshape(batch, n_feats * channels, height, width)
        q, k, v = self.qkv_dwconv(self.qkv(x_input)).chunk(3, dim=1)
        q = q.reshape(batch, n_feats, -1)
        k = k.reshape(batch, n_feats, -1)
        v = v.reshape(batch, n_feats, -1)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).reshape(batch, -1, height, width)
        out = self.project_out(out)
        out = out.reshape(batch, n_feats, channels, height, width)
        return (out + x).reshape(batch, -1, height, width)


class IlluminationEstimator(nn.Module):
    def __init__(self, n_fea_middle: int, n_fea_in: int = 4, n_fea_out: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_c = img.mean(dim=1, keepdim=True)
        x = torch.cat([img, mean_c], dim=1)
        x = self.conv1(x)
        illu_fea = self.depth_conv(x)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


class IlluminationEstimatorHeight(nn.Module):
    def __init__(self, n_fea_middle: int, n_fea_in: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(n_fea_in + 1, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=(3, 1), padding=(1, 0), stride=1, bias=True)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        mean_c = img.mean(dim=1, keepdim=True)
        x = torch.cat([img, mean_c], dim=1)
        return self.depth_conv(self.conv1(x))


class IlluminationEstimatorWidth(nn.Module):
    def __init__(self, n_fea_middle: int, n_fea_in: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(n_fea_in + 1, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=(1, 3), padding=(0, 1), stride=1, bias=True)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        mean_c = img.mean(dim=1, keepdim=True)
        x = torch.cat([img, mean_c], dim=1)
        return self.depth_conv(self.conv1(x))


def _run_croma(blocks: nn.ModuleList, x: torch.Tensor, ill_h: torch.Tensor, ill_w: torch.Tensor) -> torch.Tensor:
    for block in blocks:
        x = block(x, ill_h, ill_w)
    return x


def _run_brima(blocks: nn.ModuleList, x: torch.Tensor, ill_h: torch.Tensor, ill_w: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    for block in blocks:
        x = block(x, ill_h, ill_w, mask)
    return x


class Stage3MOCFormer(nn.Module):
    """
    Official MOCFormer backbone adapted to Stage3 relighting.

    Restoration features consume the Stage3 12-channel tensor. Illumination,
    cross-axis Croma guidance, and bright-filter Brima masks are derived from
    the composite RGB image, matching the official low-light model's intent.
    """

    def __init__(
        self,
        input_channels: int = 12,
        base_channels: int = 16,
        residual_scale: float = 1.0,
        num_blocks: Sequence[int] = (2, 2, 4, 4),
        num_refinement_blocks: int = 3,
        heads: Sequence[int] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_type: str = "WithBias",
        attention: bool = True,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        dim = base_channels
        self.residual_scale = residual_scale
        self.estimator = IlluminationEstimator(dim)
        self.ill_height = IlluminationEstimatorHeight(dim, n_fea_in=3)
        self.ill_width = IlluminationEstimatorWidth(dim, n_fea_in=3)

        self.patch_embed = OverlapPatchEmbed(input_channels, dim)
        self.patch_embed_mask = OverlapPatchEmbed(1, dim)

        self.encoder_1 = nn.ModuleList([Croma(dim, heads[0], ffn_expansion_factor, bias, layer_norm_type) for _ in range(num_blocks[0])])
        self.encoder_2 = nn.ModuleList([Croma(dim, heads[0], ffn_expansion_factor, bias, layer_norm_type) for _ in range(num_blocks[0])])
        self.encoder_3 = nn.ModuleList([Croma(dim, heads[0], ffn_expansion_factor, bias, layer_norm_type) for _ in range(num_blocks[0])])
        self.layer_fusion = ChannelConcat(in_dim=dim * 3)
        self.conv_fusion = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=bias)

        self.down_1 = Downsample(dim)
        self.decoder_level1_0 = nn.ModuleList([Croma(dim * 2, heads[1], ffn_expansion_factor, bias, layer_norm_type) for _ in range(num_blocks[0])])
        self.down_2 = Downsample(dim * 2)
        self.decoder_level2_0 = nn.ModuleList([Croma(dim * 4, heads[2], ffn_expansion_factor, bias, layer_norm_type) for _ in range(num_blocks[1])])
        self.down_3 = Downsample(dim * 4)
        self.decoder_level3_0 = nn.ModuleList([Croma(dim * 8, heads[3], ffn_expansion_factor, bias, layer_norm_type) for _ in range(num_blocks[2])])

        self.up3_2 = Upsample(dim * 8)
        self.decoder_level2_1 = nn.ModuleList([Croma(dim * 4, heads[2], ffn_expansion_factor, bias, layer_norm_type) for _ in range(num_blocks[1])])
        self.up2_1 = Upsample(dim * 4)
        self.decoder_level1_1 = nn.ModuleList([Croma(dim * 2, heads[1], ffn_expansion_factor, bias, layer_norm_type) for _ in range(num_blocks[0])])
        self.up2_0 = Upsample(dim * 2)

        self.coefficient_3_2 = nn.Parameter(torch.ones(2, dim * 4), requires_grad=attention)
        self.coefficient_2_1 = nn.Parameter(torch.ones(2, dim * 2), requires_grad=attention)
        self.coefficient_1_0 = nn.Parameter(torch.ones(2, dim), requires_grad=attention)
        self.skip_3_2 = nn.Conv2d(dim * 4, dim * 4, kernel_size=1, bias=bias)
        self.skip_1_0 = nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=bias)

        self.latent = nn.ModuleList([Croma(dim, heads[0], ffn_expansion_factor, bias, layer_norm_type) for _ in range(num_blocks[0])])
        self.refinement_1 = nn.ModuleList([Brima(dim, heads[0], ffn_expansion_factor, bias, layer_norm_type) for _ in range(num_refinement_blocks)])
        self.refinement_2 = nn.ModuleList([Brima(dim, heads[0], ffn_expansion_factor, bias, layer_norm_type) for _ in range(num_refinement_blocks)])
        self.refinement_3 = nn.ModuleList([Brima(dim, heads[0], ffn_expansion_factor, bias, layer_norm_type) for _ in range(num_refinement_blocks)])
        self.layer_fusion_2 = ChannelConcat(in_dim=dim * 3)
        self.conv_fusion_2 = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=bias)
        self.output = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1, bias=bias)

        if identity_init:
            nn.init.zeros_(self.output.weight)
            if self.output.bias is not None:
                nn.init.zeros_(self.output.bias)

    def _make_stage_input(self, x: torch.Tensor, composite_enhanced: torch.Tensor) -> torch.Tensor:
        if x.shape[1] >= 3:
            return torch.cat([composite_enhanced, x[:, 3:]], dim=1)
        return x

    def _make_bright_mask(self, composite: torch.Tensor) -> torch.Tensor:
        luminance = composite.mean(dim=1, keepdim=True).detach()
        flat = luminance.flatten(1)
        threshold = flat.mean(dim=1, keepdim=True) + (flat.max(dim=1, keepdim=True).values - flat.min(dim=1, keepdim=True).values) / 5.0
        threshold = threshold.view(-1, 1, 1, 1)
        mask = luminance.clone()
        mask = torch.where(mask > threshold, torch.zeros_like(mask), mask)
        return mask

    def forward(self, x: torch.Tensor, composite: torch.Tensor) -> torch.Tensor:
        _, illu_map = self.estimator(composite)
        composite_enhanced = composite * illu_map + composite
        stage_x = self._make_stage_input(x, composite_enhanced)

        ill_h0 = self.ill_height(composite_enhanced)
        ill_w0 = self.ill_width(composite_enhanced)
        mask0 = self.patch_embed_mask(self._make_bright_mask(composite_enhanced))

        enc1 = self.patch_embed(stage_x)
        enc1 = _run_croma(self.encoder_1, enc1, ill_h0, ill_w0)
        enc2 = _run_croma(self.encoder_2, enc1, ill_h0, ill_w0)
        enc3 = _run_croma(self.encoder_3, enc2, ill_h0, ill_w0)

        fusion = torch.cat([enc1.unsqueeze(1), enc2.unsqueeze(1), enc3.unsqueeze(1)], dim=1)
        fusion = self.layer_fusion(fusion)
        fusion = self.conv_fusion(fusion)

        level1_0 = self.down_1(fusion)
        ill_h1 = self.down_1(ill_h0)
        ill_w1 = self.down_1(ill_w0)
        level1_0 = _run_croma(self.decoder_level1_0, level1_0, ill_h1, ill_w1)

        level2_0 = self.down_2(level1_0)
        ill_h2 = self.down_2(ill_h1)
        ill_w2 = self.down_2(ill_w1)
        level2_0 = _run_croma(self.decoder_level2_0, level2_0, ill_h2, ill_w2)

        level3_0 = self.down_3(level2_0)
        ill_h3 = self.down_3(ill_h2)
        ill_w3 = self.down_3(ill_w2)
        level3_0 = _run_croma(self.decoder_level3_0, level3_0, ill_h3, ill_w3)

        level3_up = self.up3_2(level3_0)
        level2_1 = self.coefficient_3_2[0][None, :, None, None] * level2_0 + self.coefficient_3_2[1][None, :, None, None] * level3_up
        level2_1 = self.skip_3_2(level2_1)
        ill_h2_up = self.up3_2(ill_h3)
        ill_w2_up = self.up3_2(ill_w3)
        level2_1 = _run_croma(self.decoder_level2_1, level2_1, ill_h2_up, ill_w2_up)

        level2_up = self.up2_1(level2_1)
        level1_1 = self.coefficient_2_1[0][None, :, None, None] * level1_0 + self.coefficient_2_1[1][None, :, None, None] * level2_up
        level1_1 = self.skip_1_0(level1_1)
        ill_h1_up = self.up2_1(ill_h2_up)
        ill_w1_up = self.up2_1(ill_w2_up)
        level1_1 = _run_croma(self.decoder_level1_1, level1_1, ill_h1_up, ill_w1_up)
        level0_1 = self.up2_0(level1_1)

        fusion_latent = _run_croma(self.latent, fusion, ill_h0, ill_w0)
        out = self.coefficient_1_0[0][None, :, None, None] * fusion_latent + self.coefficient_1_0[1][None, :, None, None] * level0_1

        out1 = _run_brima(self.refinement_1, out, ill_h0, ill_w0, mask0)
        out2 = _run_brima(self.refinement_2, out1, ill_h0, ill_w0, mask0)
        out3 = _run_brima(self.refinement_3, out2, ill_h0, ill_w0, mask0)
        out_fusion = torch.cat([out1.unsqueeze(1), out2.unsqueeze(1), out3.unsqueeze(1)], dim=1)
        out = self.layer_fusion_2(out_fusion)
        out = self.conv_fusion_2(out)

        residual = self.output(out) * self.residual_scale
        return torch.clamp(composite + residual, 0.0, 1.0)
