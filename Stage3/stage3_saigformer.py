from __future__ import annotations

import numbers
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_3d(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, c)


def to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    b, _, c = x.shape
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("WithBiasLayerNorm expects a single channel dimension.")

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.body = WithBiasLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class SAI2E(nn.Module):
    def __init__(self, in_channels: int = 3, train_patch: int | Sequence[int] = 128, eps: float = 1e-1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.eps = eps
        self.train_patch = list(train_patch) if isinstance(train_patch, Sequence) else [train_patch, train_patch]
        self.offset_predict = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels, 4, 1, padding=0, bias=True),
        )
        self.modulation_predict = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1, padding=0, bias=True),
        )

    def get_center_grid(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        coords_h = torch.arange(h, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(w, device=x.device, dtype=x.dtype) + 0.5
        coords = torch.stack(torch.meshgrid(coords_w, coords_h, indexing="xy"), dim=-1)
        normalizer = torch.tensor([w, h], dtype=x.dtype, device=x.device)
        return coords / normalizer * 2.0 - 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, h, w = x.shape
        integrated_x = torch.cumsum(x, dim=-1)
        integrated_x = torch.cumsum(integrated_x, dim=-2)

        center_grid = self.get_center_grid(x).unsqueeze(0)
        normalizer = torch.tensor(
            [self.train_patch[0] / w, self.train_patch[1] / h],
            dtype=x.dtype,
            device=x.device,
        ).view(1, 1, 1, 2)

        subnet_output = self.offset_predict(x).permute(0, 2, 3, 1)
        off_w, off_h = torch.split(subnet_output, 2, dim=3)
        off_w = off_w - off_w.mean(dim=-1, keepdim=True)
        off_h = off_h - off_h.mean(dim=-1, keepdim=True)

        minimum_patch = 2
        off_w_min = torch.minimum(
            off_w.min(dim=-1, keepdim=True)[0],
            torch.zeros_like(off_w.min(dim=-1, keepdim=True)[0]) - minimum_patch / self.train_patch[0],
        )
        off_w_max = torch.maximum(
            off_w.max(dim=-1, keepdim=True)[0],
            torch.zeros_like(off_w.max(dim=-1, keepdim=True)[0]) + minimum_patch / self.train_patch[0],
        )
        off_h_min = torch.minimum(
            off_h.min(dim=-1, keepdim=True)[0],
            torch.zeros_like(off_h.min(dim=-1, keepdim=True)[0]) - minimum_patch / self.train_patch[1],
        )
        off_h_max = torch.maximum(
            off_h.max(dim=-1, keepdim=True)[0],
            torch.zeros_like(off_h.max(dim=-1, keepdim=True)[0]) + minimum_patch / self.train_patch[1],
        )

        area = (off_h_max - off_h_min) * (off_w_max - off_w_min) * self.train_patch[0] * self.train_patch[1] / 4.0
        area = area.view(batch, 1, h, w).clip(1, h * w)
        scale = self.modulation_predict(x)
        if self.eps != 0:
            mask = scale.abs() < self.eps
            safe_sign = torch.where(scale >= 0, 1.0, -1.0)
            scale = torch.where(mask, safe_sign * self.eps, scale)
        area = area * scale

        off_tl = (torch.cat([off_w_min, off_h_min], dim=-1) * normalizer + center_grid).clip(-1, 1)
        off_tr = (torch.cat([off_w_max, off_h_min], dim=-1) * normalizer + center_grid).clip(-1, 1)
        off_bl = (torch.cat([off_w_min, off_h_max], dim=-1) * normalizer + center_grid).clip(-1, 1)
        off_br = (torch.cat([off_w_max, off_h_max], dim=-1) * normalizer + center_grid).clip(-1, 1)

        a = F.grid_sample(integrated_x, off_tl, align_corners=True, padding_mode="border", mode="bilinear")
        b = F.grid_sample(integrated_x, off_tr, align_corners=True, padding_mode="border", mode="bilinear")
        c = F.grid_sample(integrated_x, off_bl, align_corners=True, padding_mode="border", mode="bilinear")
        d = F.grid_sample(integrated_x, off_br, align_corners=True, padding_mode="border", mode="bilinear")
        return (a + d - b - c) / area


class DualGatedFeedForward(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool) -> None:
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.proj_1 = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.proj_2 = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj_1 = self.proj_1(x)
        proj_2 = self.proj_2(x)
        gated_1 = proj_1 * torch.sigmoid(proj_2)
        gated_2 = proj_2 * F.gelu(proj_1)
        return self.project_out(gated_1 + gated_2)


class IlluminationGuideAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.svp_q_dwconv = nn.Conv2d(3, 3 * num_heads, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim + 3 * num_heads, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor, svp_fea: torch.Tensor) -> torch.Tensor:
        batch, channels, h, w = x.shape
        head_ch = channels // self.num_heads
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        svp_q = self.svp_q_dwconv(svp_fea)

        q = q.reshape(batch, self.num_heads, head_ch, h * w)
        svp_q = svp_q.reshape(batch, self.num_heads, 3, h * w)
        k = k.reshape(batch, self.num_heads, head_ch, h * w)
        v = v.reshape(batch, self.num_heads, head_ch, h * w)

        q = F.normalize(q, dim=-1)
        svp_q = F.normalize(svp_q, dim=-1)
        k = F.normalize(k, dim=-1)

        q = torch.cat([q, svp_q], dim=2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.reshape(batch, channels + 3 * self.num_heads, h, w)
        return self.project_out(out)


class SAIGTransformer(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_expansion_factor: float, bias: bool) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = IlluminationGuideAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = DualGatedFeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor, svp_fea: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), svp_fea)
        x = x + self.ffn(self.norm2(x))
        return x


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


class Stage3SAIGFormer(nn.Module):
    """
    Official SAIGFormer backbone adapted to Stage3.

    Differences from the official low-light model:
      * the restoration branch accepts Stage3's multi-buffer tensor;
      * SAI2E still follows the official 3-channel illumination path on the
        composite image;
      * the output residual is added back to the composite image.
    """

    def __init__(
        self,
        input_channels: int = 12,
        base_channels: int = 32,
        residual_scale: float = 1.0,
        encoder_num_blocks: Sequence[int] = (4, 6, 6, 8),
        decoder_num_blocks: Sequence[int] = (6, 6, 4, 4),
        heads: Sequence[int] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        train_patch: int = 128,
        eps: float = 1e-1,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        embed_dim = base_channels
        bias = False
        self.residual_scale = residual_scale

        self.patch_embed = OverlapPatchEmbed(input_channels, embed_dim, bias=bias)
        self.svp = SAI2E(in_channels=3, train_patch=train_patch, eps=eps)

        self.encoder_level1 = nn.ModuleList([
            SAIGTransformer(embed_dim, heads[0], ffn_expansion_factor, bias) for _ in range(encoder_num_blocks[0])
        ])
        self.svp_down1_2 = nn.Conv2d(3, 3, 4, 2, 1, bias=bias, groups=3)
        self.down1_2 = Downsample(embed_dim)
        self.encoder_level2 = nn.ModuleList([
            SAIGTransformer(embed_dim * 2, heads[1], ffn_expansion_factor, bias) for _ in range(encoder_num_blocks[1])
        ])

        self.svp_down2_3 = nn.Conv2d(3, 3, 4, 2, 1, bias=bias, groups=3)
        self.down2_3 = Downsample(embed_dim * 2)
        self.encoder_level3 = nn.ModuleList([
            SAIGTransformer(embed_dim * 4, heads[2], ffn_expansion_factor, bias) for _ in range(encoder_num_blocks[2])
        ])

        self.svp_down3_4 = nn.Conv2d(3, 3, 4, 2, 1, bias=bias, groups=3)
        self.down3_4 = Downsample(embed_dim * 4)
        self.latent = nn.ModuleList([
            SAIGTransformer(embed_dim * 8, heads[3], ffn_expansion_factor, bias) for _ in range(encoder_num_blocks[3])
        ])

        self.decoder_latent = nn.ModuleList([
            SAIGTransformer(embed_dim * 8, heads[3], ffn_expansion_factor, bias)
        ])
        self.up4_3 = Upsample(embed_dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(embed_dim * 8, embed_dim * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = nn.ModuleList([
            SAIGTransformer(embed_dim * 4, heads[2], ffn_expansion_factor, bias) for _ in range(decoder_num_blocks[0])
        ])

        self.up3_2 = Upsample(embed_dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(embed_dim * 4, embed_dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList([
            SAIGTransformer(embed_dim * 2, heads[1], ffn_expansion_factor, bias) for _ in range(decoder_num_blocks[1])
        ])

        self.up2_1 = Upsample(embed_dim * 2)
        self.decoder_level1 = nn.ModuleList([
            SAIGTransformer(embed_dim * 2, heads[0], ffn_expansion_factor, bias) for _ in range(decoder_num_blocks[2])
        ])
        self.refinement = nn.ModuleList([
            SAIGTransformer(embed_dim * 2, heads[0], ffn_expansion_factor, bias) for _ in range(decoder_num_blocks[-1])
        ])

        self.output = nn.Conv2d(embed_dim * 2, 3, kernel_size=3, stride=1, padding=1, bias=bias)
        if identity_init:
            nn.init.zeros_(self.output.weight)
            if self.output.bias is not None:
                nn.init.zeros_(self.output.bias)

    def _run_blocks(self, x: torch.Tensor, svp: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        for block in blocks:
            x = block(x, svp)
        return x

    def forward(self, x: torch.Tensor, composite: torch.Tensor) -> torch.Tensor:
        svp_img_1 = self.svp(composite)

        inp_enc_level1 = self.patch_embed(x)
        inp_enc_level1 = self._run_blocks(inp_enc_level1, svp_img_1, self.encoder_level1)
        out_enc_level1 = inp_enc_level1

        inp_enc_level2 = self.down1_2(out_enc_level1)
        svp_img_2 = self.svp_down1_2(svp_img_1)
        inp_enc_level2 = self._run_blocks(inp_enc_level2, svp_img_2, self.encoder_level2)
        out_enc_level2 = inp_enc_level2

        inp_enc_level3 = self.down2_3(out_enc_level2)
        svp_img_3 = self.svp_down2_3(svp_img_2)
        inp_enc_level3 = self._run_blocks(inp_enc_level3, svp_img_3, self.encoder_level3)
        out_enc_level3 = inp_enc_level3

        latent = self.down3_4(out_enc_level3)
        svp_img_4 = self.svp_down3_4(svp_img_3)
        latent = self._run_blocks(latent, svp_img_4, self.latent)
        latent = self._run_blocks(latent, svp_img_4, self.decoder_latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3 = self._run_blocks(inp_dec_level3, svp_img_3, self.decoder_level3)

        inp_dec_level2 = self.up3_2(inp_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2 = self._run_blocks(inp_dec_level2, svp_img_2, self.decoder_level2)

        inp_dec_level1 = self.up2_1(inp_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        inp_dec_level1 = self._run_blocks(inp_dec_level1, svp_img_1, self.decoder_level1)
        out_dec_level1 = self._run_blocks(inp_dec_level1, svp_img_1, self.refinement)

        residual = self.output(out_dec_level1) * self.residual_scale
        return torch.clamp(composite + residual, 0.0, 1.0)
