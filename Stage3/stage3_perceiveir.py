from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ClipStub(nn.Module):
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        return torch.zeros(text.shape[0], 512, device=text.device, dtype=torch.float32)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        pooled = image.mean(dim=(-2, -1))
        if pooled.shape[1] >= 512:
            return pooled[:, :512]
        pad = torch.zeros(pooled.shape[0], 512 - pooled.shape[1], device=pooled.device, dtype=pooled.dtype)
        return torch.cat([pooled, pad], dim=1)


def _install_clip_stub() -> None:
    if importlib.util.find_spec("clip") is None and "clip" not in sys.modules:
        clip_module = types.ModuleType("clip")
        clip_module.clip = clip_module

        def _missing_load(*_args, **_kwargs):
            raise ImportError("clip is only needed for Perceive-IR's original CLIP inference path.")

        clip_module.load = _missing_load
        sys.modules["clip"] = clip_module


def _load_perceive_module():
    root = Path(__file__).resolve().parent / "external" / "Perceive-IR"
    src = root / "src"
    arch_file = src / "basicsr" / "archs" / "stage1_arch.py"
    if not arch_file.exists():
        raise FileNotFoundError(
            f"Perceive-IR arch was not found at {arch_file}. "
            "Clone https://github.com/House-yuyu/Perceive-IR into Stage3/external/Perceive-IR first."
        )

    _install_clip_stub()
    restormer_file = src / "basicsr" / "archs" / "restormer_arch.py"
    basicsr = types.ModuleType("basicsr")
    basicsr_archs = types.ModuleType("basicsr.archs")
    basicsr.__path__ = [str(src / "basicsr")]
    basicsr_archs.__path__ = [str(src / "basicsr" / "archs")]
    sys.modules["basicsr"] = basicsr
    sys.modules["basicsr.archs"] = basicsr_archs
    restormer_spec = importlib.util.spec_from_file_location("basicsr.archs.restormer_arch", restormer_file)
    if restormer_spec is None or restormer_spec.loader is None:
        raise ImportError(f"Could not load Perceive-IR Restormer arch from {restormer_file}")
    restormer_module = importlib.util.module_from_spec(restormer_spec)
    sys.modules["basicsr.archs.restormer_arch"] = restormer_module
    restormer_spec.loader.exec_module(restormer_module)
    basicsr.archs = basicsr_archs
    basicsr_archs.restormer_arch = restormer_module

    sys.path.insert(0, str(src))
    spec = importlib.util.spec_from_file_location("_stage3_external_perceiveir_arch", arch_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Perceive-IR arch from {arch_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stage3PerceiveIR(nn.Module):
    """Stage3 adapter around Perceive-IR with learned degradation prompts."""

    def __init__(self, input_channels: int = 12, base_channels: int = 48, residual_scale: float = 1.0) -> None:
        super().__init__()
        module = _load_perceive_module()
        self.module = module
        self.residual_scale = residual_scale
        self.pad_multiple = 8
        self.backbone = module.cyclicPrompt(
            inp_channels=input_channels,
            out_channels=input_channels,
            dim=base_channels,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias",
            model_clip=_ClipStub(),
            num_vector=8,
            iter_times=2,
        )
        self.backbone.clip_input_preprocess = nn.Identity()
        cond_width = max(64, base_channels)
        self.condition = nn.Sequential(
            nn.Conv2d(input_channels, cond_width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(cond_width, cond_width, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cond_width, 512),
            nn.LayerNorm(512),
        )
        self.out_condition = nn.Sequential(
            nn.Conv2d(input_channels, cond_width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cond_width, 512),
            nn.LayerNorm(512),
        )
        self.out = nn.Conv2d(input_channels, 3, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _forward_backbone(self, imgs: torch.Tensor) -> torch.Tensor:
        model = self.backbone
        text_feat = self.condition(imgs)
        lq_clip_feat = text_feat

        if model.num_vector != 0:
            learned = model.learnable_vector.unsqueeze(0).expand(imgs.shape[0], -1, -1)
            deg = model.deg_aware(lq_clip_feat).unsqueeze(1).expand(-1, model.num_vector, -1)
            prompt = torch.cat([learned + deg, text_feat.unsqueeze(1)], dim=1)
        else:
            prompt = text_feat

        inp_enc_level1 = model.patch_embed(imgs)
        out_enc_level1 = model.encoder_level1(inp_enc_level1)
        inp_enc_level2 = model.down1_2(out_enc_level1)
        out_enc_level2 = model.encoder_level2(inp_enc_level2)
        inp_enc_level3 = model.down2_3(out_enc_level2)
        out_enc_level3 = model.encoder_level3(inp_enc_level3)
        inp_enc_level4 = model.down3_4(out_enc_level3)
        latent = model.latent(inp_enc_level4)

        inp_dec_level3 = model.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = model.reduce_chan_level3(inp_dec_level3)

        outputs = []
        hq_clip_feat = None
        rcp_level3 = None
        rcp_level2 = None
        rcp_level1 = None
        for i in range(model.iter_times):
            out_dec_level3 = model.decoder_level3(inp_dec_level3, prompt, rcp_level3, hq_clip_feat, iter=i)

            inp_dec_level2 = model.up3_2(out_dec_level3)
            inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
            inp_dec_level2 = model.reduce_chan_level2(inp_dec_level2)
            out_dec_level2 = model.decoder_level2(inp_dec_level2, prompt, rcp_level2, hq_clip_feat, iter=i)

            inp_dec_level1 = model.up2_1(out_dec_level2)
            inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
            out_dec_level1 = model.decoder_level1(inp_dec_level1, prompt, rcp_level1, hq_clip_feat, iter=i)

            out_dec_level1 = model.refinement(out_dec_level1)
            if model.dual_pixel_task:
                out_dec_level1 = out_dec_level1 + model.skip_conv(inp_enc_level1)
                out_dec_level1 = model.output(out_dec_level1)
            else:
                out_dec_level1 = model.output(out_dec_level1) + imgs

            if i == 0 and model.iter_times > 1:
                ch_res_out = self.module.get_residue(out_dec_level1)
                rcp_feature = model.rcp_extractor(torch.cat([ch_res_out, ch_res_out, ch_res_out], dim=1))
                rcp_level1 = model.rcp_ch(rcp_feature)
                rcp_level2 = model.rcp_down1(rcp_feature)
                rcp_level3 = model.rcp_down2(rcp_level2)

                hq_feat = self.out_condition(out_dec_level1)
                hq_feat = model.hq_mlp(hq_feat)
                hq_clip_feat = torch.cat([hq_feat.unsqueeze(1), text_feat.unsqueeze(1)], dim=1)

            outputs.append(out_dec_level1)

        return torch.stack(outputs, dim=0).mean(dim=0)

    def forward(self, x: torch.Tensor, composite: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h = (self.pad_multiple - h % self.pad_multiple) % self.pad_multiple
        pad_w = (self.pad_multiple - w % self.pad_multiple) % self.pad_multiple
        if pad_h or pad_w:
            x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_in = x

        features = self._forward_backbone(x_in)[..., :h, :w]
        residual = torch.tanh(self.out(features)) * self.residual_scale
        return torch.clamp(composite + residual, 0.0, 1.0)
