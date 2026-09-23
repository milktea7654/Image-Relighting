# Low Score Diagnosis

Date: 2026-06-09

## Main Finding

The low scores are mostly caused by a protocol mismatch, not by a metric bug.
The previous benchmark ran the Stage3 checkpoint directly and skipped Stage4.
That is not the intended inference path for the model.

The current `result/*` checkpoints are Stage3 renderer-gap refinement models.
They are trained to predict:

```text
reference ~= composite + learned residual
```

The checkpoints use:

```text
input_names = composite, render, albedo, roughness, metallic, glass_mask
```

They do not use target light position or `target_irradiance`. In the current
VIDIT/ISR/RSR runner, `source_rgb` is used as `composite`, so the model naturally
stays close to the source image.

The intended full inference path is:

```text
source image
-> Stage4 / Stage12 scene reconstruction
-> Stage2 Mitsuba target-light render
-> composite/render/material buffers
-> Stage3 refinement
-> final relit image
```

So the direct Stage3 benchmark numbers should not be treated as the real score
for the full model.

## Diagnostic Metrics

Source image vs target is already almost equal to `ours-best`:

| Dataset | Source PSNR | Ours-best PSNR | Ours PSNR - Source | Ours wins |
| --- | ---: | ---: | ---: | ---: |
| VIDIT | 17.467 | 17.140 | -0.466 | 20/89 |
| ISR | 15.608 | 15.589 | -0.018 | 182/390 |
| RSR | 13.865 | 13.853 | -0.012 | 367/864 |

The generated `target_render_proxy` is worse than simply using the source:

| Dataset | Source PSNR | Render proxy PSNR |
| --- | ---: | ---: |
| VIDIT | 17.467 | 16.222 |
| ISR | 15.608 | 13.998 |
| RSR | 13.865 | 13.469 |

## Visual Check

See:

```text
compare/diagnostics/low_score_contact_sheet.jpg
```

The proxy render is usually a flat/global relight or simple gradient. It does
not contain target shadows, occlusion, indirect light, or geometry-aware light
transport. The model output remains close to the source image.

## Consequence

The benchmark currently asks Stage3 to solve a task it was not trained for:

```text
source image + weak metadata proxy -> target relit image
```

But Stage3 was trained for:

```text
renderer/composite output + material maps -> refined target/reference image
```

## Fix Direction

Use one of these protocols:

1. Run the full Stage4 rendering pipeline for each target light to produce real
   `render` and `composite` inputs before Stage3 refinement.
2. Train/finetune a new paired relighting model that explicitly consumes target
   light position or dense target irradiance.
3. For RGB<->X, provide a real dense target irradiance estimate instead of the
   current metadata-gradient proxy.
4. Keep a `source_rgb` baseline in all tables to catch this failure mode.

## Stage4 Runner Status

Added:

```text
compare/scripts/run_stage4_benchmark.py
```

Smoke-tested on one ISR sample with low-quality settings:

```text
compare/isr_quantitative/predictions/stage4_smoke/000000.png
compare/isr_quantitative/stage4_artifacts/stage4_smoke/000000/contact_sheet.jpg
```

Environment fixes applied:

- copied `moge2.pt` and `ppd_moge.pth` into
  `Stage4/external/Stage12_remote_run/checkpoints/`
- upgraded Stage12 torch stack to CUDA 12.8 compatible builds
- fixed Stage12 MoGe local/HF loading fallback
- fixed PPD CPU device fallback
- fixed `stage12_single_relight.py` so it does not silently reuse an old scene
  when Stage2 fails

The successful GPU smoke used:

```text
--spp 2 --final-spp 4 --iters 2 --mesh-stride 8
```

This is only a pipeline test, not a publishable metric setting.
