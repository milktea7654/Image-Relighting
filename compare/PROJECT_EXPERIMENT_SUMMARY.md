# Relighting Benchmark Project Summary

Date: 2026-06-14

## Scope

This project set up and audited a local relighting benchmark workspace under `compare`.
The main goal was to compare our Stage3/Stage4 relighting models against external
baselines on MIIW, ISR, RSR, VIDIT-style data, and our own Stage3 test set.

## External Baselines

The following external methods were downloaded or integrated under `compare`:

- `rgbx`: RGB<->X / RGB-to-X and X-to-RGB pipelines.
- `luminet`: LumiNet checkpoint and inference wrapper.
- `ic-light`: IC-Light background-conditioned and text/background inference code.
- `omnigen`: OmniGen-v1 diffusers pipeline.

`SPOTLIGHT`/ZeroComp was intentionally skipped when it required ZeroComp-related
setup.

## Datasets

Prepared or downloaded benchmark roots include:

- `compare/miiw_xyz_quantitative`
- `compare/miiw_xyz_old400`
- `compare/isr_quantitative`
- `compare/rsr_quantitative`
- `compare/vidit_quantitative`
- `compare/own_test_quantitative`

`InternScenes` was investigated. The repository was visible through Hugging Face
listing, but actual file download remained blocked by the token setting that must
allow access to public gated repositories.

## Own Test Data

The own-test benchmark root is:

```text
compare/own_test_quantitative
```

Important fields:

- `source_rgb` / `source_rgb_512`: `composite_optimized`, used as the de-lit source/input image.
- `target_rgb` / `target_rgb_512`: `reference`, used as the ground-truth target for scoring.
- `target_irradiance_512`: originally copied from the HF `shading` field.
- `reference_rgb_512`: originally also copied from the HF `shading` field, despite the misleading name.
- `target_irradiance_last_512`: newly derived from `target_rgb_512` using RGB<->X RGB-to-X.

The original naming caused confusion because `reference_rgb_512` was not target RGB;
it was a shading/irradiance proxy.

## Our Models

The own-test run contains these local prediction directories:

- `dcpt-promptir-FT`
- `hair_FT`
- `promptir_12ch`
- `restormer`
- `xrestormer-FT`
- `ours-best`

The current own-test table selected `hair_FT` as `ours-best`.

## Stage3/Stage4 Audit

The intended full local path is:

```text
source image
-> Stage4/Stage12 reconstruction
-> Stage2 target-light render
-> Stage3 refinement checkpoint
-> final relit image
```

For own-test Stage3 evaluation, missing test-set `hybrid_albedo` was regenerated
with MVInverse from `target_rgb_512`, and `glass_mask_soft` was filled as a black
mask when missing, matching the common no-glass case in the prepared data.

## RGB<->X Protocol

Original own-test RGB<->X run:

```text
source AOVs: RGB->X from source_rgb_512
target irradiance: target_irradiance_512
```

That means the original run used the HF `shading` field as target irradiance, not
an irradiance map newly estimated from the target/reference RGB.

Updated `last` run:

```text
target_rgb_512 -> RGB<->X RGB-to-X -> target_irradiance_last_512
source_rgb_512 -> source AOVs
source AOVs + target_irradiance_last_512 -> RGB<->X X-to-RGB
```

Output:

```text
compare/own_test_quantitative/predictions/rgbx-last
compare/own_test_quantitative/predictions/rgbx-last_intermediates
```

Runtime logs:

```text
compare/own_test_quantitative/logs/last_rgbx.stdout.log
compare/own_test_quantitative/logs/last_rgbx.stderr.log
compare/own_test_quantitative/predictions/rgbx-last/runtime.json
```

Observed runtime for `rgbx-last`:

```text
count: 1029
mean: 1.445 s/image
total: 1487.2 s
```

This is shorter than the original full RGB<->X run because the last run reused
source AOVs and only refreshed target irradiance plus X-to-RGB.

## LumiNet Protocol

Original own-test LumiNet run:

```text
source_rgb_512 + reference_rgb_512
```

Because `reference_rgb_512` was the HF `shading` field, this was actually:

```text
composite_optimized + shading proxy
```

Updated `last` run:

```text
source_rgb_512 + target_irradiance_last_512
```

Output:

```text
compare/own_test_quantitative/predictions/luminet-last
```

Status at the time this file was first written: still running.

## IC-Light Protocol

IC-Light background-conditioned model expects:

```text
foreground RGB + background RGB + text prompt
```

It does not consume dense irradiance/AOV maps.

Original own-test IC-Light run used:

```text
source_rgb_512 foreground + reference_rgb_512 uploaded background
```

This was incorrect for own-test because `reference_rgb_512` was the HF `shading`
field, not a background/reference RGB image.

Updated `last` run should use:

```text
source_rgb_512 foreground + target_rgb_512 uploaded background
```

Output target:

```text
compare/own_test_quantitative/predictions/ic-light-last
```

Status at the time this file was first written: queued to run after `luminet-last`.

## OmniGen Protocol

The current OmniGen runner uses:

```text
image_1 = source_rgb_512
image_2 = target_irradiance_512
```

So yes, OmniGen uses the composite/de-lit image as the first image. That is
consistent with the source side of the relighting task.

However, OmniGen is a general image-conditioned generation model. It does not
define a relighting-specific requirement like RGB<->X irradiance or LumiNet
reference-lighting image. The local runner currently uses the second image as a
target illumination/shading reference through the text prompt. If a stricter
image-reference protocol is desired, the second image should be explicitly chosen
and documented as either:

- `target_rgb_512`: target/reference RGB image, strong but may leak target content.
- `target_irradiance_last_512`: target irradiance estimated from target/reference RGB.

## Own-Test Current Scores

Before the `last` updates, the own-test table was:

| Method | Count | RMSE | PSNR | SSIM | LPIPS | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RGB<->X | 1029 | 0.275 | 11.552 | 0.311 | 0.495 | 6.924 |
| LumiNet | 1029 | 0.214 | 13.663 | 0.407 | 0.462 | 6.383 |
| IC-Light | 1029 | 0.239 | 12.694 | 0.321 | 0.490 | 2.012 |
| OmniGen | 1029 | 0.238 | 12.870 | 0.273 | 0.622 | 49.576 |
| Ours best | 1029 | 0.121 | 19.086 | 0.688 | 0.223 | 0.102 |

The final table should be regenerated after `luminet-last` and `ic-light-last`
finish.

## External Beats Ours Ranking

For the original complete external runs, examples where external PSNR is higher
than `ours-best` are stored at:

```text
compare/own_test_quantitative/results/external_beats_ours_original
```

Summary:

- IC-Light: 0 positive-gap examples.
- LumiNet: 5 positive-gap examples.
- RGB<->X: 11 positive-gap examples.

The largest RGB<->X gap was sample `000215`, where RGB<->X scored 12.938 PSNR
and `ours-best` scored 9.881 PSNR.

## Ranking Scripts

Added scripts:

- `compare/scripts/rank_own_test_psnr.py`
- `compare/scripts/rank_methods_psnr.py`
- `compare/scripts/rank_external_beats_ours.py`
- `compare/scripts/derive_target_irradiance_rgbx.py`

These generate per-image PSNR rankings, contact sheets, copied selected images,
and CSV/JSON/Markdown summaries.

## Key Open Issues

- `luminet-last` and `ic-light-last` must finish before the final last-version
  quantitative table and top-10 visual reports are complete.
- IC-Light original run should not be treated as protocol-correct for own-test
  because it used a shading proxy as uploaded background.
- OmniGen has no paper-equivalent relighting protocol here; the conditioning
  image choice must be explicitly reported.
- ISR/RSR/VIDIT proxy runs should remain marked diagnostic unless rerun under
  official paper annotations and protocols.
