# Paper-Protocol Quantitative Audit

Date: 2026-06-10

## Invalidated Runs

The current `compare/vidit_quantitative`, `compare/isr_quantitative`, and
`compare/rsr_quantitative` result tables are diagnostic proxy runs. They should
not be used as official quantitative results.

Main reasons:

- Wrong split for paper-level VIDIT/ISR/RSR quantitative.
- Proxy target irradiance/reference images were synthesized from metadata.
- Several methods were fed inputs outside their paper protocols.
- The local Stage3-only rows do not represent the intended full model path.

## DeepIntrinsicRelighting Protocol

Official local source:

```text
compare/sources/DeepIntrinsicRelighting/test_quantitative.py
compare/sources/DeepIntrinsicRelighting/options/test_quantitative_options.py
compare/sources/DeepIntrinsicRelighting/util/metric.py
```

Official quantitative annotations:

| Dataset | Official annotation | Pairs |
| --- | --- | ---: |
| ISR | `data/anno_ISR/test_quantitative_pairs_10x.txt` | 11710 |
| RSR | `data/anno_RSR/AnyLighttest_pairs.txt` | 2880 |
| VIDIT | `data/anno_VIDIT/any2any/AnyLight_test_pairs.txt` | 1200 |

Official image size and preprocessing:

- `img_size = (256, 256)`
- ISR: `preprocess = none`
- RSR: `preprocess = none`
- VIDIT: `preprocess = resize`

Official target-light encoding:

```text
[cos(pan)*0.5+0.5, sin(pan)*0.5+0.5, cos(tilt), sin(tilt), R/255, G/255, B/255]
```

Official metrics:

```text
MSE, SSIM, PSNR, LPIPS, MPS
```

The metric implementation uses PyTorch tensors in `[0, 1]`,
`pytorch_msssim.ssim`, LPIPS Alex, and `PSNR = 10 * log10(1 / MSE)`.

## Dataset Availability

Present:

```text
compare/datasets/isr_image/Image
compare/datasets/rsr_256/RSR_256
```

Missing for official VIDIT quantitative:

```text
compare/datasets/VIDIT_full
```

The previously downloaded VIDIT Track2 validation files are not the same as the
DeepIntrinsicRelighting `VIDIT_full` any-to-any quantitative set.

## Method Conditioning Audit

### RGB<->X

The existing proxy runner used source RGB to estimate AOVs, then injected a
metadata-gradient `target_irradiance_512`.

This is not paper-level quantitative unless the benchmark provides or derives
the same dense AOV/irradiance channels expected by the RGB<->X protocol.

### LumiNet

The released inference code expects:

```text
source image + reference lighting image
```

For ISR/RSR proxy runs, `reference_rgb_512` was a synthetic irradiance map, not
a real reference lighting image. That is not paper-equivalent.

### IC-Light

The released code is text- or background-conditioned relighting. The proxy run
used a synthetic/guide background as conditioning. This is not an official
paired quantitative protocol unless the paper being compared defines exactly how
IC-Light is prompted/conditioned on this benchmark.

### Ours

Do not feed the local model arbitrary proxy tensors.

The intended full path is:

```text
source image
-> Stage4/Stage12 reconstruction
-> Stage2 target-light render
-> Stage3 refinement checkpoint
-> final relit image
```

Different Stage3 checkpoints can reuse cached Stage4 buffers, but the buffers
must come from the correct official sample and target-light condition.

## MIIW Estimated-XYZ Protocol

MIIW provides real paired images and chrome/gray probes, but not exact physical
point-light coordinates. The new local benchmark root is:

```text
compare/miiw_xyz_quantitative
```

This root uses the target chrome/gray probe to estimate the dominant incident
light direction in camera coordinates:

```text
target chrome probe highlight -> reflected light direction -> target XYZ
```

Each sample stores:

- source RGB and target RGB for scoring
- target chrome/gray probes
- `target_light.xyz`
- `target_light.rgb`
- `target_light.stage4_point_light_position`
- `target_light.stage4_point_light_rgb`
- `target_light.stage4_env_rgb`
- `target_light.npz`

For Stage4, the raw probe direction is not used as a radius-2 world-space
position. Stage4 places its own optimized source point grid near the camera,
typically around `z ~= -0.2` in camera coordinates. The MIIW active control
therefore stores:

```text
raw_probe_xyz              = chrome-probe direction * model_radius
direction_xyz_model_radius = Stage4-front direction * model_radius
stage4_point_light_position = direction projected to z=-0.2, x/y clamped
```

The raw probe estimate is kept for auditability; the projected near-plane
position is what Stage4 receives.

This benchmark should be named **MIIW estimated-xyz relighting benchmark**. It
is suitable for testing whether a method can use the estimated target light
condition to relight toward the paired target image, but it is not an exact
point-light ground-truth benchmark.

The 300-pair split is stratified by source-vs-target PSNR:

```text
100 hard / 100 medium / 100 easy
```

The default split excludes the SDK frontal directions:

```text
FRONTAL_DIRECTIONS = [2, 3, 19, 20, 21, 22, 24]
NOFRONTAL_DIRECTIONS = [i for i in range(25) if i not in FRONTAL_DIRECTIONS]
```

This avoids mixing direct visible-flash cases into the bounce-light estimated
XYZ control benchmark.

Report both full-image and changed-region metrics:

```text
Full PSNR, Changed PSNR, Gain over source, SSIM, LPIPS, RMSE
```

## Required Fix

1. Build new official benchmark roots from official annotations:
   `isr_paper_quantitative`, `rsr_paper_quantitative`, and
   `vidit_paper_quantitative`.
2. Download/prepare `VIDIT_full` before VIDIT paper quantitative.
3. Implement a paper metric evaluator matching
   `DeepIntrinsicRelighting/util/metric.py`.
4. For each external method, either:
   - run the method using its paper-defined conditioning on the official split,
     or
   - mark the method as not protocol-compatible for that table.
5. Run ours only through the full Stage4 path for official target conditions.
6. Report final numbers only from these rebuilt paper roots.
