# Relighting Benchmark

## Current Status

The VIDIT/ISR/RSR numbers generated on 2026-06-10 are diagnostic proxy runs,
not paper-protocol quantitative results. Do not use those tables as official
benchmark numbers.

The mismatch is material:

- VIDIT used NTIRE2021 Track2 validation Pair/guide files instead of the
  DeepIntrinsicRelighting `VIDIT_full` any-to-any test annotation.
- ISR used `data/anno_ISR/val_pairs.txt` with 390 pairs instead of
  `data/anno_ISR/test_quantitative_pairs_10x.txt` with 11710 pairs.
- RSR used `data/anno_RSR/AnyLightval_pairs.txt` with 864 pairs instead of
  `data/anno_RSR/AnyLighttest_pairs.txt` with 2880 pairs.
- ISR/RSR target irradiance/reference inputs were metadata-gradient proxies, not
  paper-protocol conditioning or real dense light transport.
- Local Stage3 checkpoints were also evaluated in a direct/proxy mode before the
  full Stage4 path was available. Those rows are diagnostic only.

The official quantitative pipeline must be rebuilt from the paper protocols
before reporting final numbers.

## Goal

Compare RGB↔X, LumiNet, IC-Light, and Ours best on the same real paired MIIW
split for Table 1.

## Scoring Dataset

- Dataset: MIIW, MIT CSAIL Multi-Illumination Images in the Wild test split
- Local path: `compare/miiw_quantitative`
- Source archive: `assets/downloads/miiw_test_jpg.zip`
- Prepared samples: `samples/<sample_id>/`
- Manifest: `manifest.jsonl`
- Count: 300 paired samples
- Scoring resolution: 256 x 256
- Baseline input resolution: 512 x 512 square copies
- Original MIIW copy: 1500 x 1000 mip2 JPG
- Ground truth: `target_rgb.png`

Each sample fixes scene and camera. `source_rgb.png` and `target_rgb.png`
are the 256 x 256 scoring copies and differ by light direction only.
`source_rgb_512.png` and `target_rgb_512.png` are the baseline inference copies.
`source_rgb_full.jpg` and `target_rgb_full.jpg` keep the original MIIW files.
Probe images for both light directions are kept with the sample metadata so
model-specific target-light conditioning can be derived without changing the
scoring split.

## Table 1

```text
Table 1: MIIW real paired quantitative
Method    RMSE ↓    PSNR ↑    SSIM ↑    LPIPS ↓    Runtime ↓
RGB↔X
LumiNet
IC-Light
Ours best
```

Generated files:

- `results/table1.md`
- `results/table1_metrics.csv`
- `results/table1_metrics.json`

## Methods

| Method | Prediction folder | Notes |
| --- | --- | --- |
| RGB↔X | `predictions/rgbx/` | Use `source_rgb_512.png` and run RGB→X with `--max-side 512`; X→RGB output can stay 512 x 512 and will be resized for scoring. |
| LumiNet | `predictions/luminet/` | Use `source_rgb_512.png` and `target_rgb_512.png`/target-light reference at LumiNet's 512 x 512 inference size. |
| IC-Light | `predictions/ic-light/` | Use the 512 x 512 source copy with `image_width=512`, `image_height=512`. Save only the final relit RGB image for scoring. |
| Ours best | `predictions/ours-best/` | Save the best model output per sample id. |

Prediction filenames must match the manifest sample ids:

```text
predictions/rgbx/000000.png
predictions/luminet/000000.png
predictions/ic-light/000000.png
predictions/ours-best/000000.png
```

## Metrics

- RMSE, lower is better
- PSNR, higher is better
- SSIM, higher is better
- LPIPS, lower is better when the optional LPIPS dependency is installed
- Runtime, average seconds per image when `runtime.json` is present

The evaluator compares each prediction with the matching 256 x 256
`target_rgb.png`. If a baseline prediction is saved at native resolution, the
evaluator resizes it to 256 x 256 with LANCZOS before metric calculation.
Missing predictions are counted in the JSON/CSV outputs and shown as `-` in the
Markdown table until results are available.

## Reproducibility

For each method, keep method-specific intermediate files outside the final
prediction folder when possible. Save the inference script/config, checkpoint or
model id, random seed, manifest path, and runtime metadata next to that method's
outputs.

## VIDIT / ISR / RSR Diagnostic Runs

Prepared and evaluated under `compare/`. These runs are diagnostic only and
must not be reported as paper-protocol quantitative results.

Important: the first VIDIT/ISR/RSR tables used direct Stage3 checkpoint
inference for the local `result/*` folders. Those rows are useful as a diagnostic
baseline, but they are not the intended full model protocol. The intended model
inference path is Stage4:

```text
source image -> Stage12/Stage2 reconstruction and target-light render
-> Stage3 refinement -> final relit image
```

Use `compare/scripts/run_stage4_benchmark.py` for the full model path. A
low-quality ISR smoke output is available at
`compare/isr_quantitative/predictions/stage4_smoke/000000.png`.

| Dataset | Local path | Split used | Samples |
| --- | --- | --- | ---: |
| VIDIT | `compare/vidit_quantitative` | NTIRE2021 Track2 public validation | 90 |
| ISR | `compare/isr_quantitative` | `data/anno_ISR/val_pairs.txt` | 390 |
| RSR | `compare/rsr_quantitative` | `data/anno_RSR/AnyLightval_pairs.txt` | 864 |

Prediction outputs are separated by dataset:

```text
compare/<dataset>_quantitative/predictions/rgbx/
compare/<dataset>_quantitative/predictions/luminet/
compare/<dataset>_quantitative/predictions/ic-light/
compare/<dataset>_quantitative/predictions/<local-model>/
compare/<dataset>_quantitative/predictions/ours-best/
```

RGB<->X intermediates are stored in
`compare/<dataset>_quantitative/predictions/rgbx_intermediates/`.

External baselines were fed 512 x 512 inputs. Local `result/*` models were fed
the 256 x 256 benchmark tensors. VIDIT uses the public guide image as the target
reference for external relighting. ISR/RSR use the available paired light
metadata to generate target-irradiance proxy images because dense irradiance maps
are not provided by those datasets.

Detailed metrics are written to:

```text
compare/vidit_quantitative/results/table.md
compare/isr_quantitative/results/table.md
compare/rsr_quantitative/results/table.md
```

Summary:

| Dataset | Method | RMSE | PSNR | SSIM | LPIPS | Runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| VIDIT | RGB<->X | 0.510 | 5.943 | 0.076 | 0.722 | 7.021 |
| VIDIT | LumiNet | 0.275 | 11.472 | 0.135 | 0.545 | 5.565 |
| VIDIT | IC-Light | 0.213 | 14.098 | 0.458 | 0.411 | 1.969 |
| VIDIT | Ours best (`xrestormer-FT`) | 0.168 | 17.140 | 0.569 | 0.338 | 0.153 |
| ISR | RGB<->X | 0.336 | 9.684 | 0.344 | 0.485 | 7.034 |
| ISR | LumiNet | 0.224 | 13.265 | 0.494 | 0.343 | 5.561 |
| ISR | IC-Light | 0.256 | 12.108 | 0.440 | 0.382 | 1.964 |
| ISR | Ours best (`promptir_12ch`) | 0.179 | 15.589 | 0.693 | 0.238 | 0.068 |
| RSR | RGB<->X | 0.326 | 9.909 | 0.320 | 0.573 | 7.076 |
| RSR | LumiNet | 0.269 | 11.650 | 0.494 | 0.489 | 5.563 |
| RSR | IC-Light | 0.271 | 11.522 | 0.414 | 0.445 | 1.963 |
| RSR | Ours best (`promptir_12ch`) | 0.217 | 13.853 | 0.634 | 0.385 | 0.068 |
