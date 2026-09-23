# compare

This folder keeps the relighting baselines and the active quantitative
benchmark setup.

## Active Benchmark

Table 1 evaluates real paired relighting on MIIW:

```text
Table 1: MIIW real paired quantitative
Method    RMSE ↓    PSNR ↑    SSIM ↑    LPIPS ↓    Runtime ↓
RGB↔X
LumiNet
IC-Light
Ours best
```

- Dataset: MIT CSAIL Multi-Illumination Images in the Wild test set
- Prepared split: `miiw_quantitative/manifest.jsonl`
- Samples: 300 source-target pairs
- Scoring resolution: 256 x 256
- Baseline input resolution: 512 x 512 square copies
- Original MIIW copy: 1500 x 1000 mip2 JPG
- Results: `miiw_quantitative/results/table1.md`

Each pair uses the same scene and camera with different light directions. The
source image, target image, 512 baseline copies, original JPG copies, source
probes, target probes, and per-pair metadata are saved under
`miiw_quantitative/samples/<sample_id>/`.

## Baselines

- `rgbx/`: RGB↔X repository
- `luminet/`: LumiNet repository and environment
- `ic-light/`: IC-Light repository, environment, and local model files
- `run_paper.py`: local launcher for available baseline entrypoints
- `rgbx_rgb2x_batch.py`, `rgbx_x2rgb_single.py`: RGB↔X helper wrappers

Old inactive generated dataset, output, temporary, and auxiliary benchmark
folders were removed from this compare workspace because Table 1 only needs
RGB↔X, LumiNet, IC-Light, and Ours best on MIIW.

## Prediction Contract

Save final predictions here:

```text
miiw_quantitative/predictions/rgbx/<sample_id>.png
miiw_quantitative/predictions/luminet/<sample_id>.png
miiw_quantitative/predictions/ic-light/<sample_id>.png
miiw_quantitative/predictions/ours-best/<sample_id>.png
```

Optional runtime files can be saved as
`miiw_quantitative/predictions/<method>/runtime.json`, mapping sample id to
seconds.

Predictions can be saved at native method resolution. The evaluator resizes each
prediction to the 256 x 256 scoring target before computing Table 1 metrics.

## Commands

```powershell
cd compare\miiw_quantitative
.\.venv\Scripts\python.exe .\scripts\prepare_miiw_pairs.py --count 300 --force
.\.venv\Scripts\python.exe .\scripts\evaluate_table1.py
```

The evaluator writes Markdown, CSV, and JSON outputs under
`miiw_quantitative/results/`.
