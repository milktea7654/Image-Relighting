# MIIW Real Paired Quantitative

Table 1 evaluates real paired relighting on the MIIW test set.

Dataset source: MIT CSAIL Multi-Illumination Images in the Wild. The official
test JPG archive contains 30 scenes, each captured under 25 lighting
conditions.

## Layout

```text
miiw_quantitative/
  assets/
    downloads/miiw_test_jpg.zip
    miiw_test_jpg/
  samples/
    000000/
      source_rgb.png
      target_rgb.png
      source_rgb_512.png
      target_rgb_512.png
      source_rgb_full.jpg
      target_rgb_full.jpg
      source_probe_gray.jpg
      source_probe_chrome.jpg
      target_probe_gray.jpg
      target_probe_chrome.jpg
      metadata.json
  manifest.jsonl
  predictions/
    rgbx/
    luminet/
    ic-light/
    ours-best/
  results/
    table1_metrics.json
    table1_metrics.csv
    table1.md
```

## Pair Selection And Sizes

The split contains 300 source-target pairs. It uses all 30 test scenes and
chooses 10 target lighting conditions per scene. Target directions are balanced
across all 25 MIIW light directions.

Each pair fixes scene and camera; only light direction changes. The prepared
files keep three resolution roles:

- `source_rgb.png`, `target_rgb.png`: 256 x 256 scoring copies for Table 1 and
  Ours best.
- `source_rgb_512.png`, `target_rgb_512.png`: 512 x 512 square copies for
  LumiNet, IC-Light, and RGB↔X baseline inference.
- `source_rgb_full.jpg`, `target_rgb_full.jpg`: original MIIW mip2 JPG files,
  currently 1500 x 1000.

Predictions may be saved at each method's native output size. The evaluator
resizes predictions to the 256 x 256 scoring target before computing metrics.

Recommended baseline settings:

- LumiNet: use the 512 x 512 source/reference copies.
- IC-Light: use `image_width=512` and `image_height=512`.
- RGB↔X: use `source_rgb_512.png` and pass `--max-side 512` for RGB→X.

## Expected Predictions

Put one prediction per sample under:

```text
predictions/<method>/<sample_id>.png
```

For example:

```text
predictions/rgbx/000000.png
predictions/luminet/000000.png
predictions/ic-light/000000.png
predictions/ours-best/000000.png
```

Optional runtime files can be placed at `predictions/<method>/runtime.json`:

```json
{
  "000000": 1.23,
  "000001": 1.19
}
```

Runtime is reported as average seconds per image when available.

## Commands

```powershell
cd compare\miiw_quantitative
.\.venv\Scripts\python.exe .\scripts\prepare_miiw_pairs.py --count 300 --force
.\.venv\Scripts\python.exe .\scripts\evaluate_table1.py
```
