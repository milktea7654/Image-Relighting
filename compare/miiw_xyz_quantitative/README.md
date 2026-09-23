# MIIW Estimated-XYZ Relighting Benchmark

This root contains paired MIIW samples prepared from source and target captures.
The target light is estimated from the target chrome/gray probe as a dominant incident light direction in camera coordinates.
It is not an exact physical point-light annotation.

Each sample stores:

- source RGB and target RGB for scoring
- target chrome/gray probes
- `target_light.xyz`, `target_light.rgb`, and fixed source-probe `target_light.stage4_env_rgb`
- `target_light.npz` for model loaders that prefer arrays

The default split uses MIIW SDK-style non-frontal directions only:

- excluded frontal directions: [2, 3, 19, 20, 21, 22, 24]
- allowed directions: [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 23]

Pair selection is stratified by source-vs-target PSNR:

- hard: 34
- medium: 33
- easy: 33

Recommended Stage4 run:

```powershell
python compare/scripts/run_stage4_benchmark.py --benchmark-root compare/miiw_xyz_quantitative --method-id stage4-miiw-xyz --use-point-light
```

The default keeps ambient fixed from the source gray probe and changes only the estimated target point light.
Use `--point-only-target` only for a point-light-only ablation with zero ambient.

Builder settings:

```json
{
  "source_root": "D:\\coding\\relighting\\compare\\miiw_quantitative\\assets\\miiw_test_jpg",
  "output_root": "D:\\coding\\relighting\\compare\\miiw_xyz_quantitative",
  "count": 100,
  "hard": 34,
  "medium": 33,
  "easy": 33,
  "exclude_dirs": [
    2,
    3,
    19,
    20,
    21,
    22,
    24
  ],
  "force": true,
  "model_radius": 2.0,
  "stage4_z_depth": 0.2,
  "stage4_xy_limit": 0.75,
  "point_scale": 1.0,
  "point_ceiling": 120.0,
  "env_scale": 0.2,
  "env_floor": 0.02,
  "highlight_percentile": 99.5,
  "gray_subtract_weight": 0.8,
  "sphere_threshold": 0.03,
  "changed_threshold": 0.05,
  "no_flip_z_for_stage4": false,
  "allow_stage4_positive_z": false
}
```
