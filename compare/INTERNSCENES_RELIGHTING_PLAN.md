# InternScenes Relighting Data Plan

## Status

This is a design note only. Do not treat it as a generated dataset.

InternScenes is useful as a large simulatable indoor scene source, not as a
ready-made relighting ground-truth dataset. We should generate the paired
lighting ourselves with a fixed camera and fixed scene geometry.

## Source Dataset

- HF dataset repo: `InternRobotics/InternScenes`
- Access: gated, but readable with the current HF login after license approval.
- License: CC BY-NC-SA 4.0.
- Structure observed from the repo:
  - `Scenes_info/scene_name_list_final.json`
  - `Layout_info.tar.gz`
  - `InternScenes_Gen/Layout_info.tar.gz`
  - `asset_library/<source>/*.glb`
  - split archives under `InternScenes_Gen/Asset_Library/...`

The dataset card mentions `OpenRobotLab/InternScenes`, but that repo currently
returns 404 through the HF API. Use `InternRobotics/InternScenes`.

## Goal

Generate a small verified relighting dataset first, then scale.

Initial smoke target:

- 10 scenes
- 3 camera views per scene
- 5 target lights per view
- 150 paired samples
- 256 x 256 model tensors
- 512 x 512 external-baseline copies

Scale target after verification:

- 300+ paired samples per split or per scene group.

## Sample Contract

Each sample must fix camera, geometry, materials, and object placement.
Only the light changes.

Required files:

- `source_rgb.png`
- `target_rgb.png`
- `source_rgb_512.png`
- `target_rgb_512.png`
- `render_optimized.png`
- `target_irradiance.png`
- `target_irradiance_512.png`
- `hybrid_albedo.png`
- `roughness.png`
- `metallic.png`
- `normal.png`
- `glass_mask_soft.png`
- `light/target_light.json`
- `metadata.json`

`light/target_light.json` must include:

- `env_rgb`
- `point_light_position`
- `point_light_rgb`
- `light_task`
- `camera`
- coordinate-system note

## Rendering Route

Preferred first implementation: Stage12/Mitsuba route.

1. Download minimal scene layout metadata.
2. Pick a small scene subset.
3. Resolve each selected scene's required asset ids from `layout.json`.
4. Download only those GLB assets.
5. Convert each InternScenes scene into a Stage12 package:
   - `scene.glb` or `scene_uv.obj`
   - `camera.json`
   - material maps when available, otherwise scalar fallbacks
6. Use the existing Stage12/Mitsuba code to render:
   - source condition
   - target condition
   - albedo/roughness/metallic/normal buffers
7. Use black `glass_mask_soft` unless a transparent/glass object detector is
   added later.

Fallback route: Isaac Sim.

Use Isaac Sim only if the Stage12 conversion cannot preserve room geometry or
materials well enough. InternScenes officially mentions Isaac Sim 4.1.0 for USD
rendering, but this is heavier and less consistent with the current relighting
pipeline.

## Light Task Grid

For each camera, generate target lights from a controlled set:

- `left_high`
- `right_high`
- `front_low`
- `back_high`
- `ceiling_top`
- optional colored variants: warm, cool, red, blue

The point light position should be defined relative to the camera or room bounds,
then converted to the Stage12/Mitsuba scene coordinate system and saved in the
metadata.

Recommended first grid:

```text
env_rgb:
  low:  [0.03, 0.03, 0.03]
  mid:  [0.12, 0.12, 0.12]

point_light_rgb:
  white: [35, 35, 35]
  warm:  [42, 32, 22]
  cool:  [24, 32, 45]

position source:
  derive from camera frame and scene bbox
```

## Validation Before Scaling

For the first 10 samples:

- Verify source and target differ only by lighting.
- Verify point light is in front/side/top as intended, not behind geometry by
  mistake.
- Verify `target_irradiance` is not all black and is aligned with target RGB.
- Verify metallic surfaces respond to light direction.
- Run one local model on the generated samples and inspect source/target/pred.

## Implementation Units

1. `compare/scripts/internscenes_probe.py`
   - Check HF access.
   - List scene names.
   - Estimate required files for a small subset.

2. `compare/scripts/download_internscenes_subset.py`
   - Download selected layouts and only required GLB assets.

3. `compare/scripts/build_internscenes_stage12_package.py`
   - Convert layout JSON and assets into Stage12-compatible scene packages.

4. `compare/scripts/render_internscenes_relighting.py`
   - Generate source/target paired lighting with saved light metadata.

5. `compare/scripts/prepare_internscenes_benchmark.py`
   - Write `manifest.jsonl`, 256/512 copies, and metric-ready folder layout.

## Risks

- Full InternScenes assets are very large. Use sparse download only.
- Some materials may be GLB embedded PBR, not separated albedo/roughness/metallic
  maps. If maps cannot be extracted reliably, use MVInverse/material fallback for
  Stage3 inputs.
- Coordinate systems must be audited. Save both room/world coordinates and
  camera-relative direction for each light.
- Isaac Sim dependency is heavy; avoid it unless the Mitsuba route fails.
