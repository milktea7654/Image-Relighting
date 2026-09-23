# Project Overview: Stage1 to Stage4 Relighting Pipeline

Date: 2026-06-14

## 一句話目標

這個 project 的目標是做一個 **single-image controllable relighting system**：

給一張真實室內照片和一組目標光源條件，系統先從單張圖重建可渲染的場景與材質，再用物理 renderer 產生新光照下的 intermediate render，最後用學習式模型把 renderer artifact 修掉，輸出更接近真實照片的 relighting result。

整體不是單純 image-to-image，也不是只靠 diffusion 直接改圖，而是：

```text
single RGB image
-> scene / material reconstruction
-> physically based relighting render
-> neural refinement
-> final relit image
```

## 整體流程

```text
Source RGB image
  |
  v
Stage1: Scene and material reconstruction
  - geometry, camera, masks
  - albedo, normal, roughness, metallic, irradiance prior
  |
  v
Stage2: Mitsuba rendering and light optimization
  - render under target light
  - composite foreground render with preserved background
  - save optimized / requested light condition
  |
  v
Stage3: Learned refinement model
  - input: composite, render, hybrid_albedo, roughness, metallic, glass_mask
  - output: photo-real relit prediction
  |
  v
Stage4: End-to-end single-image inference wrapper
  - one request JSON
  - calls Stage1 + Stage2 backbone
  - prepares Stage3 buffers
  - writes final_relit.png
```

## Stage1: Scene And Material Reconstruction

Main folder:

```text
Stage1_ver2
```

Stage1 takes the source photo and converts it into a self-contained scene package that Stage2 can render.

The current Stage1 path uses:

- **MoGe / PixelPerfect-depth style geometry** to estimate depth, camera intrinsics, point cloud, mesh, and foreground/background masks.
- **Ouroboros RGB-to-X predictions** to estimate material and shading-related image maps:
  - `albedo.png`
  - `normal.png`
  - `roughness.png`
  - `metallic.png`
  - `irradiance.png`

Important Stage1 outputs include:

```text
scene_uv.obj
scene.glb
scene.ply
camera.json
reference.png
geometry_mask.png
background.png
background_mask.png
albedo.png
normal.png
roughness.png
metallic.png
irradiance.png
scene_mitsuba.json
optimize_config.json
manifest.json
```

The important design decision is that Stage1 exports an image-space UV mesh when possible. That lets Stage2 use `albedo`, `roughness`, and `metallic` as real bitmap textures in Mitsuba instead of collapsing everything into one average material.

`irradiance.png` should be treated as a weak lighting prior or auxiliary prediction. It is not the exact ground-truth target light.

## Stage2: Physical Rendering And Light Optimization

Main folder:

```text
Stage2
```

Stage2 takes the Stage1 scene package and renders it with Mitsuba. Its role is to give the system physical control over lighting, geometry, shadows, and material response.

Stage2 uses the reconstructed geometry, camera, masks, and material maps, then either estimates or applies light parameters depending on the run mode. The light representation can include environment RGB and point-light style controls.

Important Stage2 outputs include:

```text
render_initial.png
render_optimized.png
render_optimized.exr
composite_optimized.png
residual.png
light/initial_light.json
light/optimized_light.json
light/optimized_point_lights.json
optimization_log.json
manifest.json
encoder_pairs/
renderer_pairs/
```

The two most important images for later stages are:

- `render_optimized.png`: the direct Mitsuba render under the optimized or requested light.
- `composite_optimized.png`: the rendered foreground composited with the preserved image-space background.

This stage is where light position, light color, environment intensity, object geometry, shadows, and metallic/roughness response are supposed to be physically grounded. If the light coordinate is wrong, the whole relighting direction will be wrong before Stage3 ever sees the image.

## Stage3: Neural Refinement Model

Main folder:

```text
Stage3
```

Stage3 is the learned model that turns Stage2's physically controlled but imperfect render into a cleaner, more photo-real image.

The current Stage3 dataset wrapper maps Hugging Face / local fields like this:

```text
reference          -> target image
composite_optimized -> composite input
render_optimized    -> render input
hybrid_albedo       -> albedo input
roughness           -> roughness input
metallic            -> metallic input
glass_mask_soft     -> glass mask input
```

The strongest Stage3 setup is the multi-channel input:

```text
composite
render
hybrid_albedo
roughness
metallic
glass_mask
```

The target is:

```text
reference
```

Stage3's job is not to decide the light from scratch. Stage2 has already provided the relit render and composite under the target light. Stage3 learns the renderer-to-photo correction:

- reduce render noise and geometric artifacts
- restore texture detail
- improve color and contrast
- handle metal-like reflections better through the `metallic` and `roughness` channels
- protect glass/specular regions through `hybrid_albedo` and `glass_mask`

The repo contains multiple Stage3 backbones and checkpoints, including the five model variants stored under:

```text
result
```

Those checkpoints are different learned refiners over the same general Stage3 input concept.

## Stage4: End-To-End Relighting Inference

Main folder:

```text
Stage4
```

Stage4 is the production-style wrapper around the whole pipeline. It should not rewrite Stage1 or Stage2. Its job is to run the full system for one image and one target lighting request.

The main entry is:

```text
Stage4/stage4_relight_single.py
```

A Stage4 request provides fields such as:

```json
{
  "input_image": "D:\\coding\\relighting\\Stage4\\input\\room.jpg",
  "scene_id": "room",
  "output_name": "room",
  "light_mode": "env",
  "env_rgb": [0.2, 0.2, 0.2],
  "point_light_position": [0.0, 0.0, 0.0],
  "point_light_rgb": [40.0, 0.0, 0.0],
  "spp": 32,
  "final_spp": 128,
  "iters": 80,
  "mesh_stride": 2,
  "stage3_checkpoint": "path/to/checkpoint.pt"
}
```

Stage4 performs these steps:

1. Calls the Stage1 + Stage2 backbone under `Stage4/external/Stage12_remote_run`.
2. Produces Stage2 render buffers:
   - `render_optimized.png`
   - `composite_optimized.png`
3. Builds glass-aware buffers with `build_glassaware_single.py`:
   - `hybrid_albedo.png`
   - `glass_mask.png`
   - `glass_mask_soft.png`
4. Writes a Stage3 input JSONL with `make_stage3_input_single.py`.
5. Runs the selected Stage3 checkpoint.
6. Copies the final result to:

```text
Stage4/outputs/<output_name>/final_relit.png
```

So Stage4 is the correct place to run the final system when we want a complete relighting output.

## What The Model Is Actually Given

For our final model, the important point is that Stage3 should receive the same kind of buffers it was trained with:

```text
composite_optimized
render_optimized
hybrid_albedo
roughness
metallic
glass_mask or glass_mask_soft
```

The model should not be fed a random source image in place of `composite_optimized`, and it should not lose the material channels if the checkpoint was trained with them.

The light control should be decided before Stage3:

```text
target light condition
-> Stage2 render / composite
-> Stage3 refinement
```

That means the correctness of `point_light_position`, `point_light_rgb`, and `env_rgb` matters at Stage2/Stage4 time. Stage3 can refine the image, but it cannot reliably fix a target light that was placed in the wrong coordinate frame.

## Why The Pipeline Has Four Stages

Each stage solves a different failure mode:

- Stage1 gives the system geometry and material grounding from a single real image.
- Stage2 gives controllable light transport, shadows, and material response through a renderer.
- Stage3 bridges the gap between imperfect inverse graphics output and real photo appearance.
- Stage4 makes the whole process runnable as one inference pipeline.

This division is important because direct image-to-image relighting often changes appearance without knowing scene geometry, while pure rendering from single-image reconstruction often looks too synthetic. This project combines both sides.

## Training Data Concept

The Stage3 training pair is built around a renderer-to-photo correction task:

```text
input buffers:
  composite_optimized
  render_optimized
  hybrid_albedo
  roughness
  metallic
  glass_mask

target:
  reference
```

The model learns to map physically meaningful but imperfect render buffers back to the real target image distribution.

At inference time, the target light condition creates a new Stage2 render/composite first, and Stage3 refines that into the final relit output.

## Role Of `compare`

The `compare` folder is for evaluation and benchmark integration. It is not the core project pipeline.

It contains scripts and outputs for comparing our model against external baselines such as RGB-X, LumiNet, IC-Light, and OmniGen on datasets like MIIW, ISR, RSR, and our own test split.

Those experiments are useful for quantitative tables, but the project itself is defined by the Stage1 to Stage4 relighting system above.

## Important Notes

- `256 x 256` is the current Stage3 model input/output resolution. It does not mean every external benchmark or original dataset image should be treated as inherently 256 x 256.
- `irradiance.png` from Stage1 is an estimated auxiliary map, not exact target lighting ground truth.
- For MIIW-like data, exact physical xyz lights may not be available. If xyz is estimated from probes, the benchmark should be labeled as estimated-control rather than exact point-light control.
- Metallic and glass behavior depends heavily on keeping `metallic`, `roughness`, `hybrid_albedo`, and `glass_mask` aligned with the checkpoint's expected input.
- The final inference path should use Stage4 with the intended checkpoint from `result` or the configured Stage4 checkpoint path.
