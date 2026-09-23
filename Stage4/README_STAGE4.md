# Stage4 — Single-scene relighting pipeline

## Goal

Given one input photo and a desired lighting condition, Stage4 should:

1. reconstruct the scene / materials
2. render the scene under the desired light condition
3. run Stage3 refinement to produce the final relit image

## Recommended design

**Best starting point:** reuse **Stage12_remote_run** as the Stage1+Stage2 backbone.

Why:
- it already does scene construction
- it already emits Mitsuba-ready assets
- it already handles Stage2 rendering / optimization
- Stage4 is basically **Stage12 inference for one scene + Stage3 refinement**

So Stage4 should be a **wrapper/orchestrator**, not a full rewrite.

## What to place in the Stage4 folder

Suggested layout:

D:\relighting\Stage4
├── stage4_relight_single.py
├── build_glassaware_single.py
├── make_stage3_input_single.py
├── relight_request_example.json
├── README_STAGE4.md
├── checkpoints\stage3\checkpoint_final.pt
├── external\Stage12_remote_run\...
├── external\mvinverse\...
├── external\pixel-perfect-depth\...
├── tools\infer_stage3_relight_unlabeled.py
├── tools\stage3_unet.py
├── inputs\
├── work\
└── outputs\

## What you still need

You already copied:
- venv
- mvinverse
- pixel-perfect-depth
- infer_stage3_relight_unlabeled.py
- stage3_unet.py

You should also copy:

### From Stage12_remote_run
- stage12_stream_worker_fast.py
- any python modules imported by it
- the helper files it depends on for:
  - scene construction
  - Mitsuba packaging
  - Stage2 rendering / optimization

### From Stage3/tools
- your best Stage3 checkpoint
  - recommended path:
    D:\relighting\Stage4\checkpoints\stage3\checkpoint_final.pt

### Runtime libraries
Make sure the Stage4 env has:
- torch
- torchvision
- numpy
- pillow
- mitsuba
- drjit
- opencv-python
- open3d (if Stage1 helpers need it)
- any packages needed by MV-Inverse / PixelPerfect
- any packages needed by stage12_stream_worker_fast.py

## Practical architecture

### Step A — build scene
Input:
- source photo

Output:
- scene mesh / PLY / GLB
- camera.json
- albedo / roughness / metallic
- optional normals
- background / source image references

Borrow this from Stage12 Stage1 code.

### Step B — relight render
Input:
- scene package
- target light specification

Output:
- render_optimized.png
- composite_optimized.png

Borrow this from Stage12 Stage2 code.

### Step C — build glass-aware buffers
Input:
- source photo
- albedo

Output:
- hybrid_albedo.png
- glass_mask.png
- glass_mask_soft.png

### Step D — Stage3 refinement
Input:
- composite
- render
- hybrid_albedo
- roughness
- metallic
- glass_mask_soft or glass_mask

Output:
- final relit image

## Recommendation

Do **not** rewrite Stage1 and Stage2 from scratch in Stage4.

Instead:
- keep Stage12 code as the engine
- let Stage4 be the single-image inference wrapper
