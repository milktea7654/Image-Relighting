"""
MoGe Scene Reconstruction Pipeline  –  main.py
==========================================

Directory layout expected:
  ./
  ├── main.py                   ← this file
  ├── asset/
  │   ├── input/                ← put your source images here
  │   ├── pointcloud/           ← intermediate: .ply files
  │   ├── mesh/                 ← intermediate: .obj mesh
  │   ├── materials/            ← intermediate: PBR maps from Ouroboros
  │   └── output/               ← final .glb files
  ├── MoGe/
  └── Ouroboros/

Usage
-----
  # Run on all images in asset/input/
  python main.py

  # Run on a specific image
  python main.py --image asset/input/photo.jpg

  # Skip a branch if you already have outputs from a previous run
  python main.py --skip_depth
  python main.py --skip_ouro

  # Default uses fast MoGe mesh.glb
  python main.py

  # Optional old Poisson fallback from MoGe pointcloud.ply
  python main.py --use_poisson --mesh_depth 8
"""

import argparse
import os
import sys
import shutil
import subprocess
from pathlib import Path


# ──────────────────────────────────────────────────────────────────
# Paths  (all relative to the project root, i.e. where main.py lives)
# ──────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).parent.resolve()

MOGE_DIR   = ROOT / "MoGe"
OURO_DIR   = ROOT / "Ouroboros"

ASSET_DIR  = ROOT / "asset"
INPUT_DIR  = ASSET_DIR / "input"
PCD_DIR    = ASSET_DIR / "pointcloud"
MESH_DIR   = ASSET_DIR / "mesh"
MAT_DIR    = ASSET_DIR / "materials"
OUTPUT_DIR = ASSET_DIR / "output"
BG_DIR     = ASSET_DIR / "background"
MOGE_RAW_DIR = ASSET_DIR / "moge_raw"

# Ouroboros checkpoint  (HuggingFace repo ID or a local path)
OURO_CKPT  = "Y-Research-Group/Ouroboros"

# Supported input extensions
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Ouroboros modalities to extract
MODALITIES = ["normals", "albedo", "irradiance", "roughness", "metallicity"]


# ──────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────

def ensure_dirs():
    for d in (INPUT_DIR, PCD_DIR, MESH_DIR, MAT_DIR, BG_DIR, OUTPUT_DIR, MOGE_RAW_DIR):
        d.mkdir(parents=True, exist_ok=True)


def run(cmd: list, cwd: Path = None):
    """Run a subprocess, inheriting stdout/stderr so progress is visible."""
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"\n▶  {cmd_str}")
    print("─" * 70)
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"          # skip HF Hub network checks (model is cached)
    env["OPENCV_IO_ENABLE_OPENEXR"] = "1" # suppress OpenEXR codec warning
    result = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env)
    if result.returncode != 0:
        sys.exit(f"\n✗  Command failed (exit {result.returncode})")


def find_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMG_EXTS)


def header(title: str):
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


# ──────────────────────────────────────────────────────────────────
# Step 1  –  MoGe  →  point cloud + image-grid mesh
# ──────────────────────────────────────────────────────────────────

def write_camera_metadata_from_moge(image: Path, moge_dir: Path, dst_meta: Path):
    """
    MoGe saves fov.json, but the rest of this pipeline wants camera intrinsics.
    Convert fov_x/fov_y + saved image size into a PixelPerfect-compatible camera.json.
    """
    import json
    import math
    from PIL import Image as PILImage

    fov_path = moge_dir / "fov.json"
    image_path = moge_dir / "image.jpg"
    if not fov_path.exists():
        print(f"⚠ MoGe fov.json not found: {fov_path}")
        return

    if image_path.exists():
        with PILImage.open(image_path) as im:
            resize_W, resize_H = im.size
    else:
        with PILImage.open(image) as im:
            resize_W, resize_H = im.size

    with open(fov_path, "r", encoding="utf-8") as f:
        fov = json.load(f)

    fov_x = math.radians(float(fov["fov_x"]))
    fov_y = math.radians(float(fov.get("fov_y", fov["fov_x"])))

    fx = resize_W / (2.0 * math.tan(fov_x / 2.0))
    fy = resize_H / (2.0 * math.tan(fov_y / 2.0))
    cx = resize_W / 2.0
    cy = resize_H / 2.0

    meta = {
        "source": "MoGe",
        "intrinsic": [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        "resize_W": int(resize_W),
        "resize_H": int(resize_H),
        "saved_ply_flip": [1.0, -1.0, -1.0],
        "coordinate_note": "MoGe mesh export uses OpenGL-style coordinates after [1,-1,-1] flip.",
        "fov_x": float(fov["fov_x"]),
        "fov_y": float(fov.get("fov_y", fov["fov_x"])),
    }
    with open(dst_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)



def make_background_outputs_from_moge(image: Path, moge_dir: Path) -> dict[str, Path]:
    """
    Create background assets for stage 2 from MoGe's depth/mask outputs.

    Output files:
      asset/background/<stem>_background.png
      asset/background/<stem>_geometry_mask.png
      asset/background/<stem>_background_mask.png
      asset/background/<stem>_background_meta.json

    Meaning:
      - geometry_mask: pixels rendered by the foreground/depth mesh
      - background_mask: far / invalid pixels to preserve as image-space background
      - background.png: original RGB with non-background pixels blacked out
    """
    import json
    import numpy as np
    import cv2
    from PIL import Image as PILImage

    stem = image.stem
    BG_DIR.mkdir(parents=True, exist_ok=True)

    # MoGe infer writes image.jpg resized/cropped to the same space as depth/mask.
    image_src = moge_dir / "image.jpg"
    if image_src.exists():
        rgb = np.asarray(PILImage.open(image_src).convert("RGB"))
    else:
        rgb = np.asarray(PILImage.open(image).convert("RGB"))

    H, W = rgb.shape[:2]

    # MoGe mask marks valid reconstructed pixels.
    mask_path = moge_dir / "mask.png"
    if mask_path.exists():
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)
        valid_mask = mask_img > 127
    else:
        valid_mask = np.ones((H, W), dtype=bool)

    # Prefer depth.npy if present, otherwise EXR if OpenCV supports it.
    depth = None
    depth_npy = moge_dir / "depth.npy"
    depth_exr = moge_dir / "depth.exr"
    if depth_npy.exists():
        depth = np.load(depth_npy).astype(np.float32)
    elif depth_exr.exists():
        try:
            depth = cv2.imread(str(depth_exr), cv2.IMREAD_UNCHANGED)
            if depth is not None:
                if depth.ndim == 3:
                    depth = depth[:, :, 0]
                depth = cv2.resize(depth.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        except Exception:
            depth = None

    background_percentile = float(getattr(make_background_outputs_from_moge, "background_percentile", 0.98))
    max_geometry_depth = float(getattr(make_background_outputs_from_moge, "max_geometry_depth", 100.0))

    if depth is not None:
        depth = cv2.resize(depth.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        finite = np.isfinite(depth) & (depth > 0) & valid_mask
        if np.any(finite):
            percentile_cut = float(np.quantile(depth[finite], background_percentile))
        else:
            percentile_cut = float("inf")

        cutoff = percentile_cut
        if max_geometry_depth > 0:
            cutoff = min(cutoff, max_geometry_depth)

        geometry_mask = valid_mask & np.isfinite(depth) & (depth > 0) & (depth <= cutoff)
        background_mask = ~geometry_mask
        meta = {
            "source": "MoGe",
            "mode": "depth_percentile_background_split",
            "background_percentile": background_percentile,
            "max_geometry_depth": max_geometry_depth,
            "depth_cutoff": cutoff,
            "has_depth": True,
            "note": "background_mask is preserved as image-space background for stage 2 compositing/loss.",
        }
    else:
        # Fallback: invalid MoGe pixels become background.
        geometry_mask = valid_mask
        background_mask = ~geometry_mask
        meta = {
            "source": "MoGe",
            "mode": "mask_only_background_split",
            "has_depth": False,
            "note": "depth.exr was unavailable/unreadable; only MoGe valid mask was used.",
        }

    # Clean masks a little to avoid pinholes.
    geometry_u8 = (geometry_mask.astype(np.uint8) * 255)
    background_u8 = (background_mask.astype(np.uint8) * 255)

    bg = rgb.copy()
    bg[~background_mask] = 0

    bg_path = BG_DIR / f"{stem}_background.png"
    geom_path = BG_DIR / f"{stem}_geometry_mask.png"
    bgmask_path = BG_DIR / f"{stem}_background_mask.png"
    meta_path = BG_DIR / f"{stem}_background_meta.json"

    PILImage.fromarray(bg).save(bg_path)
    cv2.imwrite(str(geom_path), geometry_u8)
    cv2.imwrite(str(bgmask_path), background_u8)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"✔  Background split → {BG_DIR.relative_to(ROOT)}")
    return {
        "background": bg_path,
        "geometry_mask": geom_path,
        "background_mask": bgmask_path,
        "background_meta": meta_path,
    }


def run_moge_infer(image: Path, args) -> Path:
    """
    Calls official MoGe inference from ./MoGe.

    MoGe writes:
      asset/moge_raw/<stem>/
        image.jpg
        depth.exr
        points.exr
        mask.png
        normal.png
        fov.json
        mesh.glb
        pointcloud.ply
    """
    header(f"Step 1 / 4  –  MoGe depth / mesh   [{image.name}]")

    out_root = MOGE_RAW_DIR
    out_root.mkdir(parents=True, exist_ok=True)

    infer_script = MOGE_DIR / "moge" / "scripts" / "infer.py"
    if not infer_script.exists():
        sys.exit(
            f"\n✗  MoGe inference script not found at:\n"
            f"   {infer_script}\n\n"
            f"   Your project layout should contain:\n"
            f"   {MOGE_DIR.relative_to(ROOT)}\\moge\\scripts\\infer.py"
        )

    cmd = [
        sys.executable,
        infer_script,
        "--input", image,
        "--output", out_root,
        "--version", "v2",
        "--device", getattr(args, "moge_device", "cuda"),
        "--resolution_level", str(getattr(args, "moge_resolution_level", 9)),
        "--threshold", str(getattr(args, "moge_threshold", 0.04)),
        "--maps",
        "--glb",
        "--ply",
    ]

    resize = getattr(args, "moge_resize", None)
    if resize and int(resize) > 0:
        cmd += ["--resize", str(int(resize))]

    pretrained = getattr(args, "moge_pretrained", None)
    if pretrained:
        cmd += ["--pretrained", pretrained]

    if getattr(args, "moge_fp16", False):
        cmd += ["--fp16"]

    run(cmd, cwd=MOGE_DIR)
    moge_out = out_root / image.stem
    make_background_outputs_from_moge.background_percentile = getattr(args, "background_percentile", 0.98)
    make_background_outputs_from_moge.max_geometry_depth = getattr(args, "max_geometry_depth", 100.0)
    make_background_outputs_from_moge(image, moge_out)
    return moge_out


def copy_moge_outputs_for_image(image: Path) -> tuple[Path, Path | None]:
    """Copy MoGe outputs into asset/pointcloud and asset/mesh."""
    src_dir = MOGE_RAW_DIR / image.stem
    src_pcd = src_dir / "pointcloud.ply"
    src_mesh_glb = src_dir / "mesh.glb"
    src_mesh_ply = src_dir / "mesh.ply"
    dst_pcd = PCD_DIR / f"{image.stem}.ply"
    dst_mesh_glb = MESH_DIR / f"{image.stem}.glb"
    dst_mesh_ply = MESH_DIR / f"{image.stem}.ply"
    dst_meta = PCD_DIR / f"{image.stem}_camera.json"

    if not src_pcd.exists():
        sys.exit(
            f"\n✗  Expected MoGe point cloud not found:\n"
            f"   {src_pcd}\n"
            f"   Check the MoGe output above for errors."
        )

    shutil.copy2(src_pcd, dst_pcd)
    print(f"\n✔  MoGe point cloud → {dst_pcd.relative_to(ROOT)}")

    fast_mesh: Path | None = None
    if src_mesh_glb.exists():
        shutil.copy2(src_mesh_glb, dst_mesh_glb)
        fast_mesh = dst_mesh_glb
        print(f"✔  MoGe mesh GLB → {dst_mesh_glb.relative_to(ROOT)}")
    elif src_mesh_ply.exists():
        shutil.copy2(src_mesh_ply, dst_mesh_ply)
        fast_mesh = dst_mesh_ply
        print(f"✔  MoGe mesh PLY → {dst_mesh_ply.relative_to(ROOT)}")
    else:
        print(f"⚠  MoGe mesh not found: {src_mesh_glb}")

    write_camera_metadata_from_moge(image, src_dir, dst_meta)
    if dst_meta.exists():
        print(f"✔  Camera metadata → {dst_meta.relative_to(ROOT)}")

    return dst_pcd, fast_mesh


def run_depth(image: Path, args=None) -> tuple[Path, Path | None]:
    """MoGe replacement for the old MoGe depth step."""
    run_moge_infer(image, args)
    return copy_moge_outputs_for_image(image)


def write_image_list(images: list[Path], name: str) -> Path:
    """Write absolute image paths to a temporary txt file. Kept for Ouroboros batch mode."""
    list_path = ASSET_DIR / f"_{name}_image_list.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for img in images:
            f.write(str(img.resolve()) + "\n")
    return list_path


def copy_background_outputs_for_image(image: Path) -> dict[str, Path]:
    """Return already-created MoGe background split files from asset/background/."""
    copied: dict[str, Path] = {}
    patterns = {
        "background": f"{image.stem}_background.png",
        "geometry_mask": f"{image.stem}_geometry_mask.png",
        "background_mask": f"{image.stem}_background_mask.png",
        "background_meta": f"{image.stem}_background_meta.json",
    }
    for key, name in patterns.items():
        path = BG_DIR / name
        if path.exists():
            copied[key] = path
        else:
            print(f"⚠ Background split file not found: {path}")
    return copied



def run_depth_batch(
    images: list[Path],
    mesh_stride: int = 1,
    max_depth_jump: float = 0.08,
    relative_depth_jump: float = 0.08,
    max_geometry_depth: float = 100.0,
    background_percentile: float = 0.98,
    args=None,
) -> dict[str, tuple[Path, Path | None]]:
    """
    Batch wrapper. MoGe can accept a folder, but running per image makes output
    copying deterministic for arbitrary --image paths and keeps logs simple.
    """
    header(f"Step 1 / 4 – MoGe depth / mesh  [{len(images)} image(s)]")
    outputs: dict[str, tuple[Path, Path | None]] = {}
    for image in images:
        run_moge_infer(image, args)
        ply, fast_mesh = copy_moge_outputs_for_image(image)
        outputs[image.stem] = (ply, fast_mesh)
    return outputs


# ──────────────────────────────────────────────────────────────────
# Step 2  –  Point cloud  →  mesh  (Poisson surface reconstruction)
# ──────────────────────────────────────────────────────────────────

def run_meshing(ply: Path, depth: int = 9, normal_map: Path | None = None) -> Path:
    header(f"Step 2 / 4  –  Point Cloud → Mesh   [{ply.name}]  depth={depth}")

    import open3d as o3d
    import numpy as np

    pcd = o3d.io.read_point_cloud(str(ply))
    print(f"   Loaded {len(pcd.points):,} points")

    # Estimate normals when missing
    if not pcd.has_normals():
        print("   Estimating normals…")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(100)

    print("   Running Poisson surface reconstruction…")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    # Trim low-density floater vertices (bottom 2%)
    densities = np.asarray(densities)
    keep = densities > np.quantile(densities, 0.02)
    mesh.remove_vertices_by_mask(~keep)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    print(f"   Mesh: {len(mesh.vertices):,} vertices  {len(mesh.triangles):,} faces")

    # Transfer vertex colours from the point cloud (nearest-neighbour)
    if pcd.has_colors():
        tree = o3d.geometry.KDTreeFlann(pcd)
        verts = np.asarray(mesh.vertices)
        colors = np.zeros((len(verts), 3))
        for i, v in enumerate(verts):
            _, idx, _ = tree.search_knn_vector_3d(v, 1)
            colors[i] = np.asarray(pcd.colors)[idx[0]]
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # ── Override vertex normals from Ouroboros normal map ────────────────────
    # The Poisson mesh normals are derived from the coarse geometry, which
    # causes staircase shading. The Ouroboros normal map encodes the true
    # high-frequency surface detail. We project each vertex to image-space,
    # sample the normal map, decode RGB→[-1,1] normals, and write them back.
    # This makes the blocky mesh shade smoothly without changing its geometry.
    if normal_map and normal_map.exists():
        print("   Applying Ouroboros normal map to vertex normals…")
        from PIL import Image as PILImage
        nimg = PILImage.open(normal_map).convert("RGB")
        W, H = nimg.size
        nmap = np.array(nimg).astype(np.float32) / 255.0  # (H, W, 3) in [0,1]

        verts = np.asarray(mesh.vertices)

        # Planar projection: same mapping used for albedo in run_combine
        x = verts[:, 0]
        y = verts[:, 1]
        x_n = (x - x.min()) / (x.max() - x.min() + 1e-9)
        y_n = (y - y.min()) / (y.max() - y.min() + 1e-9)
        px = np.clip((x_n * (W - 1)).astype(int), 0, W - 1)
        py = np.clip(((1 - y_n) * (H - 1)).astype(int), 0, H - 1)  # flip Y

        # Decode: RGB [0,1] → normal [-1,1]  (standard tangent-space convention)
        sampled = nmap[py, px]                   # (N, 3)
        normals = sampled * 2.0 - 1.0            # (N, 3) in [-1, 1]

        # Normalise (guard against zero vectors from flat/clipped regions)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths = np.where(lengths < 1e-6, 1.0, lengths)
        normals = normals / lengths

        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
        print(f"   Normal map applied  ({W}×{H}  →  {len(verts):,} vertex normals)")
    else:
        # Fall back to smooth geometry-based normals (better than face normals)
        mesh.compute_vertex_normals()
        print("   Using geometry-based smooth vertex normals (no normal map supplied)")

    # Save as PLY — OBJ does not support vertex colors, they would be silently
    # dropped on load. PLY preserves them natively.
    mesh_path = MESH_DIR / f"{ply.stem}.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_vertex_colors=True)
    print(f"\n✔  Mesh  →  {mesh_path.relative_to(ROOT)}")
    return mesh_path


# ──────────────────────────────────────────────────────────────────
# Step 3  –  Ouroboros rgb2x  →  PBR material maps
# ──────────────────────────────────────────────────────────────────

def run_ouroboros(image: Path) -> dict[str, Path]:
    """
    Calls Ouroboros/rgb2x/inference.py from inside the rgb2x/ subfolder,
    exactly mirroring the original shell script:
      cd Ouroboros/rgb2x && python inference.py ...
    This ensures all internal relative imports and model_cache paths resolve correctly.
    Maps are saved to asset/materials/<stem>/<modality>.png
    """
    header(f"Step 3 / 4  –  Ouroboros rgb2x   [{image.name}]")

    out_dir = MAT_DIR / image.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "inference.py",          # relative — cwd is rgb2x/
            "--checkpoint",     OURO_CKPT,
            "--modality",       *MODALITIES,
            "--condition",      "rgb",
            "--noise",          "gaussian",
            "--seed",           "0",
            "--input_rgb_path", image,
            "--output_dir",     out_dir,
        ],
        cwd=OURO_DIR / "rgb2x",     # ← run from inside rgb2x/, same as the shell script
    )

    # ── Discover output files by scanning the entire out_dir tree ──────────
    # Ouroboros output layout varies between versions; rather than guessing,
    # we walk every .png under out_dir and match by modality keyword in the name.
    all_pngs = list(out_dir.rglob("*.png"))

    if not all_pngs:
        # Nothing found at all — print the full tree so the user can investigate
        print(f"   ✗  No PNG files found under {out_dir}")
        print(f"   Scanning full output dir tree:")
        for p in sorted(out_dir.rglob("*")):
            print(f"      {p.relative_to(out_dir)}")
    else:
        print(f"   Found {len(all_pngs)} PNG(s) under {out_dir.relative_to(ROOT)}:")
        for p in sorted(all_pngs):
            print(f"      {p.relative_to(out_dir)}")

    maps: dict[str, Path] = {}
    for mod in MODALITIES:
        # Match: filename contains the modality name (case-insensitive)
        matches = [p for p in all_pngs if mod.lower() in p.name.lower()]
        if matches:
            maps[mod] = matches[0]   # take first match if multiple
        else:
            print(f"   ⚠  Map not found for '{mod}' – will be omitted from GLB")

    print(f"\n✔  Material maps: {', '.join(maps.keys())}")
    return maps



def collect_ouro_maps_for_stem(stem: str) -> dict[str, Path]:
    """Find Ouroboros maps under asset/materials/<stem>/ recursively."""
    stem_mat_dir = MAT_DIR / stem
    all_pngs = list(stem_mat_dir.rglob("*.png")) if stem_mat_dir.exists() else []
    maps: dict[str, Path] = {}
    for mod in MODALITIES:
        matches = [p for p in all_pngs if mod.lower() in p.name.lower()]
        if matches:
            maps[mod] = matches[0]
    return maps


def run_ouroboros_batch(images: list[Path]) -> dict[str, dict[str, Path]]:
    """Batch Ouroboros for all images in one subprocess. Requires patched inference.py."""
    header(f"Step 3 / 4 – Batch Ouroboros rgb2x  [{len(images)} image(s)]")

    list_path = write_image_list(images, "ouro")
    run(
        [
            sys.executable,
            "inference.py",
            "--checkpoint", OURO_CKPT,
            "--modality", *MODALITIES,
            "--condition", "rgb",
            "--noise", "gaussian",
            "--seed", "0",
            "--input_rgb_path", list_path,
            "--output_dir", MAT_DIR,
            "--output_stem_subdirs",
        ],
        cwd=OURO_DIR / "rgb2x",
    )

    outputs: dict[str, dict[str, Path]] = {}
    for image in images:
        maps = collect_ouro_maps_for_stem(image.stem)
        outputs[image.stem] = maps
        if maps:
            print(f"✔ {image.name}: maps → {', '.join(maps)}")
        else:
            print(f"⚠ {image.name}: no Ouroboros maps found")
    return outputs


# ──────────────────────────────────────────────────────────────────
# Step 4  –  Combine mesh + PBR maps  →  .glb
# ──────────────────────────────────────────────────────────────────

def run_combine(mesh_obj: Path, maps: dict[str, Path], reference_image: Path | None = None, args=None) -> Path:
    """
    Combines mesh + Ouroboros AOV maps into a .glb.

    Current stage-1 representation:
      - geometry: PixelPerfect MoGe fast mesh, or optional Poisson mesh
      - base color: Ouroboros albedo projected to vertex COLOR_0
      - shading normals: Ouroboros camera-space normals projected to vertex normals
      - roughness / metallicity / irradiance / normal maps: copied as sidecar files
      - Mitsuba-ready PLY mesh, Mitsuba scene seed, reference/masks, and optimizer config

    Note: the mesh still has no UVs, so roughness/metallic/normal PNGs are not
    embedded as true glTF texture slots yet. They are sidecar assets for stage 2.
    """
    header(f"Step 4 / 4  –  Combine → GLB   [{mesh_obj.name}]")

    try:
        import json
        import trimesh
        import numpy as np
        from PIL import Image as PILImage
    except ImportError:
        sys.exit(
            "\n✗  Missing packages: trimesh / pillow\n"
            "   Fix:  uv add trimesh pillow pyglet"
        )

    # ── Load mesh ────────────────────────────────────────────────────────────
    scene_or_mesh = trimesh.load(str(mesh_obj), process=False)
    if isinstance(scene_or_mesh, trimesh.Scene):
        geoms = list(scene_or_mesh.geometry.values())
        if not geoms:
            sys.exit("✗  No geometry found in the mesh file.")
        mesh = geoms[0]
    else:
        mesh = scene_or_mesh

    print(f"   Mesh: {len(mesh.vertices):,} vertices")

    # ── Load PixelPerfect camera metadata ────────────────────────────────────
    # The fast mesh / point cloud was exported after applying [1,-1,-1].
    # To project image-space Ouroboros maps back to vertices, undo that flip,
    # project with the same intrinsics, sample maps, then re-apply the flip to
    # normals so they live in the exported mesh coordinate system.
    meta_path = PCD_DIR / f"{mesh_obj.stem}_camera.json"
    if not meta_path.exists():
        print(f"   ⚠  Camera metadata not found: {meta_path}")
        print("   ⚠  Cannot camera-project Ouro albedo/normals; keeping mesh colors/normals as-is")
    else:
        with open(meta_path, "r") as f:
            meta = json.load(f)

        K = np.asarray(meta["intrinsic"], dtype=np.float32)
        resize_W = int(meta["resize_W"])
        resize_H = int(meta["resize_H"])
        flip = np.asarray(meta.get("saved_ply_flip", [1.0, -1.0, -1.0]), dtype=np.float32)

        verts_saved = np.asarray(mesh.vertices, dtype=np.float32)
        verts_cam = verts_saved / flip[None, :]  # undo saved PLY flip

        x = verts_cam[:, 0]
        y = verts_cam[:, 1]
        z = verts_cam[:, 2]
        valid = z > 1e-6

        u = K[0, 0] * x / z + K[0, 2]
        v = K[1, 1] * y / z + K[1, 2]

        inside = (
            valid &
            (u >= 0) & (u <= resize_W - 1) &
            (v >= 0) & (v <= resize_H - 1)
        )

        px = np.clip(np.round(u).astype(np.int32), 0, resize_W - 1)
        py = np.clip(np.round(v).astype(np.int32), 0, resize_H - 1)

        # ── Image-space UVs for MoGe single-view mesh ────────────────────────
        # Because the mesh is reconstructed from the input view, the correct
        # first UV unwrap is just the camera projection:
        #   U = projected_pixel_x / image_width
        #   V = 1 - projected_pixel_y / image_height
        # Store it on the mesh object so we can export scene_uv.obj later.
        uv = np.zeros((len(verts_saved), 2), dtype=np.float32)
        uv[:, 0] = np.clip(u / max(1, resize_W - 1), 0.0, 1.0)
        uv[:, 1] = np.clip(1.0 - (v / max(1, resize_H - 1)), 0.0, 1.0)
        mesh.metadata["image_space_uv"] = uv

        # ── Override vertex colors from Ouroboros albedo ─────────────────────
        if maps.get("albedo") and maps["albedo"].exists():
            print("   Projecting Ouroboros albedo to vertex colors…")

            albedo_img = PILImage.open(maps["albedo"]).convert("RGB")
            albedo_img = albedo_img.resize((resize_W, resize_H), PILImage.BILINEAR)
            alb = np.asarray(albedo_img, dtype=np.uint8)

            vc = np.zeros((len(verts_saved), 3), dtype=np.uint8)
            vc[inside] = alb[py[inside], px[inside]]

            # Preserve old colors for vertices that project outside the image.
            try:
                old_vc = mesh.visual.vertex_colors
                if old_vc is not None and len(old_vc) == len(verts_saved):
                    vc[~inside] = np.asarray(old_vc)[~inside, :3]
            except Exception:
                pass

            alpha = np.full((len(vc), 1), 255, dtype=np.uint8)
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh,
                vertex_colors=np.concatenate([vc, alpha], axis=1)
            )
            print(f"   ✔ using Ouro albedo as COLOR_0 ({inside.sum():,}/{len(verts_saved):,} vertices projected)")
        else:
            print("   ⚠  No Ouro albedo found; keeping existing vertex colors")

        # ── Override vertex normals from Ouroboros normals ───────────────────
        # Ouroboros prompt says camera-space normals. Decode RGB [0,1] -> [-1,1],
        # sample by the same camera projection used for albedo, normalize, then
        # apply the PixelPerfect export flip so normals match the GLB mesh axes.
        if maps.get("normals") and maps["normals"].exists():
            print("   Projecting Ouroboros normal map to vertex normals…")

            nimg = PILImage.open(maps["normals"]).convert("RGB")
            nimg = nimg.resize((resize_W, resize_H), PILImage.BILINEAR)
            nmap = np.asarray(nimg, dtype=np.float32) / 255.0

            sampled = nmap[py[inside], px[inside]]
            sampled = sampled * 2.0 - 1.0

            normals = np.zeros((len(verts_saved), 3), dtype=np.float32)
            normals[inside] = sampled

            # For unprojected vertices, fall back to geometry normals if available.
            try:
                geom_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
                if geom_normals.shape == normals.shape:
                    normals[~inside] = geom_normals[~inside]
            except Exception:
                pass

            # Match exported coordinate convention.
            normals *= flip[None, :]

            lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            bad = lengths[:, 0] < 1e-8
            if np.any(bad):
                normals[bad] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                lengths[bad] = 1.0
            normals = normals / lengths

            # Trimesh may not expose a normal setter in every version, but glTF
            # export reads mesh.vertex_normals. Setting both paths makes this robust.
            try:
                mesh.vertex_normals = normals
            except Exception:
                pass
            try:
                mesh._cache["vertex_normals"] = normals
            except Exception:
                pass

            print(f"   ✔ using Ouro normals for smooth shading ({inside.sum():,}/{len(verts_saved):,} vertices projected)")
        else:
            print("   ⚠  No Ouro normals found; keeping geometry normals")

    # ── Export self-contained stage-1 package ────────────────────────────────
    # Each image gets its own folder:
    #   asset/output/<stem>/
    #     scene.glb
    #     camera.json
    #     manifest.json
    #     albedo.png / roughness.png / metallic.png / normal.png / irradiance.png
    #     background.png / geometry_mask.png / background_mask.png / background_meta.json
    package_dir = OUTPUT_DIR / mesh_obj.stem
    package_dir.mkdir(parents=True, exist_ok=True)

    glb_path = package_dir / "scene.glb"
    mesh.export(str(glb_path))
    size_mb = glb_path.stat().st_size / 1024 / 1024
    print(f"\n✔  GLB  →  {glb_path.relative_to(ROOT)}  ({size_mb:.1f} MB)")

    # Also export a UV OBJ. This is the important path for bitmap roughness /
    # metallic / albedo in Mitsuba, because PLY is reliable for vertex colors
    # but not a good carrier for ordinary image texture coordinates.
    uv_obj_path = None
    try:
        uv = mesh.metadata.get("image_space_uv")
        if uv is not None and maps.get("albedo") and maps["albedo"].exists():
            tex_img = PILImage.open(maps["albedo"]).convert("RGB")
            tex_mesh = mesh.copy()
            tex_mesh.visual = trimesh.visual.TextureVisuals(
                uv=uv,
                image=tex_img,
                material=trimesh.visual.material.SimpleMaterial(
                    image=tex_img,
                    diffuse=[255, 255, 255, 255],
                ),
            )
            uv_obj_path = package_dir / "scene_uv.obj"
            tex_mesh.export(str(uv_obj_path))
            print(f"   UV textured OBJ → {uv_obj_path.relative_to(ROOT)}")
    except Exception as e:
        print(f"   ⚠ Could not export UV textured OBJ: {e}")
        uv_obj_path = None

    # Mitsuba prefers its mesh loaders (PLY/OBJ/serialized) for stage-2 rendering.
    # Keep GLB for Blender/preview, and export PLY for Mitsuba.
    ply_path = package_dir / "scene.ply"
    try:
        mesh.export(str(ply_path))
        print(f"   Mitsuba mesh → {ply_path.relative_to(ROOT)}")
    except Exception as e:
        print(f"   ⚠ Could not export scene.ply for Mitsuba: {e}")
        ply_path = None

    # ── Copy camera metadata beside GLB ──────────────────────────────────────
    camera_file = None
    cam_src = PCD_DIR / f"{mesh_obj.stem}_camera.json"
    if cam_src.exists():
        cam_dst = package_dir / "camera.json"
        shutil.copy2(cam_src, cam_dst)
        camera_file = cam_dst.name
        print(f"   Camera metadata → {cam_dst.relative_to(ROOT)}")
    else:
        print(f"   ⚠  Camera metadata not found: {cam_src}")

    # ── Copy original input image as reference.png for render/loss target ─────
    reference_file = None
    ref_src = reference_image if reference_image and reference_image.exists() else None
    if ref_src is None:
        for ext in IMG_EXTS:
            candidate = INPUT_DIR / f"{mesh_obj.stem}{ext}"
            if candidate.exists():
                ref_src = candidate
                break
    if ref_src is not None:
        try:
            ref_img = PILImage.open(ref_src).convert("RGB")
            ref_dst = package_dir / "reference.png"
            ref_img.save(ref_dst)
            reference_file = ref_dst.name
            print(f"   Reference image → {ref_dst.relative_to(ROOT)}")
        except Exception as e:
            print(f"   ⚠ Could not copy reference image {ref_src}: {e}")
    else:
        print("   ⚠ Reference image not found; stage-2 loss target will be missing")

    # ── Copy PBR sidecar maps into package ───────────────────────────────────
    sidecars = {}
    sidecar_name = {
        "albedo": "albedo",
        "roughness": "roughness",
        "metallicity": "metallic",
        "normals": "normal",
        "irradiance": "irradiance",
    }

    for mod, src in maps.items():
        if not src.exists():
            continue
        clean = sidecar_name.get(mod, mod)
        dst = package_dir / f"{clean}.png"
        shutil.copy2(src, dst)
        sidecars[clean] = dst.name

    # ── Copy PixelPerfect geometry/background split into package ─────────────
    background_files = {}
    bg_source_names = {
        "background_image": f"{mesh_obj.stem}_background.png",
        "geometry_mask": f"{mesh_obj.stem}_geometry_mask.png",
        "background_mask": f"{mesh_obj.stem}_background_mask.png",
        "background_meta": f"{mesh_obj.stem}_background_meta.json",
    }
    bg_output_names = {
        "background_image": "background.png",
        "geometry_mask": "geometry_mask.png",
        "background_mask": "background_mask.png",
        "background_meta": "background_meta.json",
    }
    for key, source_name in bg_source_names.items():
        src = BG_DIR / source_name
        if src.exists():
            dst = package_dir / bg_output_names[key]
            shutil.copy2(src, dst)
            background_files[key] = dst.name

    # ── Mitsuba scene seed + optimization config ─────────────────────────────
    # These are JSON recipes for stage 2. They avoid hard-coding paths later.
    mitsuba_scene_file = None
    optimize_config_file = None

    sensor = {
        "type": "perspective",
        "camera_metadata": camera_file,
        "resolution": None,
        "intrinsics": None,
        "coordinate_note": "Mesh is already exported in pixelperfect_export_flip_[1,-1,-1] coordinates.",
    }
    if camera_file is not None:
        try:
            with open(package_dir / camera_file, "r") as f:
                camera_meta = json.load(f)
            K_meta = camera_meta.get("intrinsic")
            if K_meta is not None:
                sensor["intrinsics"] = {
                    "fx": K_meta[0][0],
                    "fy": K_meta[1][1],
                    "cx": K_meta[0][2],
                    "cy": K_meta[1][2],
                }
            sensor["resolution"] = {
                "width": camera_meta.get("resize_W"),
                "height": camera_meta.get("resize_H"),
            }
            if K_meta is not None and camera_meta.get("resize_W"):
                import math
                fx = float(K_meta[0][0])
                width = float(camera_meta.get("resize_W"))
                if fx > 0:
                    sensor["fov_x_degrees"] = float(2.0 * math.degrees(math.atan(width / (2.0 * fx))))
        except Exception as e:
            print(f"   ⚠ Could not read camera metadata for Mitsuba scene seed: {e}")

    mitsuba_scene = {
        "schema": "mitsuba_stage2_scene_seed_v1",
        "name": mesh_obj.stem,
        "preferred_mesh": "scene_uv.obj" if uv_obj_path is not None else ("scene.ply" if ply_path is not None else "scene.glb"),
                "mesh_shape": {
            "type": "obj" if uv_obj_path is not None else ("ply" if ply_path is not None else "external"),
            "filename": "scene_uv.obj" if uv_obj_path is not None else ("scene.ply" if ply_path is not None else "scene.glb"),
            "face_normals": False,
        },
        "sensor": sensor,
        "integrator": {
            "type": "prb",
            "max_depth": 8,
        },
        "emitter_initialization": {
            "type": "constant",
            "radiance": [1.0, 1.0, 1.0],
            "note": "Use irradiance.png only as a weak lighting prior/initialization, not as ground-truth lighting.",
        },
        "bsdf_initialization": {
            "type": "principled",
            "base_color": "albedo.png via image-space UV, with vertex COLOR_0 fallback",
            "roughness_texture": sidecars.get("roughness"),
            "metallic_texture": sidecars.get("metallic"),
            "normal_sidecar": sidecars.get("normal"),
            "has_uv": bool(uv_obj_path),
            "uv_mesh": "scene_uv.obj" if uv_obj_path is not None else None,
            "note": "Image-space UVs are generated by camera-projecting vertices. This enables bitmap albedo/roughness/metallic in Mitsuba.",
        },
        "targets": {
            "reference": reference_file,
            "geometry_mask": background_files.get("geometry_mask"),
            "background": background_files.get("background_image"),
            "background_mask": background_files.get("background_mask"),
        },
        "coordinate_system": "pixelperfect_export_flip_[1,-1,-1]",
    }
    mitsuba_scene_path = package_dir / "scene_mitsuba.json"
    with open(mitsuba_scene_path, "w") as f:
        json.dump(mitsuba_scene, f, indent=2)
    mitsuba_scene_file = mitsuba_scene_path.name

    optimize_config = {
        "schema": "stage2_optimization_config_v1",
        "loss": {
            "target": reference_file,
            "mask": background_files.get("geometry_mask"),
            "background_composite": background_files.get("background_image"),
            "use_l1": True,
            "use_multiscale": False,
        },
        "optimize": {
            "env_radiance": True,
            "key_light": False,
            "camera_pose": False,
            "mesh_vertices": False,
            "vertex_albedo": False,
            "roughness_scalar_or_texture": False,
            "background": False,
        },
        "initial_values": {
            "env_radiance": [1.0, 1.0, 1.0],
            "samples_per_pixel": 64,
        },
        "notes": [
            "Start by optimizing lighting only; keep geometry/material fixed.",
            "Use geometry_mask for mesh-render loss; composite or ignore background pixels.",
            "irradiance.png is a learned prior and should not be treated as ground truth.",
        ],
    }
    optimize_config_path = package_dir / "optimize_config.json"
    with open(optimize_config_path, "w") as f:
        json.dump(optimize_config, f, indent=2)
    optimize_config_file = optimize_config_path.name

    # ── Manifest for stage 2 / Blender / Mitsuba ─────────────────────────────
    manifest = {
        "schema": "stage1_reconstruction_package_v1",
        "name": mesh_obj.stem,
        "mesh": glb_path.name,
        "uv_mesh": "scene_uv.obj" if uv_obj_path is not None else None,
        "mitsuba_mesh": "scene_uv.obj" if uv_obj_path is not None else ("scene.ply" if ply_path is not None else None),
        "camera_metadata": camera_file,
        "reference_image": reference_file,
        "mitsuba_scene": mitsuba_scene_file,
        "optimize_config": optimize_config_file,
        "albedo_mode": "vertex_color_from_ouro_albedo" if maps.get("albedo") else "mesh_existing_vertex_color",
        "normal_mode": "vertex_normals_from_ouro_normals" if maps.get("normals") else "geometry_normals",
        "has_uv": bool(uv_obj_path),
        "has_background": bool(background_files),
        "background_files": background_files,
        "background_mode": "camera_background_layer_from_far_depth_split" if background_files else "none",
        "sidecar_maps": sidecars,
        "coordinate_system": "pixelperfect_export_flip_[1,-1,-1]",
        "recommended_stage2_use": {
            "render_mesh_blender": "scene.glb",
            "render_mesh_mitsuba": "scene_uv.obj" if uv_obj_path is not None else ("scene.ply" if ply_path is not None else None),
            "base_color": "albedo.png through image-space UV; vertex COLOR_0 fallback",
            "composite_background": background_files.get("background_image"),
            "geometry_mask": background_files.get("geometry_mask"),
            "background_mask": background_files.get("background_mask"),
        },
        "note": "Self-contained package for Blender + Mitsuba stage 2. GLB is for Blender/preview; PLY + scene_mitsuba.json is the preferred Mitsuba entry. Image-space UVs are exported as scene_uv.obj when albedo is available; roughness/metallic PNGs share that same UV layout. Background files define non-geometry far/sky layer for compositing/rendering.",
    }
    manifest_path = package_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if sidecars:
        print(f"   PBR sidecar maps copied → {package_dir.relative_to(ROOT)}/")
    if background_files:
        print(f"   Background/mask files copied → {package_dir.relative_to(ROOT)}/")
    print(f"   Mitsuba scene seed → {(package_dir / mitsuba_scene_file).relative_to(ROOT)}")
    print(f"   Optimization config → {(package_dir / optimize_config_file).relative_to(ROOT)}")
    print(f"   Manifest → {manifest_path.relative_to(ROOT)}")

    return glb_path


# ──────────────────────────────────────────────────────────────────
# Per-image orchestration
# ──────────────────────────────────────────────────────────────────

def process_image(image: Path, args) -> Path:
    print(f"\n\n{'▓' * 70}")
    print(f"  IMAGE :  {image.name}")
    print(f"{'▓' * 70}")

    stem = image.stem

    # ── Step 1: depth → point cloud ─────────────────────────────
    fast_mesh = MESH_DIR / f"{stem}.glb"
    if not fast_mesh.exists():
        fast_mesh = MESH_DIR / f"{stem}.ply"
    if args.skip_depth:
        ply = PCD_DIR / f"{stem}.ply"
        if not ply.exists():
            sys.exit(f"✗  --skip_depth set but {ply.relative_to(ROOT)} not found")
        if not fast_mesh.exists():
            fast_mesh = None
        print(f"⏩  Skipping depth  →  using {ply.relative_to(ROOT)}")
    else:
        ply, fast_mesh = run_depth(image, args)

    # ── Step 3: Ouroboros material maps ──────────────────────────
    # Run BEFORE meshing so the normal map is available to smooth the mesh.
    if args.skip_ouro:
        stem_mat_dir = MAT_DIR / stem
        # Use the same rglob scan as the live path — works regardless of filename convention
        all_pngs = list(stem_mat_dir.rglob("*.png")) if stem_mat_dir.exists() else []
        maps = {}
        for mod in MODALITIES:
            matches = [p for p in all_pngs if mod.lower() in p.name.lower()]
            if matches:
                maps[mod] = matches[0]
        if not maps:
            sys.exit(
                f"✗  --skip_ouro set but no PNGs found in "
                f"{stem_mat_dir.relative_to(ROOT)}\n"
                f"   Files there: {[p.name for p in stem_mat_dir.rglob('*') if p.is_file()] if stem_mat_dir.exists() else 'directory does not exist'}"
            )
        print(f"⏩  Skipping Ouroboros  →  found: {', '.join(maps)}")
    else:
        maps = run_ouroboros(image)

    normal_map = maps.get("normals")   # may be None if Ouroboros skipped / failed

    # ── Step 2: choose mesh ──────────────────────────────────────
    # Default: use the MoGe fast mesh exported by PixelPerfect.
    # Optional: --use_poisson keeps the old slow Poisson reconstruction path.
    if args.skip_mesh:
        mesh_obj = MESH_DIR / f"{stem}.glb"
        if not mesh_obj.exists():
            mesh_obj = MESH_DIR / f"{stem}.ply"
        if not mesh_obj.exists():
            mesh_obj = MESH_DIR / f"{stem}.obj"
        if not mesh_obj.exists():
            sys.exit(f"✗  --skip_mesh set but {(MESH_DIR / stem).relative_to(ROOT)}.[ply|obj] not found")
        print(f"⏩  Skipping meshing  →  using {mesh_obj.relative_to(ROOT)}")
    elif args.use_poisson:
        mesh_obj = run_meshing(ply, depth=args.mesh_depth, normal_map=normal_map)
    else:
        if fast_mesh is None or not fast_mesh.exists():
            sys.exit(
                f"✗  MoGe fast mesh not found for {stem}.\n"
                "   Re-run without --skip_depth, or use --use_poisson as a fallback."
            )
        mesh_obj = fast_mesh
        print(f"⚡  Using MoGe fast mesh → {mesh_obj.relative_to(ROOT)}")

    # ── Step 4: combine → GLB ────────────────────────────────────
    return run_combine(mesh_obj, maps, reference_image=image, args=args)


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scene reconstruction pipeline: image(s) → Blender-ready .glb",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", type=str, default=None, help="Path to one input image. Omit to process every image in asset/input/.")
    parser.add_argument("--mesh_depth", type=int, default=9, help="Poisson reconstruction depth 8–11. Only used with --use_poisson. Default: 9")
    parser.add_argument("--use_poisson", action="store_true", help="Use the old slow Open3D Poisson reconstruction instead of the MoGe fast mesh.")
    parser.add_argument("--mesh_stride", type=int, default=1, help="Use every Nth depth pixel for the fast grid mesh. 1 = full resolution, 2 = lighter preview mesh.")
    parser.add_argument("--max_depth_jump", type=float, default=0.08, help="Minimum absolute depth-jump threshold for fast-mesh triangle filtering.")
    parser.add_argument("--relative_depth_jump", type=float, default=0.08, help="Also allow depth jumps up to this fraction of local depth; useful for tunnels/far geometry.")
    parser.add_argument("--max_geometry_depth", type=float, default=100.0, help="Hard safety cap: pixels farther than this are treated as background, not mesh geometry. Use <=0 to disable.")
    parser.add_argument("--background_percentile", type=float, default=0.98, help="Adaptive far-background cutoff percentile among valid depths. 0.98 removes farthest ~2% as background.")
    parser.add_argument("--no_batch", action="store_true", help="Disable batch inference and use the older per-image subprocess flow.")
    parser.add_argument("--moge_pretrained", type=str, default=None, help="Optional local MoGe checkpoint path or HF repo id. Omit to use MoGe default v2.")
    parser.add_argument("--moge_device", type=str, default="cuda", help="MoGe device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--moge_resolution_level", type=int, default=9, help="MoGe inference resolution level, 0-9. Higher is finer/slower.")
    parser.add_argument("--moge_threshold", type=float, default=0.04, help="MoGe depth-edge removal threshold for mesh export. Smaller removes more edges.")
    parser.add_argument("--moge_resize", type=int, default=0, help="Optional MoGe resize size. 0 keeps original image size.")
    parser.add_argument("--moge_fp16", action="store_true", help="Use fp16 for MoGe inference.")
    parser.add_argument("--skip_depth", action="store_true", help="Skip MoGe; reuse existing .ply from asset/pointcloud/ and mesh from asset/mesh/.")
    parser.add_argument("--skip_mesh", action="store_true", help="Skip mesh selection/reconstruction; reuse existing .ply/.obj from asset/mesh/.")
    parser.add_argument("--skip_ouro", action="store_true", help="Skip Ouroboros; reuse existing maps from asset/materials/.")
    args = parser.parse_args()

    ensure_dirs()

    if args.image:
        images = [Path(args.image).resolve()]
        if not images[0].exists():
            sys.exit(f"✗ Image not found: {images[0]}")
    else:
        images = find_images(INPUT_DIR)
        if not images:
            sys.exit(f"\n✗ No images found in {INPUT_DIR.relative_to(ROOT)}\n  Drop a .jpg or .png there, or pass --image path/to/file.jpg\n")

    print(f"\n  {len(images)} image(s) queued:  " f"{', '.join(i.name for i in images)}")

    use_batch = (not args.no_batch) and len(images) > 1

    if not use_batch:
        results = []
        for img in images:
            glb = process_image(img, args)
            results.append(glb)
    else:
        if args.skip_depth:
            depth_outputs = {}
            for image in images:
                ply = PCD_DIR / f"{image.stem}.ply"
                fast_mesh = MESH_DIR / f"{image.stem}.glb"
                if not fast_mesh.exists():
                    fast_mesh = MESH_DIR / f"{image.stem}.ply"
                if not ply.exists():
                    sys.exit(f"✗ --skip_depth set but {ply.relative_to(ROOT)} not found")
                depth_outputs[image.stem] = (ply, fast_mesh if fast_mesh.exists() else None)
            print("⏩ Skipping batch MoGe depth")
        else:
            depth_outputs = run_depth_batch(
                images,
                mesh_stride=args.mesh_stride,
                max_depth_jump=args.max_depth_jump,
                relative_depth_jump=args.relative_depth_jump,
                max_geometry_depth=args.max_geometry_depth,
                background_percentile=args.background_percentile,
                args=args,
            )

        if args.skip_ouro:
            maps_outputs = {image.stem: collect_ouro_maps_for_stem(image.stem) for image in images}
            print("⏩ Skipping batch Ouroboros")
        else:
            maps_outputs = run_ouroboros_batch(images)

        results = []
        for image in images:
            print(f"\n\n{'▓' * 70}")
            print(f"  COMBINE : {image.name}")
            print(f"{'▓' * 70}")

            ply, fast_mesh = depth_outputs[image.stem]
            maps = maps_outputs.get(image.stem, {})
            normal_map = maps.get("normals")

            if args.skip_mesh:
                mesh_obj = MESH_DIR / f"{image.stem}.glb"
                if not mesh_obj.exists():
                    mesh_obj = MESH_DIR / f"{image.stem}.ply"
                if not mesh_obj.exists():
                    mesh_obj = MESH_DIR / f"{image.stem}.obj"
                if not mesh_obj.exists():
                    sys.exit(f"✗ --skip_mesh set but {(MESH_DIR / image.stem).relative_to(ROOT)}.[ply|obj] not found")
                print(f"⏩ Skipping meshing → using {mesh_obj.relative_to(ROOT)}")
            elif args.use_poisson:
                mesh_obj = run_meshing(ply, depth=args.mesh_depth, normal_map=normal_map)
            else:
                if fast_mesh is None or not fast_mesh.exists():
                    sys.exit(f"✗ MoGe fast mesh not found for {image.stem}.\n  Re-run without --skip_depth, or use --use_poisson as a fallback.")
                mesh_obj = fast_mesh
                print(f"⚡ Using MoGe fast mesh → {mesh_obj.relative_to(ROOT)}")

            results.append(run_combine(mesh_obj, maps, reference_image=image, args=args))

    header("All done")
    for glb in results:
        print(f"  ✅  {glb.relative_to(ROOT)}")
    print()
    print("  Open in Blender:  File → Import → glTF 2.0  (.glb / .gltf)")
    print()


if __name__ == "__main__":
    main()