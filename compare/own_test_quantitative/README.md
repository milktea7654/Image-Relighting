# Own Test Quantitative

Split: `test`

Samples: 1029

Stage3 inputs use `composite_optimized`, `render_optimized`, `hybrid_albedo`, `roughness`, `metallic`, and `glass_mask_soft`. Missing HF image columns are written as zero images and recorded in each sample metadata.

External RGBX/LumiNet/IC-Light runners use `source_rgb_512` and target `shading` as `target_irradiance_512`/`reference_rgb_512`; they do not consume the Stage2 point-light JSON directly.
