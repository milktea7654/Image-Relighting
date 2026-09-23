# Compare Setup Status

## Ready

- MIIW quantitative split: `compare/miiw_quantitative`
- MIIW samples: 300 paired samples
- MIIW scoring copies: 256 x 256
- MIIW baseline input copies: 512 x 512
- MIIW original copies: 1500 x 1000 mip2 JPG
- MIIW manifest: `compare/miiw_quantitative/manifest.jsonl`
- Table 1 evaluator: `compare/miiw_quantitative/scripts/evaluate_table1.py`
- Table 1 output: `compare/miiw_quantitative/results/table1.md`
- RGB↔X repo: `compare/rgbx`
- RGB↔X wrappers: `compare/rgbx_rgb2x_batch.py`,
  `compare/rgbx_x2rgb_single.py`
- IC-Light repo: `compare/ic-light`
- IC-Light env: `compare/ic-light/.venv`
- IC-Light weights: `compare/ic-light/models`
- IC-Light RMBG local model: `compare/ic-light/hf_models/RMBG-1.4`
- LumiNet repo: `compare/luminet`

## Needs Method Outputs

Table 1 metrics are not filled until predictions are generated and saved under:

```text
compare/miiw_quantitative/predictions/rgbx/
compare/miiw_quantitative/predictions/luminet/
compare/miiw_quantitative/predictions/ic-light/
compare/miiw_quantitative/predictions/ours-best/
```

Each prediction must be named `<sample_id>.png`.

## Removed From Active Compare Workspace

- Old generated paired dataset folder
- Old output folders
- Inactive download scripts/data targets
- Inactive auxiliary benchmark files
- Temporary input/tool folders

## Commands

```powershell
cd compare\miiw_quantitative
.\.venv\Scripts\python.exe .\scripts\prepare_miiw_pairs.py --count 300 --force
.\.venv\Scripts\python.exe .\scripts\evaluate_table1.py
```

## Notes

- LPIPS is optional in the evaluator. If the `lpips` and compatible PyTorch
  packages are installed in `miiw_quantitative/.venv`, LPIPS will be reported;
  otherwise the LPIPS column stays `-`.
- Runtime is optional and is read from
  `predictions/<method>/runtime.json` when available.
