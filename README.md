# Relighting Pipeline

這是一個以單張室內影像為輸入的可控制 relighting 研究專案。系統先從輸入影像重建可渲染的場景與材質，再以 Mitsuba 在目標光源下重新渲染，最後用學習式 refinement model 修正 renderer 的誤差，輸出 relit image。

```text
single RGB image
    -> scene / material reconstruction
    -> physically based relighting render
    -> neural refinement
    -> final relit image
```

## Pipeline stages

| Path | Purpose | Main entry point |
| --- | --- | --- |
| `Stage1_ver1/` | 目前主要的MVINVERSE + PIXEL PERFECT | `Stage1_ver1/main.py` |
| `Stage1_ver2/` | 舊版 scene reconstruction baseline | `Stage1_ver2/main.py` |
| `Stage2/` | Mitsuba rendering 與 light optimization | `Stage2/main.py` |
| `Stage3/` | refinement model 的訓練、推論與多模型比較 | `Stage3/train_channel_ablation.py`, `Stage3/infer_stage3_run.py` |
| `Stage4/` | 單張影像的 end-to-end relighting wrapper | `Stage4/stage4_relight_single.py` |
| `compare/` | RGB↔X、LumiNet、IC-Light 與本方法的 benchmark 工具 | `compare/README.md` |

更完整的資料流與輸出格式請參考 [`PROJECT_OVERVIEW_STAGE1_TO_STAGE4.md`](PROJECT_OVERVIEW_STAGE1_TO_STAGE4.md)。Stage 4 的設計說明在 [`Stage4/README_STAGE4.md`](Stage4/README_STAGE4.md)。

## Requirements

- Python 3.10
- [uv](https://docs.astral.sh/uv/)（各 stage 以自己的 `pyproject.toml` / `uv.lock` 管理依賴）
- NVIDIA GPU、相容的 CUDA / PyTorch 環境（完整 Stage 1–4 pipeline 建議使用 GPU）
- Git LFS 或其他模型儲存方式不是本 repository 的必要條件；checkpoint 與大型資料檔需另外下載

各 stage 是獨立環境，不要在 repository 根目錄建立一個共用環境來取代它們。CUDA wheel、Mitsuba variant 與模型 checkpoint 需依執行平台調整。

## Quick start

先安裝 `uv`，再分別建立需要的 stage 環境：

```powershell
cd Stage1_ver2
uv sync
uv run python main.py --help

cd ..\Stage2
uv sync
uv run python main.py --help

cd ..\Stage3
uv sync
uv run python infer_stage3_run.py --help

cd ..\Stage4
uv sync
uv run python stage4_relight_single.py --help
```

完整推論前，請準備：

1. 輸入影像。
2. Stage 1 / Stage 2 所需的模型權重與外部 repository。
3. Stage 3 refinement checkpoint。
4. 一份以 [`configs/relight_request.example.json`](configs/relight_request.example.json) 為基礎的本機 request JSON。

Stage 4 request 的路徑是相對於執行指令時的工作目錄；建議在 `Stage4/` 目錄執行，並把 request 內的 `input_image` 與 `stage3_checkpoint` 改成自己的檔案：

```powershell
cd Stage4
Copy-Item ..\configs\relight_request.example.json relight_request.local.json
# 編輯 relight_request.local.json，填入本機影像與 checkpoint 路徑
uv run python stage4_relight_single.py --request relight_request.local.json
```

輸出會寫入 `Stage4/outputs/<output_name>/`；這類產物已被 `.gitignore` 排除。

## Clone with external repositories

部分第三方模型與 baseline 以 Git submodule 記錄固定版本。第一次 clone 後執行：

```powershell
git submodule update --init --recursive
```

`Stage3/` 與 `Stage4/` 的主程式則直接納入這個 repository，方便 clone 後取得完整 wrapper 與訓練程式；原本留在本機的子 repo 歷史不會被提交。

## Benchmark

比較實驗與評估命令集中在 `compare/`。請先閱讀 [`compare/README.md`](compare/README.md)，再依需要準備資料集與 baseline 權重：

```powershell
cd compare
uv sync
```

目前的數值摘要可參考 [`benchmark.md`](benchmark.md)。資料集、prediction、模型權重與 runtime log 不會隨 repository 提交。

## Repository policy

本 repository 只提交可重建研究流程所需的程式碼、設定與文件。以下內容保留在本機，但不應上傳到 GitHub：

- `.venv*`、uv cache、Python cache 與 nested repository 的 `.git`
- 原始 / 下載資料集、個人輸入影像與 benchmark samples
- `*.pt`、`*.pth`、`*.ckpt`、`*.safetensors` 等模型權重
- `result/`、`runs/`、`outputs/`、`work/` 與 renderer 產物
- 大型 zip / archive 與 Hugging Face cache

專案內的 `MoGe`、`Ouroboros`、baseline 與其他 `external` 目錄可能包含上游程式碼；發布前請保留其原始 LICENSE，並確認各上游專案與資料集的再發布條款。

## Known limitations

- 完整流程不是單一 `pip install` 就能完成，因為 Stage 1–4 的依賴與 CUDA 版本不同。
- 預訓練權重與資料集不在 Git repository 內；沒有它們只能執行程式碼檢查，無法重現完整影像結果。
- 多數 benchmark 結果依賴本機資料路徑與外部 baseline，因此 README 只提供流程入口，不宣稱 clone 後即可直接得到相同數值。
