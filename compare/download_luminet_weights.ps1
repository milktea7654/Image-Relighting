$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Join-Path $root "luminet"
$python = Join-Path $repo ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "LumiNet venv not found at $python"
}

Push-Location $root
try {
    $code = @'
from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id = "xyxingx/LumiNet"
out = Path("luminet/models").resolve()
out.mkdir(parents=True, exist_ok=True)

for filename in ["LumiNet.ckpt", "new_decoder.ckpt", "last.pth.tar"]:
    print(f"[download] {filename}", flush=True)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(out),
        resume_download=True,
    )
    print(f"[ok] {path}", flush=True)
'@
    $code | & $python -
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
