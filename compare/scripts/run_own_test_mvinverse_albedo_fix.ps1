param(
    [string]$Python = "Stage4\.venv\Scripts\python.exe",
    [string]$Stage12Python = "Stage4\external\Stage12_remote_run\.venv\Scripts\python.exe",
    [string]$BenchmarkRoot = "compare\own_test_quantitative"
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path $BenchmarkRoot).Path
$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$statusLog = Join-Path $logDir "mvinverse_albedo_fix_status.log"

function Write-Status {
    param([string]$Message)
    $line = "$(Get-Date -Format o) $Message"
    $line | Tee-Object -FilePath $statusLog -Append
}

function Run-Step {
    param(
        [string]$Name,
        [string]$Exe,
        [string[]]$Arguments
    )
    $logPath = Join-Path $logDir "$Name.log"
    Write-Status "START $Name"
    & $Exe @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) {
        Write-Status "FAIL $Name exit=$LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-Status "DONE $Name"
}

$inputRoot = Join-Path $root "_mvinverse_inputs"
$mviRoot = Join-Path $root "_mvinverse_outputs"

Run-Step "prepare_mvinverse_inputs" $Python @(
    "compare\scripts\fix_own_test_albedo_from_mvinverse.py",
    "prepare-inputs",
    "--benchmark-root", $root,
    "--force"
)

Run-Step "mvinverse_test_albedo" $Stage12Python @(
    "Stage4\external\Stage12_remote_run\single_image_batch_inference.py",
    "--image_list", (Join-Path $inputRoot "image_list.txt"),
    "--save_path", $mviRoot,
    "--device", "cuda"
)

Run-Step "apply_mvinverse_albedo" $Python @(
    "compare\scripts\fix_own_test_albedo_from_mvinverse.py",
    "apply",
    "--benchmark-root", $root,
    "--mvinverse-root", $mviRoot
)

Run-Step "rerun_own_stage3" $Python @(
    "compare\scripts\run_stage3_own_test.py",
    "--benchmark-root", $root,
    "--models", "dcpt-promptir-FT", "hair_FT", "promptir_12ch", "restormer", "xrestormer-FT",
    "--overwrite"
)

Run-Step "evaluate_mvinverse_albedo" $Python @(
    "compare\scripts\evaluate_benchmark.py",
    "--benchmark-root", $root
)

Write-Status "ALL_DONE"
