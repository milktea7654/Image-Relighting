param(
    [string]$Python = "Stage4\.venv\Scripts\python.exe",
    [string]$BenchmarkRoot = "compare\own_test_quantitative",
    [int]$Limit = 0,
    [int]$RgbxRgb2xSteps = 20,
    [int]$RgbxX2rgbSteps = 20,
    [int]$LumiNetSteps = 25,
    [int]$IcLightSteps = 20,
    [switch]$RunOmniGen,
    [int]$OmniGenSteps = 30
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path $BenchmarkRoot).Path
$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$statusLog = Join-Path $logDir "external_full_status.log"

function Write-Status {
    param([string]$Message)
    $line = "$(Get-Date -Format o) $Message"
    $line | Tee-Object -FilePath $statusLog -Append
}

function Run-Step {
    param(
        [string]$Name,
        [string[]]$Arguments
    )

    $logPath = Join-Path $logDir "$Name.log"
    Write-Status "START $Name"
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $oldPreference
    if ($exitCode -ne 0) {
        Write-Status "FAIL $Name exit=$exitCode"
        exit $exitCode
    }
    Write-Status "DONE $Name"
}

$limitArgs = @()
if ($Limit -gt 0) {
    $limitArgs = @("--limit", "$Limit")
}

$rgbxArgs = @(
    "compare\scripts\run_rgbx_benchmark.py",
    "--benchmark-root", $root,
    "--rgb2x-steps", "$RgbxRgb2xSteps",
    "--x2rgb-steps", "$RgbxX2rgbSteps"
)
$rgbxArgs += $limitArgs
$rgbxArgs += "--overwrite"
Run-Step "rgbx_full" $rgbxArgs

$luminetArgs = @(
    "compare\scripts\run_luminet_benchmark.py",
    "--benchmark-root", $root,
    "--steps", "$LumiNetSteps"
)
$luminetArgs += $limitArgs
$luminetArgs += "--overwrite"
Run-Step "luminet_full" $luminetArgs

$icLightArgs = @(
    "compare\scripts\run_ic_light_benchmark.py",
    "--benchmark-root", $root,
    "--steps", "$IcLightSteps"
)
$icLightArgs += $limitArgs
$icLightArgs += "--overwrite"
Run-Step "ic_light_full" $icLightArgs

if ($RunOmniGen) {
    $omnigenArgs = @(
        "compare\scripts\run_omnigen_benchmark.py",
        "--benchmark-root", $root,
        "--steps", "$OmniGenSteps"
    )
    $omnigenArgs += $limitArgs
    $omnigenArgs += "--overwrite"
    Run-Step "omnigen_full" $omnigenArgs
}

Run-Step "evaluate_full" @(
    "compare\scripts\evaluate_benchmark.py",
    "--benchmark-root", $root
)

Write-Status "ALL_DONE"
