param(
    [string]$Python = "Stage4\.venv\Scripts\python.exe",
    [string]$BenchmarkRoot = "compare\own_test_quantitative",
    [int]$Steps = 30,
    [int]$Limit = 0,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path $BenchmarkRoot).Path
$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$statusLog = Join-Path $logDir "omnigen_full_status.log"
$runLog = Join-Path $logDir "omnigen_full.log"

function Write-Status {
    param([string]$Message)
    $line = "$(Get-Date -Format o) $Message"
    $line | Tee-Object -FilePath $statusLog -Append
}

$argsList = @(
    "compare\scripts\run_omnigen_benchmark.py",
    "--benchmark-root", $root,
    "--steps", "$Steps"
)

if ($Limit -gt 0) {
    $argsList += @("--limit", "$Limit")
}

if ($Overwrite) {
    $argsList += "--overwrite"
}

Write-Status "START omnigen_full steps=$Steps limit=$Limit overwrite=$Overwrite"
$oldPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Python @argsList 2>&1 | Tee-Object -FilePath $runLog -Append
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $oldPreference

if ($exitCode -ne 0) {
    Write-Status "FAIL omnigen_full exit=$exitCode"
    exit $exitCode
}

Write-Status "DONE omnigen_full"

Write-Status "START evaluate_after_omnigen"
$ErrorActionPreference = "Continue"
& $Python "compare\scripts\evaluate_benchmark.py" "--benchmark-root" $root 2>&1 | Tee-Object -FilePath (Join-Path $logDir "evaluate_after_omnigen.log") -Append
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $oldPreference

if ($exitCode -ne 0) {
    Write-Status "FAIL evaluate_after_omnigen exit=$exitCode"
    exit $exitCode
}

Write-Status "DONE evaluate_after_omnigen"
Write-Status "ALL_DONE"
