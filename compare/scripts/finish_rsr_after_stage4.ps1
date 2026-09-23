param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [int]$PollSeconds = 300
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$stamp] $Message"
    Write-Output $line
    Add-Content -LiteralPath $script:LogPath -Value $line
}

function Count-Pngs {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return 0
    }
    return (Get-ChildItem -LiteralPath $Path -Filter *.png -ErrorAction SilentlyContinue | Measure-Object).Count
}

function Count-Lines {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return 0
    }
    return (Get-Content -LiteralPath $Path -ErrorAction SilentlyContinue | Measure-Object).Count
}

function Test-RsrStage4Runner {
    $procs = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
        Where-Object {
            $_.CommandLine -match "run_stage4_benchmark.py" -and
            $_.CommandLine -match "rsr_quantitative" -and
            $_.CommandLine -match "--method-id stage4"
        }
    return (($procs | Measure-Object).Count -gt 0)
}

Set-Location -LiteralPath $RepoRoot

$rsrRoot = Join-Path $RepoRoot "compare\rsr_quantitative"
$predDir = Join-Path $rsrRoot "predictions\stage4"
$manifest = Join-Path $rsrRoot "manifest.jsonl"
$failureLog = Join-Path $rsrRoot "stage4_artifacts\stage4\failures.jsonl"
$script:LogPath = Join-Path $rsrRoot "finish_after_stage4.log"
$lockPath = Join-Path $rsrRoot "finish_after_stage4.lock"

if (Test-Path $lockPath) {
    Write-Output "finish script already locked: $lockPath"
    exit 0
}

New-Item -ItemType File -Path $lockPath -Force | Out-Null
try {
    $total = Count-Lines $manifest
    Write-Log "waiting for RSR Stage4: total=$total poll=${PollSeconds}s"

    while ($true) {
        $done = Count-Pngs $predDir
        $failures = Count-Lines $failureLog
        $running = Test-RsrStage4Runner
        Write-Log "stage4 progress done=$done/$total failures=$failures running=$running"

        if ($done -ge $total) {
            break
        }

        if (-not $running) {
            throw "RSR Stage4 stopped before completion: done=$done/$total failures=$failures"
        }

        Start-Sleep -Seconds $PollSeconds
    }

    $failures = Count-Lines $failureLog
    if ($failures -gt 0) {
        throw "RSR Stage4 completed with failures=$failures"
    }

    $stage3Python = Join-Path $RepoRoot "compare\ic-light\.venv\Scripts\python.exe"
    $stage3Script = Join-Path $RepoRoot "compare\scripts\run_stage3_from_stage4_buffers.py"
    Write-Log "running cached Stage3 checkpoint variants for RSR"
    & $stage3Python $stage3Script --benchmark-root $rsrRoot --overwrite --continue-on-error 2>&1 |
        Tee-Object -FilePath (Join-Path $rsrRoot "run_stage3_from_stage4_buffers.log") -Append
    if ($LASTEXITCODE -ne 0) {
        throw "cached Stage3 runner failed with exit code $LASTEXITCODE"
    }

    $evalPython = Join-Path $RepoRoot "compare\.venv\Scripts\python.exe"
    $evalScript = Join-Path $RepoRoot "compare\scripts\evaluate_benchmark.py"
    foreach ($dataset in @("vidit_quantitative", "isr_quantitative", "rsr_quantitative")) {
        $datasetRoot = Join-Path $RepoRoot "compare\$dataset"
        Write-Log "evaluating $dataset"
        & $evalPython $evalScript --benchmark-root $datasetRoot 2>&1 |
            Tee-Object -FilePath (Join-Path $datasetRoot "evaluate_benchmark.log") -Append
        if ($LASTEXITCODE -ne 0) {
            throw "evaluation failed for $dataset with exit code $LASTEXITCODE"
        }
    }

    Write-Log "finish workflow complete"
}
catch {
    Write-Log "ERROR: $($_.Exception.Message)"
    throw
}
finally {
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
}
