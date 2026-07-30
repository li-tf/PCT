param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$root = Join-Path $PSScriptRoot ("qc\smoke_" + $stamp)
$configPath = Join-Path $PSScriptRoot "simulation_config.json"
$tasksPath = Join-Path $root "tasks.json"
New-Item -ItemType Directory -Force -Path $root | Out-Null
& $Python (Join-Path $PSScriptRoot "run_mlic_replica.py") `
    --config $configPath --write-tasks-json $tasksPath
if ($LASTEXITCODE -ne 0) { throw "Could not enumerate MLIC smoke tasks." }
$tasks = Get-Content -Raw $tasksPath | ConvertFrom-Json
$reference = $tasks | Where-Object { $_.case_id -eq "e200_reference" -and [int]$_.replica -eq 0 } | Select-Object -First 1
$water = $tasks | Where-Object { $_.case_id -eq "e200_water" -and [int]$_.replica -eq 0 } | Select-Object -First 1
if ($null -eq $reference -or $null -eq $water) { throw "Smoke task selection failed." }

foreach ($task in @($reference,$water)) {
    $data = Join-Path $root ("data\" + $task.task_id)
    $qc = Join-Path $root ("tasks\" + $task.task_id)
    New-Item -ItemType Directory -Force -Path $data,$qc | Out-Null
    & $Python (Join-Path $PSScriptRoot "run_mlic_replica.py") `
        --config $configPath --task-index $task.task_index `
        --output-dir $data --qc-dir $qc --protons 200
    if ($LASTEXITCODE -ne 0) { throw ("Smoke simulation failed: {0}" -f $task.task_id) }
    $metadataPath = Join-Path $qc "task_metadata.json"
    if (-not (Test-Path $metadataPath)) { throw "Smoke metadata is missing." }
    $m = Get-Content -Raw $metadataPath | ConvertFrom-Json
    if ($m.status -ne "completed" -or [int]$m.depth_bins -ne 3500 -or `
        [double]$m.edep_sum -le 0 -or -not (Test-Path (Join-Path $qc "completed.flag"))) {
        throw ("Smoke QC failed: {0}" -f $task.task_id)
    }
}
Write-Host ("Stage 6A smoke PASS. Reference and Water depth-dose outputs: {0}" -f $root) -ForegroundColor Green
