param(
    [ValidateRange(1,64)][int]$Workers = 12,
    [string]$Python = "python",
    [string]$OutputRoot = "D:\OpenGATE\data\simulation_data",
    [string]$ConfigPath = (Join-Path $PSScriptRoot "simulation_config.json"),
    [string]$QcRoot = (Join-Path $PSScriptRoot "qc\full")
)

$ErrorActionPreference = "Stop"
$configPath = [System.IO.Path]::GetFullPath($ConfigPath)
$config = Get-Content -Raw -Path $configPath | ConvertFrom-Json
$configHash = (Get-FileHash -Algorithm SHA256 $configPath).Hash.ToLower()
$runScript = Join-Path $PSScriptRoot "run_mlic_replica.py"
$summaryScript = Join-Path $PSScriptRoot "summarize_mlic.py"
$taskList = Join-Path $QcRoot "tasks.json"
$outputPath = Join-Path ([System.IO.Path]::GetFullPath($OutputRoot)) ([string]$config.output_name)
$taskQcRoot = Join-Path $QcRoot "tasks"
$logRoot = Join-Path $QcRoot "logs"
$progressPath = Join-Path $QcRoot "progress.json"

$driveName = ([System.IO.Path]::GetPathRoot($outputPath)).TrimEnd("\")
$freeGb = (Get-PSDrive -Name $driveName.TrimEnd(":")).Free / 1GB
if ($freeGb -lt [double]$config.minimum_free_gb) {
    throw ("Only {0:N1} GB free on {1}; at least {2} GB required." -f `
        $freeGb,$driveName,$config.minimum_free_gb)
}
New-Item -ItemType Directory -Force -Path $outputPath,$QcRoot,$taskQcRoot,$logRoot | Out-Null
Copy-Item -Force $configPath (Join-Path $QcRoot "simulation_config.json")
& $Python $runScript --config $configPath --write-tasks-json $taskList
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $taskList)) {
    throw "Could not enumerate Stage 6A tasks."
}
$tasks = Get-Content -Raw -Path $taskList | ConvertFrom-Json
$protonsPerReplica = [int]$config.protons_per_case / [int]$config.replicates_per_case

function Get-TaskDataDir($Task) {
    return Join-Path (Join-Path $outputPath ([string]$Task.case_id)) ("replica_{0:D2}" -f [int]$Task.replica)
}
function Get-TaskQcDir($Task) {
    return Join-Path $taskQcRoot ([string]$Task.task_id)
}
function Test-TaskComplete($Task) {
    $dataDir = Get-TaskDataDir $Task
    $taskQc = Get-TaskQcDir $Task
    $metadataPath = Join-Path $taskQc "task_metadata.json"
    $required = @(
        (Join-Path $taskQc "completed.flag"),
        $metadataPath,
        (Join-Path $taskQc "protonct.txt"),
        (Join-Path $dataDir "depth_dose_edep.mhd")
    )
    foreach ($path in $required) {
        if (-not (Test-Path $path)) { return $false }
    }
    try {
        $m = Get-Content -Raw -Path $metadataPath | ConvertFrom-Json
        return (
            $m.status -eq "completed" -and
            [int]$m.protons -eq [int]$protonsPerReplica -and
            [string]$m.config_sha256 -eq $configHash -and
            [int]$m.depth_bins -eq [int]([double]$config.water_tank_size_mm[2] / [double]$config.depth_bin_mm)
        )
    } catch {
        return $false
    }
}
function Write-ProgressFile {
    param([string]$CurrentMessage)
    $done = $alreadyComplete + $completedThisLaunch
    $elapsedSeconds = ((Get-Date) - $startedAt).TotalSeconds
    $meanTaskSeconds = if ($completedThisLaunch -gt 0) {
        $completedRuntime / $completedThisLaunch
    } else { $null }
    $remaining = $tasks.Count - $done
    $etaSeconds = if ($null -ne $meanTaskSeconds) {
        $meanTaskSeconds * $remaining / [math]::Max(1, $Workers)
    } else { $null }
    $runningIds = @($running.Keys | Sort-Object)
    $record = [ordered]@{
        scenario_id = $config.scenario_id
        status = if ($failed.Count -gt 0) { "running_with_failures" } else { "running" }
        total_tasks = $tasks.Count
        completed_tasks = $done
        completed_this_launch = $completedThisLaunch
        already_complete = $alreadyComplete
        running_tasks = $running.Count
        pending_tasks = $pending.Count
        failed_tasks = @($failed)
        progress_fraction = $done / [double]$tasks.Count
        workers = $Workers
        running_ids = $runningIds
        protons_per_replica = $protonsPerReplica
        elapsed_seconds = $elapsedSeconds
        mean_completed_task_seconds = $meanTaskSeconds
        eta_seconds = $etaSeconds
        message = $CurrentMessage
        updated_at = (Get-Date).ToString("s")
        output_path = $outputPath
        qc_path = $QcRoot
    }
    $temporary = $progressPath + ".tmp"
    $record | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -Path $temporary
    Move-Item -Force $temporary $progressPath
}

$pending = [System.Collections.Generic.Queue[object]]::new()
$alreadyComplete = 0
foreach ($task in $tasks) {
    if (Test-TaskComplete $task) { $alreadyComplete++ } else { $pending.Enqueue($task) }
}
$running = @{}
$failed = [System.Collections.Generic.List[string]]::new()
$completedThisLaunch = 0
$completedRuntime = 0.0
$startedAt = Get-Date
Write-Host ""
Write-Host "=== Stage 6A virtual MLIC ===" -ForegroundColor Cyan
$caseCount = [int]$tasks.Count / [int]$config.replicates_per_case
Write-Host ("{0} cases; {1} independent replicas; {2:N0} protons/replica; workers={3}" -f `
    $caseCount,$tasks.Count,$protonsPerReplica,$Workers)
Write-Host ("Already complete={0}; pending={1}; output={2}" -f `
    $alreadyComplete,$pending.Count,$outputPath)
Write-ProgressFile "launcher started"

while ($pending.Count -gt 0 -or $running.Count -gt 0) {
    while ($pending.Count -gt 0 -and $running.Count -lt $Workers) {
        $task = $pending.Dequeue()
        $dataDir = Get-TaskDataDir $task
        $taskQc = Get-TaskQcDir $task
        New-Item -ItemType Directory -Force -Path $dataDir,$taskQc | Out-Null
        $flag = Join-Path $taskQc "completed.flag"
        if (Test-Path $flag) { Remove-Item -Force $flag }
        $stdout = Join-Path $logRoot ($task.task_id + ".stdout.log")
        $stderr = Join-Path $logRoot ($task.task_id + ".stderr.log")
        $arguments = @(
            $runScript, "--config", $configPath,
            "--task-index", [int]$task.task_index,
            "--output-dir", $dataDir,
            "--qc-dir", $taskQc,
            "--protons", $protonsPerReplica
        )
        $process = Start-Process -FilePath $Python -ArgumentList $arguments `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
            -NoNewWindow -PassThru
        $running[$task.task_id] = [PSCustomObject]@{
            Process = $process
            Task = $task
            StartedAt = Get-Date
        }
        Write-Host ("Started {0}, PID={1}" -f $task.task_id,$process.Id)
    }
    Start-Sleep -Seconds 3
    foreach ($key in @($running.Keys)) {
        $item = $running[$key]
        if ($item.Process.HasExited) {
            $item.Process.WaitForExit()
            $item.Process.Refresh()
            $runtime = ((Get-Date) - $item.StartedAt).TotalSeconds
            if (Test-TaskComplete $item.Task) {
                $completedThisLaunch++
                $completedRuntime += $runtime
                Write-Host ("Finished {0} ({1:N1} s)" -f $key,$runtime) -ForegroundColor Green
            } else {
                $failed.Add($key)
                Write-Host ("FAILED {0}; see qc\\full\\logs" -f $key) -ForegroundColor Red
            }
            $running.Remove($key)
        }
    }
    $done = $alreadyComplete + $completedThisLaunch
    $elapsed = (Get-Date) - $startedAt
    $message = "Progress {0}/{1} ({2:P1}); running={3}; pending={4}; failed={5}; elapsed={6:hh\:mm\:ss}" -f `
        $done,$tasks.Count,($done/[double]$tasks.Count),$running.Count,$pending.Count,$failed.Count,$elapsed
    Write-Host $message
    Write-ProgressFile $message
}

if ($failed.Count -gt 0) {
    Write-ProgressFile "incomplete; rerun the same command to retry failed tasks"
    Write-Host "Some tasks failed. Rerun the same command to retry." -ForegroundColor Yellow
    exit 1
}

Write-Host "All replicas complete. Extracting R80 and MLIC-RSP..." -ForegroundColor Cyan
& $Python $summaryScript --config $configPath --data-dir $outputPath --qc-dir $QcRoot
if ($LASTEXITCODE -ne 0) { throw "All simulations completed but MLIC summary failed." }
$final = Get-Content -Raw $progressPath | ConvertFrom-Json
$final.status = "completed"
$final.completed_tasks = $tasks.Count
$final.running_tasks = 0
$final.pending_tasks = 0
$final.progress_fraction = 1.0
$final.message = "simulation and R80/MLIC-RSP summary completed"
$final.updated_at = (Get-Date).ToString("s")
$temporary = $progressPath + ".tmp"
$final | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -Path $temporary
Move-Item -Force $temporary $progressPath
Set-Content -Encoding ASCII -Path (Join-Path $QcRoot "completed.flag") `
    -Value ("scenario={0}`ncompleted={1}" -f $config.scenario_id,(Get-Date).ToString("s"))
Write-Host "Stage 6A virtual MLIC completed and summarized." -ForegroundColor Green
