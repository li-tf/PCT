param(
    [ValidateRange(1,64)][int]$Workers = 12,
    [string]$Python = "python",
    [string]$OutputRoot = "D:\OpenGATE\data\simulation_data"
)

$ErrorActionPreference = "Stop"
$configPath = Join-Path $PSScriptRoot "simulation_config.json"
$config = Get-Content -Raw -Path $configPath | ConvertFrom-Json
$configHash = (Get-FileHash -Algorithm SHA256 -Path $configPath).Hash.ToLower()
$outputPath = Join-Path ([System.IO.Path]::GetFullPath($OutputRoot)) ([string]$config.output_name)
$qcPath = Join-Path $PSScriptRoot "qc\full"
$runsQcPath = Join-Path $qcPath "runs"
$logPath = Join-Path $qcPath "logs"
$runScript = Join-Path $PSScriptRoot "run_one_angle.py"
$projections = [int]$config.projections
$protons = [int]$config.protons_per_projection
$expectedRoots = @("PhaseSpaceIn.root","PhaseSpaceOut.root","PrimaryTrajectory.root")
New-Item -ItemType Directory -Force -Path $outputPath,$qcPath,$runsQcPath,$logPath | Out-Null

$driveRoot = [System.IO.Path]::GetPathRoot($outputPath)
$drive = [System.IO.DriveInfo]::new($driveRoot)
$freeGB = $drive.AvailableFreeSpace / 1GB
Write-Host ("Free space on {0}: {1:N1} GB" -f $driveRoot,$freeGB)
if ($freeGB -lt [double]$config.minimum_free_gb) {
    throw ("MLP truth pilot requires at least {0:N0} GB free" -f [double]$config.minimum_free_gb)
}

function Test-RunComplete([int]$Angle) {
    $runDir = Join-Path $outputPath ("run_{0:D3}" -f $Angle)
    $runQc = Join-Path $runsQcPath ("run_{0:D3}" -f $Angle)
    foreach ($name in $expectedRoots) {
        if (-not (Test-Path (Join-Path $runDir $name))) { return $false }
    }
    foreach ($name in @("completed.flag","run_metadata.json","protonct.txt")) {
        if (-not (Test-Path (Join-Path $runQc $name))) { return $false }
    }
    $metadata = Get-Content -Raw (Join-Path $runQc "run_metadata.json") | ConvertFrom-Json
    if ($metadata.status -ne "completed") { return $false }
    if ($metadata.config_sha256 -ne $configHash -or `
        [int]$metadata.protons_per_projection -ne $protons) {
        throw "Completed output has a different config hash or proton count: $runDir"
    }
    return $true
}

$pending = [System.Collections.Generic.Queue[int]]::new()
$already = 0
for ($angle=0; $angle -lt $projections; $angle++) {
    if (Test-RunComplete $angle) { $already++ } else { $pending.Enqueue($angle) }
}
$running = @{}
$failed = [System.Collections.Generic.List[int]]::new()
$completed = 0
$started = Get-Date
Write-Host ""
Write-Host "=== Heterogeneous primary-trajectory MLP truth pilot ===" -ForegroundColor Cyan
Write-Host ("Angles={0}; protons/projection={1:N0}; total={2:N0}; workers={3}; complete={4}; pending={5}" -f `
    $projections,$protons,($projections*$protons),$Workers,$already,$pending.Count)
Write-Host ("Output={0}" -f $outputPath)

while ($pending.Count -gt 0 -or $running.Count -gt 0) {
    while ($pending.Count -gt 0 -and $running.Count -lt $Workers) {
        $angle = $pending.Dequeue()
        $runDir = Join-Path $outputPath ("run_{0:D3}" -f $angle)
        $runQc = Join-Path $runsQcPath ("run_{0:D3}" -f $angle)
        New-Item -ItemType Directory -Force -Path $runDir,$runQc | Out-Null
        $flag = Join-Path $runQc "completed.flag"
        if (Test-Path $flag) { Remove-Item -Force $flag }
        $arguments = @(
            $runScript,"--config",$configPath,"--angle",$angle,
            "--output-dir",$runDir,"--qc-dir",$runQc,
            "--protons-per-projection",$protons
        )
        $process = Start-Process -FilePath $Python -ArgumentList $arguments `
            -RedirectStandardOutput (Join-Path $logPath ("run_{0:D3}.stdout.log" -f $angle)) `
            -RedirectStandardError (Join-Path $logPath ("run_{0:D3}.stderr.log" -f $angle)) `
            -NoNewWindow -PassThru
        $running[$angle] = $process
        Write-Host ("Started angle {0:D3}, PID={1}" -f $angle,$process.Id)
    }
    Start-Sleep -Seconds 3
    foreach ($angle in @($running.Keys)) {
        $process = $running[$angle]
        if ($process.HasExited) {
            $process.WaitForExit()
            $process.Refresh()
            if (Test-RunComplete $angle) {
                $completed++
                Write-Host ("Finished angle {0:D3}" -f $angle) -ForegroundColor Green
            } else {
                $failed.Add([int]$angle)
                Write-Host ("FAILED angle {0:D3}; see qc\full\logs" -f $angle) -ForegroundColor Red
            }
            $running.Remove($angle)
        }
    }
    $done = $already + $completed
    $elapsed = (Get-Date) - $started
    $eta = "unknown"
    if ($completed -gt 0) {
        $rate = $completed / [Math]::Max($elapsed.TotalSeconds,1.0)
        $eta = [TimeSpan]::FromSeconds(($projections-$done)/$rate).ToString("hh\:mm\:ss")
    }
    Write-Host ("Progress {0}/{1} ({2:P1}); running={3}; failed={4}; elapsed={5:hh\:mm\:ss}; ETA={6}" -f `
        $done,$projections,($done/[double]$projections),$running.Count,$failed.Count,$elapsed,$eta)
}

$summary = [ordered]@{
    scenario_id = $config.scenario_id
    output_name = $config.output_name
    projections = $projections
    protons_per_projection = $protons
    total_protons = ($projections*$protons)
    workers = $Workers
    config_sha256 = $configHash
    already_complete = $already
    completed_this_launch = $completed
    failed_angles = @($failed)
    elapsed_seconds = ((Get-Date)-$started).TotalSeconds
    finished_at = (Get-Date).ToString("s")
    output_path = $outputPath
}
$summary | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $qcPath "launcher_summary.json")
if ($failed.Count -gt 0) {
    Write-Host "Rerun the same command to retry failed angles." -ForegroundColor Yellow
    exit 1
}

$manifest = for ($angle=0; $angle -lt $projections; $angle++) {
    $runQc = Join-Path $runsQcPath ("run_{0:D3}" -f $angle)
    $metadata = Get-Content -Raw (Join-Path $runQc "run_metadata.json") | ConvertFrom-Json
    [PSCustomObject]@{
        angle_index = $angle
        angle_degrees = $metadata.angle_degrees
        status = $metadata.status
        protons_per_projection = $metadata.protons_per_projection
        random_seed = $metadata.random_seed
        elapsed_seconds = $metadata.elapsed_seconds
        root_bytes = $metadata.output_bytes
        phase_in_primary = $metadata.root_qc.'PhaseSpaceIn.root'.primary_entries
        phase_out_primary = $metadata.root_qc.'PhaseSpaceOut.root'.primary_entries
        trajectory_events = $metadata.root_qc.'PrimaryTrajectory.root'.unique_primary_events
        trajectory_steps = $metadata.root_qc.'PrimaryTrajectory.root'.entries
        mean_steps_per_event = $metadata.root_qc.'PrimaryTrajectory.root'.steps_per_event_mean
        sampled_max_step_mm = $metadata.root_qc.'PrimaryTrajectory.root'.sampled_step_length_max_mm
    }
}
$manifest | Export-Csv -NoTypeInformation -Encoding UTF8 (Join-Path $qcPath "result_manifest.csv")
Write-Host "MLP truth pilot completed and manifest generated." -ForegroundColor Green
