param(
    [ValidateRange(1, 128)]
    [int]$Workers = 4,
    [string]$Python = "python",
    [string]$OutputDir = "D:\OpenGATE\data\simulation_data\results0716",
    [string]$QcDir = (Join-Path $PSScriptRoot "qc"),
    [int]$StartAngle = 0,
    [int]$EndAngle = 719,
    [int]$ProtonsPerProjection = 0
)

$ErrorActionPreference = "Stop"
$scriptPath = Join-Path $PSScriptRoot "run_angle.py"
$configPath = Join-Path $PSScriptRoot "simulation_config.json"
$outputPath = [System.IO.Path]::GetFullPath($OutputDir)
$qcPath = [System.IO.Path]::GetFullPath($QcDir)
$logPath = Join-Path $qcPath "logs"
$runsQcPath = Join-Path $qcPath "runs"

if ($StartAngle -lt 0 -or $EndAngle -gt 719 -or $StartAngle -gt $EndAngle) {
    throw "Require 0 <= StartAngle <= EndAngle <= 719"
}

New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
New-Item -ItemType Directory -Force -Path $qcPath | Out-Null
New-Item -ItemType Directory -Force -Path $logPath | Out-Null
New-Item -ItemType Directory -Force -Path $runsQcPath | Out-Null

$config = Get-Content -Raw -Path $configPath | ConvertFrom-Json
if ($ProtonsPerProjection -le 0) {
    $ProtonsPerProjection = [int]$config.protons_per_projection
}

$pending = [System.Collections.Generic.Queue[int]]::new()
$alreadyComplete = 0
for ($angle = $StartAngle; $angle -le $EndAngle; $angle++) {
    $runDir = Join-Path $outputPath ("run_{0:D3}" -f $angle)
    $runQcDir = Join-Path $runsQcPath ("run_{0:D3}" -f $angle)
    $completeFlag = Join-Path $runQcDir "completed.flag"
    $requiredOutputs = @(
        $completeFlag,
        (Join-Path $runQcDir "run_metadata.json"),
        (Join-Path $runDir "PhaseSpaceIn.root"),
        (Join-Path $runDir "PhaseSpaceOut.root"),
        (Join-Path $runQcDir "protonct.txt")
    )
    $isComplete = $true
    foreach ($requiredOutput in $requiredOutputs) {
        if (-not (Test-Path $requiredOutput)) {
            $isComplete = $false
        }
    }
    if ($isComplete) {
        $alreadyComplete++
    } else {
        $pending.Enqueue($angle)
    }
}

$total = $EndAngle - $StartAngle + 1
$running = @{}
$failed = [System.Collections.Generic.List[int]]::new()
$completedThisLaunch = 0
$startedAt = Get-Date

Write-Host "test0713 native-Windows OpenGATE launcher"
Write-Host ("Angles: {0}-{1} ({2} total)" -f $StartAngle, $EndAngle, $total)
Write-Host ("Protons/projection: {0:N0}" -f $ProtonsPerProjection)
Write-Host ("Parallel processes: {0}" -f $Workers)
Write-Host ("Already complete: {0}; pending: {1}" -f $alreadyComplete, $pending.Count)
Write-Host ("Output: {0}" -f $outputPath)
Write-Host ("QC: {0}" -f $qcPath)

while ($pending.Count -gt 0 -or $running.Count -gt 0) {
    while ($pending.Count -gt 0 -and $running.Count -lt $Workers) {
        $angle = $pending.Dequeue()
        $runDir = Join-Path $outputPath ("run_{0:D3}" -f $angle)
        New-Item -ItemType Directory -Force -Path $runDir | Out-Null
        $runQcDir = Join-Path $runsQcPath ("run_{0:D3}" -f $angle)
        New-Item -ItemType Directory -Force -Path $runQcDir | Out-Null
        # A pending directory may contain a stale flag from an interrupted or
        # incomplete earlier attempt.  Only a flag created by this process is
        # allowed to mark the new attempt as successful.
        $oldCompleteFlag = Join-Path $runQcDir "completed.flag"
        if (Test-Path $oldCompleteFlag) {
            Remove-Item -Force $oldCompleteFlag
        }
        $stdout = Join-Path $logPath ("run_{0:D3}.stdout.log" -f $angle)
        $stderr = Join-Path $logPath ("run_{0:D3}.stderr.log" -f $angle)
        $arguments = @(
            $scriptPath,
            "--config", $configPath,
            "--angle", $angle,
            "--output-dir", $runDir,
            "--qc-dir", $runQcDir,
            "--protons-per-projection", $ProtonsPerProjection
        )
        $process = Start-Process `
            -FilePath $Python `
            -ArgumentList $arguments `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -NoNewWindow `
            -PassThru
        $running[$angle] = $process
        Write-Host ("Started angle {0:D3}, PID={1}" -f $angle, $process.Id)
    }

    Start-Sleep -Seconds 3
    foreach ($angle in @($running.Keys)) {
        $process = $running[$angle]
        if ($process.HasExited) {
            # On native Windows, Start-Process may report HasExited before the
            # redirected streams and ExitCode property have been refreshed.
            # WaitForExit also guarantees that stdout/stderr redirection has
            # finished before the files are inspected.
            $process.WaitForExit()
            $process.Refresh()
            $runDir = Join-Path $outputPath ("run_{0:D3}" -f $angle)
            $runQcDir = Join-Path $runsQcPath ("run_{0:D3}" -f $angle)
            $completeFlag = Join-Path $runQcDir "completed.flag"
            $requiredOutputs = @(
                $completeFlag,
                (Join-Path $runQcDir "run_metadata.json"),
                (Join-Path $runDir "PhaseSpaceIn.root"),
                (Join-Path $runDir "PhaseSpaceOut.root"),
                (Join-Path $runQcDir "protonct.txt")
            )
            $outputsComplete = $true
            foreach ($requiredOutput in $requiredOutputs) {
                if (-not (Test-Path $requiredOutput)) {
                    $outputsComplete = $false
                }
            }
            # completed.flag is written only after sim.run() and all required
            # output checks succeed, so it is the authoritative signal.  Some
            # PowerShell/Windows combinations expose a null/stale ExitCode for
            # redirected child processes even after HasExited becomes true.
            if ($outputsComplete) {
                $completedThisLaunch++
                Write-Host (
                    "Finished angle {0:D3}, exit={1}" -f $angle, $process.ExitCode
                ) -ForegroundColor Green
            } else {
                $failed.Add([int]$angle)
                Write-Host (
                    "FAILED angle {0:D3}, exit={1}; see qc\logs\run_{0:D3}.stderr.log" -f `
                    $angle, $process.ExitCode
                ) -ForegroundColor Red
            }
            $running.Remove($angle)
        }
    }

    $done = $alreadyComplete + $completedThisLaunch
    $elapsed = (Get-Date) - $startedAt
    Write-Host (
        "Progress {0}/{1} ({2:P1}); running={3}; failed={4}; elapsed={5:hh\:mm\:ss}" -f `
        $done, $total, ($done / [double]$total), $running.Count, $failed.Count, $elapsed
    )
}

$summary = [ordered]@{
    start_angle = $StartAngle
    end_angle = $EndAngle
    total_angles = $total
    protons_per_projection = $ProtonsPerProjection
    workers = $Workers
    already_complete = $alreadyComplete
    completed_this_launch = $completedThisLaunch
    failed_angles = @($failed)
    elapsed_seconds = ((Get-Date) - $startedAt).TotalSeconds
    finished_at = (Get-Date).ToString("s")
}
$summary | ConvertTo-Json -Depth 4 | Set-Content `
    -Encoding UTF8 `
    -Path (Join-Path $qcPath "launcher_summary.json")

if ($failed.Count -gt 0) {
    Write-Host ("Run finished with failed angles: {0}" -f ($failed -join ", ")) -ForegroundColor Red
    Write-Host "Run the same command again to retry incomplete angles."
    exit 1
}

$manifest = for ($angle = $StartAngle; $angle -le $EndAngle; $angle++) {
    $runDir = Join-Path $outputPath ("run_{0:D3}" -f $angle)
    $runQcDir = Join-Path $runsQcPath ("run_{0:D3}" -f $angle)
    $metadata = Get-Content -Raw -Path (Join-Path $runQcDir "run_metadata.json") | ConvertFrom-Json
    [PSCustomObject]@{
        angle_index = $angle
        angle_degrees = $metadata.angle_degrees
        status = $metadata.status
        protons_per_projection = $metadata.protons_per_projection
        random_seed = $metadata.random_seed
        elapsed_seconds = $metadata.elapsed_seconds
        phase_space_in_bytes = (Get-Item (Join-Path $runDir "PhaseSpaceIn.root")).Length
        phase_space_out_bytes = (Get-Item (Join-Path $runDir "PhaseSpaceOut.root")).Length
    }
}
$manifest | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $qcPath "result_manifest.csv")

Write-Host "All requested angles completed." -ForegroundColor Green
Write-Host "Integrated result manifest and QC completed."
