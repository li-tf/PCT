param(
    [Parameter(Mandatory=$true)][string]$Scenario,
    [ValidateRange(1,128)][int]$Workers = 12,
    [string]$Python = "python",
    [string]$OutputRoot = "D:\OpenGATE\data\simulation_data",
    [string]$QcRoot = (Join-Path $PSScriptRoot "qc")
)

$ErrorActionPreference = "Stop"
$scenarioPath = [System.IO.Path]::GetFullPath($Scenario)
$raw = Get-Content -Raw -Path $scenarioPath | ConvertFrom-Json
$basePath = Join-Path (Split-Path $scenarioPath) $raw.base_config
$config = Get-Content -Raw -Path $basePath | ConvertFrom-Json
foreach ($property in $raw.PSObject.Properties) {
    if ($property.Name -ne "base_config") {
        $config | Add-Member -NotePropertyName $property.Name -NotePropertyValue $property.Value -Force
    }
}
$scenarioId = [string]$config.scenario_id
$projections = [int]$config.projections
$protons = [int]$config.protons_per_projection
$outputPath = Join-Path ([System.IO.Path]::GetFullPath($OutputRoot)) $config.output_name
$qcPath = Join-Path ([System.IO.Path]::GetFullPath($QcRoot)) $scenarioId
$runScript = Join-Path $PSScriptRoot "run_ct_angle.py"
$logPath = Join-Path $qcPath "logs"
$runsQcPath = Join-Path $qcPath "runs"
New-Item -ItemType Directory -Force -Path $outputPath,$qcPath,$logPath,$runsQcPath | Out-Null
Copy-Item -Force $scenarioPath (Join-Path $qcPath "scenario_config.json")
Copy-Item -Force $basePath (Join-Path $qcPath "base_ct.json")

function Test-RunComplete([int]$Angle) {
    $runDir = Join-Path $outputPath ("run_{0:D3}" -f $Angle)
    $runQcDir = Join-Path $runsQcPath ("run_{0:D3}" -f $Angle)
    $required = @(
        (Join-Path $runQcDir "completed.flag"),
        (Join-Path $runQcDir "run_metadata.json"),
        (Join-Path $runQcDir "protonct.txt"),
        (Join-Path $runDir "PhaseSpaceIn.root"),
        (Join-Path $runDir "PhaseSpaceOut.root")
    )
    foreach ($path in $required) { if (-not (Test-Path $path)) { return $false } }
    $metadata = Get-Content -Raw -Path (Join-Path $runQcDir "run_metadata.json") | ConvertFrom-Json
    if ($metadata.status -ne "completed") { return $false }
    if ($metadata.scenario_id -ne $scenarioId -or [int]$metadata.protons_per_projection -ne $protons) {
        throw "Existing completed output does not match scenario/proton count: $runDir"
    }
    return $true
}

$pending = [System.Collections.Generic.Queue[int]]::new()
$alreadyComplete = 0
for ($angle = 0; $angle -lt $projections; $angle++) {
    if (Test-RunComplete $angle) { $alreadyComplete++ } else { $pending.Enqueue($angle) }
}
$running = @{}
$failed = [System.Collections.Generic.List[int]]::new()
$completedThisLaunch = 0
$startedAt = Get-Date
Write-Host ""
Write-Host ("=== {0} ===" -f $scenarioId) -ForegroundColor Cyan
Write-Host ("Angles={0}; protons/projection={1:N0}; workers={2}" -f $projections,$protons,$Workers)
Write-Host ("Already complete={0}; pending={1}" -f $alreadyComplete,$pending.Count)
Write-Host ("Output={0}" -f $outputPath)
Write-Host ("QC={0}" -f $qcPath)

while ($pending.Count -gt 0 -or $running.Count -gt 0) {
    while ($pending.Count -gt 0 -and $running.Count -lt $Workers) {
        $angle = $pending.Dequeue()
        $runDir = Join-Path $outputPath ("run_{0:D3}" -f $angle)
        $runQcDir = Join-Path $runsQcPath ("run_{0:D3}" -f $angle)
        New-Item -ItemType Directory -Force -Path $runDir,$runQcDir | Out-Null
        $flag = Join-Path $runQcDir "completed.flag"
        if (Test-Path $flag) { Remove-Item -Force $flag }
        $stdout = Join-Path $logPath ("run_{0:D3}.stdout.log" -f $angle)
        $stderr = Join-Path $logPath ("run_{0:D3}.stderr.log" -f $angle)
        $arguments = @(
            $runScript, "--config", $scenarioPath, "--angle", $angle,
            "--output-dir", $runDir, "--qc-dir", $runQcDir,
            "--protons-per-projection", $protons
        )
        $process = Start-Process -FilePath $Python -ArgumentList $arguments `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
            -NoNewWindow -PassThru
        $running[$angle] = $process
        Write-Host ("Started angle {0:D3}, PID={1}" -f $angle,$process.Id)
    }
    Start-Sleep -Seconds 3
    foreach ($angle in @($running.Keys)) {
        $process = $running[$angle]
        if ($process.HasExited) {
            $process.WaitForExit(); $process.Refresh()
            if (Test-RunComplete $angle) {
                $completedThisLaunch++
                Write-Host ("Finished angle {0:D3}" -f $angle) -ForegroundColor Green
            } else {
                $failed.Add([int]$angle)
                Write-Host ("FAILED angle {0:D3}; see {1}" -f $angle,(Join-Path $logPath ("run_{0:D3}.stderr.log" -f $angle))) -ForegroundColor Red
            }
            $running.Remove($angle)
        }
    }
    $done = $alreadyComplete + $completedThisLaunch
    $elapsed = (Get-Date) - $startedAt
    $eta = "unknown"
    if ($completedThisLaunch -gt 0) {
        $rate = $completedThisLaunch / [Math]::Max($elapsed.TotalSeconds,1.0)
        $remaining = $projections - $done
        $etaSpan = [TimeSpan]::FromSeconds($remaining / $rate)
        $eta = $etaSpan.ToString("hh\:mm\:ss")
    }
    Write-Host ("Progress {0}/{1} ({2:P1}); running={3}; failed={4}; elapsed={5:hh\:mm\:ss}; ETA={6}" -f `
        $done,$projections,($done/[double]$projections),$running.Count,$failed.Count,$elapsed,$eta)
}

$summary = [ordered]@{
    scenario_id = $scenarioId; output_name = $config.output_name
    projections = $projections; protons_per_projection = $protons; workers = $Workers
    already_complete = $alreadyComplete; completed_this_launch = $completedThisLaunch
    failed_angles = @($failed); elapsed_seconds = ((Get-Date)-$startedAt).TotalSeconds
    finished_at = (Get-Date).ToString("s"); output_path = $outputPath
}
$summary | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 -Path (Join-Path $qcPath "launcher_summary.json")
if ($failed.Count -gt 0) {
    Write-Host "Rerun the same command to retry incomplete angles." -ForegroundColor Yellow
    exit 1
}

$manifest = for ($angle = 0; $angle -lt $projections; $angle++) {
    $runDir = Join-Path $outputPath ("run_{0:D3}" -f $angle)
    $runQcDir = Join-Path $runsQcPath ("run_{0:D3}" -f $angle)
    $m = Get-Content -Raw -Path (Join-Path $runQcDir "run_metadata.json") | ConvertFrom-Json
    [PSCustomObject]@{
        angle_index=$angle; angle_degrees=$m.angle_degrees; status=$m.status
        protons_per_projection=$m.protons_per_projection; random_seed=$m.random_seed
        elapsed_seconds=$m.elapsed_seconds
        phase_space_in_bytes=(Get-Item (Join-Path $runDir "PhaseSpaceIn.root")).Length
        phase_space_out_bytes=(Get-Item (Join-Path $runDir "PhaseSpaceOut.root")).Length
    }
}
$manifest | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $qcPath "result_manifest.csv")
Write-Host ("Scenario {0} completed." -f $scenarioId) -ForegroundColor Green
