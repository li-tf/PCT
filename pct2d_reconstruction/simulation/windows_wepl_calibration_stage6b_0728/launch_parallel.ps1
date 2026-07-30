param(
    [ValidateRange(1,64)][int]$Workers = 12,
    [string]$Python = "python",
    [string]$OutputRoot = "D:\OpenGATE\data\simulation_data"
)

$ErrorActionPreference = "Stop"
$configPath = Join-Path $PSScriptRoot "simulation_config.json"
$config = Get-Content -Raw -Path $configPath | ConvertFrom-Json
$configHash = (Get-FileHash -Algorithm SHA256 $configPath).Hash.ToLower()
$runScript = Join-Path $PSScriptRoot "run_case.py"
$qcRoot = Join-Path $PSScriptRoot "qc\full"
$casesPath = Join-Path $qcRoot "cases.json"
$taskQcRoot = Join-Path $qcRoot "cases"
$logRoot = Join-Path $qcRoot "logs"
$progressPath = Join-Path $qcRoot "progress.json"
$outputPath = Join-Path ([IO.Path]::GetFullPath($OutputRoot)) ([string]$config.output_name)

$driveName = ([IO.Path]::GetPathRoot($outputPath)).TrimEnd("\").TrimEnd(":")
$freeGb = (Get-PSDrive -Name $driveName).Free / 1GB
if ($freeGb -lt [double]$config.minimum_free_gb) {
    throw ("Only {0:N1} GB free; at least {1} GB required." -f $freeGb,$config.minimum_free_gb)
}
New-Item -ItemType Directory -Force -Path $outputPath,$qcRoot,$taskQcRoot,$logRoot | Out-Null
Copy-Item -Force $configPath (Join-Path $qcRoot "simulation_config.json")
& $Python $runScript --config $configPath --write-cases-json $casesPath
if ($LASTEXITCODE -ne 0) { throw "Could not enumerate calibration cases." }
$parsedCases = Get-Content -Raw $casesPath | ConvertFrom-Json
# Normalize the top-level JSON array explicitly for Windows PowerShell 5.1.
# Without this, some installations retain one nested Object[] and later pass
# all case indices to one Python process.
$cases = @()
foreach ($item in $parsedCases) { $cases += $item }
if ($cases.Count -ne 84) {
    throw ("Expected 84 calibration cases, found {0}" -f $cases.Count)
}

function Get-DataDir($case) { Join-Path $outputPath ([string]$case.case_id) }
function Get-QcDir($case) { Join-Path $taskQcRoot ([string]$case.case_id) }
function Test-Complete($case) {
    $data = Get-DataDir $case
    $qc = Get-QcDir $case
    $metadata = Join-Path $qc "case_metadata.json"
    foreach ($path in @(
        (Join-Path $qc "completed.flag"), $metadata,
        (Join-Path $data "PhaseSpaceIn.root"),
        (Join-Path $data "PhaseSpaceOut.root")
    )) { if (-not (Test-Path $path)) { return $false } }
    try {
        $m = Get-Content -Raw $metadata | ConvertFrom-Json
        return $m.status -eq "completed" -and
            [int]$m.protons -eq [int]$config.protons_per_case -and
            [string]$m.config_sha256 -eq $configHash
    } catch { return $false }
}
function Write-Progress($message,$status="running") {
    $done = $already + $completed
    $elapsed = ((Get-Date) - $started).TotalSeconds
    $mean = if ($completed -gt 0) { $completedRuntime / $completed } else { $null }
    $eta = if ($null -ne $mean) { $mean * ($cases.Count-$done) / [math]::Max(1,$Workers) } else { $null }
    $record = [ordered]@{
        status=$status; total_cases=$cases.Count; completed_cases=$done
        running_cases=$running.Count; pending_cases=$pending.Count
        failed_cases=@($failed); progress_fraction=$done/[double]$cases.Count
        workers=$Workers; elapsed_seconds=$elapsed; eta_seconds=$eta
        message=$message; output_path=$outputPath; updated_at=(Get-Date).ToString("s")
    }
    $tmp = $progressPath + ".tmp"
    $record | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $tmp
    Move-Item -Force $tmp $progressPath
}

$pending = [Collections.Generic.Queue[object]]::new()
$already = 0
foreach ($case in $cases) {
    if (Test-Complete $case) { $already++ } else { $pending.Enqueue($case) }
}
$running = @{}
$failed = [Collections.Generic.List[string]]::new()
$completed = 0
$completedRuntime = 0.0
$started = Get-Date
Write-Host ("Stage 6B: {0} cases, {1:N0} total protons, workers={2}" -f `
    $cases.Count,($cases.Count*[int]$config.protons_per_case),$Workers) -ForegroundColor Cyan
Write-Progress "launcher started"

while ($pending.Count -gt 0 -or $running.Count -gt 0) {
    while ($pending.Count -gt 0 -and $running.Count -lt $Workers) {
        $case = $pending.Dequeue()
        $data = Get-DataDir $case
        $qc = Get-QcDir $case
        New-Item -ItemType Directory -Force -Path $data,$qc | Out-Null
        $stdout = Join-Path $logRoot ($case.case_id + ".stdout.log")
        $stderr = Join-Path $logRoot ($case.case_id + ".stderr.log")
        $arguments = @(
            $runScript,"--config",$configPath,"--case-index",[int]$case.case_index,
            "--output-dir",$data,"--qc-dir",$qc,
            "--protons",[int]$config.protons_per_case
        )
        $process = Start-Process -FilePath $Python -ArgumentList $arguments `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
            -NoNewWindow -PassThru
        $running[$case.case_id] = [PSCustomObject]@{
            Process=$process; Case=$case; StartedAt=Get-Date
        }
        Write-Host ("Started {0}, PID={1}" -f $case.case_id,$process.Id)
    }
    Start-Sleep -Seconds 3
    foreach ($key in @($running.Keys)) {
        $item = $running[$key]
        if ($item.Process.HasExited) {
            $item.Process.WaitForExit()
            $runtime = ((Get-Date)-$item.StartedAt).TotalSeconds
            if (Test-Complete $item.Case) {
                $completed++; $completedRuntime += $runtime
                Write-Host ("Finished {0} ({1:N1}s)" -f $key,$runtime) -ForegroundColor Green
            } else {
                $failed.Add($key)
                Write-Host ("FAILED {0}; inspect qc\full\logs" -f $key) -ForegroundColor Red
            }
            $running.Remove($key)
        }
    }
    $done = $already+$completed
    $message = "Progress {0}/{1} ({2:P1}); running={3}; pending={4}; failed={5}" -f `
        $done,$cases.Count,($done/[double]$cases.Count),$running.Count,$pending.Count,$failed.Count
    Write-Host $message
    Write-Progress $message
}
if ($failed.Count -gt 0) {
    Write-Progress "incomplete; rerun the same command" "incomplete"
    exit 1
}
Write-Progress "all Stage-6B calibration cases completed" "completed"
Set-Content -Encoding ASCII (Join-Path $qcRoot "completed.flag") "stage6b completed"
Write-Host "Stage 6B simulation completed." -ForegroundColor Green
