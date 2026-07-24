param(
    [ValidateRange(1,128)][int]$Workers = 12,
    [string]$Python = "python",
    [string]$OutputRoot = "D:\OpenGATE\data\simulation_data",
    [string]$QcRoot = (Join-Path $PSScriptRoot "qc")
)

$ErrorActionPreference = "Stop"
$configPath = Join-Path $PSScriptRoot "material_scan_config.json"
$config = Get-Content -Raw -Path $configPath | ConvertFrom-Json
$runScript = Join-Path $PSScriptRoot "run_material_case.py"
$summaryScript = Join-Path $PSScriptRoot "summarize_material_scan.py"
$enumerationPath = Join-Path (Join-Path $PSScriptRoot "qc") "material_cases.json"
& $Python $runScript --config $configPath --write-cases-json $enumerationPath
if ($LASTEXITCODE -ne 0) { throw "Could not enumerate material cases" }
if (-not (Test-Path $enumerationPath)) { throw "Material case list was not created: $enumerationPath" }
$cases = Get-Content -Raw -Path $enumerationPath | ConvertFrom-Json
$outputPath = Join-Path ([System.IO.Path]::GetFullPath($OutputRoot)) $config.output_name
$qcPath = Join-Path ([System.IO.Path]::GetFullPath($QcRoot)) $config.scenario_id
$casesQcPath = Join-Path $qcPath "cases"
$logPath = Join-Path $qcPath "logs"
New-Item -ItemType Directory -Force -Path $outputPath,$qcPath,$casesQcPath,$logPath | Out-Null
Copy-Item -Force $configPath (Join-Path $qcPath "material_scan_config.json")

function Test-CaseComplete($Case) {
    $dataDir = Join-Path $outputPath $Case.case_id
    $caseQc = Join-Path $casesQcPath $Case.case_id
    $required = @(
        (Join-Path $caseQc "completed.flag"), (Join-Path $caseQc "case_metadata.json"),
        (Join-Path $caseQc "protonct.txt"), (Join-Path $dataDir "PhaseSpaceIn.root"),
        (Join-Path $dataDir "PhaseSpaceOut.root")
    )
    foreach ($path in $required) { if (-not (Test-Path $path)) { return $false } }
    $m = Get-Content -Raw -Path (Join-Path $caseQc "case_metadata.json") | ConvertFrom-Json
    return ($m.status -eq "completed" -and [int]$m.protons -eq [int]$config.protons_per_case)
}

$pending = [System.Collections.Generic.Queue[object]]::new()
$alreadyComplete = 0
foreach ($case in $cases) { if (Test-CaseComplete $case) { $alreadyComplete++ } else { $pending.Enqueue($case) } }
$running = @{}
$failed = [System.Collections.Generic.List[string]]::new()
$completedThisLaunch = 0
$startedAt = Get-Date
Write-Host ""
Write-Host "=== s6_material_energy_scan ===" -ForegroundColor Cyan
Write-Host ("Cases={0}; protons/case={1:N0}; workers={2}; pending={3}" -f $cases.Count,$config.protons_per_case,$Workers,$pending.Count)

while ($pending.Count -gt 0 -or $running.Count -gt 0) {
    while ($pending.Count -gt 0 -and $running.Count -lt $Workers) {
        $case = $pending.Dequeue()
        $dataDir = Join-Path $outputPath $case.case_id
        $caseQc = Join-Path $casesQcPath $case.case_id
        New-Item -ItemType Directory -Force -Path $dataDir,$caseQc | Out-Null
        $flag = Join-Path $caseQc "completed.flag"; if (Test-Path $flag) { Remove-Item -Force $flag }
        $stdout = Join-Path $logPath ($case.case_id + ".stdout.log")
        $stderr = Join-Path $logPath ($case.case_id + ".stderr.log")
        $arguments = @($runScript,"--config",$configPath,"--case-index",$case.case_index,
            "--output-dir",$dataDir,"--qc-dir",$caseQc,"--protons",$config.protons_per_case)
        $process = Start-Process -FilePath $Python -ArgumentList $arguments `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr -NoNewWindow -PassThru
        $running[$case.case_id] = [PSCustomObject]@{ Process=$process; Case=$case }
        Write-Host ("Started {0}, PID={1}" -f $case.case_id,$process.Id)
    }
    Start-Sleep -Seconds 3
    foreach ($key in @($running.Keys)) {
        $item = $running[$key]; $process = $item.Process
        if ($process.HasExited) {
            $process.WaitForExit(); $process.Refresh()
            if (Test-CaseComplete $item.Case) {
                $completedThisLaunch++; Write-Host ("Finished {0}" -f $key) -ForegroundColor Green
            } else {
                $failed.Add($key); Write-Host ("FAILED {0}; see qc logs" -f $key) -ForegroundColor Red
            }
            $running.Remove($key)
        }
    }
    $done = $alreadyComplete + $completedThisLaunch
    $elapsed = (Get-Date)-$startedAt
    Write-Host ("Material scan {0}/{1} ({2:P1}); running={3}; failed={4}; elapsed={5:hh\:mm\:ss}" -f `
        $done,$cases.Count,($done/[double]$cases.Count),$running.Count,$failed.Count,$elapsed)
}

$launchSummary = [ordered]@{
    scenario_id=$config.scenario_id; case_count=$cases.Count; protons_per_case=$config.protons_per_case
    workers=$Workers; already_complete=$alreadyComplete; completed_this_launch=$completedThisLaunch
    failed_cases=@($failed); elapsed_seconds=((Get-Date)-$startedAt).TotalSeconds
    finished_at=(Get-Date).ToString("s"); output_path=$outputPath
}
$launchSummary | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 -Path (Join-Path $qcPath "launcher_summary.json")
if ($failed.Count -gt 0) { Write-Host "Rerun to retry incomplete cases." -ForegroundColor Yellow; exit 1 }
& $Python $summaryScript --config $configPath --data-dir $outputPath --qc-dir $qcPath
if ($LASTEXITCODE -ne 0) { throw "Material scan completed but summary failed" }
Write-Host "Material energy scan completed and summarized." -ForegroundColor Green
