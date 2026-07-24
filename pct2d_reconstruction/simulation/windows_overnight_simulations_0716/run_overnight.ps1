param(
    [ValidateRange(1,128)][int]$Workers = 12,
    [string]$Python = "python",
    [string]$OutputRoot = "D:\OpenGATE\data\simulation_data",
    [double]$MinimumFreeGB = 90.0
)

$ErrorActionPreference = "Stop"
$outputFull = [System.IO.Path]::GetFullPath($OutputRoot)
New-Item -ItemType Directory -Force -Path $outputFull | Out-Null
$rootName = [System.IO.Path]::GetPathRoot($outputFull).TrimEnd('\').TrimEnd(':')
$drive = Get-PSDrive -Name $rootName
$freeGB = $drive.Free / 1GB
Write-Host ("Free space on {0}: {1:N1} GB" -f $drive.Name,$freeGB)
if ($freeGB -lt $MinimumFreeGB) {
    throw ("At least {0:N1} GB free space is required; found {1:N1} GB" -f $MinimumFreeGB,$freeGB)
}

$started = Get-Date
$masterQc = Join-Path $PSScriptRoot "qc"
New-Item -ItemType Directory -Force -Path $masterQc | Out-Null
$records = @()

function Run-Step([string]$Name, [scriptblock]$Action) {
    $stepStart = Get-Date
    Write-Host ""
    Write-Host ("######## {0} ########" -f $Name) -ForegroundColor Magenta
    & $Action
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $Name" }
    $script:records += [PSCustomObject]@{
        step=$Name; status="PASS"; elapsed_seconds=((Get-Date)-$stepStart).TotalSeconds
    }
}

Run-Step "s6_material_energy_scan" {
    & (Join-Path $PSScriptRoot "launch_material_scan.ps1") -Workers $Workers -Python $Python `
        -OutputRoot $outputFull -QcRoot $masterQc
}

$scenarioOrder = @(
    "s1_aluminium_air_full.json",
    "s2_water_vacuum_pilot.json",
    "s3_water_air_pilot.json",
    "s4_material_calibration_air_pilot.json",
    "s5_resolution_air_pilot.json"
)
foreach ($scenarioName in $scenarioOrder) {
    $scenario = Join-Path (Join-Path $PSScriptRoot "scenarios") $scenarioName
    Run-Step ([System.IO.Path]::GetFileNameWithoutExtension($scenarioName)) {
        & (Join-Path $PSScriptRoot "launch_ct_scenario.ps1") -Scenario $scenario `
            -Workers $Workers -Python $Python -OutputRoot $outputFull -QcRoot $masterQc
    }
}

$records | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $masterQc "overnight_steps.csv")
$summary = [ordered]@{
    status="PASS"; workers=$Workers; output_root=$outputFull
    started_at=$started.ToString("s"); finished_at=(Get-Date).ToString("s")
    elapsed_seconds=((Get-Date)-$started).TotalSeconds; steps=$records
}
$summary | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -Path (Join-Path $masterQc "overnight_summary.json")
Write-Host ""
Write-Host ("ALL OVERNIGHT SIMULATIONS COMPLETED in {0:hh\:mm\:ss}" -f ((Get-Date)-$started)) -ForegroundColor Green
