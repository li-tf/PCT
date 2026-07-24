param([string]$OutputRoot = "D:\OpenGATE\data\simulation_data")

$scenarioPaths = Get-ChildItem (Join-Path $PSScriptRoot "scenarios") -Filter "s*.json" | Sort-Object Name
foreach ($scenarioPath in $scenarioPaths) {
    $config = Get-Content -Raw -Path $scenarioPath.FullName | ConvertFrom-Json
    $qc = Join-Path (Join-Path $PSScriptRoot "qc") $config.scenario_id
    $data = Join-Path $OutputRoot $config.output_name
    $complete = if (Test-Path (Join-Path $qc "runs")) { (Get-ChildItem (Join-Path $qc "runs") -Filter "completed.flag" -Recurse).Count } else { 0 }
    $bytes = if (Test-Path $data) { (Get-ChildItem $data -Filter "*.root" -Recurse | Measure-Object Length -Sum).Sum } else { 0 }
    Write-Host ("{0,-42} {1,3}/720  {2,8:N2} GB" -f $config.scenario_id,$complete,($bytes/1GB))
}
$materialQc = Join-Path (Join-Path $PSScriptRoot "qc") "s6_material_energy_scan"
$materialData = Join-Path $OutputRoot "results0717_s6_material_energy_scan"
$materialComplete = if (Test-Path (Join-Path $materialQc "cases")) { (Get-ChildItem (Join-Path $materialQc "cases") -Filter "completed.flag" -Recurse).Count } else { 0 }
$materialBytes = if (Test-Path $materialData) { (Get-ChildItem $materialData -Filter "*.root" -Recurse | Measure-Object Length -Sum).Sum } else { 0 }
Write-Host ("{0,-42} {1,3}/52   {2,8:N2} GB" -f "s6_material_energy_scan",$materialComplete,($materialBytes/1GB))
