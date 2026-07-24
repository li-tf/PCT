param([string]$OutputRoot = "D:\OpenGATE\data\simulation_data")
$config = Get-Content -Raw (Join-Path $PSScriptRoot "simulation_config.json") | ConvertFrom-Json
$qc = Join-Path $PSScriptRoot "qc\full"; $data = Join-Path $OutputRoot ([string]$config.output_name)
$complete = if (Test-Path (Join-Path $qc "runs")) { (Get-ChildItem (Join-Path $qc "runs") -Filter "completed.flag" -Recurse).Count } else { 0 }
$failed = if (Test-Path (Join-Path $qc "runs")) { (Get-ChildItem (Join-Path $qc "runs") -Filter "run_metadata.json" -Recurse | Where-Object { (Get-Content -Raw $_.FullName | ConvertFrom-Json).status -eq "failed" }).Count } else { 0 }
$bytes = if (Test-Path $data) { (Get-ChildItem $data -Filter "*.root" -Recurse | Measure-Object Length -Sum).Sum } else { 0 }
Write-Host ("Compact 3-D progress: {0}/{1}; failed metadata={2}; ROOT={3:N2} GB" -f $complete,[int]$config.projections,$failed,($bytes/1GB))
Write-Host ("Data: {0}" -f $data); Write-Host ("QC:   {0}" -f $qc)
