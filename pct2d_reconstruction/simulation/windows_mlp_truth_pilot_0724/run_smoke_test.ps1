param([string]$Python = "python")
$ErrorActionPreference = "Stop"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$root = Join-Path $PSScriptRoot ("qc\smoke_" + $stamp)
$data = Join-Path $root "data\run_000"
$qc = Join-Path $root "run_000"
New-Item -ItemType Directory -Force -Path $data,$qc | Out-Null

& $Python (Join-Path $PSScriptRoot "run_one_angle.py") `
    --config (Join-Path $PSScriptRoot "simulation_config.json") `
    --angle 0 --output-dir $data --qc-dir $qc `
    --protons-per-projection 200

if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $qc "completed.flag"))) {
    throw "MLP truth smoke simulation failed"
}
$metadata = Get-Content -Raw (Join-Path $qc "run_metadata.json") | ConvertFrom-Json
$expectedRoots = @("PhaseSpaceIn.root","PhaseSpaceOut.root","PrimaryTrajectory.root")
$actualRoots = @($metadata.root_qc.PSObject.Properties | ForEach-Object { $_.Name })
$missing = @($expectedRoots | Where-Object { $_ -notin $actualRoots })
$trajectory = $metadata.root_qc.'PrimaryTrajectory.root'
if ($metadata.status -ne "completed" -or $missing.Count -ne 0 -or `
    [int]$trajectory.entries -le [int]$trajectory.unique_primary_events -or `
    [double]$trajectory.sampled_step_length_max_mm -gt 1.011 -or `
    [int]$metadata.insert_validation.overlaps -ne 0) {
    throw ("MLP truth smoke QC failed: status={0}; missing={1}; trajectory entries={2}; events={3}; max step={4}" -f `
        $metadata.status,($missing -join ","),$trajectory.entries,`
        $trajectory.unique_primary_events,$trajectory.sampled_step_length_max_mm)
}
Write-Host ("MLP truth smoke PASS. QC: {0}" -f $root) -ForegroundColor Green
