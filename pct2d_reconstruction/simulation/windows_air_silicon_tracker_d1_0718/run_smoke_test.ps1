param([string]$Python = "python")
$ErrorActionPreference = "Stop"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$root = Join-Path $PSScriptRoot ("qc\smoke_" + $stamp)
$data = Join-Path $root "data\run_000"
$qc = Join-Path $root "run_000"
New-Item -ItemType Directory -Force -Path $data,$qc | Out-Null
& $Python (Join-Path $PSScriptRoot "run_angle.py") --config (Join-Path $PSScriptRoot "simulation_config.json") --angle 0 --output-dir $data --qc-dir $qc --protons-per-projection 2000
if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $qc "completed.flag"))) { throw "D1 smoke test failed" }
$m = Get-Content -Raw (Join-Path $qc "run_metadata.json") | ConvertFrom-Json
$expectedRoots = @(
    "PhaseSpaceIn.root",
    "PhaseSpaceOut.root",
    "TrackerUpstream1.root",
    "TrackerUpstream2.root",
    "TrackerDownstream1.root",
    "TrackerDownstream2.root"
)
$actualRoots = @($m.root_qc.PSObject.Properties | ForEach-Object { $_.Name })
$missingRoots = @($expectedRoots | Where-Object { $_ -notin $actualRoots })
if ($m.status -ne "completed" -or $actualRoots.Count -ne $expectedRoots.Count -or $missingRoots.Count -ne 0) {
    throw ("D1 smoke ROOT QC failed: status={0}; ROOT count={1}; missing={2}" -f `
        $m.status,$actualRoots.Count,($missingRoots -join ","))
}
Write-Host ("D1 smoke test PASS. QC: {0}" -f $root) -ForegroundColor Green
