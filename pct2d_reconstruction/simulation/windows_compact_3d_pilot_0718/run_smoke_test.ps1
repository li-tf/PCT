param([string]$Python = "python")
$ErrorActionPreference = "Stop"; $stamp = Get-Date -Format "yyyyMMdd_HHmmss"; $root = Join-Path $PSScriptRoot ("qc\smoke_" + $stamp)
$data=Join-Path $root "data\run_000"; $qc=Join-Path $root "run_000"; New-Item -ItemType Directory -Force -Path $data,$qc | Out-Null
& $Python (Join-Path $PSScriptRoot "run_angle.py") --config (Join-Path $PSScriptRoot "simulation_config.json") --angle 0 --output-dir $data --qc-dir $qc --protons-per-projection 2000
if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $qc "completed.flag"))) { throw "Compact 3-D smoke test failed" }
$m=Get-Content -Raw (Join-Path $qc "run_metadata.json") | ConvertFrom-Json
$expectedRoots = @("PhaseSpaceIn.root", "PhaseSpaceOut.root")
$actualRoots = @($m.root_qc.PSObject.Properties | ForEach-Object { $_.Name })
$missingRoots = @($expectedRoots | Where-Object { $_ -notin $actualRoots })
if ($m.status -ne "completed" -or $actualRoots.Count -ne $expectedRoots.Count -or `
    $missingRoots.Count -ne 0 -or $m.sphere_validation.overlaps -ne 0) {
    throw ("Compact 3-D smoke QC failed: status={0}; ROOT count={1}; missing={2}; overlaps={3}" -f `
        $m.status,$actualRoots.Count,($missingRoots -join ","),$m.sphere_validation.overlaps)
}
Write-Host ("Compact 3-D smoke test PASS. QC: {0}" -f $root) -ForegroundColor Green
