param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$root = Join-Path $PSScriptRoot ("qc\smoke_" + $stamp)
$data = Join-Path $root "data"
New-Item -ItemType Directory -Force -Path $data | Out-Null
$runner = Join-Path $PSScriptRoot "run_ct_angle.py"
$scenarioPaths = Get-ChildItem (Join-Path $PSScriptRoot "scenarios") -Filter "s*.json" | Sort-Object Name
$results = @()
foreach ($scenario in $scenarioPaths) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($scenario.Name)
    Write-Host ("Smoke testing {0}" -f $name) -ForegroundColor Cyan
    $out = Join-Path $data $name
    $qc = Join-Path $root $name
    & $Python $runner --config $scenario.FullName --angle 0 --output-dir $out --qc-dir $qc --protons-per-projection 2000
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $qc "completed.flag"))) {
        throw "Smoke test failed: $name"
    }
    $results += [PSCustomObject]@{ scenario=$name; status="PASS"; protons=2000 }
}
Write-Host "Smoke testing material scan" -ForegroundColor Cyan
$materialOut = Join-Path $data "s6_material_energy_scan"
$materialQc = Join-Path $root "s6_material_energy_scan"
& $Python (Join-Path $PSScriptRoot "run_material_case.py") --config (Join-Path $PSScriptRoot "material_scan_config.json") `
    --case-index 0 --output-dir $materialOut --qc-dir $materialQc --protons 2000
if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $materialQc "completed.flag"))) {
    throw "Smoke test failed: material scan"
}
$results += [PSCustomObject]@{ scenario="s6_material_energy_scan"; status="PASS"; protons=2000 }
Write-Host "Smoke testing longest Air slab" -ForegroundColor Cyan
$materialAirOut = Join-Path $data "s6_material_air_2000mm"
$materialAirQc = Join-Path $root "s6_material_air_2000mm"
& $Python (Join-Path $PSScriptRoot "run_material_case.py") --config (Join-Path $PSScriptRoot "material_scan_config.json") `
    --case-index 51 --output-dir $materialAirOut --qc-dir $materialAirQc --protons 2000
if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $materialAirQc "completed.flag"))) {
    throw "Smoke test failed: 2000 mm Air slab"
}
$results += [PSCustomObject]@{ scenario="s6_air_2000mm"; status="PASS"; protons=2000 }
$results | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $root "smoke_summary.csv")
Write-Host ("All smoke tests PASS. QC: {0}" -f $root) -ForegroundColor Green
