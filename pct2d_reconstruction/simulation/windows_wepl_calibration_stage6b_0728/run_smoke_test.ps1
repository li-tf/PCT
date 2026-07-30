param([string]$Python = "python")
$ErrorActionPreference = "Stop"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$root = Join-Path $PSScriptRoot ("qc\smoke_" + $stamp)
$config = Join-Path $PSScriptRoot "simulation_config.json"
$cases = Join-Path $root "cases.json"
New-Item -ItemType Directory -Force $root | Out-Null
& $Python (Join-Path $PSScriptRoot "run_case.py") --config $config --write-cases-json $cases
if ($LASTEXITCODE -ne 0) { throw "Could not enumerate cases." }
$parsedCases = Get-Content -Raw $cases | ConvertFrom-Json
# Windows PowerShell 5.1 may preserve a JSON top-level array as one nested
# pipeline object.  Explicit foreach normalization prevents .case_index from
# expanding to all 84 indices when it is passed to Python.
$all = @()
foreach ($item in $parsedCases) { $all += $item }
if ($all.Count -ne 84) {
    throw ("Expected 84 smoke case definitions, found {0}" -f $all.Count)
}
foreach ($case in @($all[0],$all[$all.Count-1])) {
    $data = Join-Path $root ("data\"+$case.case_id)
    $qc = Join-Path $root ("cases\"+$case.case_id)
    & $Python (Join-Path $PSScriptRoot "run_case.py") --config $config `
        --case-index $case.case_index --output-dir $data --qc-dir $qc --protons 500
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $qc "completed.flag"))) {
        throw ("Smoke failed: {0}" -f $case.case_id)
    }
}
Write-Host ("Stage 6B smoke PASS: {0}" -f $root) -ForegroundColor Green
