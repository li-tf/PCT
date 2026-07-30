param([string]$ProgressPath = (Join-Path $PSScriptRoot "qc\full\progress.json"))
$ErrorActionPreference = "Stop"
if (-not (Test-Path $ProgressPath)) {
    Write-Host "No progress file yet." -ForegroundColor Yellow
    exit 0
}
$p = Get-Content -Raw $ProgressPath | ConvertFrom-Json
$eta = if ($null -eq $p.eta_seconds) { "unknown" } else {
    [TimeSpan]::FromSeconds([double]$p.eta_seconds).ToString("hh\:mm\:ss")
}
Write-Host ("status={0}; progress={1}/{2} ({3:P1})" -f `
    $p.status,$p.completed_cases,$p.total_cases,[double]$p.progress_fraction)
Write-Host ("running={0}; pending={1}; failed={2}; ETA={3}" -f `
    $p.running_cases,$p.pending_cases,@($p.failed_cases).Count,$eta)
Write-Host $p.message
