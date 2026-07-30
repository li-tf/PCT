param(
    [string]$OutputRoot = "D:\OpenGATE\data\simulation_data",
    [string]$ConfigPath = (Join-Path $PSScriptRoot "simulation_config.json"),
    [string]$QcRoot = (Join-Path $PSScriptRoot "qc\full")
)

$config = Get-Content -Raw ([System.IO.Path]::GetFullPath($ConfigPath)) | ConvertFrom-Json
$progressPath = Join-Path $QcRoot "progress.json"
$dataPath = Join-Path $OutputRoot ([string]$config.output_name)
if (-not (Test-Path $progressPath)) {
    Write-Host "Stage 6A has not been launched yet." -ForegroundColor Yellow
    Write-Host ("Expected progress file: {0}" -f $progressPath)
    exit 0
}
$p = Get-Content -Raw $progressPath | ConvertFrom-Json
function Format-Duration($Seconds) {
    if ($null -eq $Seconds) { return "estimating" }
    $span = [TimeSpan]::FromSeconds([double]$Seconds)
    return ("{0:D2}:{1:D2}:{2:D2}" -f [int]$span.TotalHours,$span.Minutes,$span.Seconds)
}
$bytes = if (Test-Path $dataPath) {
    (Get-ChildItem $dataPath -File -Recurse | Measure-Object Length -Sum).Sum
} else { 0 }
Write-Host "Stage 6A virtual MLIC" -ForegroundColor Cyan
Write-Host ("Status:    {0}" -f $p.status)
Write-Host ("Progress:  {0}/{1} ({2:P1})" -f `
    $p.completed_tasks,$p.total_tasks,[double]$p.progress_fraction)
Write-Host ("Running:   {0}; pending: {1}; failed: {2}" -f `
    $p.running_tasks,$p.pending_tasks,@($p.failed_tasks).Count)
Write-Host ("Elapsed:   {0}" -f (Format-Duration $p.elapsed_seconds))
Write-Host ("Rough ETA: {0}" -f (Format-Duration $p.eta_seconds))
Write-Host ("Data size: {0:N3} GB" -f ($bytes / 1GB))
if (@($p.running_ids).Count -gt 0) {
    Write-Host ("Current:   {0}" -f (@($p.running_ids) -join ", "))
}
Write-Host ("Updated:   {0}" -f $p.updated_at)
Write-Host ("Data:      {0}" -f $dataPath)
Write-Host ("QC:        {0}" -f $QcRoot)
if (@($p.failed_tasks).Count -gt 0) {
    Write-Host ("Failed:    {0}" -f (@($p.failed_tasks) -join ", ")) -ForegroundColor Red
}
