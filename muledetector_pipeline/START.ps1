$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& docker info --format '{{.ServerVersion}}' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Start Docker Desktop with Linux containers, then retry.' }
$manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'HANDOVER.json') -Raw | ConvertFrom-Json
$loadImages = $false
foreach ($image in $manifest.docker_images.PSObject.Properties) {
    $actualId = & docker image inspect --format '{{.Id}}' $image.Name 2>$null
    if ($LASTEXITCODE -ne 0 -or $actualId -ne $image.Value) { $loadImages = $true }
}
if ($loadImages) {
    Write-Host 'Loading the bundled images. This is needed only on first use.'
    $archive = Join-Path $PSScriptRoot 'images\pipeline-images.tar.gz'
    if (-not (Test-Path -LiteralPath $archive)) { throw 'The image archive is missing. Extract the complete handover ZIP first.' }
    if ($manifest.image_archive_sha256) {
        $archiveHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($archiveHash -ne $manifest.image_archive_sha256) { throw 'The image archive checksum differs. Re-copy the complete handover ZIP.' }
    }
    & docker load --input $archive
    if ($LASTEXITCODE -ne 0) { throw 'Could not load the bundled Docker images.' }
}
& docker compose up -d
if ($LASTEXITCODE -ne 0) { throw 'Startup failed. Check Docker Desktop and whether port 8081 is already occupied.' }
Write-Host 'Starting Airflow. Check readiness with: docker compose ps'
Write-Host 'Open http://localhost:8081 once airflow-api is healthy.'
Write-Host 'Run the demo with: powershell -ExecutionPolicy Bypass -File .\RUN_DEMO.ps1'
