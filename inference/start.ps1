# Start the handover pipeline (Kafka, Spark, Airflow) plus frozen-model inference and the live network UI.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$pipeline = Join-Path (Get-Location) 'muledetector_pipeline'
& docker info --format '{{.ServerVersion}}' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Start Docker Desktop with Linux containers, then retry.' }

# Load the bundled pipeline images once (same check as muledetector_pipeline\START.ps1).
$manifest = Get-Content -LiteralPath (Join-Path $pipeline 'HANDOVER.json') -Raw | ConvertFrom-Json
$loadImages = $false
foreach ($image in $manifest.docker_images.PSObject.Properties) {
    # 'image ls' prints nothing for a missing image; 'image inspect' writes to stderr, which
    # Windows PowerShell 5.1 turns into a terminating error under ErrorActionPreference=Stop.
    $actualId = & docker image ls --no-trunc --format '{{.ID}}' $image.Name | Select-Object -First 1
    if ($actualId -ne $image.Value) { $loadImages = $true }
}
if ($loadImages) {
    Write-Host 'Loading the bundled pipeline images (first use only).'
    & docker load --input (Join-Path $pipeline 'images\pipeline-images.tar.gz')
    if ($LASTEXITCODE -ne 0) { throw 'Could not load the bundled Docker images.' }
}

New-Item -ItemType Directory -Force 'inference\outputs' | Out-Null
& docker compose -f muledetector_pipeline/compose.yaml -f inference/compose.yaml up -d --build
if ($LASTEXITCODE -ne 0) { throw 'Startup failed. Check Docker Desktop and whether ports 8081/8765 are free.' }
Write-Host 'Airflow:          http://localhost:8081  (ready when airflow-api is healthy)'
Write-Host 'Live networks:    http://localhost:8765'
Write-Host 'Start a replay:   powershell -ExecutionPolicy Bypass -File .\inference\run_replay.ps1'
