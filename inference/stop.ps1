# Stop everything; keeps pipeline data, checkpoints, the Airflow database and inference outputs.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
& docker compose -f muledetector_pipeline/compose.yaml -f inference/compose.yaml stop
