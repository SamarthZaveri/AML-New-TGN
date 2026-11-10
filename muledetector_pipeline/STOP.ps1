$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& docker compose stop
if ($LASTEXITCODE -ne 0) { throw 'Could not stop the handover services.' }
Write-Host 'Stopped. Data, checkpoints, logs and database volumes were retained.'
