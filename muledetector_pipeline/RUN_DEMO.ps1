$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& docker compose exec -T airflow-scheduler python /opt/handover/trigger_demo.py
if ($LASTEXITCODE -ne 0) { throw 'Demo could not be triggered. Wait for startup, then check: docker compose logs --tail=50 airflow-scheduler' }
Write-Host 'Demo submitted. Follow aml_replay at http://localhost:8081'
