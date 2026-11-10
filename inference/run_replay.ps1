# Trigger the aml_replay DAG on a transactions CSV (IBM format). The inference service scores it as batches finish.
#   .\inference\run_replay.ps1                                  # bundled 1,000-row demo (HI-Medium rows; smoke test only)
#   .\inference\run_replay.ps1 -Csv C:\data\HI-Small_Trans.csv               # 10,000 per pipeline batch
#   .\inference\run_replay.ps1 -Csv C:\data\HI-Small_Trans.csv -BatchSize 1000
param(
    [string]$Csv = '',
    [int]$BatchSize = 10000,   # = model training batch; the UI still updates every 1,000 (UI_CHUNK)
    [int]$FeatureJobs = 5      # run all five feature components in parallel (pipeline default is 2)
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$windows = 'muledetector_pipeline\data\streaming\replay_windows'
if ($Csv -eq '') {
    $name = 'final_demo_replay_1000.csv'
} else {
    $name = [System.IO.Path]::GetFileName($Csv) -replace '[^A-Za-z0-9_.-]', '_'
    Copy-Item -LiteralPath $Csv -Destination (Join-Path $windows $name) -Force
}
$local = Resolve-Path (Join-Path $windows $name)
$rows = -1  # header
foreach ($line in [System.IO.File]::ReadLines($local)) { if ($line) { $rows++ } }
# Compact JSON without spaces: it is passed unquoted through sh inside the container.
$conf = @{ replay_row_count = $rows; max_offsets_per_trigger = $BatchSize; concurrent_feature_jobs = $FeatureJobs;
           prepared_replay_path = "/opt/muledetector/data/streaming/replay_windows/$name"; verify_features = $false } |
        ConvertTo-Json -Compress
$confName = "$name.conf.json"
[System.IO.File]::WriteAllText((Join-Path (Resolve-Path $windows) $confName), $conf)
$compose = @('-f', 'muledetector_pipeline/compose.yaml', '-f', 'inference/compose.yaml')
& docker compose @compose exec -T airflow-scheduler airflow dags unpause aml_replay
if ($LASTEXITCODE -ne 0) { throw 'Airflow is not ready yet. Wait for airflow-api to be healthy, then retry.' }
& docker compose @compose exec -T airflow-scheduler sh -c "airflow dags trigger aml_replay --conf `$(cat /opt/muledetector/data/streaming/replay_windows/$confName)"
if ($LASTEXITCODE -ne 0) { throw 'Could not trigger aml_replay. Check: docker compose logs --tail=50 airflow-scheduler' }
Write-Host "Triggered aml_replay: $rows transactions, $BatchSize per batch."
Write-Host 'Watch scoring: docker compose -f muledetector_pipeline/compose.yaml -f inference/compose.yaml logs -f inference'
Write-Host 'Live networks: http://localhost:8765'
