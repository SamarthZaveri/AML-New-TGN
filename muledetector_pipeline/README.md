# MuleDetector Docker handover

This folder runs the exact published pipeline from GitHub commit
`78437bc2782579ef07bd1f00c6269281d82f647a`.
Yesterday's unpublished target-isolation edits are NOT included. All 24 Account,
15 Bank, 15 Entity and 9 directed-edge features remain unchanged.

## First run on Windows (Intel/AMD)

1. Install and open Docker Desktop. Use Linux containers. Allocate around 8 GB
   of memory to Docker and allow enough disk for the extracted bundle, images,
   output and checkpoints. No host Python, Spark, Java, Git or Astro CLI is needed.
2. Extract the entire ZIP to a local folder outside OneDrive, for example
   `C:\AML\pipeline`. Keep `compose.yaml`, `data`, `images`, and scripts together.
3. Open PowerShell in that folder and run:

   powershell -ExecutionPolicy Bypass -File .\START.ps1

   First use loads the three included Docker images; later starts reuse them.
   Nothing is installed into your ML model's Python environment. The image
   includes the Kafka connector JARs as well as Python and Java dependencies.
4. Open http://localhost:8081. The local Airflow interface requires no login.
   This local-only configuration is for integration and evaluation, not public hosting.
5. Trigger the included demo:

   powershell -ExecutionPolicy Bypass -File .\RUN_DEMO.ps1

   Follow `aml_replay` in Airflow. It processes 1,000 transactions in two batches
   of 500. The independent recomputation is off; audit, schema checks, ready
   manifests and the final report still run. Wait for DAG discovery before
   triggering; if the script reports that the DAG is missing, wait briefly and retry.

The equivalent normal startup after images have been loaded is simply:

   docker compose up -d

Kafka and PostgreSQL are on a private Compose network and do not occupy your
host's 9092 or 5432 ports. Airflow binds only to 127.0.0.1:8081. If that port is
occupied, set `$env:AIRFLOW_PORT = '8082'` before startup and use that URL.
The scripts' printed links assume the default port.

## Connect the ML model

Keep the model in its existing environment. After each batch is ready, read:

   data/streaming/model_updates/run_id=<run_id>/batch_id=<batch_id>/

The six tables are `account_features.parquet`, `bank_features.parquet`,
`entity_features.parquet`, `account_to_account_edges.parquet`,
`account_to_bank.parquet`, and `entity_to_account.parquet`.
Read them only once their `manifest.json` has status `ready`.

Each package contains affected graph rows with cumulative feature values. Upsert
by the table keys; replace totals rather than adding them again. Save the last
applied `(run_id, batch_id)` durably. A model in another container should mount
this same data folder, preferably read-only for input. Model inference, graph
upserts, tensor indexing, prediction storage and model recovery remain model code.

The ready-package feed in `source/streaming/operations/model_update_feed.py` uses
only the Python standard library. The detailed original integration contract is
`source/MODEL_INTEGRATION.md`. The prepared historical graph lives under
`data/processed/spark_static_full`; it is not automatically loaded into native
streaming state. Offline `account_labels.parquet` is a target file, not an input
feature table. This handover preserves the published prediction-time semantics;
it does not fix or certify the separate model's reported temporal leakage.

Final reports appear in `data/streaming/reports`; per-batch audit JSON files are
in `data/streaming/audit/metrics`. Packages are consumable before the final report.

## Other replay sizes and the full raw dataset

The included CSV has exactly 1,000 transactions. A different prepared replay must
have the row count configured in Airflow. Use Trigger with configuration:

```json
{"replay_row_count":1000,"max_offsets_per_trigger":500,"prepared_replay_path":"/opt/muledetector/data/streaming/replay_windows/final_demo_replay_1000.csv","verify_features":false}
```

`support/demo.json` records the shipped demo settings. The same settings are
baked into the image's demo helper. To change them, use Airflow's trigger dialog.
To scan raw data instead, supply `data/raw/HI-Medium_Trans.csv` and omit
`prepared_replay_path` from the Airflow run configuration. The 3 GB raw CSV is not
included. Independent feature validation can be enabled with `verify_features:true`.
Do not reuse static metadata from a different historical cutoff without revisiting
the evaluation design. The current historical cutoff is 2022/09/01 00:30.

## Stop, restart and diagnose

   powershell -ExecutionPolicy Bypass -File .\STOP.ps1
   powershell -ExecutionPolicy Bypass -File .\START.ps1

These retain the database, outputs and checkpoints. Starting services does not
automatically publish another replay. Trigger a new run to get a fresh run ID.
Do not remove checkpoint files while a failed run may need recovery, and do not
use volume-deletion commands as an ordinary stop procedure.

Check status and logs:

   docker compose ps --all
   docker compose logs --tail=100 airflow-scheduler
   docker compose logs --tail=100 airflow-dag-processor

Expected `airflow-init` status is Exited (0). It performs database migrations;
it is not a continuously running service. Test results and source/image hashes
are recorded in HANDOVER.json and VERIFICATION.json. The bundle contains no
previous replay outputs or checkpoints.

## Source and rebuilding

`source/` is a fresh export of the exact published GitHub commit, not a copy of
the author's locally edited working directory. The wrapper adds Compose startup,
image packaging and demo launch only; the source pipeline is unchanged. The
provided image is linux/amd64. Source is included for inspection and future work.
Rebuilding is optional and needs internet:

   docker build -t aml-pipeline-handover:78437bc .

A rebuild can change image IDs because the base runtime and OS packages are not
fully reproducible over time. For the verified handover, use the included image
archive whose exact IDs are in HANDOVER.json. No image registry account is needed.

References: https://github.com/RN90210/aml_pipeline/tree/78437bc2782579ef07bd1f00c6269281d82f647a
and https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html
