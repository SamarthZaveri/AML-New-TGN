# MuleDetector data pipeline

One implementation, one Airflow workflow, one normal way to run it.

This repository contains the verified shared Kafka → Spark pipeline. It produces
graph updates for the separately owned ML model. There are no alternative
streaming runners or implementation-selection settings.

## What runs

```text
Prepare an ordered replay CSV → publish to Kafka
→ one Spark application prepares each micro-batch once
→ native Account, directed-edge, Bank, and Entity processors update RocksDB state
→ write six tables, audit them, and publish a ready manifest
→ repeat for the next batch → final run report
```

The model can consume a ready batch while Spark processes the next one. All
24 Account, 9 directed-edge, 15 Bank, and 15 Entity features are retained.
This workflow uses `availableNow`: it processes the transactions available when
the query starts, then finishes. It is a bounded replay workflow, not an always-on
production service.

## Setup once

Install Docker Desktop and Astro CLI. The Airflow image installs Java 21 and the
same Python library versions used by the verified implementation; host Spark,
Java, Hadoop, and ML libraries are not required for the documented workflow.
Ports 8080 (Airflow) and 9092 (Kafka) must be available.

Clone this repository, open PowerShell in its root, then run:

```powershell
$env:AML_PROJECT_ROOT = (Get-Location).Path
docker compose up -d
Push-Location orchestration/airflow
astro dev start
Pop-Location
```

`AML_PROJECT_ROOT` must be the absolute repository path. The scheduler mounts it
at `/opt/muledetector`. The Kafka and Airflow configuration contains no personal
Windows paths.

Place the raw IBM files in:

```text
data/raw/HI-Medium_Trans.csv
data/raw/HI-Medium_accounts.csv
```

Prepare the historical graph and fixed metadata once. This is input preparation,
not another streaming implementation. From `orchestration/airflow`, open the
scheduler shell:

```text
astro dev bash --scheduler
```

Inside that shell:

```text
cd /opt/muledetector
python -m preprocessing.prepare --transactions-path data/raw/HI-Medium_Trans.csv --accounts-path data/raw/HI-Medium_accounts.csv
exit
```

Preparation uses transactions strictly before `2022/09/01 00:30`, writes the
historical baseline to `data/processed/spark_static_full/`, and creates
`data/processed/account_entity_lookup.parquet`. Existing preparation outputs are
protected from overwriting. If these prepared datasets have already been supplied,
put them at those locations and skip preparation. Large datasets and generated
outputs are deliberately excluded from Git.

## Run the workflow

From `orchestration/airflow`, use this command for a normal run:

```text
astro dev run dags unpause aml_replay
astro dev run dags trigger aml_replay --conf '{"verify_features": false}'
```

Monitor the `aml_replay` DAG at http://localhost:8080.
Every run gets its own topic, run ID, output directories, and checkpoints.

The default replay contains 1,000 transactions in batches of 500. Each transaction
produces outgoing and incoming Account events, so a clean batch contains 1,000
Account events. Both sides of an invalid transaction are rejected together.

The supported run settings are:

| Setting | Default | Meaning |
| --- | --- | --- |
| `replay_row_count` | 1000 | Number of transactions to publish |
| `max_offsets_per_trigger` | 500 | Kafka transactions per Spark batch in this single-partition replay |
| `prepared_replay_path` | empty | Container path to an already prepared, ordered CSV; its row count must match |
| `concurrent_feature_jobs` | 2 | Maximum simultaneous feature output actions in the one application |
| `verify_features` | true | Independently recompute expected features for acceptance testing |

`verify_features: false` omits only independent recomputation. Auditing, schema
validation, package readiness, and the final report still run. Unknown settings
are rejected. There is no pipeline-version selector.

The replay starts at `2022/09/01 00:30`; keep this aligned with the historical
cutoff when preparing data for training/evaluation. Kafka uses one partition for
the reproducible input-order/batch validation used here.

## Output and model connection

Every completed valid batch has this layout:

```text
data/streaming/model_updates/run_id=<run_id>/batch_id=<batch_id>/
  account_features.parquet/
  bank_features.parquet/
  entity_features.parquet/
  account_to_account_edges.parquet/
  account_to_bank.parquet/
  entity_to_account.parquet/
  manifest.json
```

Read [MODEL_INTEGRATION.md](MODEL_INTEGRATION.md) for the keys, cumulative-feature
semantics, ordered package feed, and model-owned cursor. A rejected-only batch has
audit data and an explicit skip rather than a feature package. The model does
not have to wait for the final run report to consume individually audited ready
packages.

## Code map

| Location | Responsibility |
| --- | --- |
| `orchestration/airflow/dags/aml_replay.py` | The only Airflow workflow |
| `streaming/ingestion/pipeline.py` | The only streaming implementation |
| `streaming/ingestion/events.py`, `audit.py`, `outputs.py` | Helpers used by that implementation |
| `streaming/state/` | The four required state processors and their calculation helper |
| `preprocessing/` | Historical baseline and fixed metadata preparation |
| `streaming/operations/model_update_feed.py` | Model-facing package discovery |
| `validation/`, `tests/` | Development checks; not alternative pipelines |

The exact feature names and ordering are defined in
`preprocessing/spark_static_pipeline.py`. `Is Laundering` is an offline label and
never a model input feature.

## Checkpoints and verification

Retain a run's prepared events, component checkpoints, receipts, completion
records, and Kafka checkpoint together. They live under its checkpoint root.
Never run two drivers against the same root. A changed source/run identity is
rejected rather than silently reusing incompatible state. Use a new root for a
new independent replay.

Native queries use RocksDB and UTC. Account/edge have eight shuffle partitions;
Bank/Entity retain one. The large Account/Entity lookup is not broadcast. Only
stateless input and static metadata are cached; a stateful output executes once
for its final write.

See [VERIFICATION.md](VERIFICATION.md) for the cleanup verification and development
test commands. Deployment and the ML model's graph update/scoring code belong to
the integration stage.
