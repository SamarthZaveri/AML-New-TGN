# Connect the ML model

The data pipeline supplies completed graph-update packages. The model consumer
must load them, update its graph, run the desired predictions, and save its
last successfully applied `(run_id, batch_id)`.

## Six-table contract

| Table | Key | Contents |
| --- | --- | --- |
| `account_features.parquet` | `account_id` | 24 cumulative Account features |
| `bank_features.parquet` | `bank_id` | 15 Bank features, including static account count |
| `entity_features.parquet` | `entity_id` | 15 Entity features, including static type indicators |
| `account_to_account_edges.parquet` | `sender_account_id`, `receiver_account_id` | 9 directed-edge features |
| `account_to_bank.parquet` | `account_id`, `bank_id` | Stable mappings affected by this batch |
| `entity_to_account.parquet` | `entity_id`, `account_id` | Stable mappings affected by this batch |

Feature columns are doubles and retain the canonical column order from
`preprocessing/spark_static_pipeline.py`. Validate the model's expected names and
order against the manifest before loading tensors. Keep preprocessing/scaling
consistent between training and inference.

A package contains affected rows, not a complete graph snapshot. Upsert by the
keys above. If a count changes from 10 to 12, replace it with 12; do not add 12 to
the previous 10. Temporal features are calculated as of the emitted micro-batch.

For a fresh run, native state starts empty. Feature values accumulate valid
transactions from the beginning of that replay. Historical baseline features are
not automatically loaded into native state. The separately prepared historical
graph supplies context; agree on these existing run-local feature semantics when
training the model. Do not assume an incremental row includes pre-cutoff history.

Spark batch size and ML training batch size are separate choices. A batch of 500
transactions can affect different numbers of Accounts, Banks, Entities, and edges.

## Read ready packages in order

The feed uses only the Python standard library and does not start Spark:

The model does not need to install this repository's Spark runtime requirements
into its training environment. It can read Parquet with its own compatible data
libraries and use this lightweight feed.

```python
from pathlib import Path
from streaming.operations.model_update_feed import ready_batches

# Supply these functions in the ML integration.
cursor = load_last_applied_batch(run_id)  # -1 before the first batch
for package in ready_batches(
    Path("data/streaming/model_updates"),
    Path("data/streaming/audit/metrics"),
    run_id,
    cursor,
):
    if package["status"] == "ready":
        apply_graph_updates_and_save_cursor(package)
    else:
        save_skipped_batch_cursor(package)
```

The two `save`/`apply` functions above are model-side integration hooks, not
implemented model code. Persist the graph update and cursor consistently. Advance
the cursor only after successful application; retries with an old cursor return
the same package again. The pipeline does not claim exactly-once execution of a
separate model process.

Returned records include table paths, column lists, counts, and the batch audit.
The feed checks the audit and manifest, returns consecutive batches only, and
stops at the first unfinished batch. An all-rejected batch is explicitly returned
as `skipped_no_valid_events`, with zero model table counts.

A ready package is published only after all native component commits, successful
table writes, and schema checks. The consumer can apply batch 0 while batch 1 is
processing. Apply your rejection policy using the returned audit record. The final
report summarizes the whole run; offline training/evaluation can additionally
require the completed report and independent feature comparison.

The ML integration must still decide which affected nodes to score, how to keep
stable tensor node indices as the graph grows, where predictions are stored, and
how graph state/cursors are recovered. These responsibilities are not alternative
data-pipeline implementations.
