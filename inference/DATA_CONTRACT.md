# Inference data contract: MuleDetector Stage 2 on IBM AML HI-Small

**Version:** 1.2 (2026-11-10)
**Status:** frozen. Any change to a section marked **FROZEN** needs a new contract version.

**Changes from v1.1:** static node features now default to **neutral** (§6a). The first demo showed that streamed account features push every account to about 0.94. Retraining is ruled out.

**Changes from v1.0**
- Networks are now built live, batch by batch (§10), replacing the offline post-processing step.
- A batch's transactions are found through its Kafka offsets (§4).
- Hub mappings come from the static entity lookup (§5).
- The model state is saved every N batches (§8).
- Docker and Airflow wiring (§10).

This contract defines every input, transform and output between:
1. the Docker/Airflow feature pipeline (the `78437bc` handover),
2. the frozen Stage 2 model,
3. network post-processing, and
4. the visualisation server.

The words MUST, MUST NOT and SHOULD carry their usual meaning.

---

## 1. Frozen model **FROZEN**

### Checkpoint

| Item | Value |
|---|---|
| Checkpoint | `runs/txn_separate_seeds/seed_42/best.pt`, to be copied to `inference/model/best.pt` |
| SHA-256 | `37d44ad8a62f01b7fe5b5dbf45580d99a0898938af2512c0a99fcd977714d084` |
| Size | 4,016,853 bytes |
| Training dataset fingerprint | `0c23ac5c96ba015cb7ea7ebc4729f904d223376e934935940a1493f39a61b0f1` |
| Trained on | IBM AML HI-Medium, chronological 70/15/15 split, epoch 8 (best validation account AUPRC 0.496) |
| Model class | `stage2.transaction_model.TransactionAccountTGN`, `mode='separate'` |
| Parameters | 995,333 learned weights, all independent of which node they're applied to |
| Checkpoint keys | `weights`, `thresholds`, `dataset_metadata`, `settings`, `epoch`, `validation`, `fingerprint` |

### Architecture settings (`settings['model']`)

| Setting | Value |
|---|---|
| `account_feat_dim` | 24 |
| `bank_feat_dim` | 15 |
| `entity_feat_dim` | 15 |
| `edge_feat_dim` | **49** |
| `memory_dim` | 128 |
| `time_dim` | 64 |
| `num_attention_heads` | 4 |
| `max_neighbors` | 20 |
| `dropout` | 0.1, inactive in `eval()` |
| `enable_relay` | true |

The node counts (`num_account_nodes`, `num_bank_nodes`, `num_entity_nodes`) are **not** frozen. They only size the per-node state tables, and inference sets them to the HI-Small node registry's capacity (Section 5).

### Runtime settings

| Setting | Value |
|---|---|
| Micro-batch size | 10,000 events |
| Relay through bank and entity hubs | every micro-batch |
| Mode | `model.eval()` under `torch.no_grad()` |

### Which heads are used

| Output | Model key | Used? |
|---|---|---|
| Account risk | `sigmoid(base_account_logits)` | **YES**: the account score |
| Transaction risk | `sigmoid(transaction_logits)` | **YES**: the transaction score |
| Blended account score | `final_account_logits` | **NO**: untrained fixed blend, scored worse |
| Gradient-boosting stacker | `seed_*/stacker.joblib` | **NO**: not part of the frozen model |
| Chain heads | `chain_*` | **NO**: frozen and unused |

### Decision thresholds

These were fitted on HI-Medium validation and MUST NOT be re-tuned on HI-Small.

| Score | Threshold |
|---|---|
| Account (`account_base`) | **0.7912804484367371** |
| Transaction | **0.9086251258850098** |

### Reference scores (HI-Medium test, seed 42)

| Score | AUPRC | Precision | Recall | F1 |
|---|---|---|---|---|
| Account | 0.508 | 0.859 | 0.358 | 0.505 |
| Transaction | 0.637 | 0.814 | 0.415 | 0.549 |

These are for comparison only.

**Loading.** Build the model with the HI-Small capacities, then copy `checkpoint['weights'][name]` into each named parameter. The checkpoint holds weights only: it contains no memory, neighbor or feature state. Inference always starts from empty state.

---

## 2. End-to-end flow

```
replay CSV ─► Airflow aml_replay ─► Kafka (1 topic per run) ─► [A] Spark ─► model_updates/run_id=R/batch_id=k/
     │                                                                     + audit/metrics/run_id=R/batch_id=k.json
     │                                                                              │
     └──────────── [B] rows min_offset..max_offset of the replay CSV ───────────────┤
                                                                                    ▼
                          [C] inference consumer (docker service `inference`): score batch k, then apply package k
                                                                                    │
                                    ├─► inference/outputs/run_id=R/{transaction_scores, account_scores}
                                    └─► [D] live network builder (in the same process, every batch)
                                                │
                                    inference/outputs/run_id=R/network_live/{top.json, deltas/}
                                                ▼
                          [E] live UI (docker service `network-ui`, http://localhost:8765), pushed over SSE
```

**Orchestration**
- Airflow's `aml_replay` DAG drives A.
- C and D run as one long-lived service that follows the newest run as its batches become ready. This is why each batch shows up as soon as Spark finishes it.
- The pipeline image doesn't contain torch, so the model runs in its own container and never inside Airflow.

---

## 3. Input A: feature packages from the Docker pipeline

### Location

`data/streaming/model_updates/run_id=<run_id>/batch_id=<k>/`. The model container MUST mount the same `data/` directory.

### Readiness rules

- Packages are discovered **only** through `streaming.operations.model_update_feed.ready_batches`. It checks the audit, the manifest and the `_SUCCESS` markers, and it returns consecutive batches only.
- `status == "ready"`: score the batch, then apply the package.
- `status == "skipped_no_valid_events"`: record the skip in the cursor. Nothing is scored.
- Batches are processed in strictly increasing `batch_id` order, with no gaps.
- The manifest's `columns[table]` MUST equal `[key, *config columns]` (below). Any difference stops the run.

### Tables

Each `.parquet` is a directory and MUST be read whole. `_SUCCESS` and `.crc` files are ignored.

| Table | Key | Columns used, **in this exact order** |
|---|---|---|
| `account_features.parquet/` | `account_id` | the 24 columns of `config.ACCOUNT_FEATURE_COLUMNS`: `outgoing_txn_count, incoming_txn_count, outgoing_unique_counterparties, incoming_unique_counterparties, repeat_counterparty_ratio, self_transfer_ratio, outgoing_total_paid, incoming_total_received, outgoing_avg_paid, incoming_avg_received, outgoing_amount_std, incoming_amount_std, payment_currency_diversity, receiving_currency_diversity, currency_switch_ratio, payment_format_diversity, unique_connected_banks, cross_bank_txn_ratio, odd_hour_txn_ratio, weekend_txn_ratio, active_span_hours, hours_since_last_txn, max_txns_per_hour, avg_inter_txn_gap_hours` |
| `bank_features.parquet/` | `bank_id` | the 15 columns of `config.BANK_FEATURE_COLUMNS`: `bank_account_count, outgoing_txn_count, incoming_txn_count, outgoing_total_paid, incoming_total_received, outgoing_avg_paid, incoming_avg_received, unique_sender_accounts, unique_receiver_accounts, counterparty_bank_count, currency_diversity, payment_format_diversity, cross_bank_txn_ratio, odd_hour_txn_ratio, weekend_txn_ratio` |
| `entity_features.parquet/` | `entity_id` | the 15 columns of `config.ENTITY_FEATURE_COLUMNS`: `entity_type_sole_proprietorship, entity_type_partnership, entity_type_corporation, entity_type_country, entity_type_other, owned_account_count, connected_bank_count, outgoing_txn_count, incoming_txn_count, outgoing_total_paid, incoming_total_received, unique_counterparty_accounts, cross_bank_txn_ratio, odd_hour_txn_ratio, payment_format_diversity` |
| `account_to_bank.parquet/` | `(account_id, bank_id)` | the account's relay hub. Exactly one bank per account. |
| `entity_to_account.parquet/` | `(entity_id, account_id)` | the account's relay hub. Exactly one entity per account. |
| `account_to_account_edges.parquet/` | `(sender_account_id, receiver_account_id)` | **not used** by the model; may be ignored |

### Feature value rules

- **Types:** IDs are strings. `bank_id` is cast to string. Feature columns are float64.
- **Finite values:** all values MUST be finite. A NaN or inf is a contract violation, and the consumer MUST fail the batch, never fill it.
- **Missing or extra columns:** a missing column is a contract violation. Extra columns are ignored.
- **Upsert, never add:** each row replaces the stored raw vector for its key.
- **Cumulative semantics:** values describe the state at the **end** of batch `k` (handover §6–7). Streaming totals start empty and cover only this replay.

---

## 4. Input B: transaction events **FROZEN schema**

Feature packages don't contain transactions, and the model needs every transaction. They come from the run's replay CSV.

**Source:** `data/streaming/replay_windows/<run_id>.csv`, which Airflow copies before publishing.

**Alignment rule:**
- Each run publishes to its own single-partition topic, in stable `Timestamp` order, as `producer.py` does. So Kafka offset `i` is row `i` of the CSV after a stable sort on `Timestamp`.
- Batch `k`'s transactions are rows `minimum_offset..maximum_offset` from its audit's `partition_offsets`.
- The row count MUST equal `kafka_transaction_count`, or the run stops.

**Invalid rows:** rows failing the checks below are excluded from scoring and counted as `invalid_transactions` in the manifest. They aren't silently filled.

### Event fields

| Field | Type | Rule | Source in the IBM CSV |
|---|---|---|---|
| `event_idx` | int64 | unique within a run, in publish order | row number |
| `timestamp_unix` | int64 | seconds, UTC | `Timestamp` (`YYYY/MM/DD HH:MM`, read as UTC) |
| `sender_account_id` | string | `"<from_bank>::<account>"` | `From Bank`, `Account` |
| `receiver_account_id` | string | `"<to_bank>::<account>"` | `To Bank`, `Account.1` |
| `from_bank` | string | sender's bank id | `From Bank` |
| `to_bank` | string | receiver's bank id | `To Bank` |
| `amount_paid` | float64 | finite, ≥ 0 | `Amount Paid` |
| `amount_received` | float64 | finite, ≥ 0 | `Amount Received` |
| `payment_currency` | string | exact IBM spelling | `Payment Currency` |
| `receiving_currency` | string | exact IBM spelling | `Receiving Currency` |
| `payment_format` | string | exact IBM spelling | `Payment Format` |

- **Account id format:** it MUST match the pipeline's `account_id`. Example: `291683::83F3021F0`.
- **Labels:** `Is Laundering` MUST NOT reach the model. It may be carried only into `inference/eval/labels.parquet` (Section 9).
- **Ordering:** events are sorted by `timestamp_unix` with a stable sort. Ties keep `event_idx` order.

---

## 5. Node registry (new accounts)

The model works on accounts it never saw: all weights are shared across nodes. Per-node state lives in tables of fixed capacity.

| Rule | Value |
|---|---|
| Capacity | Taken from the replay CSV at run start: accounts and entities = unique accounts + `--account-margin` (1,024); banks = unique banks + `--bank-margin` (256) |
| Index assignment | First come, first served. An ID gets the next free row the first time it appears in a transaction or a package. Rows are never reused within a run. |
| Persistence | The registry (`id → row`) is saved with the state (Section 8). |
| Capacity overflow | Fail the run. Never drop events. |
| Bank of an account | From `account_to_bank`, or else from the transaction itself: the sender's bank is `from_bank`, the receiver's is `to_bank`. |
| Entity of an account | From the replay's static `account_entity_lookup.parquet` on first sight (metadata, not leakage); otherwise from `entity_to_account`. If neither has it, `-1`: no entity relay. |
| Lookup must match the dataset | HI-Small needs a lookup built from `HI-Small_accounts.csv`. With the shipped HI-Medium lookup, the pipeline rejects every HI-Small transaction as `missing_entity_id`. |
| An account with no feature row yet | Its raw feature vector is **all zeros**, meaning no history. It's then transformed like any other row (Section 6). |

---

## 6. Preprocessing **FROZEN**

All statistics come from `checkpoint['dataset_metadata']['preprocessing']`, fitted on HI-Medium's training window. They MUST NOT be refitted on HI-Small.

### Static node features

This applies to account, bank and entity features. For each group, take `state = preprocessing[group]` and apply `stage2.preprocessing.transform(x, state)`:
1. On columns where `signed_log` is true: `x = sign(x) * log1p(|x|)`.
2. Standardise: `x = (x - mean) / std`.
3. Clip to `±clip` (5.0), then cast to float32.

Columns are in the `state['columns']` order (identical to Section 3).

### Transaction edge vector (49 dimensions, in this order)

| Dims | Content | Transform |
|---|---|---|
| 0–1 | `amount_paid`, `amount_received` | `preprocessing['edge']`: signed log on both columns, then standardise and clip to ±6 |
| 2–8 | `sender==receiver`; `from_bank!=to_bank`; `payment_currency!=receiving_currency`; `sin(2π·h/24)`; `cos(2π·h/24)`; `sin(2π·d/7)`; `cos(2π·d/7)` | `preprocessing['transaction_numeric']`: standardise, then clip to ±5 |
| 9–24 | one-hot `payment_currency` (16 slots) | slot 0 = unseen value; slots 1–15 = `transaction_vocabularies['payment_currency']` in listed order |
| 25–40 | one-hot `receiving_currency` (16 slots) | same scheme |
| 41–48 | one-hot `payment_format` (8 slots) | slot 0 = unseen; slots 1–7 = `ACH, Bitcoin, Cash, Cheque, Credit Card, Reinvestment, Wire` |

- `h` is the UTC hour plus minutes/60. `d` is the UTC day of week, with Monday = 0.
- Currency vocabulary, in order: `Australian Dollar, Bitcoin, Brazil Real, Canadian Dollar, Euro, Mexican Peso, Ruble, Rupee, Saudi Riyal, Shekel, Swiss Franc, UK Pound, US Dollar, Yen, Yuan`.
- The reference implementation is `stage2/transaction_dataset.py:_events` and `stage2/dataset.py:_events`. The consumer MUST reuse that code path with the frozen states and MUST NOT re-implement it.

### 6a. Static feature mode: `--static-features neutral` (default) or `stream`

**Why `stream` fails**
- Training used full-history account features. Streamed features are cumulative up to the current batch.
- A streamed `hours_since_last_txn` is about 0 for every active account. That's a z-score of −5.0, the clip limit. In training, that value only occurred for accounts active at the very end of the data, almost all of which were laundering.
- Result: every account scores about 0.94.

**`neutral`**
- Every account, bank and entity row holds the **median transformed training value** from `inference/model/neutral_features.json`, which `inference/feature_ablation.py` writes.
- Packages then supply only hub mappings.
- Scores come entirely from the stream: TGN memory, relay through bank and entity hubs, attention over neighbours, and the 49 transaction features. All of these are computed as of each event in training and in inference.

**Choosing the mode**
- `inference/feature_ablation.py` replays HI-Medium with the frozen weights under the variants `full`, `no_time` and `neutral`. It writes `inference/model/ablation.json`.
- `neutral` is used if its validation metrics are acceptable. Its HI-Medium numbers are the honest expectation for streaming.

### Time

- Model time is float64 **hours** since the first event of the inference run.
- Only time differences enter the model, so the origin doesn't matter, but it MUST stay fixed for the whole run.
- The origin is recorded in the state.

---

## 7. Inference procedure (per package)

For each package `k`, in order:

1. Load package `k`'s transactions (Section 4) and sort them.
2. **Score first, with the state as of the end of batch `k-1`.** For each 10,000-event micro-batch, in time order:
   - Run `model(source_ids, dest_ids, edge_features, timestamps, do_relay=True, score_accounts=True)`. This one call scores the transactions and then updates memory, the relay hubs and the neighbor lists.
   - Record the transaction scores, and the account scores for the accounts it touched.
3. **Then apply package `k`'s features:** upsert the raw vectors (Section 3), re-transform the affected rows (Section 6), and write them into the model's account, bank and entity feature tables and hub maps.
4. Write the outputs for `k` (Section 9), update the live network (Section 10), and advance the in-memory cursor to `k`.
5. Every `--state-every` batches (50) and at the end of the run, save the state and the cursor together (Section 8).

**Why features lag by one package.** Package `k`'s features include all of batch `k`. Using them while scoring batch `k`'s own transactions would show each transaction totals that include later transactions. That is a lookahead leak, and the handover warns against it (§7).

### Warm-up

| Rule | Value |
|---|---|
| Count | `--warmup-records`, default **500,000** transactions, counted from the first event of the run |
| Scoring | Every transaction and account is scored from the very first one. No re-scoring, and no second pass. |
| Flag | Rows with a cumulative event position ≤ the warm-up count get `in_warmup = true` |
| Metrics | Headline metrics use `in_warmup = false` rows only (Section 9) |

---

## 8. State, cursor and restart

**Location:** `inference/outputs/run_id=<R>/state/`. It's written atomically: write to a temporary file, then rename.

| File | Contents |
|---|---|
| `cursor.json` | `{run_id, last_batch_id, skipped_batch_ids, events_processed, invalid_transactions, unseen_categories, time_origin_unix, checkpoint_sha256, contract_version}`. Written **last**, so it only ever points at fully saved state. |
| `model_state.pt` | all model buffers: memory, last-update times, seen flags, neighbor lists, feature tables, hub maps, risk sums and counts |
| `host_state.pkl` | node registry, latest raw feature vectors, latest per-account scores, and the live network graph |

**Restart rules**
- Load the state and resume at `last_batch_id + 1`.
- Batches after the cursor are re-scored from that saved state, and their output files are overwritten. Each package therefore takes effect on the saved state exactly once.
- If `checkpoint_sha256` or `contract_version` doesn't match, refuse to resume.

---

## 9. Outputs

**Location:** `inference/outputs/run_id=<R>/`. All outputs are Parquet. Scores are probabilities in [0, 1].

### `transaction_scores/batch_id=<k:08d>.parquet`
One row per transaction.

| Column | Type | Meaning |
|---|---|---|
| `run_id` | string | |
| `batch_id` | int64 | |
| `event_idx` | int64 | |
| `timestamp_unix` | int64 | |
| `sender_account_id` | string | |
| `receiver_account_id` | string | |
| `amount_paid` | float64 | |
| `amount_received` | float64 | |
| `transaction_risk_score` | float32 | `sigmoid(transaction_logits)` |
| `transaction_flag` | bool | score ≥ 0.9086251258850098 |
| `in_warmup` | bool | |

### `account_scores/batch_id=<k:08d>.parquet`
One row per account touched in batch `k`, holding its last-touch score in that batch.

| Column | Type | Meaning |
|---|---|---|
| `run_id` | string | |
| `batch_id` | int64 | |
| `account_id` | string | |
| `scored_at_unix` | int64 | |
| `risk_score` | float32 | `sigmoid(base_account_logits)`, the frozen account score |
| `account_flag` | bool | score ≥ 0.7912804484367371 |
| `transaction_risk_mean` | float32 | running mean of the transaction scores the account took part in |
| `event_count` | int64 | |
| `in_warmup` | bool | |

### `account_scores_latest.parquet`
The latest row per `account_id` over all batches. It has the same columns, minus `batch_id`, and is written at the end of the run. It's also valid input for the offline `stage2/network_postprocess.py`.

### Naming rule
The account score column MUST be called `risk_score`, and **no other score column may be in this file**. `network_postprocess.read_scores` takes the first match among `risk_score, mule_score, score, probability`.

### Run manifest: `manifest.json`
It is written at the end of the run and holds:
- the final cursor fields (Section 8);
- `warmup_records`;
- all four thresholds;
- network totals;
- `finished_at_unix`.

### Offline evaluation (`inference/eval/`), never read by the model

| Rule | Value |
|---|---|
| Labels | `run_id=<R>/eval/labels_transactions.parquet` (`event_idx, is_laundering`), split off the replay CSV before scoring. Account label = was ever in a laundering transaction in this replay. |
| Command | `python -m inference.evaluate --run-dir inference/outputs/run_id=<R>`, which writes `eval/metrics.json` |
| Headline metrics | AUPRC, precision, recall and F1 at the frozen thresholds, on rows with `in_warmup = false` |
| Also reported | all rows together |
| Prohibitions | MUST NOT re-tune thresholds, select subsets after looking at results, or change the model based on HI-Small results |

---

## 10. Live networks, visualisation and deployment

### [D] Live network builder (`inference/live_network.py`)

It runs inside the consumer after every scored batch. It keeps the rules of `stage2/network_postprocess.py` but applies them incrementally.

**Bins**

| Item | Red | Yellow | Green |
|---|---|---|---|
| Account (`risk_score`) | ≥ **0.7913**, the model threshold | ≥ 0.55 | below |
| Transaction (`transaction_risk_score`) | ≥ **0.9086**, the model threshold | ≥ 0.55 | below |
| Network (`network_score`) | ≥ 0.7913 | ≥ 0.55 | below |

**Rules**

| Rule | Value |
|---|---|
| Which transactions enter the graph | The transaction is red or yellow, **or** either endpoint account is red or yellow at that batch. Green-to-green transactions don't enter. |
| Nodes | Accounts. A node is recoloured with its latest score every time it's touched. |
| Edges | One per (sender, receiver) pair: count, first/last time, amounts paid and received, maximum transaction score |
| Networks | Connected components (union-find). They only grow and merge; nothing is ever removed. On a merge, the larger network keeps its id. |
| Flow paths | A→B→C. Each incoming transaction into B is matched once, to B's first later outgoing transaction. Kept if the gap is ≤ 72 h and out/in amount is in [0.70, 1.30]. At most 200 per intermediary. `path_risk` = mean of the three account scores. |
| Rescoring | After each batch, every network touched by a new transaction or a recoloured node is rescored |
| `network_score` | `0.45·max + 0.20·mean + 0.15·min(1, log1p(paths)/ln 11) + 0.10·min(1, log1p(value)/ln 1,000,001) + 0.10·min(1, 5·density)` (unchanged) |
| Display set | The **top 25** networks by `network_score`, re-ranked after every batch. Networks enter and leave as scores change. |
| Warm-up | Built from the first transaction. Nodes and edges whose latest activity is in warm-up are drawn faded. |
| Render cap | Per network: the 300 riskiest accounts, the 1,000 riskiest edges among them, and a hidden-node count |

**Outputs** (`inference/outputs/run_id=<R>/network_live/`)

| File | Contents |
|---|---|
| `deltas/<frame:08d>.json` | `{frame, batch_id, stats, top: [summary ×25, ranked], entered, exited, merges: [[absorbed, survivor]], details: {component_id: {summary, nodes, edges, hidden_nodes}}}`. `details` covers only top-25 networks that changed in this frame. The model scores a whole pipeline batch at once, then the network is updated in frames of `--ui-chunk` (1,000) transactions, in time order. |
| `top.json` | The same shape, with `details` for all of the current top 25. Late-joining browsers start from it. |

A network `summary` has these fields: `component_id, network_score, network_bin, account_count, red_accounts, yellow_accounts, edge_count, transaction_count, suspicious_value, flow_path_count, risk_score_max, risk_score_mean`.

At the end of the run, the full graph is also written to `run_id=<R>/network/` in the offline schema (`network_nodes`, `network_edges`, `flow_paths`, `networks`.parquet). Those files can be served with `network.server --data-dir`.

### [E] Live UI (`network/server.py --live-root`, page `network/static/live.html`)

- **Endpoints:**
  - `GET /` → the live page.
  - `GET /api/live/state` → `top.json` for the run named in `outputs/LATEST`.
  - `GET /api/live/stream?after=<frame>&run_id=<R>` → Server-Sent Events. It sends one `batch` event per delta, and a `reset` event when a new run starts. It honours `Last-Event-ID` on reconnect.
- **Page:**
  - The ranked top-25 list, with rank movement, newly entered networks and flashing updates.
  - The selected network's graph, which grows in place: new nodes and edges are added, existing ones recoloured. Selection follows #1 by default, and a click pins a network instead.
  - Header counters: transactions scored, red and yellow transactions, accounts, networks, warm-up badge.
- **Port:** 8765. It MUST NOT clash with Airflow on 8081.

### Deployment (Docker Compose)

The handover package lives in `muledetector_pipeline/`, unchanged. `inference/compose.yaml` is an override file layered on top of it:

| Service | Image | Does |
|---|---|---|
| postgres, kafka, airflow-* | handover images (loaded from `muledetector_pipeline/images/`) | `aml_replay`: replay → Kafka → Spark → packages |
| `inference` | `mule-inference:seed42`, built from `inference/Dockerfile` (CPU torch 2.11, frozen `best.pt`) | `python -m inference.consumer --watch`: follows the newest run and writes scores and network deltas |
| `network-ui` | same image | `python -m network.server --live-root`, on http://localhost:8765 |

**Mounts:** pipeline `data/` → `/pipeline/data`, read-only; `inference/outputs/` → `/app/inference/outputs`, read-write for `inference` and read-only for the UI.

**Scripts** (PowerShell, run from `ml_ly_latest/`):
- `inference\start.ps1`
- `inference\run_replay.ps1 [-Csv <file>] [-BatchSize 10000] [-FeatureJobs 5]`. 10,000 per pipeline batch matches the model's training batch size. All five feature components run in parallel.
- `inference\stop.ps1`

**Environment knobs:**
- `LOOKUP`: entity lookup, relative to the pipeline `data/` folder. One value is used by both Airflow and the consumer.
- `WARMUP_RECORDS`
- `UI_CHUNK`: transactions per live frame, default 1,000.
- `MIN_SECONDS_PER_FRAME`: slows the frames down to watch the growth.
- Spark checkpoints and RocksDB state go in the Docker volume `pipeline-checkpoints`, not on the Windows bind mount.
- `NETWORK_UI_PORT`

---

## 11. Validation checklist (run before every inference run)

| Check | On failure |
|---|---|
| `sha256(best.pt)` equals Section 1 | refuse to run |
| `settings['model']` dimensions equal Section 1, and `edge_feat_dim == 49` | refuse |
| Package column sets and order equal Section 3 | stop the run |
| Batch offset range matches `kafka_transaction_count` (Section 4) | stop the run |
| Event fields complete, finite and non-negative (Section 4) | exclude the row; count it as `invalid_transactions` |
| Unseen currency or format count > 0 | allowed; log the count |
| Registry within capacity | fail the run |
| `is_laundering` not present in any model input frame | refuse |
| Package order has no gaps; `batch_id` not already applied | refuse |
| Output `risk_score` within [0, 1] and finite | fail the batch |

---

## 12. Known limitations (disclosed, not fixed)

1. **Training-time feature leak.** The model was trained on HI-Medium static features computed over the full history, including `hours_since_last_txn` and `active_span_hours`. Streamed values mean something different and make every account look like a mule (§6a). Without a retrain, the static inputs are neutralised, and the model runs on its stream-derived signal alone.
2. **Streaming totals start empty.** Static feature values early in the run are far below HI-Medium's training distribution. The warm-up flag covers the evaluation side of this.
3. **"Ever laundered" account labels.** These are used for offline evaluation only.
4. **Networks never split.** Once two accounts are linked, they stay in the same network even if their scores drop later. Those accounts are recoloured instead.
5. **Possible giant network.** One busy yellow or red account can pull many counterparties into one network. The top-25 view and the render cap keep the UI usable, but watch whether one network dominates the ranking.
6. **Pipeline throughput.** The handover demo took 302 s for 2 batches of 500. At 1,000 per batch, all of HI-Small (about 5M transactions, about 5,000 batches) would take days in the pipeline. Model scoring isn't the bottleneck. Use a bounded replay window or larger batches for full-dataset runs.
