# Offline Network Post-processing

This layer runs after account scoring over a completed historical graph. It is
separate from the real-time inference engine: it does not retain memory or
alter account scores. It converts a score export into connected suspicious
networks and evidence paths for an investigator or visual graph UI.

## Input

`--account-scores` is a Parquet file with one row per account and these fields:

```text
account_id, risk_score
```

`mule_score`, `score`, and `probability` are accepted aliases for
`risk_score`. Scores must be probabilities in `[0, 1]`.

## Run

First export one completed-history risk score per account from the selected
checkpoint. Use the *same batch size* as training so the checkpoint fingerprint
matches:

```powershell
python -m stage2.export_scores `
  --checkpoint runs\stage2_chrono_5\seed_42\best.pt `
  --output runs\stage2_chrono_5\account_scores.parquet `
  --batch-size 10000
```

Then build the offline investigation graph:

```powershell
python -m stage2.network_postprocess `
  --events-file data\processed\event_stream\events.parquet `
  --account-scores runs\stage2_chrono_5\account_scores.parquet `
  --output-dir runs\stage2_chrono_5\networks `
  --red-threshold 0.80 --yellow-threshold 0.55 --context-min-score 0.15
```

The module retains accounts above `context-min-score`, extracts transactions
where both endpoints are retained, creates weakly connected components, and
then reconstructs ordered two-hop flow paths. A flow path requires an inbound
payment followed by an outbound payment through the same intermediary within
72 hours, with outbound/inbound amount ratio between 0.70 and 1.30.

The amount and time criteria are evidence filters, not a declaration of money
laundering. They are deliberately configurable.

## Outputs

| File | Contents |
|---|---|
| `network_nodes.parquet` | Account score, alert bin, and network ID. |
| `network_edges.parquet` | Directed account-to-account links aggregated across completed history. |
| `flow_paths.parquet` | Time-ordered source → intermediary → target transfers with time and amount consistency. |
| `networks.parquet` | Component-level counts, value, risk aggregation, path count and network score. |
| `manifest.json` | Inputs, thresholds, and output sizes for reproducibility. |

`network_score` is a transparent prioritisation score, not a second trained
model. It combines maximum and mean account risk, path count, suspicious
value, and component density. The account model remains the source of risk;
the network score helps order investigations.

## Scaling controls

The default five-million-event cap prevents an overly broad score threshold
from materialising a huge offline subgraph. If it is reached, raise
`context-min-score` first. Only increase `--max-candidate-events` when the
machine has sufficient RAM and a deliberately bounded investigation scope.
