# Transaction and account experiments

Implementation added without executing training, imports, tests or syntax checks, per user request. Existing Stage 2 trainer and checkpoints are retained.

## Experiment definitions

Separate: account TGN/TGAT classification and a transaction MLP trained alongside it. Transaction head consumes detached pre-batch endpoint representations and event attributes; transaction loss cannot update the shared temporal backbone. Final account risk is (1-alpha)*account_probability + alpha*mean_incident_transaction_probability. The two heads share computed context but are not two independently trained whole models. Alpha defaults to a fixed 0.5; choose/tune it using validation only.

Fused: transaction head consumes differentiable endpoint context. Its predictions accumulate as incident transaction mean risk and count. Account final readout learns from account embedding, mean transaction risk and log count. Both account loss and transaction loss update the shared model. The original account classifier is retained as a diagnostic but does not receive a direct account loss in fused mode; do not interpret its metrics as an independently trained baseline.

Transaction predictions use pre-batch account memories/static projections, amounts, time-since-last-activity encoding and transaction attributes. Account predictions are batch-end, after current transactions and optional relay. Transactions within one batch do not update each other's pre-batch transaction-head context.

Features: normalized log amounts, self transfer, cross bank, currency mismatch, cyclical UTC hour and day-of-week, and one-hot payment currency, receiving currency and payment format. Vocabularies and amount/feature normalization fit only training time prefix; unseen categories map to a dedicated bucket. No is_laundering or future aggregate edge statistics enter the model. event_idx identifies each exported transaction.

Transaction labels are the original is_laundering labels from events.parquet, never inferred from account labels. Training transaction loss uses all early-window transactions, including transfers involving held-out account IDs. Thus account-label holdout is not a holdout of all supervision associated with those accounts. Interpret improvements with this protocol stated; a stricter held-out-endpoint experiment would require an additional edge mask.

## Truncated BPTT

--bptt-batches 1 keeps no gradient history across optimizer windows. Values >1 preserve sparse live account/bank/entity memories and transaction-risk sums across consecutive batches. One averaged window loss is backpropagated, gradients clipped, optimizer stepped, then live state detached. Neighbor attention reads live memories inside a window. Dense history remains detached for storage. Windows, not individual batches, are safe checkpoint/interrupt boundaries. This is truncated BPTT, not full-history BPTT. It trades GPU memory and Python sparse-state lookup overhead for cross-batch gradients. Try 2 with 1,000-event batches; reduce batch/window size on OOM. No runtime memory guarantee has been measured.

## Run from PowerShell

```powershell
cd C:\Users\Samarth\Desktop\GNN\ml_ly_latest
.\venv\Scripts\Activate.ps1

python -m stage2.train_transactions --mode separate --run-dir runs/txn_separate_bptt2 --batch-size 1000 --bptt-batches 2 --num-epochs 20 --num-seeds 1 --export-scores

python -m stage2.train_transactions --mode fused --run-dir runs/txn_fused_bptt2 --batch-size 1000 --bptt-batches 2 --num-epochs 20 --num-seeds 1 --export-scores
```

Run sequentially. Ctrl+C finishes the current optimizer window and saves latest.pt. Resume using the same command with --resume. To disable cross-batch gradients use --bptt-batches 1 and a new run directory; do not change a resumed run's settings. For a variance study use --num-seeds 3 in a new directory. No changes to the original stage2_chrono_5 run.

Results: per-seed history/results/checkpoints and top-level summary.json with separate transaction, base-account and final-account metrics. Best epoch chosen by validation final-account AP; each task gets its own validation-selected F1 threshold, frozen for test. Transaction metrics cover all events in the test time window; account metrics cover held-out test accounts active in that window. Scores export contains original event IDs, transaction probabilities/labels, endpoints and timestamps, plus final account scores.

The network frontend is not automatically rewired: its existing geometric-mean edge heuristic remains until transaction_scores.parquet is explicitly integrated. The new exports provide learned per-transaction scores for that later integration. Static account/bank/entity aggregates remain offline and require cutoff-safe reconstruction for prospective claims. Old and new checkpoints have different architecture, feature dimensions and dataset fingerprints and cannot be interchanged. Rich event feature storage and TBPTT increase host/GPU memory demand.
