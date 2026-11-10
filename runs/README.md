# Stage 2 experiments: what we ran, what we found, what we use

This file covers the Stage 2 temporal GNN experiments under `runs/`.
The static Spark feature pipeline is described in the top-level `README.md`.

## TL;DR

- **Final model:** one model, separate mode, seed 42: `runs/txn_separate_seeds/seed_42/best.pt` (epoch 8). It was picked because it has the highest validation AUPRC of the 3 seeds.
- **Inference:**
  - Account risk = `base_account_risk_score`. Do **not** use `risk_score`, the fixed 0.5 blend.
  - Transaction risk = `transaction_risk_score`.
  - Thresholds are the ones stored in `best.pt`.
- **Test AUPRC:**

  | Score | Full test | Clean test | Positive rate |
  |---|---|---|---|
  | Account | **0.508** | **0.396** | 2.4% |
  | Transaction | **0.637** | **0.461** | 0.18% |

  The clean test drops the end-of-data tail ([Known issues](#known-issues)).
- Averaging the 3 seeds would score higher (0.523 account, 0.665 transaction), but it was left out to keep inference to a single model.
- The gradient-boosting stacker is **not** used. It added only about +0.01 per seed.
- The reported numbers are **optimistic**, because two account features leak future information ([Known issues](#known-issues)). Treat the clean-test numbers as the upper bound you can honestly expect.

## Setup shared by all runs

**Data**
- IBM AML HI-Medium: 31,898,238 transactions (events).
- Static account, bank and entity features come from the Spark pipeline in `data/processed/spark_static_full/`.

**Split**
- Events are sorted by time and split chronologically 70 / 15 / 15:
  - train: events [0, 22,328,508)
  - val: [22,328,508, 27,112,237)
  - test: [27,112,237, 31,898,238)
- Account labels are also held out, using a stratified random 70 / 15 / 15 account split with seed 42.
- Test scores only held-out test accounts that are active in the test window: 216,900 accounts, 2.37% positive. Transaction metrics cover all 4,786,001 test-window transactions, 0.18% positive.

**Model**
- TGN memory: a GRU with separate messages for incoming and outgoing transactions.
- Relay messages through bank and entity hub nodes.
- TGAT attention over each node's last 20 neighbors, with Time2Vec time encoding.
- Truncated BPTT over 2 batches of 10,000 events.
- Focal loss (gamma 2), with class weights capped at 50 for accounts and 500 for transactions.
- Hidden size 128, time dimension 64, 4 attention heads, learning rate 1e-4, up to 20 epochs, patience 5.

**Evaluation**
- Replay train, then val, then test, starting from empty memory.
- An account's score is its score after its last transaction in the window.
- The decision threshold is the F1-max threshold on validation, frozen for test.
- AUPRC is sklearn's average precision.

**Heads in the transaction trainer**
- `account_base`: account head on the GNN embedding.
- `transaction`: per-transaction head.
- `account_final`: a combination of the two, which depends on the mode:
  - **Separate mode:** the account loss trains the backbone and the base head. The transaction head reads detached features, so it doesn't change the backbone. `account_final` is a fixed, untrained 0.5 blend of the base score and mean transaction risk.
  - **Fused mode:** the transaction loss trains the backbone. The account loss trains only the final head. The base head gets no gradient.

## Experiments

### Account-only trainer (`stage2.train`)

| Run | What it is | Test account AUPRC | Precision / recall at val threshold | Comparable? |
|---|---|---|---|---|
| `stage2` | First run. Random account split; trained over the **full** event stream. | 0.440 | 0.81 / 0.32 | **No.** Different split, and training saw test-period events. Don't resume or compare. |
| `stage2_chrono_5` | Chronological split (same as above), account loss only. Best epoch 7. | 0.457 | 0.62 / 0.37 | Yes, same test set |

### Joint account + transaction trainer (`stage2.train_transactions`)

All of these use seed 42 and BPTT over 2 batches of 10,000.

| Run | Mode | Best epoch | Account base | Account final | Transaction | Notes |
|---|---|---|---|---|---|---|
| `txn_fused_bptt2` | fused | 9 | 0.094 (untrained) | 0.482 | 0.664 | Base head gets no gradient in fused mode, so its score is meaningless |
| `txn_separate_bptt2` | separate | 14 | **0.493** | 0.386 | **0.675** | Epoch was chosen by `account_final`, which is a bug (see below) |

Separate mode's base head was the best account model, and it beat the account-only trainer (0.493 vs 0.457). Training a transaction head alongside the account head helps the account head.

### Final: separate mode, 3 seeds (`txn_separate_seeds`)

Same settings as `txn_separate_bptt2`, with two fixes:

1. **Epoch selection.** In separate mode, the best epoch is now chosen by validation `account_base` AUPRC. Before, it used `account_final`, an untrained blend that peaked at epoch 14 more or less by chance.
2. **Validation exports.** `--export-scores` now also writes validation-window scores (`scores_val/`), which the stacker needs.

**Test AUPRC per seed**

| Seed | Best epoch | Account base (full) | Account base (clean) | Transaction (full) | Transaction (before 09-17) |
|---|---|---|---|---|---|
| 42 | 8 | 0.508 | 0.396 | 0.637 | 0.461 |
| 7 | 18 | 0.488 | 0.374 | 0.669 | 0.508 |
| 2024 | 3 (stopped early at 8) | 0.464 | 0.346 | 0.616 | 0.409 |
| **Mean ± std** | | 0.487 ± 0.018 | 0.372 ± 0.020 | 0.641 ± 0.022 | |
| **Average of the 3 seeds' scores** | | **0.523** | **0.412** | **0.665** | **0.482** |

At the frozen threshold, the base head reaches precision **0.82** and recall **0.35**, mean over seeds. It's confident on the accounts it flags but misses about two thirds of laundering accounts.

`account_final` (the fixed blend) scored 0.362 ± 0.007. It's consistently worse than the base head, so don't use it.

**Same seed, old vs new.** Comparing seed 42 in the earlier run with seed 42 here:
- The new run is better for accounts: 0.508 vs 0.493 on full test, 0.396 vs 0.382 on clean test.
- It's worse for transactions: 0.637 vs 0.675.
- The only difference is the stopping epoch, 8 vs 14. Account validation AUPRC peaks around epoch 8 and then sags, while transaction AUPRC keeps rising through epoch 14 and beyond.

### Gradient-boosting stacker (`stage2.fit_stacker`)

**How it's built**
- One stacker per seed: a HistGradientBoosting model fitted on validation-window scores.
- Inputs are the base account logit plus, for sent and for received transactions, summary statistics of transaction risk (mean, max, top-3, log count, count above 0.5, count above 0.9).
- The threshold is chosen from 5-fold out-of-fold predictions on validation.

**Results (test AUPRC, mean ± std over seeds)**

| | Full test | Clean test |
|---|---|---|
| Base head | 0.487 ± 0.018 | 0.372 ± 0.020 |
| Stacked | 0.496 ± 0.015 | 0.380 ± 0.017 |

- **Gain:** about +0.01. It's positive for every seed, but almost zero for seed 42.
- **Trade-off:** precision drops from 0.82 to 0.77, and recall rises from 0.35 to 0.37.
- **Verdict:** averaging the three seeds gains more (+0.036) for less complexity, so the stacker is left out of the frozen model.

### Abandoned: `new_stage_2`

We started a from-scratch rewrite with three changes:
- account labels that depend on time,
- attention that knows transaction direction and edge features,
- account context fed into the transaction head.

It was deleted before a full run, and no results exist. Those ideas are still on the [Next steps](#next-steps) list.

## Known issues

These affect every run above, so they don't change the ranking between runs, but they do inflate the absolute numbers.

1. **End-of-data tail.**
   - From 2022-09-17 to 09-28 there are only about 7k events, but about 58% of them are laundering.
   - The tail holds 4,084 of the 8,661 positive transactions in the test window. Every one of the 742 test accounts active in it is positive.
   - This makes the full-test numbers look easier than they are. The "clean" columns drop those accounts, or for transactions, everything on or after 09-17.
2. **Static features leak the future.**
   - The Spark account features are computed over the whole history, including the val and test windows.
   - Two features are strong leaks. Using either feature alone to rank held-out test accounts gives AUPRC well above the 0.020 base rate:
     - `hours_since_last_txn`: 0.155
     - `active_span_hours`: 0.152
   - The other 22 account features give 0.018–0.085 on their own.
   - Bank and entity features are also computed over full history, but they leak less.
3. **"Ever laundered" account labels.**
   - An account counts as positive for its whole history.
   - 30% of positive training touches come before the account's first laundering transaction.
   - 14,724 positive accounts have no laundering at all in the training window.
4. **Noisy epoch selection and seed spread.**
   - The best epochs were 8, 18 and 3, and single seeds range from 0.46 to 0.51 on accounts.
   - Differences under about 0.02–0.03 can't be trusted from one run.

## What we use: the frozen model

**Model:** `runs/txn_separate_seeds/seed_42/best.pt`, which holds the weights and the validation thresholds.
- Its exports are in `seed_42/scores_val/` and `seed_42/scores/`.
- Its metrics are in `seed_42/history.json` and `seed_42/results.json`.

**Scoring.**
- Account risk = `base_account_risk_score`.
- Transaction risk = `transaction_risk_score`.
- Use the `account_base` and `transaction` thresholds saved in `best.pt`.

**Why seed 42.** It has the highest validation AUPRC of the 3 seeds (0.496 vs 0.473 and 0.444), so it was chosen without looking at test. Seeds 7 and 2024 are kept only as evidence of how much results vary between seeds.

**Reproduce**
```powershell
venv\Scripts\python -m stage2.train_transactions --mode separate --run-dir runs/txn_separate_seeds --seeds 42 7 2024 --batch-size 10000 --bptt-batches 2 --num-epochs 20 --export-scores
venv\Scripts\python -m stage2.fit_stacker --run-dir runs/txn_separate_seeds
```

Notes for running:
- Add `--resume` to continue after Ctrl+C or a crash.
- Seeds that haven't started yet begin fresh.
- Each seed folder is about 2.2 GB, mostly `latest.pt`.
- On an 8 GB GPU, training uses about 7.1 GB and once ran out of memory. Resuming worked.

## Next steps

1. Build an inference adapter for the streaming pipeline packages, using seed 42 only.
2. Retrain without the leaking features. At minimum, drop `hours_since_last_txn` and `active_span_hours`. Ideally, compute all static features from the training window only or as of each event's time.
3. Compare against a simple baseline, such as gradient boosting on static and transaction features only, so the GNN's added value is measured.
4. Revisit the `new_stage_2` ideas: time-aware labels, attention that knows direction and edge features, and account context in the transaction head.
