# Stage 2 v3 handoff

## Baseline

Original nine-file upload and failed run: 31.9M events, 2.077M accounts, 2.02% positives; epoch-2 validation AP 0.0272, F1 0.0406. The original train/eval representations differed, relay detached upstream learning paths, time/feature handling was unsafe, and recovery was best-epoch-only. See ORIGINAL_VS_V3.md for the full comparison. Do not mistake the user-provided intermediate revision for the baseline.

## Implemented

- Shared TGAT/static/count readout; frozen-weight replay for validation/test, with the same relay schedule.
- Live batch-local bank/entity relay, trainable account/bank/entity feature encoders, directional transaction messages, zero historical initialization.
- Relative float64 hours, explicit first-seen flags, untouched transaction clocks during relay, log time encoding.
- Deterministic recent-event ring updates across both directions, safe all-masked attention, current-batch neighbors included before batch-end scoring.
- Training-fitted saved transforms and source hashes; explicit rejection of ambiguous IDs/mappings and invalid numeric input.
- Correct common-cutoff exponential flow accounting; training-only auxiliary supervision; unknown chain labels masked; positive-only binary membership disabled.
- Primary-only objective defaults; optional flow/chain losses and relay ablation; validation-only threshold selection and richer metrics.
- Full batch-boundary checkpoints including histories, optimizer, RNG, cursor, losses and compatibility checks; atomic writes; safe SIGINT; pre-replay recovery checkpoint.
- Compact best weights and docs/tests.

## Protocol decisions

Transductive stratified account split, not a future-time holdout. Batch-end post-observation scoring, cached last-touch logits for each account. History is truncated between batches; messages within a batch are aggregated, not sequential GRU updates per event. Static data and Stage 1 signals need an upstream as-of/leakage audit before live-detection claims.

## Validation

17 CPU tests passed, including two Spark Parquet directory regressions. Exact resumed versus uninterrupted model/optimizer/RNG comparison passed with dropout and optional losses enabled. Gradient, scoring-equivalence, neighbor-overflow, flow-decay, masked-attention, transform, data-validation, threshold, small-overfit, failed-write and replay-interruption checks passed. Tests used Python 3.12.14/PyTorch 2.14.0+cpu. Code is written for Python 3.11+; Python 3.11 syntax was checked separately. Full user dataset and CUDA execution are untested.

## Next real-data work

Run corrected primary-only baseline; inspect train vs validation AP and predicted-positive rates. Compare relay and auxiliary objectives on validation only. Audit feature and chain generation if performance remains weak. The project root config and source datasets were not provided, so column names, label provenance, source timestamp units, relationship cardinality and as-of correctness remain dependent on the user's pipeline. No claimed full-data metric improvement or throughput improvement yet.
