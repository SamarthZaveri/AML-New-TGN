# Validation report

## Result

**17 regression tests passed** on the final package. The recorded suite completed in 2.992 seconds. See `TEST_OUTPUT.txt` for the full captured output.

Environment actually used:

| Component | Version |
|---|---|
| Python | 3.12.14 |
| PyTorch | 2.14.0+cpu |
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| scikit-learn | 1.8.0 |
| PyArrow | 25.0.1 |
| Device | CPU |

The implementation uses Python 3.11-compatible syntax, verified with `ast.parse(feature_version=(3, 11))` for every Python file. Execution on Python 3.11 and CUDA was not available in this environment. Retain your existing GPU PyTorch installation; the table records testing, not a requirement to upgrade your environment.

## Checks performed

| Test | What it establishes |
|---|---|
| Spark multipart input directories | All input datasets load as directories containing part files and `_SUCCESS`; changed part content changes the digest |
| Directory identity | Digests are location-independent, detect renames and reject empty datasets; regular-file hashes are unchanged |
| Backbone gradient paths, three optimization steps | Incoming/outgoing transaction networks, account/bank/entity projections, GRUs and both directions of both relay relations receive finite, nonzero gradients |
| Shared readout | Evaluation-mode forward logits match chunked `predict_accounts` at the same cutoff; empty inference requests work |
| First-seen clocks and history | First observation has zero elapsed time, prior counts exclude current batch, relay does not change transaction clocks |
| Neighbor overflow and restore | Mixed incoming/outgoing events exceeding capacity keep the true latest entries; restored pointers continue correctly |
| Empty-neighbor attention | Float32 and float16 forward/backward remain finite for entirely masked neighborhoods |
| Conservation decay | Both sides decay correctly after a 72-hour gap; earlier contributions within a batch are discounted to batch end |
| Tiny overfit | Classification loss on a fixed small fixture falls below 15% of its initial value after 200 small optimization steps |
| Transform fit/round-trip | Held-out outliers do not affect fitted training statistics; serialized transform reproduces output |
| Dataset ordering/precision/chain unknowns | Unsorted events become chronological, fine relative times retain float64 precision, unlisted chains stay unknown, identity hashes are reproducible, duplicate IDs are rejected |
| Multi-owner rejection | An ambiguous account-to-entity relationship fails explicitly |
| Threshold metrics | A validation-derived threshold improves F1 on a controlled example without changing ranking |
| Checkpoint compatibility | Changed data identity or settings are rejected |
| Exact interrupted continuation | With dropout and both auxiliary weights enabled, an interruption after batch 2 resumes to exactly matching final model, conservation state, optimizer tensors and Torch RNG state versus uninterrupted CPU training |
| Failed atomic write | A simulated disk-write failure leaves the prior checkpoint byte-for-byte unchanged and cleans the temporary file |
| Interrupted replay | A simulated stop during validation preserves the completed-training cursor; resume rebuilds replay and completes evaluation |

The stop tests set the same flag used by the SIGINT handler at controlled points. They do not simulate power loss or a real Windows console Ctrl+C event. Atomic-write testing injects an exception; it does not establish durability against every filesystem failure.

## Interpretation and limits

These tests establish software properties, not real-data detection performance. The printed AP/F1 numbers in the test log come from tiny synthetic fixtures with very short training used to exercise recovery. They are not model-quality benchmarks and must not be compared with the user's 2.02%-prevalence run.

Not performed:

- Training or evaluation on the 31,898,238-event source dataset.
- GPU throughput, VRAM or checkpoint-size profiling at full scale.
- CUDA bitwise-resume validation.
- Audit of root `config.py`, feature-generation scripts, Stage 1 tracing code, source timestamp units, or label provenance; those were not supplied.
- Verification that static features and chain targets were available at each prediction cutoff.
- A future-time holdout experiment or production deployment test.

The next necessary evidence is a corrected primary-only run on the user's data, followed by controlled validation-based ablations. No numerical F1/AP improvement is promised in advance.
