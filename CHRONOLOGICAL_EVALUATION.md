# Stage 2 chronological evaluation (five-epoch run)

Use these files in the existing project, replacing `stage2/dataset.py` and
`stage2/train.py`, and adding the updated `tests/test_stage2_v3.py`. From the
project root in PowerShell:

```powershell
python -m unittest discover -s tests -p test_stage2_v3.py -v
python -m stage2.train --split-mode chronological --num-seeds 1 --num-epochs 5 --batch-size 10000 --patience 5 --run-dir runs\stage2_chrono_5
```

The event stream is sorted by timestamp and divided into earliest 70% train,
next 15% validation, and latest 15% test. Simultaneous timestamps stay in one
window. Training updates weights from the train events alone. At each epoch,
validation replays train then validation events with frozen weights; it scores
only held-out validation-label accounts active in the validation window. The
best checkpoint is selected by validation average precision, with its decision
threshold fitted on validation. After training, test replays train, validation,
then test and scores only held-out test-label accounts active in the test
window, using the saved threshold. No test events affect training or validation.
Edge amount normalization fits only early events connecting train-label
accounts. Recovery checkpoints bind the split, source hashes, and settings.

The label groups remain stratified, randomly disjoint account sets. This
protocol tests future event windows with withheld account labels; it does not
establish a fully prospective deployment score. Static account, bank, entity,
ownership, and chain input files must be snapshots available by the training
cutoff. If they contain information computed over the full 28-day stream, their
lookahead remains and the reported metrics are optimistic. Account labels also
need event-time availability to assess when an alert could first be made.
Scores are last observed *after* each account's final micro-batch in its
evaluation window; the metrics do not score each incoming transaction.

Do not resume the old `runs\stage2` account-split experiment: it trained over
the complete event stream. Start a new run directory. This environment has
only the synthetic fixture, so the included test completes a five-epoch CPU
run on 160 synthetic events. The 31.9-million-event run must be executed on
your machine with the Parquet data and CUDA installation. To resume the new
run after stopping, use the same command and add `--resume`.
