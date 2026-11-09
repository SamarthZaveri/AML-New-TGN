# MuleDetector Stage 2 v3

Corrected temporal account classification and resumable training.

This package replaces the Stage 2 implementation supplied with the original failed training run. The main comparison is **original upload → this delivered version**, not merely the intermediate revision. Your intermediate fixes informed this implementation and are credited in the comparison.

Read [Original vs v3](docs/stage2/ORIGINAL_VS_V3.md) for the technical explanation and [Runbook](docs/stage2/RUNBOOK.md) for commands. [Validation](docs/stage2/VALIDATION.md) records actual tests and limitations. [Handoff](docs/stage2/UPDATES.md) gives a compact summary for the next coding session.

## Install into your existing project

Keep your existing root `config.py`, data, environment, Stage 1, and other project files. The original `config.py` and source datasets were not provided; this package uses their existing feature-column and path contracts. It does not invent feature names.

Back up `stage2`, then extract the archive at the project root. It contains:

- All nine original Stage 2 filenames. Eight are replaced; `__init__.py` is retained.
- Two new modules: `stage2/checkpointing.py` and `stage2/preprocessing.py`.
- `tests/test_stage2_v3.py` and documentation under `docs/stage2/`.
- This guide and `requirements-stage2.txt`.

No deletions are required. Existing `config.py`, root README, Stage 1 code, and old checkpoints are not overwritten. **Old checkpoints cannot resume this architecture.** Use a new run directory.

## Quick start (PowerShell, activated existing venv)

```powershell
Set-Location C:\Users\Samarth\Desktop\GNN\ml_ly_latest
Copy-Item .\stage2 ('.\stage2_backup_' + (Get-Date -Format 'yyyyMMdd_HHmmss')) -Recurse
Expand-Archive "$env:USERPROFILE\Downloads\MuleDetector_Stage2_v3.zip" -DestinationPath . -Force
python -m unittest discover -s tests -p test_stage2_v3.py -v
python -m stage2.train --num-seeds 1 --num-epochs 3 --batch-size 10000 --run-dir runs\stage2_v3
```

Adjust the ZIP path if you downloaded it elsewhere. Keep your working CUDA PyTorch installation; do not replace it with the CPU wheel used for package testing.

The first run uses relay, focal classification, and **zero auxiliary-loss weights**. This intentionally establishes a working classifier before assessing optional objectives. Configuration values for the old auxiliary defaults are not inherited.

Press Ctrl+C once to finish the current batch and save. Resume with the same command plus `--resume`:

```powershell
python -m stage2.train --num-seeds 1 --num-epochs 3 --batch-size 10000 --run-dir runs\stage2_v3 --resume
```

## Verified and not yet verified

17 CPU regression tests passed, including exact uninterrupted-versus-resumed model/optimizer/RNG comparison, differentiable relay, a small overfit check, consistent scoring, flow decay, and deterministic neighbor retention. No full-data or CUDA performance result is claimed. The synthetic fixtures validate software behavior; they are not evidence of useful mule-detection accuracy.

Spark Parquet folders are supported by the corrected fingerprint helper. See `docs/stage2/PARQUET_DIRECTORY_FIX.md`.
