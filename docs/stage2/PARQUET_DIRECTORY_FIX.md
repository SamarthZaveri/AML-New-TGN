# Spark Parquet directory hotfix

The initial v3 fingerprint helper called `open(path, 'rb')` for each input. Spark commonly writes a directory named `account_labels.parquet` containing part files. pandas can read that directory, but opening it as a regular file raises `PermissionError` on Windows. This was a regression in v3's new fingerprinting code, not a model-training failure.

The corrected helper accepts regular files and directories for every input: labels, static features, metadata edges, events, and chain signals. Directory hashing processes files recursively in sorted relative-path order. Relative names and content hashes are encoded unambiguously. Moving the complete directory does not change the hash; changing, renaming, adding, or removing a file does. Empty directories fail explicitly. Genuine file-access errors still propagate.

Only `stage2/dataset.py` changes production behavior. The test update covers all eight input Parquet datasets as multi-part directories, content changes, directory relocation, renaming and the unchanged regular-file digest.

No data conversion, deletion, administrator session or checkpoint reset is required for this startup failure. No training checkpoint was created before the reported exception. Existing regular-file hashes remain unchanged.

Apply `MuleDetector_Stage2_v3_Parquet_Fix.zip` at the project root. It contains the corrected dataset module, updated regression tests and updated validation documents. The full v3 archive has also been refreshed.

If the failing path is an actual file rather than a directory, investigate its real access permissions; this patch intentionally does not bypass them. In PowerShell, inspect it with:

```powershell
(Get-Item .\data\processed\spark_static_full\account_labels.parquet).PSIsContainer
```
