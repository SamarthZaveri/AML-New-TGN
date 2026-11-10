"""One-time preparation of the historical baseline and fixed metadata lookup."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys

if sys.platform == "win32":
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions-path", type=Path, required=True)
    parser.add_argument("--accounts-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/spark_static_full"))
    parser.add_argument("--cutoff", default="2022/09/01 00:30")
    args = parser.parse_args()
    lookup_path = args.output_dir.parent / "account_entity_lookup.parquet"
    for path in (args.transactions_path, args.accounts_path):
        if not path.is_file():
            parser.error(f"Input CSV not found: {path}")
    for path in (args.output_dir, lookup_path):
        if path.exists():
            raise FileExistsError(f"Preparation output already exists: {path}. Use the existing prepared data or choose a fresh directory.")
    os.environ["AML_TRANSACTIONS_PATH"] = str(args.transactions_path.resolve())
    os.environ["AML_ACCOUNTS_PATH"] = str(args.accounts_path.resolve())
    from preprocessing import spark_static_pipeline, build_account_metadata_lookup

    previous_arguments = sys.argv
    try:
        sys.argv = ["prepare", "--full", "--end-timestamp", args.cutoff,
                    "--output-dir", str(args.output_dir), "--write-account-events"]
        spark_static_pipeline.main()
        sys.argv = ["prepare", "--output-dir", str(lookup_path)]
        build_account_metadata_lookup.main()
    finally:
        sys.argv = previous_arguments


if __name__ == "__main__":
    main()
