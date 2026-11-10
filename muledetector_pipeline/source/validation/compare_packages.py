"""Compare completed clean-version packages with a previously verified run.

This reads bounded Parquet batches without starting Spark. It is a regression
tool only, not a second implementation of the processing pipeline.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TABLE_KEYS = {
    "account_features": ["account_id"],
    "bank_features": ["bank_id"],
    "entity_features": ["entity_id"],
    "account_to_account_edges": ["sender_account_id", "receiver_account_id"],
    "account_to_bank": ["account_id", "bank_id"],
    "entity_to_account": ["entity_id", "account_id"],
}


def read_table(directory):
    files = sorted(directory.glob("*.parquet"))
    if not files or not (directory / "_SUCCESS").is_file():
        raise ValueError(f"Completed Parquet table not found: {directory}")
    return pa.concat_tables([pq.ParquetFile(path).read() for path in files])


def compare(expected_root, actual_root):
    expected_batches = {p.name for p in expected_root.glob("batch_id=*") if p.is_dir()}
    actual_batches = {p.name for p in actual_root.glob("batch_id=*") if p.is_dir()}
    if not expected_batches or expected_batches != actual_batches:
        raise ValueError(f"Batch sets differ: expected={sorted(expected_batches)}, actual={sorted(actual_batches)}")
    results = []
    for batch in sorted(expected_batches, key=lambda name: int(name.split("=")[1])):
        for table, keys in TABLE_KEYS.items():
            try:
                expected = read_table(expected_root / batch / f"{table}.parquet")
                actual = read_table(actual_root / batch / f"{table}.parquet")
                if expected.column_names != actual.column_names:
                    raise ValueError("Column order differs")
                if list(expected.schema.types) != list(actual.schema.types):
                    raise ValueError("Column types differ")
                left, right = expected.to_pandas(), actual.to_pandas()
                for label, frame in (("expected", left), ("actual", right)):
                    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
                        raise ValueError(f"{label} has null or duplicate complete keys")
                left = left.sort_values(keys).reset_index(drop=True)
                right = right.sort_values(keys).reset_index(drop=True)
                pd.testing.assert_frame_equal(left[keys], right[keys], check_exact=True)
                features = [name for name in expected.column_names if name not in keys]
                if features:
                    np.testing.assert_allclose(left[features].to_numpy(dtype=float),
                                               right[features].to_numpy(dtype=float),
                                               atol=1e-6, rtol=0, equal_nan=True)
                result = {"batch": batch, "table": table, "status": "passed", "rows": len(right)}
            except Exception as error:
                result = {"batch": batch, "table": table, "status": "failed", "error": str(error)}
            results.append(result)
            print(f"{batch} table={table} {'PASS' if result['status'] == 'passed' else 'FAIL: ' + result['error']}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-run-dir", type=Path, required=True)
    parser.add_argument("--actual-run-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    args = parser.parse_args()
    results = compare(args.expected_run_dir, args.actual_run_dir)
    passed = all(row["status"] == "passed" for row in results)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps({"status": "passed" if passed else "failed", "comparisons": results}, indent=2) + "\n")
    print(f"overall={'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
