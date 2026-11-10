"""Validate six tables and atomically publish a ready manifest."""
from __future__ import annotations
from pathlib import Path
import json
import pyarrow.parquet as pq
from preprocessing.spark_static_pipeline import ACCOUNT_FEATURE_COLUMNS, ACCOUNT_TO_ACCOUNT_EDGE_FEATURE_COLUMNS, BANK_FEATURE_COLUMNS, ENTITY_FEATURE_COLUMNS

EXPECTED_TABLE_COLUMNS = {
    "account_features": ["account_id", *ACCOUNT_FEATURE_COLUMNS],
    "bank_features": ["bank_id", *BANK_FEATURE_COLUMNS],
    "entity_features": ["entity_id", *ENTITY_FEATURE_COLUMNS],
    "account_to_account_edges": [
        "sender_account_id",
        "receiver_account_id",
        *ACCOUNT_TO_ACCOUNT_EDGE_FEATURE_COLUMNS,
    ],
    "account_to_bank": ["account_id", "bank_id"],
    "entity_to_account": ["entity_id", "account_id"],
}


def validate_table_and_count_rows(
    table_directory: Path,
    expected_columns: list[str],
) -> int:
    """Validate every Parquet footer and sum its metadata row count."""
    if not table_directory.is_dir():
        raise FileNotFoundError(
            f"Required table directory not found: {table_directory}"
        )

    parquet_files = sorted(table_directory.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"Required table contains no Parquet files: {table_directory}"
        )

    row_count = 0
    for parquet_path in parquet_files:
        try:
            parquet_file = pq.ParquetFile(parquet_path)
            actual_columns = parquet_file.schema_arrow.names
            row_count += parquet_file.metadata.num_rows
        except Exception as error:
            raise ValueError(
                f"Unable to read Parquet metadata: {parquet_path}"
            ) from error

        if actual_columns != expected_columns:
            raise ValueError(
                f"Schema mismatch for {parquet_path}. "
                f"Expected columns {expected_columns}, "
                f"found {actual_columns}."
            )

    return row_count


def package_batch(
    batch_directory: Path,
    run_id: str,
    batch_id: int,
) -> None:
    """Validate one complete batch and write its ready manifest."""
    counts = {}
    columns = {}

    for table_name, expected_columns in EXPECTED_TABLE_COLUMNS.items():
        table_directory = batch_directory / f"{table_name}.parquet"
        counts[table_name] = validate_table_and_count_rows(
            table_directory,
            expected_columns,
        )
        columns[table_name] = expected_columns

    manifest = {
        "kind": "incremental_model_update",
        "version": 1,
        "run_id": run_id,
        "batch_id": batch_id,
        "status": "ready",
        "description": (
            "These are only the rows affected by this Kafka micro-batch; "
            "feature values are cumulative through this batch."
        ),
        "counts": counts,
        "columns": columns,
    }

    manifest_path = batch_directory / "manifest.json"
    temporary_manifest_path = manifest_path.with_suffix(".json.tmp")
    temporary_manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest_path.replace(manifest_path)

    print(
        f"Packaged run_id={run_id}, batch_id={batch_id}: "
        "6 tables ready."
    )
