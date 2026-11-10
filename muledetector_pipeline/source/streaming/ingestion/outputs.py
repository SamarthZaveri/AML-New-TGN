"""Final writes for the single native streaming pipeline."""
from __future__ import annotations
from pathlib import Path
from pyspark import StorageLevel
from pyspark.sql import DataFrame, functions as F
from streaming.state.stateful_account_processor import finalize_stateful_account_feature_updates
from streaming.state.stateful_edge_processor import finalize_stateful_account_edge_updates
from streaming.state.stateful_bank_processor import finalize_stateful_bank_feature_updates
from streaming.state.stateful_entity_processor import finalize_stateful_entity_feature_updates


def _batch_output_path(output_dir, run_id, batch_id, table_name):
    return output_dir / f"run_id={run_id}" / f"batch_id={batch_id}" / f"{table_name}.parquet"


def validate_account_mappings(account_mappings: DataFrame) -> None:
    """Reject a batch containing an unstable Bank or Entity mapping."""
    conflicts = (
        account_mappings
        .groupBy("account_id")
        .agg(
            F.countDistinct("bank_id").alias("bank_count"),
            F.countDistinct("entity_id").alias("entity_count"),
        )
        .filter(
            (F.col("bank_count") > 1)
            | (F.col("entity_count") > 1)
        )
        .limit(1)
        .collect()
    )

    if not conflicts:
        return

    conflict = conflicts[0]
    problems = []
    if conflict["bank_count"] > 1:
        problems.append("multiple Banks")
    if conflict["entity_count"] > 1:
        problems.append("multiple Entities")

    raise ValueError(
        f"Account {conflict['account_id']} maps to "
        f"{' and '.join(problems)} in the current batch."
    )

def write_account_feature_batch(
    account_feature_updates: DataFrame,
    batch_id: int,
    output_dir: Path,
    run_id: str,
) -> None:
    """Write the 24-feature Account updates produced by one micro-batch."""
    account_features = finalize_stateful_account_feature_updates(
        account_feature_updates
    )

    batch_output_dir = _batch_output_path(output_dir, run_id, batch_id, 'account_features')

    account_features.write.mode("overwrite").parquet(
        str(batch_output_dir)
    )

    print(
        f"Batch {batch_id}: Account feature updates written to "
        f"{batch_output_dir}"
    )

def write_account_edge_batch(
    account_edge_updates: DataFrame,
    batch_id: int,
    output_dir: Path,
    run_id: str,
) -> None:
    """Write finalized directed Account-edge updates for one batch."""
    account_edges = finalize_stateful_account_edge_updates(
        account_edge_updates
    )
    output_path = _batch_output_path(output_dir, run_id, batch_id, 'account_to_account_edges')

    account_edges.write.mode("overwrite").parquet(str(output_path))

def write_bank_feature_batch(
    bank_updates: DataFrame,
    batch_id: int,
    output_dir: Path,
    run_id: str,
    bank_account_counts: DataFrame,
) -> None:
    """Finalize and write one micro-batch of Bank feature updates."""
    bank_features = finalize_stateful_bank_feature_updates(
        bank_updates,
        bank_account_counts,
    )
    output_path = _batch_output_path(output_dir, run_id, batch_id, 'bank_features')

    bank_features.write.mode("overwrite").parquet(str(output_path))

def write_entity_feature_batch(
    entity_updates: DataFrame,
    batch_id: int,
    output_dir: Path,
    run_id: str,
    entity_types: DataFrame,
) -> None:
    """Finalize and write one micro-batch of Entity updates."""
    entity_features = finalize_stateful_entity_feature_updates(
        entity_updates,
        entity_types,
    )
    output_path = _batch_output_path(output_dir, run_id, batch_id, 'entity_features')

    entity_features.write.mode("overwrite").parquet(str(output_path))

def write_account_relation_batch(
    valid_account_events: DataFrame,
    batch_id: int,
    output_dir: Path,
    run_id: str,
) -> None:
    """Validate and write stable Account relations for one batch."""
    account_mappings = valid_account_events.select(
        "account_id",
        "bank_id",
        "entity_id",
    ).distinct().persist(StorageLevel.MEMORY_AND_DISK)

    try:
        validate_account_mappings(account_mappings)

        account_to_bank = account_mappings.select(
            "account_id",
            "bank_id",
        ).distinct()
        entity_to_account = account_mappings.select(
            "entity_id",
            "account_id",
        ).distinct()

        account_to_bank.write.mode("overwrite").parquet(
            str(
                _batch_output_path(output_dir, run_id, batch_id, "account_to_bank")
            )
        )
        entity_to_account.write.mode("overwrite").parquet(
            str(
                _batch_output_path(output_dir, run_id, batch_id, "entity_to_account")
            )
        )
    finally:
        account_mappings.unpersist()
