"""Batch audit helpers, not a separate streaming application."""
from __future__ import annotations
from pathlib import Path
import json
import time
from pyspark import StorageLevel
from pyspark.sql import DataFrame, functions as F
from streaming.ingestion.events import split_valid_and_rejected_events

def rejected_batch_output_path(
    rejected_output_dir: Path,
    run_id: str,
    batch_id: int,
) -> Path:
    """Return the deterministic rejected-event path for one batch."""
    return (
        rejected_output_dir
        / f"run_id={run_id}"
        / f"batch_id={batch_id}"
        / "rejected_account_events.parquet"
    )


def metrics_batch_output_path(
    metrics_output_dir: Path,
    run_id: str,
    batch_id: int,
) -> Path:
    """Return the deterministic JSON metrics path for one batch."""
    return (
        metrics_output_dir
        / f"run_id={run_id}"
        / f"batch_id={batch_id}.json"
    )


def write_audit_batch(
    batch_events: DataFrame,
    batch_id: int,
    run_id: str,
    rejected_output_dir: Path,
    metrics_output_dir: Path,
    totals: dict,
    release_input: bool = True,
) -> None:
    """Validate, retain rejections, and record metrics for one batch.

    A shared stateless-input owner may retain the input cache for another write
    using release_input=False; it must unpersist that input in its own finally.
    """
    started = time.perf_counter()
    batch_events = batch_events.persist(StorageLevel.MEMORY_AND_DISK)
    valid_events = None
    rejected_events = None

    try:
        valid_events, rejected_events = split_valid_and_rejected_events(
            batch_events
        )
        valid_events = valid_events.persist(StorageLevel.MEMORY_AND_DISK)
        rejected_events = rejected_events.persist(
            StorageLevel.MEMORY_AND_DISK
        )

        kafka_transactions = (
            batch_events
            .select(
                "kafka_topic",
                "kafka_partition",
                "kafka_offset",
            )
            .distinct()
        )
        kafka_transaction_count = kafka_transactions.count()
        valid_event_count = valid_events.count()
        rejected_event_count = rejected_events.count()
        expected_event_count = kafka_transaction_count * 2

        partition_offset_rows = (
            kafka_transactions
            .groupBy("kafka_topic", "kafka_partition")
            .agg(
                F.min("kafka_offset").alias("minimum_offset"),
                F.max("kafka_offset").alias("maximum_offset"),
            )
            .orderBy("kafka_topic", "kafka_partition")
            .collect()
        )
        partition_offsets = [
            {
                "kafka_topic": str(row["kafka_topic"]),
                "kafka_partition": int(row["kafka_partition"]),
                "minimum_offset": int(row["minimum_offset"]),
                "maximum_offset": int(row["maximum_offset"]),
            }
            for row in partition_offset_rows
        ]

        rejected_events.write.mode("overwrite").parquet(
            str(
                rejected_batch_output_path(
                    rejected_output_dir,
                    run_id,
                    batch_id,
                )
            )
        )

        validation_passed = (
            valid_event_count + rejected_event_count
            == expected_event_count
        )
        processing_seconds = round(
            time.perf_counter() - started,
            3,
        )

        metrics_record = {
            "run_id": run_id,
            "batch_id": batch_id,
            "kafka_transaction_count": kafka_transaction_count,
            "expected_account_event_count": expected_event_count,
            "valid_account_event_count": valid_event_count,
            "rejected_account_event_count": rejected_event_count,
            "processing_seconds": processing_seconds,
            "partition_offsets": partition_offsets,
            "validation_status": (
                "passed" if validation_passed else "failed"
            ),
        }

        metrics_path = metrics_batch_output_path(
            metrics_output_dir,
            run_id,
            batch_id,
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_metrics_path = metrics_path.with_suffix(".json.tmp")
        temporary_metrics_path.write_text(
            json.dumps(metrics_record, indent=2),
            encoding="utf-8",
        )
        temporary_metrics_path.replace(metrics_path)

        totals["kafka_transactions"] += kafka_transaction_count
        totals["valid_events"] += valid_event_count
        totals["rejected_events"] += rejected_event_count
        totals["batches"] += 1
        totals["processing_seconds"] += processing_seconds
        totals["validation_passed"] = (
            totals["validation_passed"] and validation_passed
        )
    finally:
        if valid_events is not None:
            valid_events.unpersist()
        if rejected_events is not None:
            rejected_events.unpersist()
        if release_input:
            batch_events.unpersist()
