"""Independent feature recomputation for tests only; not a pipeline runner."""
from __future__ import annotations
from pathlib import Path
from pyspark.sql import DataFrame, SparkSession, functions as F
from preprocessing.spark_static_pipeline import (ACCOUNT_FEATURE_COLUMNS, BANK_FEATURE_COLUMNS, ENTITY_FEATURE_COLUMNS, compute_directional_account_features, compute_shared_account_features, compute_temporal_account_features, finalize_account_features, compute_base_account_to_account_edges, finalize_account_to_account_edges)

def build_account_features(
    account_events,
    snapshot_time=None,
):
    """
    Reuse the exact static feature formulas over the streamed account-event log.

    The output is the same 24-feature Account contract used by the graph/model.
    """
    all_accounts = account_events.select("account_id").distinct()

    directional_features = compute_directional_account_features(
        account_events
    )
    shared_features = compute_shared_account_features(account_events)
    temporal_features = compute_temporal_account_features(
        account_events,
        snapshot_time=snapshot_time,
    )

    feature_stage = (
        all_accounts
        .join(directional_features, on="account_id", how="left")
        .join(shared_features, on="account_id", how="left")
        .join(temporal_features, on="account_id", how="left")
        .fillna(0)
    )

    return finalize_account_features(feature_stage)


def rebuild_transaction_rows(account_events):
    """
    Reconstruct one transaction row from each outgoing Account event.

    Every source transaction creates exactly one outgoing event, so this retains
    the fields required by the static 9-feature edge formulas.
    """
    return (
        account_events
        .filter(F.col("direction") == "outgoing")
        .select(
            F.col("account_id").alias("sender_account_id"),
            F.col("counterparty_account_id").alias(
                "receiver_account_id"
            ),
            F.col("amount").alias("Amount Paid"),
            F.col("payment_format").alias("Payment Format"),
            F.col("event_time"),
            F.col("is_currency_switch"),
        )
    )


def build_account_edges(
    account_events,
    snapshot_time=None,
):
    """Build the exact nine Account-to-Account edge features."""
    transactions = rebuild_transaction_rows(account_events)
    base_edges = compute_base_account_to_account_edges(transactions)

    return finalize_account_to_account_edges(
        base_edges,
        transactions,
        snapshot_time=snapshot_time,
    )


def spark_read_parquet(path: Path) -> DataFrame:
    """Read a Parquet table through the active Spark session."""
    return SparkSession.getActiveSession().read.parquet(str(path))


def load_bank_account_counts(
    static_output_dir: Path,
) -> DataFrame:
    """
    Load stable Bank metadata.

    This is safe to cache for an entire streaming run because the number of
    accounts registered at each Bank does not change during the replay.
    """
    return (
        spark_read_parquet(
            static_output_dir / "node_bank_features.parquet"
        )
        .select("bank_id", "bank_account_count")
    )


def load_entity_types(
    static_output_dir: Path,
) -> DataFrame:
    """
    Load stable Entity-type metadata.

    These one-hot Entity type columns are fixed metadata, not transaction
    behaviour, so they can be reused by every micro-batch.
    """
    entity_type_columns = [
        column_name
        for column_name in ENTITY_FEATURE_COLUMNS
        if column_name.startswith("entity_type_")
    ]

    return (
        spark_read_parquet(
            static_output_dir / "node_entity_features.parquet"
        )
        .select("entity_id", *entity_type_columns)
    )


def build_bank_features(
    account_events: DataFrame,
    static_output_dir: Path | None = None,
    bank_account_counts: DataFrame | None = None,
) -> DataFrame:
    """Build exact Bank features for Banks affected by streamed events."""
    if bank_account_counts is None:
        if static_output_dir is None:
            raise ValueError(
                "Provide static_output_dir or bank_account_counts."
            )

        bank_account_counts = load_bank_account_counts(
            static_output_dir
        )

    bank_events = account_events.select(
        "bank_id",
        "counterparty_bank_id",
        F.col("account_id").alias("participant_account_id"),
        "direction",
        "amount",
        "currency",
        "payment_format",
        "is_cross_bank",
        "is_odd_hour",
        "is_weekend",
    )

    active_banks = bank_events.select("bank_id").distinct()
    outgoing = bank_events.filter(F.col("direction") == "outgoing")
    incoming = bank_events.filter(F.col("direction") == "incoming")

    outgoing_features = outgoing.groupBy("bank_id").agg(
        F.count("*").alias("outgoing_txn_count"),
        F.sum("amount").alias("outgoing_total_paid"),
        F.avg("amount").alias("outgoing_avg_paid"),
        F.countDistinct("participant_account_id").alias(
            "unique_sender_accounts"
        ),
    )

    incoming_features = incoming.groupBy("bank_id").agg(
        F.count("*").alias("incoming_txn_count"),
        F.sum("amount").alias("incoming_total_received"),
        F.avg("amount").alias("incoming_avg_received"),
        F.countDistinct("participant_account_id").alias(
            "unique_receiver_accounts"
        ),
    )

    shared_features = bank_events.groupBy("bank_id").agg(
        F.countDistinct("counterparty_bank_id").alias(
            "counterparty_bank_count"
        ),
        F.countDistinct("currency").alias("currency_diversity"),
        F.countDistinct("payment_format").alias(
            "payment_format_diversity"
        ),
        F.avg("is_cross_bank").alias("cross_bank_txn_ratio"),
        F.avg("is_odd_hour").alias("odd_hour_txn_ratio"),
        F.avg("is_weekend").alias("weekend_txn_ratio"),
    )

    features = (
        active_banks
        .join(bank_account_counts, on="bank_id", how="left")
        .join(outgoing_features, on="bank_id", how="left")
        .join(incoming_features, on="bank_id", how="left")
        .join(shared_features, on="bank_id", how="left")
        .fillna(0)
    )

    return features.select(
        F.col("bank_id"),
        *[
            F.col(feature_name).cast("double").alias(feature_name)
            for feature_name in BANK_FEATURE_COLUMNS
        ],
    )


def build_entity_features(
    account_events: DataFrame,
    static_output_dir: Path | None = None,
    entity_types: DataFrame | None = None,
) -> DataFrame:
    """Build exact Entity features for Entities affected by streamed events."""
    if entity_types is None:
        if static_output_dir is None:
            raise ValueError(
                "Provide static_output_dir or entity_types."
            )

        entity_types = load_entity_types(static_output_dir)

    entity_events = account_events.select(
        "entity_id",
        "bank_id",
        "account_id",
        "counterparty_account_id",
        "direction",
        "amount",
        "payment_format",
        "is_cross_bank",
        "is_odd_hour",
    )

    active_entities = entity_events.select("entity_id").distinct()
    outgoing = entity_events.filter(F.col("direction") == "outgoing")
    incoming = entity_events.filter(F.col("direction") == "incoming")

    ownership_features = entity_events.groupBy("entity_id").agg(
        F.countDistinct("account_id").alias("owned_account_count"),
        F.countDistinct("bank_id").alias("connected_bank_count"),
    )

    outgoing_features = outgoing.groupBy("entity_id").agg(
        F.count("*").alias("outgoing_txn_count"),
        F.sum("amount").alias("outgoing_total_paid"),
    )

    incoming_features = incoming.groupBy("entity_id").agg(
        F.count("*").alias("incoming_txn_count"),
        F.sum("amount").alias("incoming_total_received"),
    )

    shared_features = entity_events.groupBy("entity_id").agg(
        F.countDistinct("counterparty_account_id").alias(
            "unique_counterparty_accounts"
        ),
        F.avg("is_cross_bank").alias("cross_bank_txn_ratio"),
        F.avg("is_odd_hour").alias("odd_hour_txn_ratio"),
        F.countDistinct("payment_format").alias(
            "payment_format_diversity"
        ),
    )

    features = (
        active_entities
        .join(entity_types, on="entity_id", how="left")
        .join(ownership_features, on="entity_id", how="left")
        .join(outgoing_features, on="entity_id", how="left")
        .join(incoming_features, on="entity_id", how="left")
        .join(shared_features, on="entity_id", how="left")
        .fillna(0)
    )

    return features.select(
        F.col("entity_id"),
        *[
            F.col(feature_name).cast("double").alias(feature_name)
            for feature_name in ENTITY_FEATURE_COLUMNS
        ],
    )


def build_incremental_account_features(
    state_events: DataFrame,
    current_batch_events: DataFrame,
) -> DataFrame:
    affected_account_ids = current_batch_events.select(
        "account_id"
    ).distinct()

    affected_account_history = state_events.join(
        affected_account_ids,
        on="account_id",
        how="inner",
    )

    global_snapshot_time = state_events.agg(
        F.max("event_time").alias("snapshot_time")
    )

    return build_account_features(
        affected_account_history,
        snapshot_time=global_snapshot_time,
    )


def build_incremental_account_edges(
    state_events: DataFrame,
    current_batch_events: DataFrame,
) -> DataFrame:
    """
    Recompute exact edge features only for directed pairs touched by a batch.

    Each selected edge uses its complete transaction history from replay start
    through the current micro-batch.
    """
    current_transactions = rebuild_transaction_rows(
        current_batch_events
    )

    affected_edge_keys = current_transactions.select(
        "sender_account_id",
        "receiver_account_id",
    ).distinct()

    affected_edge_events = (
        state_events
        .filter(F.col("direction") == "outgoing")
        .join(
            affected_edge_keys,
            (
                (F.col("account_id")
                 == F.col("sender_account_id"))
                & (F.col("counterparty_account_id")
                   == F.col("receiver_account_id"))
            ),
            how="inner",
        )
        .select(state_events["*"])
    )

    global_snapshot_time = state_events.agg(
        F.max("event_time").alias("snapshot_time")
    )

    return build_account_edges(
        account_events=affected_edge_events,
        snapshot_time=global_snapshot_time,
    )


def build_incremental_bank_features(
    state_events: DataFrame,
    current_batch_events: DataFrame,
    static_output_dir: Path,
    bank_account_counts: DataFrame | None = None,
) -> DataFrame:
    """Update only Banks touched by the current micro-batch."""
    affected_banks = current_batch_events.select(
        "bank_id"
    ).distinct()

    affected_bank_history = state_events.join(
        affected_banks,
        on="bank_id",
        how="inner",
    )

    return build_bank_features(
        affected_bank_history,
        static_output_dir=static_output_dir,
        bank_account_counts=bank_account_counts,
    )


def build_incremental_entity_features(
    state_events: DataFrame,
    current_batch_events: DataFrame,
    static_output_dir: Path,
    entity_types: DataFrame | None = None,
) -> DataFrame:
    """Update only Entities touched by the current micro-batch."""
    affected_entities = current_batch_events.select(
        "entity_id"
    ).distinct()

    affected_entity_history = state_events.join(
        affected_entities,
        on="entity_id",
        how="inner",
    )

    return build_entity_features(
        affected_entity_history,
        static_output_dir=static_output_dir,
        entity_types=entity_types,
    )


def build_incremental_relations(
    current_batch_events: DataFrame,
) -> tuple[DataFrame, DataFrame]:
    """Create only the stable Account mappings touched by one batch."""
    account_to_bank = current_batch_events.select(
        "account_id",
        "bank_id",
    ).distinct()

    entity_to_account = current_batch_events.select(
        "entity_id",
        "account_id",
    ).distinct()

    multiple_bank_mappings = (
        account_to_bank
        .groupBy("account_id")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )
    multiple_entity_mappings = (
        entity_to_account
        .groupBy("account_id")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

    if multiple_bank_mappings != 0:
        raise ValueError(
            "An affected Account maps to multiple Banks."
        )

    if multiple_entity_mappings != 0:
        raise ValueError(
            "An affected Account maps to multiple Entities."
        )

    return account_to_bank, entity_to_account
