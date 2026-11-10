"""Transaction preparation for the single streaming application."""
from __future__ import annotations
from pathlib import Path
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F

REQUIRED_EVENT_COLUMNS = [
    "account_id",
    "counterparty_account_id",
    "bank_id",
    "counterparty_bank_id",
    "entity_id",
    "event_time",
    "amount",
    "currency",
    "payment_format",
]


def add_transaction_flags(transactions: DataFrame) -> DataFrame:
    """Create transaction-level flags from the fixed feature contract."""
    return (
        transactions
        .withColumn(
            "sender_account_id",
            F.concat_ws(
                "::",
                F.col("From Bank").cast("string"),
                F.col("Account"),
            ),
        )
        .withColumn(
            "receiver_account_id",
            F.concat_ws(
                "::",
                F.col("To Bank").cast("string"),
                F.col("`Account.1`"),
            ),
        )
        .withColumn(
            "is_currency_switch",
            (
                F.col("payment_currency")
                != F.col("Receiving Currency")
            ).cast("int"),
        )
        .withColumn(
            "is_cross_bank",
            (
                F.col("From Bank") != F.col("To Bank")
            ).cast("int"),
        )
        .withColumn(
            "is_odd_hour",
            F.hour("event_time").between(0, 5).cast("int"),
        )
        .withColumn(
            "is_weekend",
            F.dayofweek("event_time").isin([1, 7]).cast("int"),
        )
        .withColumn(
            "is_self_transfer",
            (
                F.col("sender_account_id")
                == F.col("receiver_account_id")
            ).cast("int"),
        )
    )


def enrich_transaction_entities(
    transactions: DataFrame,
    entity_lookup: DataFrame,
) -> DataFrame:
    """Attach sender and receiver Entity IDs with static lookups."""
    sender_lookup = entity_lookup.select(
        F.col("account_id").alias("_sender_lookup_account_id"),
        F.col("entity_id").alias("sender_entity_id"),
    )
    receiver_lookup = entity_lookup.select(
        F.col("account_id").alias("_receiver_lookup_account_id"),
        F.col("entity_id").alias("receiver_entity_id"),
    )

    return (
        transactions
        .join(
            sender_lookup,
            F.col("sender_account_id")
            == F.col("_sender_lookup_account_id"),
            "left",
        )
        .drop("_sender_lookup_account_id")
        .join(
            receiver_lookup,
            F.col("receiver_account_id")
            == F.col("_receiver_lookup_account_id"),
            "left",
        )
        .drop("_receiver_lookup_account_id")
    )


def add_transaction_rejection_reason(
    transactions: DataFrame,
) -> DataFrame:
    """Add one atomic validation reason for both transaction sides."""
    return transactions.withColumn(
        "_transaction_rejection_reason",
        F.concat_ws(
            "; ",
            F.when(
                F.col("Account").isNull()
                | F.col("`Account.1`").isNull(),
                F.lit("missing_account_id"),
            ),
            F.when(
                F.col("From Bank").isNull()
                | F.col("To Bank").isNull(),
                F.lit("missing_bank_id"),
            ),
            F.when(
                F.col("sender_entity_id").isNull()
                | F.col("receiver_entity_id").isNull(),
                F.lit("missing_entity_id"),
            ),
            F.when(
                F.col("event_time").isNull(),
                F.lit("missing_event_time"),
            ),
            F.when(
                F.col("Amount Paid").isNull()
                | F.col("Amount Received").isNull(),
                F.lit("missing_amount"),
            ),
            F.when(
                F.col("payment_currency").isNull()
                | F.col("Receiving Currency").isNull(),
                F.lit("missing_currency"),
            ),
            F.when(
                F.col("payment_format").isNull(),
                F.lit("missing_payment_format"),
            ),
        ),
    )


def build_account_events(transactions: DataFrame) -> DataFrame:
    """Create exactly one outgoing and one incoming event per transaction."""
    common_columns = [
        "event_time",
        "payment_format",
        "is_currency_switch",
        "is_cross_bank",
        "is_odd_hour",
        "is_weekend",
        "is_self_transfer",
        "kafka_topic",
        "kafka_partition",
        "kafka_offset",
        "kafka_ingested_at",
        "_transaction_rejection_reason",
    ]

    outgoing_event = F.struct(
        F.col("sender_account_id").alias("account_id"),
        F.col("receiver_account_id").alias(
            "counterparty_account_id"
        ),
        F.col("From Bank").alias("bank_id"),
        F.col("To Bank").alias("counterparty_bank_id"),
        F.col("sender_entity_id").alias("entity_id"),
        F.lit("outgoing").alias("direction"),
        F.col("Amount Paid").alias("amount"),
        F.col("payment_currency").alias("currency"),
        *[F.col(name) for name in common_columns],
    )

    incoming_event = F.struct(
        F.col("receiver_account_id").alias("account_id"),
        F.col("sender_account_id").alias(
            "counterparty_account_id"
        ),
        F.col("To Bank").alias("bank_id"),
        F.col("From Bank").alias("counterparty_bank_id"),
        F.col("receiver_entity_id").alias("entity_id"),
        F.lit("incoming").alias("direction"),
        F.col("Amount Received").alias("amount"),
        F.col("Receiving Currency").alias("currency"),
        *[F.col(name) for name in common_columns],
    )

    return (
        transactions
        .select(
            F.explode(
                F.array(outgoing_event, incoming_event)
            ).alias("event")
        )
        .select("event.*")
    )


def load_entity_lookup(
    spark: SparkSession,
    account_entity_lookup_path: Path,
    cache_lookup: bool = True,
) -> DataFrame:
    """Load and materialize fixed metadata once, not once per micro-batch.

    Only this static lookup is persisted. Callers own its lifetime and should
    unpersist it when their query ends. The opt-out supports controlled replay
    benchmarks of the former uncached implementation.
    """
    if not account_entity_lookup_path.exists():
        raise FileNotFoundError(
            "Account metadata lookup not found: "
            f"{account_entity_lookup_path}"
        )

    lookup = (
        spark.read.parquet(str(account_entity_lookup_path))
        .select("account_id", "entity_id")
    )
    if cache_lookup:
        # An explicit partition count prevents AQE from coalescing cached metadata
        # differently from the streaming joins. Keep the fixed join side sorted
        # so each micro-batch does not sort the full lookup again.
        partitions = int(spark.conf.get("spark.sql.shuffle.partitions", "8"))
        lookup = lookup.repartition(partitions, "account_id")
    lookup = lookup.dropDuplicates(["account_id"])
    if cache_lookup:
        lookup = lookup.sortWithinPartitions("account_id").persist(StorageLevel.MEMORY_AND_DISK)
        try:
            lookup.count()
        except Exception:
            lookup.unpersist()
            raise
    return lookup


def split_valid_and_rejected_events(
    events: DataFrame,
) -> tuple[DataFrame, DataFrame]:
    """
    Separate valid events from rejected events.

    Rejected events retain their Kafka topic, partition, offset, event
    direction, and rejection reason for later debugging.
    """
    valid_events = (
        events
        .filter(F.col("_transaction_rejection_reason") == "")
        .drop("_transaction_rejection_reason")
    )

    rejected_events = (
        events
        .filter(F.col("_transaction_rejection_reason") != "")
        .withColumn(
            "rejection_reason",
            F.col("_transaction_rejection_reason"),
        )
        .drop("_transaction_rejection_reason")
    )

    return valid_events, rejected_events
