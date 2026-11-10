"""Spark session and Kafka parsing for the single application."""
from __future__ import annotations
from pathlib import Path
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType, StructField, StructType

KAFKA_CONNECTOR_PACKAGE = (
    "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0"
)


TRANSACTION_SCHEMA = StructType(
    [
        StructField("Timestamp", StringType(), True),
        StructField("From Bank", LongType(), True),
        StructField("Account", StringType(), True),
        StructField("To Bank", LongType(), True),
        StructField("Account.1", StringType(), True),
        StructField("Amount Received", DoubleType(), True),
        StructField("Receiving Currency", StringType(), True),
        StructField("Amount Paid", DoubleType(), True),
        StructField("Payment Currency", StringType(), True),
        # Handles the alternate key seen in a few replayed messages.
        StructField("PaymentCurrency", StringType(), True),
        StructField("Payment Format", StringType(), True),
        StructField("Is Laundering", IntegerType(), True),
    ]
)


def build_spark_session() -> SparkSession:
    """Create Spark with the Kafka connector required for readStream.format('kafka')."""
    return (
        SparkSession.builder
        .appName("MuleDetectorKafkaStreaming")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        .config("spark.jars.packages", KAFKA_CONNECTOR_PACKAGE)
        .getOrCreate()
    )


def parse_transaction_stream(kafka_stream: DataFrame) -> DataFrame:
    """Convert Kafka's raw message bytes into typed transaction columns."""
    return (
        kafka_stream
        .select(
            F.col("partition").alias("kafka_partition"),
            F.col("offset").alias("kafka_offset"),
            F.col("timestamp").alias("kafka_ingested_at"),
            F.from_json(
                F.col("value").cast("string"),
                TRANSACTION_SCHEMA,
            ).alias("transaction"),
        )
        .select(
            "kafka_partition",
            "kafka_offset",
            "kafka_ingested_at",
            "transaction.*",
        )
        .withColumn(
            "event_time",
            F.to_timestamp(F.col("Timestamp"), "yyyy/MM/dd HH:mm"),
        )
        .withColumn(
            "payment_currency",
            F.coalesce(
                F.col("Payment Currency"),
                F.col("PaymentCurrency"),
            ),
        )
        .withColumn(
            "payment_format",
            F.col("Payment Format"),
        )
        .withColumn(
            "is_laundering_offline_label",
            F.col("Is Laundering"),
        )
        .drop(
            "Payment Currency",
            "PaymentCurrency",
            "Payment Format",
            "Is Laundering",
        )
    )
