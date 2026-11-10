"""Resolve Kafka connector JARs while building, not on the friend's first replay."""
from streaming.kafka.connector import build_spark_session
spark = build_spark_session()
try:
    spark.range(1).selectExpr('id').collect()
    print('Spark and Kafka connector initialized successfully.')
finally:
    spark.stop()
