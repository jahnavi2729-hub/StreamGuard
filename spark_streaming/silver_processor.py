import logging
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, length, timestamp_millis, trim
from pyspark.sql.types import (
    BooleanType,
    DecimalType,
    LongType,
    StringType,
    StructField,
    StructType,
)


BRONZE_PATH = Path("/opt/delta_lake/bronze/market_trades")
SILVER_PATH = "/opt/delta_lake/silver/market_trades"
CHECKPOINT_PATH = "/opt/spark-checkpoints/silver-market-trades"
TRADE_SCHEMA = StructType(
    [
        StructField("symbol", StringType()),
        StructField("event_time", LongType()),
        StructField("trade_id", LongType()),
        StructField("price", StringType()),
        StructField("quantity", StringType()),
        StructField("trade_time", LongType()),
        StructField("is_buyer_maker", BooleanType()),
        StructField("stream", StringType()),
    ]
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("streamguard.silver")


def wait_for_bronze_table() -> None:
    delta_log = BRONZE_PATH / "_delta_log"
    while not (
        delta_log.is_dir()
        and any(
            entry.suffix == ".json" and entry.stem.isdigit()
            for entry in delta_log.iterdir()
        )
    ):
        logger.info("Waiting for the Bronze Delta table at %s", BRONZE_PATH)
        time.sleep(10)


def main() -> None:
    wait_for_bronze_table()
    spark = SparkSession.builder.appName("StreamGuardSilver").getOrCreate()

    bronze_records = spark.readStream.format("delta").load(str(BRONZE_PATH))
    parsed_records = bronze_records.select(
        from_json(col("payload"), TRADE_SCHEMA).alias("trade"),
        col("topic").alias("kafka_topic"),
        col("partition").alias("kafka_partition"),
        col("offset").alias("kafka_offset"),
        col("timestamp").alias("kafka_timestamp"),
    ).select(
        "trade.*",
        "kafka_topic",
        "kafka_partition",
        "kafka_offset",
        "kafka_timestamp",
    )

    converted_records = (
        parsed_records.withColumn("price", col("price").cast(DecimalType(38, 18)))
        .withColumn("quantity", col("quantity").cast(DecimalType(38, 18)))
        .withColumn("event_time", timestamp_millis(col("event_time")))
        .withColumn("trade_time", timestamp_millis(col("trade_time")))
    )
    valid_records = converted_records.filter(
        col("symbol").isNotNull()
        & (length(trim(col("symbol"))) > 0)
        & col("event_time").isNotNull()
        & col("trade_id").isNotNull()
        & col("price").isNotNull()
        & (col("price") > 0)
        & col("quantity").isNotNull()
        & (col("quantity") > 0)
        & col("trade_time").isNotNull()
        & col("is_buyer_maker").isNotNull()
        & col("stream").isNotNull()
        & (length(trim(col("stream"))) > 0)
    )

    deduplicated_records = valid_records.withWatermark(
        "event_time", "10 minutes"
    ).dropDuplicatesWithinWatermark(["symbol", "trade_id"])

    query = (
        deduplicated_records.writeStream.format("delta")
        .outputMode("append")
        .option("checkpointLocation", CHECKPOINT_PATH)
        .start(SILVER_PATH)
    )
    query.awaitTermination()


if __name__ == "__main__":
    main()
