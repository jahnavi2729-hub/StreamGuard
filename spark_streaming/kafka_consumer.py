import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col


def main() -> None:
    spark = SparkSession.builder.appName("StreamGuardBronze").getOrCreate()

    kafka_records = (
        spark.readStream.format("kafka")
        .option(
            "kafka.bootstrap.servers",
            os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092"),
        )
        .option("subscribe", "market-trades")
        .option("startingOffsets", "earliest")
        .load()
    )

    bronze_records = kafka_records.select(
        col("value").cast("string").alias("payload"),
        col("topic"),
        col("partition"),
        col("offset"),
        col("timestamp"),
    )

    query = (
        bronze_records.writeStream.format("delta")
        .outputMode("append")
        .option(
            "checkpointLocation",
            "/opt/spark-checkpoints/bronze-market-trades",
        )
        .start("/opt/delta_lake/bronze/market_trades")
    )
    query.awaitTermination()


if __name__ == "__main__":
    main()
