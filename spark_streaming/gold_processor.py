from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    lit,
    max as max_,
    max_by,
    min as min_,
    min_by,
    struct,
    sum as sum_,
    when,
    window,
)
from pyspark.sql.types import DecimalType


SILVER_PATH = "/opt/delta_lake/silver/market_trades"
GOLD_PATH = "/opt/delta_lake/gold/market_metrics"
CHECKPOINT_PATH = "/opt/spark-checkpoints/gold-market-metrics"
WATERMARK_DELAY = "10 minutes"
VOLUME_TYPE = DecimalType(38, 18)


def main() -> None:
    spark = (
        SparkSession.builder.appName("StreamGuardGold")
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )

    silver = (
        spark.readStream.format("delta")
        .load(SILVER_PATH)
        .withWatermark("event_time", WATERMARK_DELAY)
    )
    trade_window = window(col("event_time"), "1 minute")
    zero_volume = lit(0).cast(VOLUME_TYPE)
    total_quantity = sum_(col("quantity"))
    weighted_quantity = sum_(col("price") * col("quantity"))

    metrics = (
        silver.groupBy("symbol", trade_window.alias("event_window"))
        .agg(
            min_by(
                col("price"),
                struct(col("event_time"), col("trade_id")),
            ).alias("open"),
            max_(col("price")).alias("high"),
            min_(col("price")).alias("low"),
            max_by(
                col("price"),
                struct(col("event_time"), col("trade_id")),
            ).alias("close"),
            count(lit(1)).alias("trade_count"),
            total_quantity.alias("total_quantity"),
            weighted_quantity.alias("weighted_quantity"),
            sum_(
                when(
                    col("is_buyer_maker") == lit(False),
                    col("quantity"),
                ).otherwise(zero_volume)
            ).alias("buy_initiated_volume"),
            sum_(
                when(
                    col("is_buyer_maker") == lit(True),
                    col("quantity"),
                ).otherwise(zero_volume)
            ).alias("sell_initiated_volume"),
        )
        .withColumn(
            "vwap",
            (col("weighted_quantity") / col("total_quantity")).cast(VOLUME_TYPE),
        )
        .select(
            "symbol",
            col("event_window.start").alias("window_start"),
            col("event_window.end").alias("window_end"),
            "open",
            "high",
            "low",
            "close",
            "trade_count",
            "total_quantity",
            "vwap",
            "buy_initiated_volume",
            "sell_initiated_volume",
        )
    )

    query = (
        metrics.writeStream.format("delta")
        .outputMode("append")
        .option("checkpointLocation", CHECKPOINT_PATH)
        .start(GOLD_PATH)
    )
    query.awaitTermination()


if __name__ == "__main__":
    main()
