import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    lit,
    max as max_,
    min as min_,
    when,
)


GOLD_PATH = "/opt/delta_lake/gold/market_metrics"
METRIC_COLUMNS = (
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


def report_check(name: str, passed: bool, detail: str) -> bool:
    status = "PASS" if passed else "FAIL"
    print(f"{status}: {name} - {detail}")
    return passed


def main() -> None:
    spark = (
        SparkSession.builder.appName("ValidateStreamGuardGold")
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

    all_checks_passed = True
    try:
        gold = spark.read.format("delta").load(GOLD_PATH).cache()
        total_rows = gold.count()
        print(f"Total rows: {total_rows}")
        all_checks_passed &= report_check(
            "non-empty Gold table", total_rows > 0, f"rows={total_rows}"
        )

        distinct_symbols = (
            gold.filter(col("symbol").isNotNull())
            .select("symbol")
            .distinct()
            .count()
        )
        print(f"Distinct symbols: {distinct_symbols}")
        all_checks_passed &= report_check(
            "distinct symbols available",
            distinct_symbols > 0,
            f"count={distinct_symbols}",
        )

        null_counts = gold.agg(
            *[
                count(when(col(column).isNull(), lit(1))).alias(column)
                for column in METRIC_COLUMNS
            ]
        ).first()
        for column in METRIC_COLUMNS:
            null_count = null_counts[column]
            all_checks_passed &= report_check(
                f"no nulls in {column}",
                null_count == 0,
                f"null_rows={null_count}",
            )

        ohlc_present = (
            col("open").isNotNull()
            & col("high").isNotNull()
            & col("low").isNotNull()
            & col("close").isNotNull()
        )
        invalid_ohlc = gold.filter(
            ohlc_present
            & (
                (col("open") <= 0)
                | (col("high") <= 0)
                | (col("low") <= 0)
                | (col("close") <= 0)
                | (col("high") < col("low"))
                | (col("open") < col("low"))
                | (col("open") > col("high"))
                | (col("close") < col("low"))
                | (col("close") > col("high"))
            )
        ).count()
        all_checks_passed &= report_check(
            "valid OHLC values",
            invalid_ohlc == 0,
            f"invalid_rows={invalid_ohlc}",
        )

        vwap_present = (
            col("vwap").isNotNull()
            & col("low").isNotNull()
            & col("high").isNotNull()
        )
        invalid_vwap = gold.filter(
            vwap_present
            & (
                (col("vwap") <= 0)
                | (col("vwap") < col("low"))
                | (col("vwap") > col("high"))
            )
        ).count()
        all_checks_passed &= report_check(
            "valid VWAP values",
            invalid_vwap == 0,
            f"invalid_rows={invalid_vwap}",
        )

        event_time_range = gold.agg(
            min_("window_start").alias("first_window"),
            max_("window_end").alias("last_window"),
        ).first()
        first_window = event_time_range["first_window"]
        last_window = event_time_range["last_window"]
        print(f"Event-time range: min={first_window}, max={last_window}")
        all_checks_passed &= report_check(
            "event-time range available",
            first_window is not None and last_window is not None,
            "both bounds are present"
            if first_window is not None and last_window is not None
            else "no event-time windows found",
        )

        print(f"Overall result: {'PASS' if all_checks_passed else 'FAIL'}")
        if not all_checks_passed:
            sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
