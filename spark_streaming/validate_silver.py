import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, lit, min as min_, max as max_, sum as sum_, when


SILVER_PATH = "/opt/delta_lake/silver/market_trades"
REQUIRED_COLUMNS = (
    "symbol",
    "event_time",
    "trade_id",
    "price",
    "quantity",
    "trade_time",
    "is_buyer_maker",
    "stream",
)


def report_check(name: str, passed: bool, detail: str) -> bool:
    status = "PASS" if passed else "FAIL"
    print(f"{status}: {name} - {detail}")
    return passed


def main() -> None:
    spark = (
        SparkSession.builder.appName("ValidateStreamGuardSilver")
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
        silver = spark.read.format("delta").load(SILVER_PATH).cache()
        total_rows = silver.count()
        print(f"Total rows: {total_rows}")
        all_checks_passed &= report_check(
            "non-empty Silver table", total_rows > 0, f"rows={total_rows}"
        )

        null_counts = silver.agg(
            *[
                count(when(col(column).isNull(), lit(1))).alias(column)
                for column in REQUIRED_COLUMNS
            ]
        ).first()
        for column in REQUIRED_COLUMNS:
            null_count = null_counts[column]
            all_checks_passed &= report_check(
                f"no nulls in {column}",
                null_count == 0,
                f"null_rows={null_count}",
            )

        duplicate_groups = (
            silver.groupBy("symbol", "trade_id")
            .count()
            .filter(col("count") > 1)
        )
        duplicate_summary = duplicate_groups.agg(
            count(lit(1)).alias("duplicate_key_groups"),
            sum_(col("count") - 1).alias("extra_duplicate_rows"),
        ).first()
        duplicate_key_groups = duplicate_summary["duplicate_key_groups"]
        extra_duplicate_rows = duplicate_summary["extra_duplicate_rows"] or 0
        all_checks_passed &= report_check(
            "unique (symbol, trade_id)",
            duplicate_key_groups == 0,
            (
                f"duplicate_key_groups={duplicate_key_groups}, "
                f"extra_duplicate_rows={extra_duplicate_rows}"
            ),
        )

        non_positive_prices = silver.filter(col("price") <= 0).count()
        all_checks_passed &= report_check(
            "positive prices",
            non_positive_prices == 0,
            f"non_positive_rows={non_positive_prices}",
        )

        non_positive_quantities = silver.filter(col("quantity") <= 0).count()
        all_checks_passed &= report_check(
            "positive quantities",
            non_positive_quantities == 0,
            f"non_positive_rows={non_positive_quantities}",
        )

        event_time_range = silver.agg(
            min_("event_time").alias("min_event_time"),
            max_("event_time").alias("max_event_time"),
        ).first()
        minimum_event_time = event_time_range["min_event_time"]
        maximum_event_time = event_time_range["max_event_time"]
        print(
            "Event-time range: "
            f"min={minimum_event_time}, max={maximum_event_time}"
        )
        all_checks_passed &= report_check(
            "event-time range available",
            minimum_event_time is not None and maximum_event_time is not None,
            "both bounds are present"
            if minimum_event_time is not None and maximum_event_time is not None
            else "no event times found",
        )

        print(f"Overall result: {'PASS' if all_checks_passed else 'FAIL'}")
        if not all_checks_passed:
            sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
