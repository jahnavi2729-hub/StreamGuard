import operator
import sys
from functools import reduce

from pyspark.sql import SparkSession
from pyspark.sql.functions import abs as abs_, col, isnan
from pyspark.sql.types import BooleanType, DoubleType, IntegerType, StructField


GOLD_PATH = "/opt/delta_lake/gold/market_metrics"
ANOMALIES_PATH = "/opt/delta_lake/anomalies"
MIN_VALID_RECORDS = 30
FEATURE_COLUMNS = (
    "volume",
    "trade_count",
    "vwap",
    "buy_sell_imbalance",
    "volatility",
)
REQUIRED_COLUMNS = (
    "symbol",
    "window_start",
    "window_end",
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


def main() -> None:
    spark = (
        SparkSession.builder.appName("StreamGuardAnomalyDetector")
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

    try:
        gold = spark.read.format("delta").load(GOLD_PATH)
        featured = (
            gold.withColumn("volume", col("total_quantity").cast("double"))
            .withColumn(
                "buy_sell_imbalance",
                (
                    (
                        col("buy_initiated_volume").cast("double")
                        - col("sell_initiated_volume").cast("double")
                    )
                    / col("total_quantity").cast("double")
                ),
            )
            .withColumn(
                "volatility",
                (
                    (col("high").cast("double") - col("low").cast("double"))
                    / col("vwap").cast("double")
                ),
            )
        )
        valid_conditions = [
            col(column).isNotNull() for column in REQUIRED_COLUMNS
        ] + [
            col("total_quantity") > 0,
            col("trade_count") > 0,
            col("vwap") > 0,
            col("high") >= col("low"),
            col("low") <= col("open"),
            col("open") <= col("high"),
            col("low") <= col("close"),
            col("close") <= col("high"),
        ]
        valid_conditions.extend(
            ~isnan(col(feature).cast("double"))
            & (abs_(col(feature).cast("double")) <= sys.float_info.max)
            for feature in FEATURE_COLUMNS
        )
        valid_records = (
            featured.filter(reduce(operator.and_, valid_conditions))
            .orderBy(col("window_start").desc(), col("symbol").asc())
            .limit(10_000)
        )
        valid_record_count = valid_records.count()
        print(f"Valid Gold records selected: {valid_record_count}")

        if valid_record_count < MIN_VALID_RECORDS:
            print(
                f"Insufficient valid records: need at least {MIN_VALID_RECORDS}; "
                "skipping training and output."
            )
            return

        try:
            from sklearn.ensemble import IsolationForest
        except ImportError as exc:
            print(
                "scikit-learn is required to train IsolationForest but is not "
                "installed in this Python environment.",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc

        records = valid_records.collect()
        feature_matrix = [
            [float(record[column]) for column in FEATURE_COLUMNS]
            for record in records
        ]
        model = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
        predictions = model.fit_predict(feature_matrix)
        anomaly_scores = -model.score_samples(feature_matrix)

        output_columns = valid_records.columns
        scored_rows = [
            tuple(record[column] for column in output_columns)
            + (
                int(prediction),
                float(score),
                bool(prediction == -1),
            )
            for record, prediction, score in zip(
                records, predictions, anomaly_scores
            )
        ]
        output_schema = (
            valid_records.schema.add(
                StructField("anomaly_prediction", IntegerType(), nullable=False)
            )
            .add(StructField("anomaly_score", DoubleType(), nullable=False))
            .add(StructField("is_anomaly", BooleanType(), nullable=False))
        )
        scored_results = spark.createDataFrame(scored_rows, schema=output_schema)
        scored_results.write.format("delta").mode("overwrite").save(ANOMALIES_PATH)
        print(
            f"Saved {valid_record_count} scored records to {ANOMALIES_PATH}"
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
