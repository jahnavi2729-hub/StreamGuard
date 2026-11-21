# Spark Streaming

Code for Spark Structured Streaming jobs that consume data from Kafka, apply transformations, and write results to Delta Lake or other sinks.

`kafka_consumer.py` reads and validates normalized `market-trades` records from Kafka, then displays valid records in the Spark console. The Compose Spark service runs it using the Spark 3.5.5 Kafka connector and persists checkpoints in `checkpoints/`.