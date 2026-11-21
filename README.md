# StreamGuard 🛡️
> Real-Time Financial Market Intelligence & Isolation Forest Drift Detection Platform

StreamGuard is a modular, production-ready local streaming and lakehouse platform designed to ingest high-throughput cryptocurrency trades, standardize them into Medallion architecture Delta Lake tables (Bronze → Silver → Gold), detect market anomalies and distribution drift using unsupervised machine learning (Isolation Forest), and visualize the resulting market metrics through an interactive Plotly dashboard.

---

## 🏛️ Architecture & Medallion Data Flow

```
                                  ┌──────────────────────────────┐
                                  │   Binance WebSocket Stream   │
                                  │    (BTCUSDT & ETHUSDT)       │
                                  └──────────────┬───────────────┘
                                                 │ Normalized JSON
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │      Apache Kafka (KRaft)    │
                                  │      Topic: market-trades    │
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              DELTA LAKEHOUSE (Medallion)                               │
│                                                                                        │
│  ┌───────────────────────┐   ┌───────────────────────┐   ┌──────────────────────────┐  │
│  │     Bronze Table      │──►│     Silver Table      │──►│        Gold Table        │  │
│  │ (Raw events + Kafka   │   │  (Typed, validated &  │   │  (1-minute OHLC, VWAP,   │  │
│  │   message lineage)    │   │  deduplicated trades) │   │   volume, buy/sell flow) │  │
│  └───────────────────────┘   └───────────────────────┘   └─────────────┬────────────┘  │
└────────────────────────────────────────────────────────────────────────┼───────────────┘
                                                                         │
                                                                         ▼
                                                          ┌──────────────────────────────┐
                                                          │   Isolation Forest (ML)      │
                                                          │  (Market Anomaly Detection)  │
                                                          └──────────────┬───────────────┘
                                                                         │
                                                                         ▼
                                                          ┌──────────────────────────────┐
                                                          │   Interactive Dashboard      │
                                                          │   (Plotly / http://:8050)    │
                                                          └──────────────────────────────┘
```

1. **Ingestion Layer (`ingestion/binance_producer.py`)**: Asynchronous WebSocket client streaming combined live trade events for `BTCUSDT` and `ETHUSDT`, normalizing payload schema, handling exponential reconnection backoff, and publishing to Kafka topic `market-trades`.
2. **Streaming Broker (`kafka/`)**: Single-node Apache Kafka broker operating in KRaft mode (Kafka Raft Metadata) with dual listeners (internal `kafka:29092` for containers, external `localhost:9092` for host scripts).
3. **Bronze Layer (`spark_streaming/kafka_consumer.py`)**: Consumes raw JSON strings with message lineage (`topic`, `partition`, `offset`, `timestamp`) and appends them to `/opt/delta_lake/bronze/market_trades`.
4. **Silver Layer (`spark_streaming/silver_processor.py`)**: Deserializes JSON, validates fields (non-null positive price/quantity, valid timestamps), casts numeric types to high-precision decimals (`DecimalType(38,18)`), watermarks event time (`10 minutes`), and deduplicates within watermarks by `(symbol, trade_id)`.
5. **Gold Layer (`spark_streaming/gold_processor.py`)**: Computes 1-minute tumbling windows grouped by `symbol`, calculating:
   - **OHLC**: Open (earliest trade price), High, Low, Close (latest trade price).
   - **VWAP**: Volume-Weighted Average Price ($\sum (P \cdot Q) / \sum Q$).
   - **Order Flow**: Buy-initiated volume vs. Sell-initiated volume (`is_buyer_maker` split).
   - **Metrics**: Trade count and total aggregated quantity.
6. **ML Layer (`ml_models/anomaly_detector.py`)**: Trains an unsupervised `IsolationForest` model on feature vectors:
   - Trading Volume
   - Trade Count
   - VWAP
   - Buy/Sell Imbalance Ratio: $\frac{\text{BuyVol} - \text{SellVol}}{\text{TotalVol}}$
   - High-Low Volatility: $\frac{\text{High} - \text{Low}}{\text{VWAP}}$
7. **Serving & Dashboard Layer (`dashboard/app.py`)**: Reads Gold and Anomaly Delta parquet files, rendering an interactive Plotly dashboard with auto-refreshing candlestick charts, volume flows, and anomaly alert markers.

---

## 🚀 Quickstart — Run the Complete Application

### 1. Prerequisites
- **Docker Desktop** installed and running with WSL2 backend.
- **Python 3.10+** (with virtualenv in `.venv`).

### 2. Single Startup Command
Run the provided startup script in PowerShell:
```powershell
.\start_all.ps1
```

Or start individual components manually:
```powershell
# 1. Start Kafka and Spark Streaming services
docker compose up -d

# 2. Start Binance trade producer (in Python virtualenv)
.\.venv\Scripts\python.exe ingestion\binance_producer.py

# 3. Launch Plotly Interactive Dashboard
python dashboard\app.py 8050
```

---

## 📊 Dashboard Access

- **Live Interactive Dashboard URL**: [http://localhost:8050](http://localhost:8050)
- **Static / Offline Report**: [index.html](file:///D:/StreamGuard/dashboard/index.html)
- **Health Endpoint**: `http://localhost:8050/health` (returns `{"status": "healthy"}`)

The dashboard features:
- **BTCUSDT & ETHUSDT Candlestick (OHLC)** charts with customizable zoom and hover details.
- **VWAP overlay line** showing volume-weighted market baseline.
- **Visual Anomaly Indicators** (red alerts) highlighting anomalous market minutes.
- **Volume Breakdown** (Buy-initiated vs. Sell-initiated volume bar charts).
- **Auto-Refresh** every 10 seconds with manual "Refresh Now" trigger.

---

## 🔍 Layer-by-Layer Verification

### 1. Kafka Broker & Ingestion
```powershell
# List Kafka topics
docker exec kafka kafka-topics --bootstrap-server localhost:29092 --list

# View live trades from Kafka
docker exec kafka kafka-console-consumer --bootstrap-server localhost:29092 --topic market-trades --from-beginning --max-messages 5
```

### 2. Bronze Delta Table
```powershell
docker exec spark-kafka-consumer /opt/spark/bin/spark-submit --packages io.delta:delta-spark_2.12:3.2.0 --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog -e "SELECT count(*) FROM delta.\`/opt/delta_lake/bronze/market_trades\`"
```

### 3. Silver Delta Table
```powershell
docker exec streamguard-silver /opt/spark/bin/spark-submit --conf spark.jars.ivy=/tmp/ivy2 --packages io.delta:delta-spark_2.12:3.2.0 --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog /opt/spark-app/validate_silver.py
```

### 4. Gold Delta Table
```powershell
docker exec streamguard-gold /opt/spark/bin/spark-submit --conf spark.jars.ivy=/tmp/ivy2 --packages io.delta:delta-spark_2.12:3.2.0 --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog /opt/spark-app/validate_gold.py
```

### 5. Anomaly Detection
```powershell
docker exec streamguard-gold /opt/spark/bin/spark-submit --conf spark.jars.ivy=/tmp/ivy2 --packages io.delta:delta-spark_2.12:3.2.0 --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog /opt/spark-app/../ml_models/anomaly_detector.py
```

---

## 🛠️ Root-Cause Analysis & Fixes Applied

1. **`spark-kafka-consumer` exit code 56**:
   - **Root Cause**: Docker Compose exit code 56 occurred due to container dependency failure. The container environment previously failed when attempting to write Ivy dependencies to `/home/spark/.ivy2` which does not exist for the unprivileged `spark` user, and Kafka's health check was misconfigured across internal vs external listeners.
   - **Fix**: Pinned Ivy dependency path via `--conf spark.jars.ivy=/tmp/ivy2`, ensured `/tmp/ivy2` is created before `spark-submit`, and configured Kafka KRaft with explicit internal (`kafka:29092`) and external (`localhost:9092`) advertised listeners.
2. **Preservation of Checkpoints**:
   - Host checkpoint directories (`./spark_streaming/checkpoints:/opt/spark-checkpoints`) were strictly preserved to guarantee exactly-once streaming resumption without resetting committed Kafka offsets or Delta transactions.
3. **Continuous Gold Processing**:
   - Added `spark-gold` service in `docker-compose.yml` with `restart: unless-stopped`, continuous windowing, and watermarking.
4. **Persistent Scikit-Learn Runtime**:
   - Installed `scikit-learn==1.3.2`, `numpy`, `scipy`, and `joblib` into the Spark environment and provided a custom `spark_streaming/Dockerfile` for persistent builds.

---

## 📌 Known Limitations & Production Enhancements
- **Watermark Latency**: Gold 1-minute windows commit upon expiration of the 10-minute event-time watermark delay to handle late-arriving trade events.
- **Scale-out**: The local setup runs in `local[*]` Spark mode. For distributed clusters, configure Spark standalone cluster manager or Kubernetes operator.
- **Model Retraining**: Currently runs as a scheduled batch job over the latest 10,000 Gold windows.
