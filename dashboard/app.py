import glob
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
GOLD_DIR = BASE_DIR / "delta_lake" / "gold" / "market_metrics"
ANOMALIES_DIR = BASE_DIR / "delta_lake" / "anomalies"
OUTPUT_HTML = BASE_DIR / "dashboard" / "index.html"

PORT = 8050
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("streamguard.dashboard")


def load_gold_data() -> pd.DataFrame:
    """Load and concatenate all parquet files from the Gold Delta table."""
    parquet_files = glob.glob(str(GOLD_DIR / "*.parquet"))
    if not parquet_files:
        logger.warning(f"No parquet files found in {GOLD_DIR}")
        return pd.DataFrame()

    dfs = []
    for f in parquet_files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            logger.debug(f"Could not read {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    gold = pd.concat(dfs, ignore_index=True)
    if "window_start" in gold.columns:
        gold["window_start"] = pd.to_datetime(gold["window_start"])
        gold = gold.sort_values(by=["window_start", "symbol"]).reset_index(drop=True)
        # Deduplicate on symbol and window_start
        gold = gold.drop_duplicates(subset=["symbol", "window_start"], keep="last")

    # Cast numeric columns
    numeric_cols = [
        "open", "high", "low", "close", "vwap", "total_quantity",
        "trade_count", "buy_initiated_volume", "sell_initiated_volume"
    ]
    for col in numeric_cols:
        if col in gold.columns:
            gold[col] = pd.to_numeric(gold[col], errors="coerce")

    return gold


def load_anomalies_data() -> pd.DataFrame:
    """Load scored anomalies from the Delta anomalies folder if available."""
    parquet_files = glob.glob(str(ANOMALIES_DIR / "*.parquet"))
    if not parquet_files:
        return pd.DataFrame()

    dfs = []
    for f in parquet_files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            logger.debug(f"Could not read {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    anom = pd.concat(dfs, ignore_index=True)
    if "window_start" in anom.columns:
        anom["window_start"] = pd.to_datetime(anom["window_start"])
        anom = anom.drop_duplicates(subset=["symbol", "window_start"], keep="last")
    return anom


def compute_or_merge_anomalies(gold: pd.DataFrame, anom: pd.DataFrame) -> pd.DataFrame:
    """Merge known anomalies or compute IsolationForest on available valid records."""
    if gold.empty:
        return gold

    df = gold.copy()

    if not anom.empty and "is_anomaly" in anom.columns:
        cols_to_merge = [c for c in ["symbol", "window_start", "anomaly_prediction", "anomaly_score", "is_anomaly"] if c in anom.columns]
        df = df.merge(anom[cols_to_merge], on=["symbol", "window_start"], how="left")
    else:
        # Compute baseline feature columns
        df["volume"] = df["total_quantity"].astype(float)
        df["buy_sell_imbalance"] = (
            (df["buy_initiated_volume"].astype(float) - df["sell_initiated_volume"].astype(float))
            / df["total_quantity"].replace(0, np.nan).astype(float)
        ).fillna(0.0)
        df["volatility"] = (
            (df["high"].astype(float) - df["low"].astype(float))
            / df["vwap"].replace(0, np.nan).astype(float)
        ).fillna(0.0)

        # Run Isolation Forest if sklearn is available
        try:
            from sklearn.ensemble import IsolationForest
            features = ["volume", "trade_count", "vwap", "buy_sell_imbalance", "volatility"]
            valid_mask = df[features].notnull().all(axis=1) & (df["volume"] > 0)
            if valid_mask.sum() >= 10:
                X = df.loc[valid_mask, features].values
                iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
                preds = iso.fit_predict(X)
                scores = -iso.score_samples(X)
                df.loc[valid_mask, "anomaly_prediction"] = preds
                df.loc[valid_mask, "anomaly_score"] = scores
                df.loc[valid_mask, "is_anomaly"] = (preds == -1)
            else:
                df["is_anomaly"] = False
                df["anomaly_score"] = 0.0
                df["anomaly_prediction"] = 1
        except Exception as e:
            logger.warning(f"IsolationForest calculation skipped: {e}")
            df["is_anomaly"] = False
            df["anomaly_score"] = 0.0
            df["anomaly_prediction"] = 1

    df["is_anomaly"] = df.get("is_anomaly", False).fillna(False).astype(bool)
    df["anomaly_score"] = df.get("anomaly_score", 0.0).fillna(0.0).astype(float)
    return df


def build_dashboard_html() -> str:
    """Build a rich, dark-mode Plotly HTML dashboard."""
    gold = load_gold_data()
    anom = load_anomalies_data()
    df = compute_or_merge_anomalies(gold, anom)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_records = len(df)
    anomaly_count = int(df["is_anomaly"].sum()) if not df.empty else 0
    latest_score = float(df["anomaly_score"].iloc[-1]) if not df.empty and "anomaly_score" in df else 0.0
    latest_time = str(df["window_start"].max()) if not df.empty and "window_start" in df else "N/A"

    symbols = df["symbol"].unique() if not df.empty else ["BTCUSDT", "ETHUSDT"]

    # Create subplots for each symbol: Row 1 = Candlestick + VWAP + Anomalies, Row 2 = Volume & Buy/Sell
    figs = {}
    for sym in symbols:
        sym_df = df[df["symbol"] == sym].sort_values("window_start") if not df.empty else pd.DataFrame()
        if sym_df.empty:
            continue

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=[f"{sym} Price, VWAP & Market Anomalies (1-Min OHLC)", f"{sym} Volume & Buy/Sell Order Flow"]
        )

        # 1. Candlestick Chart
        candlestick = go.Candlestick(
            x=sym_df["window_start"],
            open=sym_df["open"],
            high=sym_df["high"],
            low=sym_df["low"],
            close=sym_df["close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        )
        fig.add_trace(candlestick, row=1, col=1)

        # 2. VWAP Line
        fig.add_trace(
            go.Scatter(
                x=sym_df["window_start"],
                y=sym_df["vwap"],
                mode="lines",
                name="VWAP",
                line=dict(color="#f39c12", width=2, dash="dash")
            ),
            row=1, col=1
        )

        # 3. Anomaly Markers
        anom_subset = sym_df[sym_df["is_anomaly"] == True]
        if not anom_subset.empty:
            fig.add_trace(
                go.Scatter(
                    x=anom_subset["window_start"],
                    y=anom_subset["high"] * 1.0005,
                    mode="markers+text",
                    name="Anomaly Alert",
                    marker=dict(symbol="triangle-down", size=14, color="#ff1744", line=dict(width=1, color="#ffffff")),
                    text=["⚠️ Anomaly" for _ in range(len(anom_subset))],
                    textposition="top center",
                    textfont=dict(color="#ff1744", size=11),
                    hoverinfo="text",
                    hovertext=[
                        f"🚨 <b>MARKET ANOMALY</b><br>Time: {r.window_start}<br>Score: {r.anomaly_score:.4f}<br>Volume: {r.total_quantity:.2f}<br>Trades: {r.trade_count}"
                        for _, r in anom_subset.iterrows()
                    ]
                ),
                row=1, col=1
            )

        # 4. Volume & Buy/Sell initiated
        fig.add_trace(
            go.Bar(
                x=sym_df["window_start"],
                y=sym_df["buy_initiated_volume"],
                name="Buy Initiated Vol",
                marker_color="#26a69a",
                opacity=0.75
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(
                x=sym_df["window_start"],
                y=sym_df["sell_initiated_volume"],
                name="Sell Initiated Vol",
                marker_color="#ef5350",
                opacity=0.75
            ),
            row=2, col=1
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#131722",
            plot_bgcolor="#1e222d",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=50, t=60, b=40),
            height=650,
            hovermode="x unified"
        )
        fig.update_xaxes(gridcolor="#2a2e39", showgrid=True)
        fig.update_yaxes(gridcolor="#2a2e39", showgrid=True)

        figs[sym] = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # Render HTML page
    charts_html = ""
    for sym, chart in figs.items():
        charts_html += f"""
        <div class="card" style="margin-bottom: 25px;">
            <div class="card-header">
                <h2>{sym} Real-Time Surveillance</h2>
            </div>
            <div class="card-body">
                {chart}
            </div>
        </div>
        """

    if not charts_html:
        charts_html = "<div class='card' style='padding: 40px; text-align: center;'><h3>Waiting for streaming data batches...</h3></div>"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="10">
    <title>StreamGuard — Real-Time Crypto Market Intelligence & Drift Monitoring</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #0b0e14;
            color: #d1d4dc;
            padding: 24px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid #2a2e39;
            margin-bottom: 24px;
        }}
        .title h1 {{
            font-size: 26px;
            color: #ffffff;
            font-weight: 700;
            letter-spacing: -0.5px;
        }}
        .title p {{
            color: #787b86;
            font-size: 14px;
            margin-top: 4px;
        }}
        .status-badge {{
            display: inline-flex;
            align-items: center;
            background: #142e2b;
            color: #26a69a;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            border: 1px solid #1e564f;
        }}
        .status-badge .dot {{
            width: 8px;
            height: 8px;
            background: #26a69a;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0% {{ box-shadow: 0 0 0 0 rgba(38, 166, 154, 0.7); }}
            70% {{ box-shadow: 0 0 0 8px rgba(38, 166, 154, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(38, 166, 154, 0); }}
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: #131722;
            border: 1px solid #2a2e39;
            border-radius: 12px;
            padding: 18px;
        }}
        .kpi-label {{
            font-size: 12px;
            color: #787b86;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
        }}
        .kpi-value {{
            font-size: 26px;
            font-weight: 700;
            color: #ffffff;
            margin-top: 8px;
        }}
        .kpi-subtext {{
            font-size: 12px;
            color: #26a69a;
            margin-top: 4px;
        }}
        .kpi-alert {{
            color: #ff1744;
        }}
        .card {{
            background: #131722;
            border: 1px solid #2a2e39;
            border-radius: 12px;
            overflow: hidden;
        }}
        .card-header {{
            padding: 16px 20px;
            border-bottom: 1px solid #2a2e39;
            background: #181c27;
        }}
        .card-header h2 {{
            font-size: 18px;
            color: #ffffff;
            font-weight: 600;
        }}
        .card-body {{
            padding: 10px;
        }}
        .refresh-btn {{
            background: #2962ff;
            color: #ffffff;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            font-size: 13px;
        }}
        .refresh-btn:hover {{
            background: #1e4bd8;
        }}
        .footer {{
            text-align: center;
            color: #5d606b;
            font-size: 12px;
            margin-top: 30px;
            padding-top: 15px;
            border-top: 1px solid #1e222d;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="title">
            <h1>🛡️ StreamGuard Platform</h1>
            <p>Real-Time Financial Market Intelligence & Isolation Forest Drift Detection</p>
        </div>
        <div>
            <span class="status-badge">
                <span class="dot"></span> Pipeline Active (Auto-refresh: 10s)
            </span>
            <button class="refresh-btn" onclick="location.reload()" style="margin-left: 12px;">🔄 Refresh Now</button>
        </div>
    </div>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Gold 1m Windows</div>
            <div class="kpi-value">{total_records}</div>
            <div class="kpi-subtext">Delta Lake Lakehouse</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Detected Anomalies</div>
            <div class="kpi-value {'kpi-alert' if anomaly_count > 0 else ''}">{anomaly_count}</div>
            <div class="kpi-subtext {'kpi-alert' if anomaly_count > 0 else ''}">Isolation Forest (outliers)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Latest Anomaly Score</div>
            <div class="kpi-value">{latest_score:.4f}</div>
            <div class="kpi-subtext">Higher = higher deviation</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Last Window Processed</div>
            <div class="kpi-value" style="font-size: 16px; margin-top: 14px;">{latest_time}</div>
            <div class="kpi-subtext">Refreshed: {now_str}</div>
        </div>
    </div>

    {charts_html}

    <div class="footer">
        StreamGuard MVP • Binance WebSocket → Kafka (KRaft) → Spark Structured Streaming → Delta Lake (Bronze / Silver / Gold) → Isolation Forest Anomaly Detection
    </div>
</body>
</html>
"""
    return html_content


def save_html_report() -> str:
    """Build and save index.html, returning its path."""
    html = build_dashboard_html()
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Dashboard saved to {OUTPUT_HTML}")
    return str(OUTPUT_HTML)


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = build_dashboard_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy", "service": "StreamGuard Dashboard"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence default request logs
        return


def run_server(port=PORT):
    save_html_report()
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    logger.info(f"Serving StreamGuard Plotly Dashboard on http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping server...")
        server.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    run_server(port)
