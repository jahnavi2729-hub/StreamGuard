import asyncio
import json
import logging
import signal
import sys
import threading
from typing import Any, Dict

import websockets
from confluent_kafka import Producer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("binance_producer")

# Kafka configuration
KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "market-trades"
producer_conf = {"bootstrap.servers": KAFKA_BROKER}
producer = Producer(producer_conf)

# Binance combined streams for BTCUSDT and ETHUSDT trades
BINANCE_WS_URL = (
    "wss://stream.binance.com:9443/stream?streams=btcusdt@trade/ethusdt@trade"
)

# Graceful shutdown flag
stop_event = asyncio.Event()
stop_requested = threading.Event()

def delivery_report(err: Any, msg: Any) -> None:
    """Called once for each message produced to indicate delivery result.
    Triggered by poll() or flush().
    """
    if err is not None:
        logger.error(f"Message delivery failed: {err}")
    else:
        logger.info(f"Message delivered to {msg.topic()} [{msg.partition()}]")

async def consume_and_forward() -> None:
    """Connect to Binance WS, normalize events, and forward to Kafka."""
    backoff = 1  # seconds, exponential backoff on reconnect
    while not stop_event.is_set():
        try:
            logger.info("Connecting to Binance WebSocket...")
            async with websockets.connect(BINANCE_WS_URL) as ws:
                logger.info("Connected to Binance WebSocket.")
                backoff = 1  # reset backoff after successful connection
                while not stop_event.is_set():
                    raw_msg = await ws.recv()
                    data = json.loads(raw_msg)
                    # Binance sends {"stream": "btcusdt@trade", "data": {...}}
                    stream = data.get("stream", "")
                    trade = data.get("data", {})
                    # Normalized payload
                    normalized: Dict[str, Any] = {
                        "symbol": trade.get("s"),
                        "event_time": trade.get("E"),
                        "trade_id": trade.get("t"),
                        "price": trade.get("p"),
                        "quantity": trade.get("q"),
                        "buyer_order_id": trade.get("b"),
                        "seller_order_id": trade.get("a"),
                        "trade_time": trade.get("T"),
                        "is_buyer_maker": trade.get("m"),
                        "stream": stream,
                    }
                    payload = json.dumps(normalized).encode("utf-8")
                    while not stop_requested.is_set():
                        try:
                            producer.produce(
                                KAFKA_TOPIC, payload, callback=delivery_report
                            )
                            break
                        except BufferError:
                            producer.poll(0.1)
                    else:
                        logger.warning(
                            "Shutdown requested while Kafka queue was full; "
                            "trade was not enqueued."
                        )
                        return
                    # Trigger delivery callbacks periodically
                    producer.poll(0)
        except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.InvalidURI) as e:
            logger.warning(f"WebSocket connection lost: {e}. Reconnecting in {backoff}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)  # cap backoff
        except Exception as exc:
            logger.exception(f"Unexpected error: {exc}")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

async def run_producer() -> None:
    loop = asyncio.get_running_loop()
    previous_handlers = {}

    def shutdown_signal(signum: int, _frame: Any) -> None:
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        stop_requested.set()
        loop.call_soon_threadsafe(stop_event.set)

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, shutdown_signal)

    consumer_task = asyncio.create_task(consume_and_forward())
    stop_task = asyncio.create_task(stop_event.wait())
    try:
        done, _ = await asyncio.wait(
            (consumer_task, stop_task), return_when=asyncio.FIRST_COMPLETED
        )
        if stop_task in done and not consumer_task.done():
            consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            if stop_task not in done:
                raise
    finally:
        for task in (consumer_task, stop_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(consumer_task, stop_task, return_exceptions=True)
        for sig, previous_handler in previous_handlers.items():
            signal.signal(sig, previous_handler)

def main() -> None:
    global stop_event
    stop_event = asyncio.Event()
    stop_requested.clear()
    try:
        asyncio.run(run_producer())
    finally:
        # Ensure any queued messages are flushed before exit
        logger.info("Flushing pending Kafka messages...")
        remaining = producer.flush(10)
        if remaining:
            logger.error(
                "%s Kafka message(s) were not confirmed before shutdown", remaining
            )
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    main()
