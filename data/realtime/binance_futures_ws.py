"""Binance USDT-M Futures WebSocket — perpetual sözleşmeler için anlık mark price ve funding rate."""

from __future__ import annotations

import json
import threading
import time
from loguru import logger

try:
    import websocket as _ws_lib
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False

# İç sembol → Binance futures sembol haritası
FUTURES_MAP: dict[str, str] = {
    "BTC-PERP":   "btcusdt",
    "ETH-PERP":   "ethusdt",
    "SOL-PERP":   "solusdt",
    "BNB-PERP":   "bnbusdt",
    "XRP-PERP":   "xrpusdt",
    "DOGE-PERP":  "dogeusdt",
    "ADA-PERP":   "adausdt",
    "AVAX-PERP":  "avaxusdt",
    "MATIC-PERP": "maticusdt",
}

REVERSE_FUTURES_MAP: dict[str, str] = {v: k for k, v in FUTURES_MAP.items()}

DEFAULT_FUTURES = ["BTC-PERP", "ETH-PERP", "SOL-PERP"]

# Bakım teminat oranı (Binance tier-1 yaklaşımı)
MAINT_MARGIN_RATE = 0.004  # %0.4


class BinanceFuturesStream:
    """
    Binance USDT-M Futures markPrice WebSocket akışı.

    markPrice stream her 3 saniyede mark price ve funding rate sağlar.

    Kullanım:
        stream = BinanceFuturesStream(["BTC-PERP", "ETH-PERP"])
        stream.start()
        price = stream.get_mark_price("BTC-PERP")
        rate  = stream.get_funding_rate("BTC-PERP")
        stream.stop()
    """

    _WS_URL = "wss://fstream.binance.com/stream?streams={streams}"

    def __init__(self, symbols: list[str]):
        self._prices: dict[str, float] = {}
        self._funding_rates: dict[str, float] = {}
        self._timestamps: dict[str, float] = {}
        self._lock = threading.Lock()
        self._ws = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._reconnect_delay = 5

        self.symbols = [s for s in symbols if s in FUTURES_MAP]
        self.streams = [f"{FUTURES_MAP[s]}@markPrice" for s in self.symbols]

        if not HAS_WEBSOCKET:
            logger.warning("websocket-client yüklü değil → pip install websocket-client")
        if not self.symbols:
            logger.warning("Binance Futures için desteklenen sembol bulunamadı.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        if not HAS_WEBSOCKET or not self.streams:
            return False
        self._running = True
        self._thread = threading.Thread(
            target=self._run_forever, daemon=True, name="BinanceFuturesWS"
        )
        self._thread.start()
        time.sleep(2)
        logger.info("Binance Futures WebSocket başlatıldı | Semboller: {}", self.symbols)
        return True

    def stop(self) -> None:
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def get_mark_price(self, symbol: str) -> float | None:
        """Güncel mark price döndürür. 30 sn'den eski veri None döner."""
        with self._lock:
            price = self._prices.get(symbol)
            ts = self._timestamps.get(symbol, 0)
        if price and (time.time() - ts) < 30:
            return price
        return None

    def get_funding_rate(self, symbol: str) -> float:
        """Mevcut funding rate (örn: 0.0001 = %0.01 her 8 saatte bir)."""
        with self._lock:
            return self._funding_rates.get(symbol, 0.0)

    def get_all_prices(self) -> dict[str, float]:
        with self._lock:
            now = time.time()
            return {
                sym: price
                for sym, price in self._prices.items()
                if (now - self._timestamps.get(sym, 0)) < 30
            }

    @property
    def connected(self) -> bool:
        return self._running and bool(self._prices)

    # ── WebSocket iç döngüsü ───────────────────────────────────────────────────

    def _run_forever(self) -> None:
        while self._running:
            try:
                self._connect()
            except Exception as e:
                logger.warning("Binance Futures WS bağlantı hatası: {}", e)
            if self._running:
                logger.info(
                    "Binance Futures WS {} sn sonra yeniden bağlanıyor...",
                    self._reconnect_delay,
                )
                time.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 60)

    def _connect(self) -> None:
        url = self._WS_URL.format(streams="/".join(self.streams))

        def on_message(ws, raw):
            try:
                data = json.loads(raw)
                ticker = data.get("data", {})
                stream = data.get("stream", "")
                binance_sym = stream.split("@")[0]
                symbol = REVERSE_FUTURES_MAP.get(binance_sym)
                if not symbol:
                    return
                price = float(ticker.get("p", 0))      # mark price
                funding = float(ticker.get("r", 0.0))  # funding rate
                if price > 0:
                    with self._lock:
                        self._prices[symbol] = price
                        self._funding_rates[symbol] = funding
                        self._timestamps[symbol] = time.time()
            except Exception as e:
                logger.debug("Futures WS parse hatası: {}", e)

        def on_error(ws, error):
            logger.warning("Binance Futures WS hata: {}", error)

        def on_open(ws):
            self._reconnect_delay = 5
            logger.debug("Binance Futures WS bağlandı.")

        def on_close(ws, code, msg):
            logger.debug("Binance Futures WS kapandı (code={}).", code)

        self._ws = _ws_lib.WebSocketApp(
            url,
            on_message=on_message,
            on_error=on_error,
            on_open=on_open,
            on_close=on_close,
        )
        self._ws.run_forever(ping_interval=20, ping_timeout=10)
