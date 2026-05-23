"""Binance WebSocket — kripto için gerçek zamanlı fiyat akışı.

Bağlantı kesilirse otomatik yeniden bağlanır.
websocket-client paketi yoksa yfinance polling'e düşer.
"""

from __future__ import annotations

import json
import threading
import time
from loguru import logger

try:
    import websocket as _websocket_lib
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False


# yfinance → Binance sembol haritası
SYMBOL_MAP: dict[str, str] = {
    "BTC-USD":  "btcusdt",
    "ETH-USD":  "ethusdt",
    "SOL-USD":  "solusdt",
    "BNB-USD":  "bnbusdt",
    "XRP-USD":  "xrpusdt",
    "DOGE-USD": "dogeusdt",
    "ADA-USD":  "adausdt",
    "AVAX-USD": "avaxusdt",
    "DOT-USD":  "dotusdt",
    "MATIC-USD":"maticusdt",
}

REVERSE_MAP: dict[str, str] = {v: k for k, v in SYMBOL_MAP.items()}


class BinancePriceStream:
    """
    Binance miniTicker WebSocket ile anlık fiyat akışı.

    Kullanım:
        stream = BinancePriceStream(["BTC-USD", "ETH-USD", "SOL-USD"])
        stream.start()
        price = stream.get_price("BTC-USD")  # float | None
        stream.stop()
    """

    _WS_URL = "wss://stream.binance.com:9443/stream?streams={streams}"

    def __init__(self, symbols: list[str]):
        self._prices: dict[str, float] = {}
        self._timestamps: dict[str, float] = {}
        self._lock = threading.Lock()
        self._ws = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._reconnect_delay = 5

        self.symbols = [s for s in symbols if s in SYMBOL_MAP]
        self.streams = [f"{SYMBOL_MAP[s]}@miniTicker" for s in self.symbols]

        if not HAS_WEBSOCKET:
            logger.warning("websocket-client yüklü değil → pip install websocket-client")
        if not self.symbols:
            logger.warning("Binance WebSocket için desteklenen sembol yok.")

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> bool:
        if not HAS_WEBSOCKET or not self.streams:
            return False

        self._running = True
        self._thread = threading.Thread(target=self._run_forever, daemon=True, name="BinanceWS")
        self._thread.start()

        # İlk bağlantı için kısa bekleme
        time.sleep(2)
        logger.info("Binance WebSocket başlatıldı | Semboller: {}", self.symbols)
        return True

    def stop(self) -> None:
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def get_price(self, symbol: str) -> float | None:
        """Anlık fiyat döndürür. 30 sn'den eski veriyi None kabul eder."""
        with self._lock:
            price = self._prices.get(symbol)
            ts = self._timestamps.get(symbol, 0)
        if price and (time.time() - ts) < 30:
            return price
        return None

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

    # ── WebSocket iç döngüsü ─────────────────────────────────────────────────

    def _run_forever(self) -> None:
        while self._running:
            try:
                self._connect()
            except Exception as e:
                logger.warning("Binance WS bağlantı hatası: {}", e)
            if self._running:
                logger.info("Binance WS {} sn sonra yeniden bağlanıyor...", self._reconnect_delay)
                time.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 60)

    def _connect(self) -> None:
        url = self._WS_URL.format(streams="/".join(self.streams))

        def on_message(ws, raw):
            try:
                data = json.loads(raw)
                ticker = data.get("data", {})
                stream = data.get("stream", "")

                # stream örn: "btcusdt@miniTicker"
                binance_sym = stream.split("@")[0]
                yf_sym = REVERSE_MAP.get(binance_sym)
                if not yf_sym:
                    return

                price = float(ticker.get("c", 0))
                if price > 0:
                    with self._lock:
                        self._prices[yf_sym] = price
                        self._timestamps[yf_sym] = time.time()
            except Exception as e:
                logger.debug("WS mesaj parse hatası: {}", e)

        def on_error(ws, error):
            logger.warning("Binance WS hata: {}", error)

        def on_open(ws):
            self._reconnect_delay = 5  # başarılı bağlantıda sıfırla
            logger.debug("Binance WS bağlandı.")

        def on_close(ws, code, msg):
            logger.debug("Binance WS kapandı (code={}).", code)

        self._ws = _websocket_lib.WebSocketApp(
            url,
            on_message=on_message,
            on_error=on_error,
            on_open=on_open,
            on_close=on_close,
        )
        self._ws.run_forever(ping_interval=20, ping_timeout=10)


# ── Fallback: yfinance polling ────────────────────────────────────────────────

class YFinancePoller:
    """
    Binance WS kullanılamıyorsa yfinance ile polling yapar.
    Her sembol için ayrı thread açmak yerine sıralı hızlı sorgular atar.
    """

    def __init__(self, symbols: list[str], interval: int = 15):
        self._prices: dict[str, float] = {}
        self._lock = threading.Lock()
        self._running = False
        self.symbols = symbols
        self.interval = interval

    def start(self) -> bool:
        self._running = True
        threading.Thread(target=self._poll_loop, daemon=True, name="YFPoller").start()
        logger.info("yfinance poller başlatıldı ({} sembol, {}s)", len(self.symbols), self.interval)
        return True

    def stop(self) -> None:
        self._running = False

    def get_price(self, symbol: str) -> float | None:
        with self._lock:
            return self._prices.get(symbol)

    def _poll_loop(self) -> None:
        import yfinance as yf
        while self._running:
            for sym in self.symbols:
                try:
                    price = float(yf.Ticker(sym).fast_info.last_price or 0)
                    if price > 0:
                        with self._lock:
                            self._prices[sym] = price
                except Exception:
                    pass
            time.sleep(self.interval)


def build_price_stream(crypto_symbols: list[str]) -> BinancePriceStream | YFinancePoller | None:
    """
    Mümkünse Binance WebSocket, değilse yfinance poller döndürür.
    Kripto sembol yoksa None.
    """
    if not crypto_symbols:
        return None

    if HAS_WEBSOCKET:
        ws = BinancePriceStream(crypto_symbols)
        if ws.streams:
            return ws

    return YFinancePoller(crypto_symbols, interval=10)
