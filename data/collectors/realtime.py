"""Gerçek zamanlı fiyat verisi — Binance WebSocket (kripto) + yfinance polling (hisse)."""

import threading
import time
from datetime import datetime, timezone
from typing import Callable

from loguru import logger

# ── Binance WebSocket ────────────────────────────────────────────────────────

try:
    from binance import ThreadedWebsocketManager
    _BINANCE_AVAILABLE = True
except ImportError:
    _BINANCE_AVAILABLE = False
    logger.warning("python-binance yüklü değil — kripto WebSocket devre dışı.")


class BinancePriceFeed:
    """Binance mini-ticker stream: canlı fiyat, hacim, 24s değişimi."""

    def __init__(self, symbols: list[str], on_price: Callable[[dict], None]):
        """
        Args:
            symbols: ["BTCUSDT", "ETHUSDT", ...]
            on_price: Her fiyat güncellemesinde çağrılır: fn({"symbol": ..., "price": ..., ...})
        """
        self.symbols = [s.upper() for s in symbols]
        self.on_price = on_price
        self._twm = None
        self._running = False

    def start(self) -> None:
        if not _BINANCE_AVAILABLE:
            logger.error("python-binance yüklü değil, pip install python-binance")
            return
        if self._running:
            return

        from config.settings import BINANCE_API_KEY, BINANCE_API_SECRET
        self._twm = ThreadedWebsocketManager(
            api_key=BINANCE_API_KEY,
            api_secret=BINANCE_API_SECRET,
        )
        self._twm.start()

        for symbol in self.symbols:
            self._twm.start_symbol_miniticker_socket(
                callback=self._handle_message,
                symbol=symbol,
            )

        self._running = True
        logger.info("Binance WebSocket başlatıldı: {}", self.symbols)

    def stop(self) -> None:
        if self._twm and self._running:
            self._twm.stop()
            self._running = False
            logger.info("Binance WebSocket durduruldu.")

    def _handle_message(self, msg: dict) -> None:
        if msg.get("e") == "error":
            logger.error("Binance WS hatası: {}", msg)
            return
        try:
            self.on_price({
                "symbol":   msg["s"],
                "price":    float(msg["c"]),
                "change_pct": float(msg["P"]),
                "volume":   float(msg["v"]),
                "high":     float(msg["h"]),
                "low":      float(msg["l"]),
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "source":   "binance_ws",
            })
        except (KeyError, ValueError) as e:
            logger.debug("WS mesaj parse hatası: {}", e)


# ── yfinance Polling ─────────────────────────────────────────────────────────

class YFinancePoller:
    """yfinance ile BIST/hisse fiyatlarını periyodik sorgular."""

    def __init__(
        self,
        symbols: list[str],
        on_price: Callable[[dict], None],
        interval_seconds: int = 60,
    ):
        """
        Args:
            symbols: ["THYAO.IS", "GARAN.IS", ...]
            on_price: Her güncellemede çağrılır
            interval_seconds: Polling aralığı (saniye)
        """
        self.symbols = symbols
        self.on_price = on_price
        self.interval = interval_seconds
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("yfinance poller başlatıldı: {} ({} sn)", self.symbols, self.interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("yfinance poller durduruldu.")

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._fetch_all()
            self._stop_event.wait(self.interval)

    def _fetch_all(self) -> None:
        try:
            import yfinance as yf
            tickers = yf.Tickers(" ".join(self.symbols))
            for symbol in self.symbols:
                try:
                    t = tickers.tickers.get(symbol)
                    if not t:
                        continue
                    info = t.fast_info
                    price = getattr(info, "last_price", None)
                    prev  = getattr(info, "previous_close", None)
                    if price is None:
                        continue
                    change_pct = ((price - prev) / prev * 100) if prev else 0.0
                    self.on_price({
                        "symbol":     symbol,
                        "price":      float(price),
                        "change_pct": round(change_pct, 2),
                        "volume":     getattr(info, "three_month_average_volume", 0) or 0,
                        "high":       getattr(info, "day_high", price) or price,
                        "low":        getattr(info, "day_low",  price) or price,
                        "timestamp":  datetime.now(tz=timezone.utc).isoformat(),
                        "source":     "yfinance",
                    })
                except Exception as e:
                    logger.debug("{} fiyat hatası: {}", symbol, e)
        except Exception as e:
            logger.error("yfinance toplu sorgu hatası: {}", e)


# ── PriceCache — ortak fiyat deposu ─────────────────────────────────────────

class PriceCache:
    """Thread-safe fiyat önbelleği. Her iki feed de buraya yazar."""

    def __init__(self) -> None:
        self._prices: dict[str, dict] = {}
        self._lock = threading.Lock()

    def update(self, data: dict) -> None:
        with self._lock:
            self._prices[data["symbol"]] = data

    def get(self, symbol: str) -> dict | None:
        with self._lock:
            return self._prices.get(symbol)

    def get_all(self) -> dict[str, dict]:
        with self._lock:
            return dict(self._prices)

    def get_price(self, symbol: str) -> float | None:
        entry = self.get(symbol)
        return entry["price"] if entry else None


# ── Tek global cache + başlatma yardımcısı ───────────────────────────────────

_cache = PriceCache()
_binance_feed: BinancePriceFeed | None = None
_yf_poller: YFinancePoller | None = None


def start_realtime_feeds(
    crypto_symbols: list[str] | None = None,
    stock_symbols: list[str] | None = None,
    stock_interval: int = 60,
) -> PriceCache:
    """
    Tüm gerçek zamanlı feed'leri başlatır ve ortak cache döner.

    Kullanım:
        cache = start_realtime_feeds(
            crypto_symbols=["BTCUSDT", "ETHUSDT"],
            stock_symbols=["THYAO.IS", "GARAN.IS"],
        )
        price = cache.get_price("BTCUSDT")
    """
    global _binance_feed, _yf_poller

    if crypto_symbols:
        _binance_feed = BinancePriceFeed(crypto_symbols, _cache.update)
        _binance_feed.start()

    if stock_symbols:
        _yf_poller = YFinancePoller(stock_symbols, _cache.update, interval_seconds=stock_interval)
        _yf_poller.start()

    return _cache


def stop_realtime_feeds() -> None:
    if _binance_feed:
        _binance_feed.stop()
    if _yf_poller:
        _yf_poller.stop()


def get_price_cache() -> PriceCache:
    return _cache
