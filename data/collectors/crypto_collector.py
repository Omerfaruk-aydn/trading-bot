"""Binance public API'den OHLCV ve ticker verisi toplayan modül."""

import hashlib
import time
from datetime import datetime, timezone
from typing import Any

import requests
from loguru import logger

from config.settings import (
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    BINANCE_REQUEST_INTERVAL_SECONDS,
    BINANCE_TESTNET,
    CACHE_TTL_SECONDS,
    CRYPTO_INTERVALS,
    CRYPTO_SYMBOLS,
)


_BINANCE_BASE = "https://testnet.binance.vision/api" if BINANCE_TESTNET else "https://api.binance.com/api"

# Basit in-memory cache: {cache_key: (timestamp, data)}
_cache: dict[str, tuple[float, Any]] = {}


def _cache_key(symbol: str, interval: str, limit: int) -> str:
    raw = f"{symbol}:{interval}:{limit}"
    return hashlib.md5(raw.encode()).hexdigest()


def _is_cached(key: str) -> bool:
    if key not in _cache:
        return False
    ts, _ = _cache[key]
    return (time.monotonic() - ts) < CACHE_TTL_SECONDS


def _get_cached(key: str) -> Any:
    _, data = _cache[key]
    return data


def _set_cache(key: str, data: Any) -> None:
    _cache[key] = (time.monotonic(), data)


def _make_request(endpoint: str, params: dict | None = None) -> Any:
    """Binance REST isteği atar; rate limit için bekleme uygular."""
    url = f"{_BINANCE_BASE}{endpoint}"
    headers = {}
    if BINANCE_API_KEY:
        headers["X-MBX-APIKEY"] = BINANCE_API_KEY

    time.sleep(BINANCE_REQUEST_INTERVAL_SECONDS)
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        logger.error("Binance isteği zaman aşımına uğradı: {}", endpoint)
        raise
    except requests.exceptions.HTTPError as exc:
        logger.error("Binance HTTP hatası: {} — {}", exc.response.status_code, endpoint)
        raise


def fetch_ohlcv(
    symbol: str,
    interval: str = "1h",
    limit: int = 500,
) -> dict:
    """
    Binance'tan OHLCV mum verisi çeker.

    Args:
        symbol: İşlem çifti (ör: BTCUSDT)
        interval: Zaman dilimi (1m, 5m, 15m, 1h, 4h, 1d)
        limit: Kaç mum (max 1000)

    Returns:
        Standart CollectorOutput formatında dict
    """
    if interval not in CRYPTO_INTERVALS:
        raise ValueError(f"Geçersiz interval: {interval}. Geçerliler: {CRYPTO_INTERVALS}")

    key = _cache_key(symbol, interval, limit)
    if _is_cached(key):
        logger.debug("Cache'den döndürülüyor: {} {}", symbol, interval)
        return _get_cached(key)

    logger.info("Binance OHLCV çekiliyor: {} {} (limit={})", symbol, interval, limit)
    raw = _make_request("/v3/klines", params={"symbol": symbol, "interval": interval, "limit": limit})

    candles = [
        {
            "open_time": datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc).isoformat(),
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5]),
            "close_time": datetime.fromtimestamp(c[6] / 1000, tz=timezone.utc).isoformat(),
            "quote_volume": float(c[7]),
            "num_trades": int(c[8]),
        }
        for c in raw
    ]

    output = {
        "source": "binance",
        "symbol": symbol,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "data_type": "ohlcv",
        "payload": {
            "interval": interval,
            "candles": candles,
        },
    }
    _set_cache(key, output)
    logger.info("OHLCV alındı: {} mum — {} {}", len(candles), symbol, interval)
    return output


def fetch_ticker(symbol: str) -> dict:
    """Anlık fiyat ve 24s istatistiklerini çeker."""
    logger.info("Ticker çekiliyor: {}", symbol)
    raw = _make_request("/v3/ticker/24hr", params={"symbol": symbol})

    output = {
        "source": "binance",
        "symbol": symbol,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "data_type": "ticker",
        "payload": {
            "price": float(raw["lastPrice"]),
            "price_change_pct": float(raw["priceChangePercent"]),
            "high_24h": float(raw["highPrice"]),
            "low_24h": float(raw["lowPrice"]),
            "volume_24h": float(raw["volume"]),
            "quote_volume_24h": float(raw["quoteVolume"]),
            "num_trades_24h": int(raw["count"]),
        },
    }
    return output


def fetch_order_book(symbol: str, depth: int = 20) -> dict:
    """Order book (bid/ask) verisini çeker."""
    logger.info("Order book çekiliyor: {} (depth={})", symbol, depth)
    raw = _make_request("/v3/depth", params={"symbol": symbol, "limit": depth})

    output = {
        "source": "binance",
        "symbol": symbol,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "data_type": "order_book",
        "payload": {
            "bids": [[float(p), float(q)] for p, q in raw["bids"]],
            "asks": [[float(p), float(q)] for p, q in raw["asks"]],
        },
    }
    return output


def collect_all(
    symbols: list[str] | None = None,
    intervals: list[str] | None = None,
    limit: int = 500,
) -> list[dict]:
    """
    Tüm sembol/interval kombinasyonları için OHLCV toplar.

    Args:
        symbols: Sembol listesi; None ise settings'ten alır
        intervals: Interval listesi; None ise settings'ten alır
        limit: Her çekim için mum sayısı

    Returns:
        CollectorOutput listesi
    """
    symbols = symbols or CRYPTO_SYMBOLS
    intervals = intervals or CRYPTO_INTERVALS
    results: list[dict] = []

    for symbol in symbols:
        for interval in intervals:
            try:
                data = fetch_ohlcv(symbol, interval, limit)
                results.append(data)
            except Exception as exc:
                logger.error("Hata: {} {} — {}", symbol, interval, exc)
                continue

    logger.info("Toplam {} veri seti toplandı.", len(results))
    return results
