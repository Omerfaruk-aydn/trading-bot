"""Forex verisi toplayan modül — yfinance birincil, OANDA opsiyonel."""

from datetime import datetime, timezone
from typing import Any

import yfinance as yf
from loguru import logger

from config.settings import (
    FOREX_SYMBOLS,
    OANDA_API_KEY,
    OANDA_ACCOUNT_ID,
    OANDA_ENVIRONMENT,
)

# OANDA entegrasyonu API key varsa aktif olur
_OANDA_AVAILABLE = bool(OANDA_API_KEY and OANDA_ACCOUNT_ID)


def _pip_value(symbol: str, price: float) -> float:
    """Çiftin pip değerini hesaplar (JPY çiftleri 0.01, diğerleri 0.0001)."""
    if "JPY" in symbol.upper():
        return 0.01
    return 0.0001


def fetch_ohlcv_yfinance(
    symbol: str,
    period: str = "1mo",
    interval: str = "1h",
) -> dict:
    """
    yfinance üzerinden forex OHLCV verisi çeker.

    Args:
        symbol: yfinance forex sembolü (ör: EURUSD=X, USDTRY=X)
        period: Kaç geriye (1d, 5d, 1mo, 3mo, 6mo, 1y)
        interval: Zaman dilimi (1m, 5m, 15m, 30m, 60m, 1h, 1d)

    Returns:
        Standart CollectorOutput formatında dict
    """
    logger.info("Forex OHLCV çekiliyor: {} period={} interval={}", symbol, period, interval)
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
    except Exception as exc:
        logger.error("yfinance forex hatası: {} — {}", symbol, exc)
        raise

    if df.empty:
        logger.warning("Boş forex verisi: {}", symbol)
        return _empty_output(symbol, "yfinance")

    last_close = float(df["Close"].iloc[-1])
    pip = _pip_value(symbol, last_close)

    candles = []
    for ts, row in df.iterrows():
        candles.append({
            "open_time": ts.tz_convert("UTC").isoformat() if ts.tzinfo else ts.isoformat(),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row.get("Volume", 0)),
        })

    output = {
        "source": "yfinance",
        "symbol": symbol,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "data_type": "ohlcv",
        "payload": {
            "interval": interval,
            "period": period,
            "pip_size": pip,
            "last_price": last_close,
            "candles": candles,
        },
    }
    logger.info("Forex OHLCV alındı: {} mum — {}", len(candles), symbol)
    return output


def fetch_ohlcv_oanda(
    instrument: str,
    granularity: str = "H1",
    count: int = 500,
) -> dict:
    """
    OANDA practice/live hesabından forex verisi çeker.

    Args:
        instrument: OANDA formatında (ör: EUR_USD, USD_TRY)
        granularity: S5, M1, M5, M15, M30, H1, H4, D
        count: Kaç mum

    Returns:
        Standart CollectorOutput formatında dict
    """
    if not _OANDA_AVAILABLE:
        raise RuntimeError("OANDA API anahtarları .env dosyasında tanımlı değil.")

    try:
        import oandapyV20
        import oandapyV20.endpoints.instruments as instruments
    except ImportError:
        raise ImportError("oandapyV20 kurulu değil: pip install oandapyV20")

    logger.info("OANDA veri çekiliyor: {} granularity={}", instrument, granularity)

    client = oandapyV20.API(
        access_token=OANDA_API_KEY,
        environment=OANDA_ENVIRONMENT,
    )
    params = {"count": count, "granularity": granularity, "price": "M"}  # M = mid price
    req = instruments.InstrumentsCandles(instrument=instrument, params=params)

    try:
        client.request(req)
        raw_candles = req.response.get("candles", [])
    except Exception as exc:
        logger.error("OANDA isteği başarısız: {} — {}", instrument, exc)
        raise

    candles = []
    for c in raw_candles:
        if not c.get("complete", False):
            continue  # tamamlanmamış mumu atla
        mid = c["mid"]
        candles.append({
            "open_time": c["time"],
            "open": float(mid["o"]),
            "high": float(mid["h"]),
            "low": float(mid["l"]),
            "close": float(mid["c"]),
            "volume": int(c.get("volume", 0)),
        })

    output = {
        "source": "oanda",
        "symbol": instrument,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "data_type": "ohlcv",
        "payload": {
            "granularity": granularity,
            "candles": candles,
        },
    }
    logger.info("OANDA veri alındı: {} mum — {}", len(candles), instrument)
    return output


def fetch_spread(symbol: str) -> dict:
    """Anlık bid/ask spread bilgisini döner (yfinance yaklaşık değer verir)."""
    logger.info("Spread çekiliyor: {}", symbol)
    try:
        ticker = yf.Ticker(symbol)
        info: dict[str, Any] = ticker.info
    except Exception as exc:
        logger.error("Spread hatası: {} — {}", symbol, exc)
        raise

    bid = info.get("bid", 0.0)
    ask = info.get("ask", 0.0)
    price = info.get("regularMarketPrice") or info.get("previousClose", 0.0)
    pip = _pip_value(symbol, float(price))
    spread_pips = (float(ask) - float(bid)) / pip if pip and bid and ask else None

    output = {
        "source": "yfinance",
        "symbol": symbol,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "data_type": "spread",
        "payload": {
            "bid": float(bid),
            "ask": float(ask),
            "spread_pips": spread_pips,
            "pip_size": pip,
        },
    }
    return output


def collect_all(
    symbols: list[str] | None = None,
    period: str = "1mo",
    interval: str = "1h",
) -> list[dict]:
    """
    Tüm forex sembolleri için veri toplar.

    Args:
        symbols: yfinance sembol listesi; None ise settings'ten alır
        period: Tarih aralığı
        interval: Zaman dilimi

    Returns:
        CollectorOutput listesi
    """
    symbols = symbols or FOREX_SYMBOLS
    results: list[dict] = []

    for symbol in symbols:
        try:
            data = fetch_ohlcv_yfinance(symbol, period=period, interval=interval)
            results.append(data)
        except Exception as exc:
            logger.error("Forex toplanamadı: {} — {}", symbol, exc)
            continue

    logger.info("Toplam {} forex veri seti toplandı.", len(results))
    return results


def _empty_output(symbol: str, source: str) -> dict:
    return {
        "source": source,
        "symbol": symbol,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "data_type": "ohlcv",
        "payload": {"candles": []},
    }
