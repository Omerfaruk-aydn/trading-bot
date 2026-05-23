"""yfinance ile BIST ve global hisse verisi toplayan modül."""

from datetime import datetime, timezone
from typing import Any

import yfinance as yf
from loguru import logger

from config.settings import STOCK_SYMBOLS


def fetch_ohlcv(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
) -> dict:
    """
    yfinance üzerinden OHLCV verisi çeker.

    Args:
        symbol: Ticker (ör: THYAO.IS, AAPL)
        period: Kaç geriye (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
        interval: Zaman dilimi (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo)

    Returns:
        Standart CollectorOutput formatında dict
    """
    logger.info("yfinance OHLCV çekiliyor: {} period={} interval={}", symbol, period, interval)
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
    except Exception as exc:
        logger.error("yfinance hatası: {} — {}", symbol, exc)
        raise

    if df.empty:
        logger.warning("Boş veri döndü: {}", symbol)
        return _empty_output(symbol)

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
            "candles": candles,
        },
    }
    logger.info("Hisse OHLCV alındı: {} mum — {}", len(candles), symbol)
    return output


def fetch_fundamentals(symbol: str) -> dict:
    """P/E, market cap, sektör gibi temel verileri çeker."""
    logger.info("Temel veriler çekiliyor: {}", symbol)
    try:
        ticker = yf.Ticker(symbol)
        info: dict[str, Any] = ticker.info
    except Exception as exc:
        logger.error("yfinance temel veri hatası: {} — {}", symbol, exc)
        raise

    payload = {
        "name": info.get("longName", ""),
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "eps": info.get("trailingEps"),
        "dividend_yield": info.get("dividendYield"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "avg_volume": info.get("averageVolume"),
        "currency": info.get("currency", "TRY"),
    }

    output = {
        "source": "yfinance",
        "symbol": symbol,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "data_type": "fundamental",
        "payload": payload,
    }
    return output


def fetch_intraday(symbol: str, interval: str = "5m") -> dict:
    """Günlük seanslar için dakikalık veri çeker (son 5 gün)."""
    return fetch_ohlcv(symbol, period="5d", interval=interval)


def collect_all(
    symbols: list[str] | None = None,
    include_fundamentals: bool = True,
) -> list[dict]:
    """
    Tüm hisse sembolleri için veri toplar.

    Args:
        symbols: Ticker listesi; None ise settings'ten alır
        include_fundamentals: Temel verileri de çek

    Returns:
        CollectorOutput listesi
    """
    symbols = symbols or STOCK_SYMBOLS
    results: list[dict] = []

    for symbol in symbols:
        try:
            daily = fetch_ohlcv(symbol, period="1y", interval="1d")
            results.append(daily)

            if include_fundamentals:
                fund = fetch_fundamentals(symbol)
                results.append(fund)
        except Exception as exc:
            logger.error("Hisse verisi toplanamadı: {} — {}", symbol, exc)
            continue

    logger.info("Toplam {} hisse veri seti toplandı.", len(results))
    return results


def _empty_output(symbol: str) -> dict:
    return {
        "source": "yfinance",
        "symbol": symbol,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "data_type": "ohlcv",
        "payload": {"candles": []},
    }
