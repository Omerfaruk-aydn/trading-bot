"""Hisse / Forex / Emtia / Endeks veri toplama — yfinance batch mode."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from data.storage import init_db, save_collector_output
from config.symbols import (
    BIST_SYMBOLS, SP500_SYMBOLS, EUROPE_SYMBOLS, ASIA_SYMBOLS,
    FOREX_PAIRS, COMMODITY_SYMBOLS, INDEX_SYMBOLS,
    TIMEFRAMES_STOCK, TIMEFRAMES_FOREX,
)

CHECKPOINT_FILE = Path("logs/stocks_checkpoint.json")
CHECKPOINT_FILE.parent.mkdir(exist_ok=True)

GROUPS = {
    "bist":      (BIST_SYMBOLS,      TIMEFRAMES_STOCK, "stock"),
    "sp500":     (SP500_SYMBOLS,     TIMEFRAMES_STOCK, "stock"),
    "europe":    (EUROPE_SYMBOLS,    TIMEFRAMES_STOCK, "stock"),
    "asia":      (ASIA_SYMBOLS,      TIMEFRAMES_STOCK, "stock"),
    "forex":     (FOREX_PAIRS,       TIMEFRAMES_FOREX, "forex"),
    "commodity": (COMMODITY_SYMBOLS, TIMEFRAMES_STOCK, "commodity"),
    "index":     (INDEX_SYMBOLS,     TIMEFRAMES_STOCK, "index"),
}

HISTORY = {
    "stock": "10y",
    "forex": "5y",
    "commodity": "10y",
    "index": "10y",
}


def _load_checkpoint() -> list[str]:
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8")).get("done", [])
        except Exception:
            pass
    return []


def _save_checkpoint(done: list[str]) -> None:
    CHECKPOINT_FILE.write_text(
        json.dumps({"done": done}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def fetch_batch(symbols: list[str], period: str, interval: str) -> dict[str, list[dict]]:
    """yfinance ile birden fazla sembolü tek HTTP isteğinde çeker."""
    import yfinance as yf
    import pandas as pd

    try:
        df = yf.download(
            symbols,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            timeout=30,
            group_by="ticker",
        )
    except Exception as exc:
        logger.warning("yfinance batch hata: {}", exc)
        return {}

    if df is None or df.empty:
        return {}

    results: dict[str, list[dict]] = {}

    if len(symbols) == 1:
        sym = symbols[0]
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
        candles = _df_to_candles(df)
        if candles:
            results[sym] = candles
    else:
        for sym in symbols:
            try:
                sub = df[sym].dropna(how="all")
                if sub.empty:
                    continue
                candles = _df_to_candles(sub)
                if candles:
                    results[sym] = candles
            except Exception:
                continue

    return results


def _df_to_candles(df) -> list[dict]:
    candles = []
    for ts, row in df.iterrows():
        try:
            ot = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
        except Exception:
            ot = ts
        try:
            candles.append({
                "open_time": ot.isoformat(),
                "open":   float(row["Open"]),
                "high":   float(row["High"]),
                "low":    float(row["Low"]),
                "close":  float(row["Close"]),
                "volume": float(row.get("Volume", 0)),
            })
        except Exception:
            continue
    return candles


def main():
    init_db()
    done = _load_checkpoint()

    logger.info("=" * 60)
    logger.info("HİSSE / FOREX / EMTİA TOPLAMA (batch mode)")
    logger.info("Tamamlanan: {} işlem", len(done))
    logger.info("=" * 60)

    for group_name, (symbols, timeframes, data_type) in GROUPS.items():
        period = HISTORY[data_type]

        for interval in timeframes:
            remaining = [
                s for s in symbols
                if f"yf:{group_name}:{s}:{interval}" not in done
            ]
            if not remaining:
                logger.info("{} {} — tümü tamamlandı", group_name, interval)
                continue

            logger.info("=== {} | {} | {} sembol ===",
                        group_name.upper(), interval, len(remaining))

            batch_size = 20
            for i in range(0, len(remaining), batch_size):
                batch = remaining[i:i + batch_size]
                batch_no = i // batch_size + 1
                total_batches = (len(remaining) + batch_size - 1) // batch_size
                logger.info("  [{}/{}] {} sembol çekiliyor...",
                            batch_no, total_batches, len(batch))

                results = fetch_batch(batch, period, interval)

                for sym, candles in results.items():
                    output = {
                        "source": f"yfinance_{group_name}",
                        "symbol": sym,
                        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                        "data_type": "ohlcv",
                        "payload": {"interval": interval, "candles": candles},
                    }
                    save_collector_output(output)
                    key = f"yf:{group_name}:{sym}:{interval}"
                    if key not in done:
                        done.append(key)
                    logger.debug("    {} — {} mum", sym, len(candles))

                # Veri gelmeyen sembolleri de tamamlandı say
                for sym in batch:
                    key = f"yf:{group_name}:{sym}:{interval}"
                    if key not in done:
                        done.append(key)

                _save_checkpoint(done)
                time.sleep(0.5)

    logger.info("=" * 60)
    logger.info("TAMAMLANDI")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
