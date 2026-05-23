"""Büyük ölçekli toplu veri çekici.

Tüm piyasalar için paralel veri toplama:
- Binance: Top 200 kripto, 5 yıl, 6 timeframe
- yfinance: BIST, S&P500, Avrupa, Asya, Forex, Emtia, Endeksler
- FRED: Makro ekonomik seriler

Özellikler:
- Kaldığı yerden devam (checkpoint)
- Rate limit yönetimi
- Otomatik hata toleransı
- İlerleme takibi
"""

from __future__ import annotations

import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
import yfinance as yf
from loguru import logger

# Thread-safe checkpoint lock
_checkpoint_lock = threading.Lock()
_db_lock = threading.Lock()

from config.symbols import (
    ASIA_SYMBOLS,
    BIST_SYMBOLS,
    COMMODITY_SYMBOLS,
    CRYPTO_TOP200,
    EUROPE_SYMBOLS,
    FOREX_PAIRS,
    FRED_SERIES,
    HISTORY_YEARS,
    INDEX_SYMBOLS,
    SP500_SYMBOLS,
    TIMEFRAMES_CRYPTO,
    TIMEFRAMES_FOREX,
    TIMEFRAMES_STOCK,
)
from data.storage import save_collector_output

CHECKPOINT_FILE = Path("logs/bulk_collect_checkpoint.json")
CHECKPOINT_FILE.parent.mkdir(exist_ok=True)

# ── Checkpoint (kaldığı yerden devam) ────────────────────────────────────────

def _load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"done": []}


def _save_checkpoint(done: list[str]) -> None:
    with _checkpoint_lock:
        CHECKPOINT_FILE.write_text(
            json.dumps({"done": done}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _mark_done(key: str, done: list[str]) -> None:
    with _checkpoint_lock:
        if key not in done:
            done.append(key)
        _save_checkpoint(done)


# ── Binance Geçmiş Veri ───────────────────────────────────────────────────────

_BINANCE_BASE = "https://api.binance.com/api"


def _binance_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
    """Binance'tan belirli tarih aralığında mum verisi çeker (1000'er parça)."""
    all_candles: list[dict] = []
    current = start_ms

    while current < end_ms:
        try:
            time.sleep(0.08)  # 1200 req/dk sınırına saygı
            resp = requests.get(
                f"{_BINANCE_BASE}/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": current,
                    "endTime": end_ms,
                    "limit": 1000,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Binance hata: {} {} — {}", symbol, interval, exc)
            time.sleep(5)
            break

        if not data:
            break

        for c in data:
            all_candles.append({
                "open_time": datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc).isoformat(),
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
                "close_time": datetime.fromtimestamp(c[6] / 1000, tz=timezone.utc).isoformat(),
                "quote_volume": float(c[7]),
                "num_trades": int(c[8]),
            })

        current = data[-1][6] + 1  # son kapanış zamanından devam
        if len(data) < 1000:
            break

    return all_candles


def collect_binance_bulk(
    symbols: list[str] | None = None,
    intervals: list[str] | None = None,
    years: int | None = None,
    checkpoint: list[str] | None = None,
) -> dict[str, int]:
    """
    Binance'tan toplu kripto OHLCV verisi çeker ve DB'ye kaydeder.

    Args:
        symbols: Çekilecek semboller (None → TOP200)
        intervals: Zaman dilimleri (None → tüm crypto TF'ler)
        years: Kaç yıl geriye (None → HISTORY_YEARS["crypto"])
        checkpoint: Daha önce tamamlanan (sembol, interval) ikilisi listesi

    Returns:
        Her sembol için kaydedilen mum sayısı
    """
    symbols = symbols or CRYPTO_TOP200
    intervals = intervals or TIMEFRAMES_CRYPTO
    years = years or HISTORY_YEARS["crypto"]
    checkpoint = checkpoint or []

    start_dt = datetime.now(tz=timezone.utc) - timedelta(days=years * 365)
    end_dt = datetime.now(tz=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    counts: dict[str, int] = {}
    total_symbols = len(symbols) * len(intervals)
    done_count = 0

    for symbol in symbols:
        for interval in intervals:
            key = f"binance:{symbol}:{interval}"
            done_count += 1

            if key in checkpoint:
                logger.debug("Atlandı (checkpoint): {}", key)
                continue

            logger.info(
                "[{}/{}] Binance çekiliyor: {} {} ({} yıl)",
                done_count, total_symbols, symbol, interval, years,
            )

            # 1m ve 5m için daha kısa tarih (veri çok büyük)
            actual_start = start_ms
            if interval == "1m":
                actual_start = int((datetime.now(tz=timezone.utc) - timedelta(days=30)).timestamp() * 1000)
            elif interval == "5m":
                actual_start = int((datetime.now(tz=timezone.utc) - timedelta(days=90)).timestamp() * 1000)
            elif interval == "15m":
                actual_start = int((datetime.now(tz=timezone.utc) - timedelta(days=180)).timestamp() * 1000)

            candles = _binance_klines(symbol, interval, actual_start, end_ms)
            if not candles:
                logger.warning("Veri gelmedi: {} {}", symbol, interval)
                continue

            output = {
                "source": "binance",
                "symbol": symbol,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "data_type": "ohlcv",
                "payload": {"interval": interval, "candles": candles},
            }
            save_collector_output(output)
            counts[key] = len(candles)
            _mark_done(key, checkpoint)
            logger.info("  → {} mum kaydedildi.", len(candles))

    return counts


# ── yfinance Toplu Çekici ─────────────────────────────────────────────────────

def _yf_fetch(
    symbol: str,
    period: str,
    interval: str,
    source_label: str,
) -> dict | None:
    """Tek sembol için yfinance OHLCV çeker, standart format döner."""
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            timeout=15,
        )
    except Exception as exc:
        logger.warning("yfinance hata: {} — {}", symbol, exc)
        return None

    if df is None or df.empty:
        return None

    # yf.download tek sembol için de MultiIndex döndürebilir — düzleştir
    import pandas as pd
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)

    candles = []
    for ts, row in df.iterrows():
        try:
            ot = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
        except Exception:
            ot = ts
        candles.append({
            "open_time": ot.isoformat(),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row.get("Volume", 0)),
        })

    return {
        "source": source_label,
        "symbol": symbol,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "data_type": "ohlcv",
        "payload": {"interval": interval, "candles": candles},
    }


def _years_to_period(years: int) -> str:
    if years >= 10:
        return "10y"
    if years >= 5:
        return "5y"
    if years >= 2:
        return "2y"
    return "1y"


def collect_yfinance_bulk(
    symbol_groups: dict[str, list[str]] | None = None,
    checkpoint: list[str] | None = None,
    max_workers: int = 1,
) -> dict[str, int]:
    """
    yfinance üzerinden BIST, S&P500, Avrupa, Asya, Forex, Emtia, Endeks verisi çeker.

    Args:
        symbol_groups: {"bist": [...], "sp500": [...], ...} — None ise hepsi
        checkpoint: Tamamlanan anahtarlar
        max_workers: Kullanılmaz, uyumluluk için tutuldu

    Returns:
        Her sembol için kaydedilen mum sayısı
    """
    checkpoint = checkpoint or []

    default_groups = {
        "bist":      (BIST_SYMBOLS,      TIMEFRAMES_STOCK, "stock",     "yfinance_bist"),
        "sp500":     (SP500_SYMBOLS,      TIMEFRAMES_STOCK, "stock",     "yfinance_us"),
        "europe":    (EUROPE_SYMBOLS,     TIMEFRAMES_STOCK, "stock",     "yfinance_eu"),
        "asia":      (ASIA_SYMBOLS,       TIMEFRAMES_STOCK, "stock",     "yfinance_asia"),
        "forex":     (FOREX_PAIRS,        TIMEFRAMES_FOREX, "forex",     "yfinance_forex"),
        "commodity": (COMMODITY_SYMBOLS,  TIMEFRAMES_STOCK, "commodity", "yfinance_commodity"),
        "index":     (INDEX_SYMBOLS,      TIMEFRAMES_STOCK, "index",     "yfinance_index"),
    }

    if symbol_groups:
        groups = {k: default_groups[k] for k in symbol_groups if k in default_groups}
    else:
        groups = default_groups

    counts: dict[str, int] = {}

    for group_name, (symbols, timeframes, history_key, source_label) in groups.items():
        years = HISTORY_YEARS.get(history_key, 5)
        period = _years_to_period(years)
        total = len(symbols) * len(timeframes)
        done = 0

        logger.info("=== {} grubu başlıyor ({} sembol × {} TF) ===",
                    group_name.upper(), len(symbols), len(timeframes))

        for symbol in symbols:
            for interval in timeframes:
                key = f"yf:{group_name}:{symbol}:{interval}"
                done += 1

                if key in checkpoint:
                    logger.debug("Atlandı: {}", key)
                    continue

                logger.info("[{}/{}] {} — {} {}", done, total, group_name, symbol, interval)

                output = _yf_fetch(symbol, period, interval, source_label)
                if output and output["payload"]["candles"]:
                    save_collector_output(output)
                    n = len(output["payload"]["candles"])
                    counts[key] = n
                    _mark_done(key, checkpoint)
                    logger.debug("  → {} mum", n)
                else:
                    logger.warning("  → Veri yok: {} {}", symbol, interval)

                time.sleep(0.3)

    return counts


# ── FRED Makro Veri ───────────────────────────────────────────────────────────

def collect_fred_bulk(
    api_key: str | None = None,
    series: dict[str, str] | None = None,
    checkpoint: list[str] | None = None,
) -> dict[str, int]:
    """
    FRED API'den makro ekonomik seri verisi çeker.

    Args:
        api_key: FRED API key (https://fred.stlouisfed.org/docs/api/api_key.html)
                 None ise yfinance üzerinden alternatif alınır
        series: {"FEDFUNDS": "açıklama", ...} — None ise default
        checkpoint: Tamamlanan anahtarlar

    Returns:
        Her seri için kaydedilen satır sayısı
    """
    checkpoint = checkpoint or []
    series = series or FRED_SERIES
    counts: dict[str, int] = {}

    for series_id, description in series.items():
        key = f"fred:{series_id}"
        if key in checkpoint:
            continue

        logger.info("FRED çekiliyor: {} ({})", series_id, description)

        if api_key:
            data = _fred_api_fetch(series_id, api_key)
        else:
            # FRED API key yoksa yfinance ile bazı makro verileri alınabilir
            data = _fred_yfinance_fallback(series_id)

        if not data:
            logger.warning("FRED veri yok: {}", series_id)
            continue

        output = {
            "source": "fred",
            "symbol": series_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "data_type": "macro",
            "payload": {
                "interval": "1mo",
                "description": description,
                "series_id": series_id,
                "candles": data,
            },
        }
        save_collector_output(output)
        counts[key] = len(data)
        _mark_done(key, checkpoint)
        logger.info("  → {} veri noktası: {}", len(data), description)
        time.sleep(0.5)

    return counts


def _fred_api_fetch(series_id: str, api_key: str) -> list[dict]:
    """FRED REST API'den veri çeker."""
    try:
        resp = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": "2000-01-01",
            },
            timeout=15,
        )
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
    except Exception as exc:
        logger.warning("FRED API hata: {} — {}", series_id, exc)
        return []

    result = []
    for o in obs:
        try:
            val = float(o["value"])
            result.append({
                "open_time": o["date"] + "T00:00:00+00:00",
                "close": val,
                "open": val, "high": val, "low": val, "volume": 0,
            })
        except (ValueError, KeyError):
            continue
    return result


def _fred_yfinance_fallback(series_id: str) -> list[dict]:
    """FRED API olmadan yfinance ile bazı makro göstergeler."""
    # Eşleşme tablosu: FRED serisi → yfinance sembolü
    mapping = {
        "DCOILWTICO": "CL=F",   # WTI petrol
        "T10Y2Y": "^TNX",       # 10Y tahvil faiz
    }
    yf_sym = mapping.get(series_id)
    if not yf_sym:
        return []

    output = _yf_fetch(yf_sym, "20y", "1mo", "yfinance_macro")
    if output:
        return output["payload"]["candles"]
    return []


# ── Ana Toplu Çalıştırma ───────────────────────────────────────────────────────

def run_full_collection(
    include_crypto: bool = True,
    include_stocks: bool = True,
    include_macro: bool = True,
    fred_api_key: str | None = None,
) -> None:
    """
    Tüm piyasalar için eksiksiz veri toplama pipeline'ı.
    Kaldığı yerden devam eder (checkpoint).

    Args:
        include_crypto: Binance kripto verisi dahil
        include_stocks: yfinance hisse/forex/emtia/endeks dahil
        include_macro: FRED makro verisi dahil
        fred_api_key: FRED API anahtarı (opsiyonel)
    """
    checkpoint_data = _load_checkpoint()
    done: list[str] = checkpoint_data.get("done", [])

    logger.info("="*60)
    logger.info("TOPLU VERİ TOPLAMA BAŞLIYOR")
    logger.info("Tamamlanan (checkpoint): {} işlem", len(done))
    logger.info("="*60)

    total_counts: dict[str, int] = {}

    if include_crypto:
        logger.info("\n[1/3] KRİPTO (Binance Top 200)")
        counts = collect_binance_bulk(checkpoint=done)
        total_counts.update(counts)
        logger.info("Kripto tamamlandı: {} sembol/interval", len(counts))

    if include_stocks:
        logger.info("\n[2/3] HİSSE + FOREX + EMTİA + ENDEKSLEr")
        counts = collect_yfinance_bulk(checkpoint=done)
        total_counts.update(counts)
        logger.info("Hisse/Forex tamamlandı: {} sembol/interval", len(counts))

    if include_macro:
        logger.info("\n[3/3] MAKRO (FRED)")
        counts = collect_fred_bulk(api_key=fred_api_key, checkpoint=done)
        total_counts.update(counts)
        logger.info("Makro tamamlandı: {} seri", len(counts))

    total_candles = sum(total_counts.values())
    logger.info("\n" + "="*60)
    logger.info("TOPLAM TOPLANAN: {:,} mum/veri noktası", total_candles)
    logger.info("Checkpoint: {}", CHECKPOINT_FILE)
    logger.info("="*60)
