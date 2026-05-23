"""NYSE/NASDAQ ABD hisse senetleri — piyasa saati kontrolü, sembol tanımları ve döviz çevirimi."""

from __future__ import annotations

import time
from datetime import datetime, time as dtime, timezone, timedelta
from typing import Literal
from loguru import logger

import yfinance as yf

DEFAULT_US_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "AMD", "NFLX", "INTC",
]

# ABD piyasa seansları (ET = Eastern Time)
# Türkiye saati = ET + 7 saat (yaz DST) veya ET + 8 saat (kış)
# Pre-market:  04:00 – 09:30 ET  →  11:00 – 16:30 TR
# Regular:     09:30 – 16:00 ET  →  16:30 – 23:00 TR
# After-market:16:00 – 20:00 ET  →  23:00 – 03:00 TR

UsSession = Literal["premarket", "regular", "aftermarket", "closed"]


def _now_et() -> tuple[datetime, bool]:
    """Şu anki ET saatini ve DST durumunu döndürür."""
    now_utc = datetime.now(timezone.utc)
    month = now_utc.month
    # ABD DST: Mart 2. Pazar – Kasım 1. Pazar (yaklaşım)
    is_dst = 3 <= month <= 10
    offset = timedelta(hours=4 if is_dst else 5)
    return now_utc - offset, is_dst


def get_us_session() -> UsSession:
    """
    ABD piyasa seansını döndürür:
      'premarket'   — 04:00-09:30 ET (TR: 11:00-16:30)
      'regular'     — 09:30-16:00 ET (TR: 16:30-23:00)
      'aftermarket' — 16:00-20:00 ET (TR: 23:00-03:00)
      'closed'      — geri kalan saatler ve hafta sonu
    """
    now_et, _ = _now_et()
    if now_et.weekday() >= 5:
        return "closed"
    t = now_et.time()
    if dtime(4, 0) <= t < dtime(9, 30):
        return "premarket"
    if dtime(9, 30) <= t < dtime(16, 0):
        return "regular"
    if dtime(16, 0) <= t < dtime(20, 0):
        return "aftermarket"
    return "closed"


def is_us_market_open() -> bool:
    """Regular seans açık mı? (Geriye uyumluluk için korundu.)"""
    return get_us_session() == "regular"


def is_us_tradeable() -> bool:
    """Pre-market, regular veya after-market — işlem yapılabilir mi?"""
    return get_us_session() != "closed"


def session_label() -> str:
    s = get_us_session()
    return {
        "premarket":   "PRE-MARKET  (04:00-09:30 ET / 11:00-16:30 TR)",
        "regular":     "REGULAR     (09:30-16:00 ET / 16:30-23:00 TR)",
        "aftermarket": "AFTER-MARKET(16:00-20:00 ET / 23:00-03:00 TR)",
        "closed":      "KAPALI",
    }[s]


def is_us_symbol(symbol: str) -> bool:
    """
    ABD hisse senedi mi?
    .IS, -USD, -USDT, -PERP, -FUT, =X ekleri yoksa NYSE/NASDAQ kabul edilir.
    """
    s = symbol.upper()
    return (
        not s.endswith(".IS")
        and "-USD" not in s
        and "-USDT" not in s
        and "-PERP" not in s
        and "-FUT" not in s
        and "=X" not in s
    )


# ── USD/TRY kur cache ─────────────────────────────────────────────────────────

_rate_cache: tuple[float, float] = (0.0, 0.0)  # (rate, timestamp)
_CACHE_TTL = 900  # 15 dakika


def get_usdtry_rate() -> float:
    """USD/TRY döviz kuru (15 dakika cache'li). Hata durumunda yaklaşık kur döner."""
    global _rate_cache
    rate, ts = _rate_cache
    if rate > 0 and (time.time() - ts) < _CACHE_TTL:
        return rate
    try:
        t = yf.Ticker("USDTRY=X")
        r = float(t.fast_info.last_price or 0)
        if r > 0:
            _rate_cache = (r, time.time())
            return r
    except Exception as e:
        logger.debug("USD/TRY kuru alınamadı: {}", e)
    return rate if rate > 0 else 38.0  # Fallback yaklaşık kur


def usd_to_tl(usd: float) -> float:
    """USD değerini TL'ye çevirir."""
    return usd * get_usdtry_rate()


def tl_to_usd(tl: float) -> float:
    """TL değerini USD'ye çevirir."""
    rate = get_usdtry_rate()
    return tl / rate if rate > 0 else 0.0
