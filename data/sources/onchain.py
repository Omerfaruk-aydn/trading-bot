"""On-Chain Veri Kaynağı — Binance Futures kamuya açık API'si.

Desteklenen metrikler:
  • Funding rate     — perp sözleşme finansman oranı
  • Open interest    — açık pozisyon değişimi (24s)
  • Long/Short ratio — büyük hesap L/S oranı
  • Liquidation heat — tasfiye bölgeleri (yaklaşım)

Tüm endpoint'ler auth gerektirmez (kamuya açık).
"""
from __future__ import annotations

import time
from functools import lru_cache
from typing import TypedDict

import requests
from loguru import logger

_FAPI   = "https://fapi.binance.com"
_TIMEOUT = 6   # saniye
_CACHE_TTL = 300  # 5 dakika


def _to_pair(symbol: str) -> str:
    """BTC-PERP veya BTC-USD → BTCUSDT."""
    base = symbol.split("-")[0].upper()
    return f"{base}USDT"


class OnChainData(TypedDict):
    funding_rate: float | None       # ör: 0.0008 = %0.08
    oi_change_pct: float | None      # 24s OI değişimi %
    long_short_ratio: float | None   # >1 = daha fazla long
    score: int                       # -3 ile +3
    reason: str


# ── API çağrıları ─────────────────────────────────────────────────────────────

def _get(url: str, params: dict | None = None) -> list | dict | None:
    try:
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.debug("onchain HTTP hatası [{}]: {}", url, e)
        return None


def get_funding_rate(symbol: str) -> float | None:
    """Anlık funding rate. Pozitif = longlar ödüyor (aşırı long)."""
    pair = _to_pair(symbol)
    data = _get(f"{_FAPI}/fapi/v1/premiumIndex", {"symbol": pair})
    if isinstance(data, dict):
        try:
            return float(data["lastFundingRate"])
        except (KeyError, TypeError, ValueError):
            pass
    return None


def get_open_interest_change(symbol: str, hours: int = 24) -> float | None:
    """Son N saatteki OI değişimi (%). Pozitif = artıyor."""
    pair   = _to_pair(symbol)
    period = "1h"
    limit  = min(hours, 48)
    data   = _get(
        f"{_FAPI}/futures/data/openInterestHist",
        {"symbol": pair, "period": period, "limit": limit},
    )
    if not isinstance(data, list) or len(data) < 2:
        return None
    try:
        current = float(data[-1]["sumOpenInterest"])
        oldest  = float(data[0]["sumOpenInterest"])
        if oldest == 0:
            return None
        return round((current - oldest) / oldest * 100, 2)
    except (KeyError, TypeError, ValueError):
        return None


def get_long_short_ratio(symbol: str) -> float | None:
    """Büyük hesap long/short oranı. >1 = daha fazla long pozisyon."""
    pair = _to_pair(symbol)
    data = _get(
        f"{_FAPI}/futures/data/globalLongShortAccountRatio",
        {"symbol": pair, "period": "1h", "limit": 1},
    )
    if isinstance(data, list) and data:
        try:
            return float(data[0]["longShortRatio"])
        except (KeyError, TypeError, ValueError):
            pass
    return None


# ── Sinyal hesaplama ──────────────────────────────────────────────────────────

def onchain_signal(symbol: str) -> OnChainData:
    """
    Sembol için on-chain sinyali hesapla.

    Puanlama mantığı:
      Funding rate:
        >+0.10%  → aşırı long, düşüş riski    → -1
        <-0.05%  → aşırı short, yükseliş riski → +1
      OI değişimi (24s):
        >+10%    → yeni para giriyor           → +1
        <-10%    → para çıkıyor               → -1
      Long/Short ratio:
        >1.8     → aşırı kalabalık long        → -1 (contrarian)
        <0.7     → aşırı kalabalık short       → +1 (contrarian)
    """
    fr   = get_funding_rate(symbol)
    oi   = get_open_interest_change(symbol, hours=24)
    ls   = get_long_short_ratio(symbol)

    score   = 0
    reasons: list[str] = []

    # Funding rate analizi
    if fr is not None:
        if fr > 0.001:      # >0.10% — longlar çok pahalı ödüyor
            score -= 1
            reasons.append(f"Yüksek funding({fr:.3%})")
        elif fr > 0.0005:   # 0.05-0.10% — hafif uzun baskı
            pass
        elif fr < -0.0005:  # Negatif funding — shortlar ödüyor, yükseliş sinyal
            score += 1
            reasons.append(f"Negatif funding({fr:.3%})")

    # Open interest
    if oi is not None:
        if oi > 12.0:
            score += 1
            reasons.append(f"OI artıyor({oi:+.1f}%)")
        elif oi < -12.0:
            score -= 1
            reasons.append(f"OI azalıyor({oi:+.1f}%)")

    # Long/Short oranı (contrarian: kalabalık taraf kaybeder)
    if ls is not None:
        if ls > 1.8:
            score -= 1
            reasons.append(f"L/S aşırı long({ls:.2f})")
        elif ls < 0.65:
            score += 1
            reasons.append(f"L/S aşırı short({ls:.2f})")

    reason = " | ".join(reasons) if reasons else "On-chain nötr"
    logger.debug("On-chain [{}]: skor={:+d} | funding={} oi={}% ls={} | {}",
                 symbol, score,
                 f"{fr:.4%}" if fr is not None else "?",
                 f"{oi:.1f}" if oi is not None else "?",
                 f"{ls:.2f}" if ls is not None else "?",
                 reason)

    return OnChainData(
        funding_rate=fr,
        oi_change_pct=oi,
        long_short_ratio=ls,
        score=score,
        reason=reason,
    )


# ── Basit in-process cache ────────────────────────────────────────────────────
# Aynı sembol için birden fazla çağrıda gereksiz API trafiğini önler

_cache: dict[str, tuple[float, OnChainData]] = {}


def onchain_signal_cached(symbol: str) -> OnChainData:
    """TTL=5dk cache'li onchain_signal."""
    now  = time.monotonic()
    hit  = _cache.get(symbol)
    if hit and now - hit[0] < _CACHE_TTL:
        return hit[1]
    result = onchain_signal(symbol)
    _cache[symbol] = (now, result)
    return result
