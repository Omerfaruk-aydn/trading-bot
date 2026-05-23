"""Piyasa Rejimi Dedektörü — Trend / Yatay / Panik."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from loguru import logger

RegimeType = Literal["trending_up", "trending_down", "sideways", "panic"]

# Piyasa başına referans sembol (rejim ölçümü için)
_REGIME_SYMBOLS = {
    "bist":   "XU100.IS",
    "us":     "SPY",
    "crypto": "BTC-USD",
}


@dataclass
class MarketRegime:
    regime: RegimeType
    adx: float
    vol_ratio: float   # mevcut ATR / 20g ort ATR
    description: str

    @property
    def is_panic(self) -> bool:
        return self.regime == "panic"

    @property
    def is_trending(self) -> bool:
        return self.regime in ("trending_up", "trending_down")

    @property
    def is_sideways(self) -> bool:
        return self.regime == "sideways"

    @property
    def is_bullish(self) -> bool:
        return self.regime == "trending_up"


def detect_regime(
    df: pd.DataFrame,
    adx_trend: float = 25.0,
    adx_sideways: float = 15.0,
    vol_panic: float = 2.2,
) -> MarketRegime:
    """
    OHLCV verisinden piyasa rejimini tespit eder.

    Rejimler:
        trending_up   — ADX yüksek, fiyat EMA50 üstünde
        trending_down — ADX yüksek, fiyat EMA50 altında
        sideways      — ADX düşük
        panic         — ATR anormal yüksek (vol_panic × ort)
    """
    try:
        from data.indicators import compute_all
        ind = compute_all(df.copy())
        adx = float(ind["adx"].iloc[-1]) if "adx" in ind else 20.0
        atr = ind["atr"].dropna() if "atr" in ind else pd.Series(dtype=float)
    except Exception:
        adx = 20.0
        atr = pd.Series(dtype=float)

    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)

    # ATR volatilite oranı
    if atr.empty:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().dropna()

    vol_ratio = 1.0
    if len(atr) >= 20:
        atr_now = float(atr.iloc[-1])
        atr_avg = float(atr.iloc[-20:].mean())
        vol_ratio = atr_now / atr_avg if atr_avg > 0 else 1.0

    # EMA50 yön
    ema50 = close.ewm(span=50, adjust=False).mean()
    above_ema = float(close.iloc[-1]) > float(ema50.iloc[-1])

    # Panik tespiti önce
    if vol_ratio >= vol_panic:
        return MarketRegime(
            "panic", adx, vol_ratio,
            f"Panik: volatilite normalin {vol_ratio:.1f}x katı",
        )

    if adx >= adx_trend:
        regime: RegimeType = "trending_up" if above_ema else "trending_down"
        direction = "yükseliş" if above_ema else "düşüş"
        return MarketRegime(regime, adx, vol_ratio,
                            f"Güçlü {direction} trendi (ADX={adx:.0f})")

    return MarketRegime("sideways", adx, vol_ratio,
                        f"Yatay piyasa (ADX={adx:.0f})")


def detect_market_regime(market: str = "bist") -> MarketRegime | None:
    """
    Piyasa indeksinden rejim tespit eder.

    Args:
        market: "bist" | "us" | "crypto"

    Returns:
        MarketRegime veya None (veri çekilemezse)
    """
    sym = _REGIME_SYMBOLS.get(market)
    if not sym:
        return None
    try:
        import yfinance as yf
        df = yf.download(sym, period="60d", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 20:
            return None
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                      for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        regime = detect_regime(df)
        logger.debug("Piyasa rejimi [{}]: {} | ADX={:.0f} | Vol={:.1f}x",
                     market, regime.regime, regime.adx, regime.vol_ratio)
        return regime
    except Exception as e:
        logger.debug("Rejim tespit edilemedi [{}]: {}", market, e)
        return None


# Rejime göre sinyal eşiği çarpanı
REGIME_THRESHOLD_MULTIPLIER: dict[RegimeType, float] = {
    "trending_up":   0.85,   # trend var → daha kolay al
    "trending_down": 1.20,   # düşüş trendi → al için daha yüksek bar
    "sideways":      1.10,   # yatay → biraz daha temkinli
    "panic":         9.99,   # panik → yeni pozisyon açma
}

REGIME_POSITION_SIZE: dict[RegimeType, float] = {
    "trending_up":   1.00,   # tam boyut
    "trending_down": 0.50,   # yarı boyut
    "sideways":      0.75,   # 3/4 boyut
    "panic":         0.00,   # pozisyon açma
}
