"""
Teknik gösterge özellik mühendisliği.
Tüm özellikler fiyat bağımsız (oran/normalize edilmiş) olarak hesaplanır.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_COLS = [
    # Momentum (fiyat getirisi)
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    # RSI
    "rsi_7", "rsi_14", "rsi_21", "rsi_14_slope",
    # MACD
    "macd_norm", "macd_hist_norm", "macd_hist_slope",
    # EMA oranları (fiyat / EMA — 1 üzeri = fiyat EMA'nın üstünde)
    "ema9_ratio", "ema21_ratio", "ema50_ratio", "ema200_ratio",
    # EMA çapraz sinyalleri
    "ema9_above_21", "ema21_above_50", "ema50_above_200",
    # Bollinger Bantları
    "bb_position", "bb_width",
    # ATR (volatilite)
    "atr_pct",
    # Hacim
    "vol_ratio_10", "vol_ratio_20",
    # Fiyat pozisyonu (52 haftalık min-max arası)
    "price_52w_pos",
    # Mum yapısı
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
    # Günlük yüksek-düşük yayılımı
    "hl_spread",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _bollinger(close: pd.Series, period: int = 20, n_std: float = 2.0):
    ma = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return upper, ma, lower


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV DataFrame'den ML özellikleri hesaplar.

    Beklenen sütunlar: Open/open, High/high, Low/low, Close/close, Volume/volume
    Döndürür: FEATURE_COLS sütunlarını içeren DataFrame (satır sayısı aynı, baştaki NaN'lar var)
    """
    # Büyük/küçük harf normalleştirme
    col_map = {col.lower(): col for col in df.columns}
    def _get(name: str) -> pd.Series:
        key = col_map.get(name.lower()) or col_map.get(name)
        if key is None:
            raise KeyError(name)
        return df[key].astype(float)

    c = _get("close")
    o = _get("open")
    h = _get("high")
    l = _get("low")
    v = _get("volume").replace(0, np.nan)

    feat = pd.DataFrame(index=df.index)

    # ── Momentum ──────────────────────────────────────────────────────────────
    feat["ret_1d"]  = c.pct_change(1)
    feat["ret_5d"]  = c.pct_change(5)
    feat["ret_10d"] = c.pct_change(10)
    feat["ret_20d"] = c.pct_change(20)

    # ── RSI ───────────────────────────────────────────────────────────────────
    feat["rsi_7"]  = _rsi(c, 7)  / 100.0
    feat["rsi_14"] = _rsi(c, 14) / 100.0
    feat["rsi_21"] = _rsi(c, 21) / 100.0
    feat["rsi_14_slope"] = feat["rsi_14"].diff(3)

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    hist = macd_line - signal_line
    feat["macd_norm"]      = macd_line / c          # fiyata normalize
    feat["macd_hist_norm"] = hist / c
    feat["macd_hist_slope"] = hist.diff(3) / c

    # ── EMA oranları ──────────────────────────────────────────────────────────
    ema9   = _ema(c, 9)
    ema21  = _ema(c, 21)
    ema50  = _ema(c, 50)
    ema200 = _ema(c, 200)

    feat["ema9_ratio"]   = c / ema9   - 1
    feat["ema21_ratio"]  = c / ema21  - 1
    feat["ema50_ratio"]  = c / ema50  - 1
    feat["ema200_ratio"] = c / ema200 - 1

    feat["ema9_above_21"]  = (ema9  > ema21).astype(float)
    feat["ema21_above_50"] = (ema21 > ema50).astype(float)
    feat["ema50_above_200"]= (ema50 > ema200).astype(float)

    # ── Bollinger Bantları ────────────────────────────────────────────────────
    bb_upper, bb_mid, bb_lower = _bollinger(c, 20, 2.0)
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    feat["bb_position"] = (c - bb_lower) / bb_range   # 0=alt, 1=üst
    feat["bb_width"]    = bb_range / bb_mid            # genişlik oranı

    # ── ATR (Volatilite) ──────────────────────────────────────────────────────
    feat["atr_pct"] = _atr(h, l, c, 14) / c

    # ── Hacim ─────────────────────────────────────────────────────────────────
    vol_ma10 = v.rolling(10, min_periods=5).mean()
    vol_ma20 = v.rolling(20, min_periods=10).mean()
    feat["vol_ratio_10"] = v / vol_ma10
    feat["vol_ratio_20"] = v / vol_ma20

    # ── 52 Haftalık Fiyat Pozisyonu ───────────────────────────────────────────
    roll252 = 252
    low_52w  = l.rolling(roll252, min_periods=60).min()
    high_52w = h.rolling(roll252, min_periods=60).max()
    rng_52w  = (high_52w - low_52w).replace(0, np.nan)
    feat["price_52w_pos"] = (c - low_52w) / rng_52w

    # ── Mum Yapısı ────────────────────────────────────────────────────────────
    candle_range = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_wick = pd.concat([c, o], axis=1).min(axis=1) - l

    feat["body_ratio"]        = body / candle_range
    feat["upper_wick_ratio"]  = upper_wick.clip(lower=0) / candle_range
    feat["lower_wick_ratio"]  = lower_wick.clip(lower=0) / candle_range
    feat["hl_spread"]         = candle_range / c

    assert list(feat.columns) == FEATURE_COLS, \
        f"Özellik listesi uyumsuz: {list(feat.columns)}"

    return feat
