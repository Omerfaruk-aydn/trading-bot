"""Teknik indikatörler — saf pandas/numpy implementasyonu.

Python 3.14 uyumlu (pandas-ta/numba gerektirmez).
Her fonksiyon bir pandas Series veya DataFrame döner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


# ── Yardımcı ──────────────────────────────────────────────────────────────────

def _validate(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame'de eksik kolonlar: {missing}")


def prepare_df(candles: list[dict]) -> pd.DataFrame:
    """
    Storage'dan gelen mum listesini indikatör hesaplamaya hazır DataFrame'e çevirir.

    Args:
        candles: get_ohlcv() çıktısı

    Returns:
        open_time index'li OHLCV DataFrame
    """
    df = pd.DataFrame(candles)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.set_index("open_time").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df


# ── Trend İndikatörleri ───────────────────────────────────────────────────────

def sma(series: pd.Series, period: int) -> pd.Series:
    """Basit hareketli ortalama."""
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Üstel hareketli ortalama."""
    return series.ewm(span=period, adjust=False).mean()


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD, Signal ve Histogram.

    Returns:
        DataFrame — kolonlar: macd, signal, histogram
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }, index=series.index)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — trend gücü."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr_val = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr_val
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr_val
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean()


def ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ichimoku Cloud.

    Returns:
        DataFrame — tenkan, kijun, senkou_a, senkou_b, chikou
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    chikou = close.shift(-26)

    return pd.DataFrame({
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b,
        "chikou": chikou,
    }, index=df.index)


# ── Momentum İndikatörleri ────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> pd.DataFrame:
    """Stochastic Oscillator (%K ve %D)."""
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    k_smooth = k.rolling(smooth_k).mean()
    d = k_smooth.rolling(d_period).mean()
    return pd.DataFrame({"stoch_k": k_smooth, "stoch_d": d}, index=df.index)


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * mad.replace(0, np.nan))


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R."""
    high_max = df["high"].rolling(period).max()
    low_min = df["low"].rolling(period).min()
    return -100 * (high_max - df["close"]) / (high_max - low_min).replace(0, np.nan)


# ── Volatilite İndikatörleri ──────────────────────────────────────────────────

def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands (üst, orta, alt, bant genişliği)."""
    mid = sma(series, period)
    std = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    width = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({
        "bb_upper": upper,
        "bb_mid": mid,
        "bb_lower": lower,
        "bb_width": width,
        "bb_pct": pct_b,
    }, index=series.index)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def keltner_channels(
    df: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    """Keltner Channels."""
    mid = ema(df["close"], ema_period)
    atr_val = atr(df, atr_period)
    return pd.DataFrame({
        "kc_upper": mid + multiplier * atr_val,
        "kc_mid": mid,
        "kc_lower": mid - multiplier * atr_val,
    }, index=df.index)


# ── Hacim İndikatörleri ───────────────────────────────────────────────────────

def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Hacim hareketli ortalaması."""
    return df["volume"].rolling(period).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume Weighted Average Price (gün içi sıfırlanmaz — yaklaşık)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cumvol = df["volume"].cumsum()
    cumtpvol = (tp * df["volume"]).cumsum()
    return cumtpvol / cumvol.replace(0, np.nan)


def chaikin_money_flow(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow."""
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
        (df["high"] - df["low"]).replace(0, np.nan)
    )
    mfv = clv * df["volume"]
    return mfv.rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, np.nan)


# ── Pattern Detection ─────────────────────────────────────────────────────────

def pivot_points(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Destek ve direnç seviyeleri (pivot point yöntemi).

    Returns:
        DataFrame — pivot, r1, r2, s1, s2
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)

    return pd.DataFrame({
        "pivot": pivot,
        "r1": r1,
        "r2": r2,
        "s1": s1,
        "s2": s2,
    }, index=df.index)


def candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temel mum formasyonları (1 = sinyal var, 0 = yok).

    Dönen kolonlar: doji, hammer, shooting_star, bullish_engulfing, bearish_engulfing
    """
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]
    body = (c - o).abs()
    candle_range = h - l

    # Doji: gövde çok küçük
    doji = (body / candle_range.replace(0, np.nan) < 0.1).astype(int)

    # Hammer: alt gölge >= 2x gövde, üst gölge küçük, aşağı trend sonrası
    lower_shadow = o.clip(lower=c) - l  # min(o,c) - low
    upper_shadow = h - o.clip(lower=c)  # high - max(o,c)
    hammer = (
        (lower_shadow >= 2 * body) &
        (upper_shadow < body) &
        (body > 0)
    ).astype(int)

    # Shooting Star: üst gölge >= 2x gövde, alt gölge küçük
    shooting_star = (
        (upper_shadow >= 2 * body) &
        (lower_shadow < body) &
        (body > 0)
    ).astype(int)

    # Bullish Engulfing
    prev_c = c.shift(1)
    prev_o = o.shift(1)
    bullish_engulfing = (
        (prev_c < prev_o) &   # önceki mum kırmızı
        (c > o) &              # şimdiki mum yeşil
        (o < prev_c) &         # şimdiki açılış önceki kapanışın altında
        (c > prev_o)           # şimdiki kapanış önceki açılışın üstünde
    ).astype(int)

    # Bearish Engulfing
    bearish_engulfing = (
        (prev_c > prev_o) &
        (c < o) &
        (o > prev_c) &
        (c < prev_o)
    ).astype(int)

    return pd.DataFrame({
        "doji": doji,
        "hammer": hammer,
        "shooting_star": shooting_star,
        "bullish_engulfing": bullish_engulfing,
        "bearish_engulfing": bearish_engulfing,
    }, index=df.index)


def volume_profile(df: pd.DataFrame, bins: int = 24) -> pd.DataFrame:
    """
    Volume Profile — her fiyat seviyesinde kaç birim işlem gördüğünü gösterir.

    Institutional trader'ların referans aldığı High Volume Node (HVN) ve
    Low Volume Node (LVN) seviyelerini bulmak için kullanılır.

    Returns:
        DataFrame — kolonlar: price_level, volume, is_hvn (HVN = ortalama üstü hacim)
    """
    price_min = float(df["low"].min())
    price_max = float(df["high"].max())
    if price_max <= price_min:
        return pd.DataFrame(columns=["price_level", "volume", "is_hvn"])

    edges  = np.linspace(price_min, price_max, bins + 1)
    vols   = np.zeros(bins, dtype=float)

    for _, row in df.iterrows():
        close = float(row["close"])
        vol   = float(row["volume"])
        # Kapanış fiyatının düştüğü bin'i bul
        idx = min(int((close - price_min) / (price_max - price_min) * bins), bins - 1)
        vols[idx] += vol

    levels = (edges[:-1] + edges[1:]) / 2
    avg_vol = float(np.mean(vols)) if vols.sum() > 0 else 1.0
    is_hvn  = vols >= avg_vol * 1.2

    return pd.DataFrame({
        "price_level": levels,
        "volume":      vols,
        "is_hvn":      is_hvn,
    })


def vwap_signal(df: pd.DataFrame, current_price: float) -> tuple[int, str]:
    """
    VWAP pozisyonuna göre basit sinyal üretir.

    Returns:
        (score, reason) — score: -1, 0 veya +1
    """
    if "vwap" not in df.columns:
        return 0, ""
    vwap_val = float(df["vwap"].iloc[-1])
    if vwap_val <= 0:
        return 0, ""
    pct = (current_price - vwap_val) / vwap_val
    if pct < -0.020:    # %2 altı → güçlü destek
        return 1, f"VWAP altında ({pct:+.1%})"
    if pct > 0.020:     # %2 üstü → direnç bölgesi
        return -1, f"VWAP üstünde ({pct:+.1%})"
    return 0, f"VWAP yakınında ({pct:+.1%})"


def rsi_divergence(df: pd.DataFrame, rsi_series: pd.Series, lookback: int = 14) -> pd.Series:
    """
    RSI-fiyat uyumsuzluğu (divergence) tespiti.

    Returns:
        Series: 1=bullish div, -1=bearish div, 0=yok
    """
    price_higher = df["close"] > df["close"].shift(lookback)
    rsi_lower = rsi_series < rsi_series.shift(lookback)
    bearish_div = (price_higher & rsi_lower).astype(int) * -1

    price_lower = df["close"] < df["close"].shift(lookback)
    rsi_higher = rsi_series > rsi_series.shift(lookback)
    bullish_div = (price_lower & rsi_higher).astype(int)

    return bearish_div + bullish_div


# ── Ana Hesaplama Fonksiyonu ───────────────────────────────────────────────────

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tüm indikatörleri tek seferde hesaplar ve DataFrame'e ekler.

    Args:
        df: prepare_df() çıktısı (OHLCV DataFrame)

    Returns:
        Tüm indikatörlerle zenginleştirilmiş DataFrame
    """
    _validate(df)
    result = df.copy()

    # Trend
    for p in [20, 50, 200]:
        result[f"sma_{p}"] = sma(df["close"], p)
    for p in [9, 21, 55]:
        result[f"ema_{p}"] = ema(df["close"], p)

    macd_df = macd(df["close"])
    result["macd"] = macd_df["macd"]
    result["macd_signal"] = macd_df["signal"]
    result["macd_hist"] = macd_df["histogram"]

    result["adx"] = adx(df)

    ichi = ichimoku(df)
    for col in ichi.columns:
        result[col] = ichi[col]

    # Momentum
    result["rsi"] = rsi(df["close"])
    stoch = stochastic(df)
    result["stoch_k"] = stoch["stoch_k"]
    result["stoch_d"] = stoch["stoch_d"]
    result["cci"] = cci(df)
    result["williams_r"] = williams_r(df)

    # Volatilite
    bb = bollinger_bands(df["close"])
    for col in bb.columns:
        result[col] = bb[col]
    result["atr"] = atr(df)
    kc = keltner_channels(df)
    for col in kc.columns:
        result[col] = kc[col]

    # Hacim
    result["obv"] = obv(df)
    result["volume_sma20"] = volume_sma(df)
    result["vwap"] = vwap(df)
    result["cmf"] = chaikin_money_flow(df)

    # Pattern
    pp = pivot_points(df)
    for col in pp.columns:
        result[col] = pp[col]
    patterns = candlestick_patterns(df)
    for col in patterns.columns:
        result[col] = patterns[col]
    result["rsi_divergence"] = rsi_divergence(df, result["rsi"])

    logger.debug("İndikatörler hesaplandı: {} satır, {} kolon", len(result), len(result.columns))
    return result


# ── LLM için Doğal Dil Özeti ──────────────────────────────────────────────────

def generate_summary(df: pd.DataFrame, symbol: str) -> str:
    """
    Hesaplanmış indikatörlerden LLM'e gönderilecek doğal dil özeti üretir.

    Args:
        df: compute_all() çıktısı
        symbol: İşlem sembolü (ör: BTCUSDT)

    Returns:
        Türkçe doğal dil analiz metni
    """
    if df.empty or len(df) < 2:
        return f"{symbol} için yeterli veri yok."

    row = df.iloc[-1]
    prev = df.iloc[-2]
    close = row["close"]
    prev_close = prev["close"]

    lines: list[str] = []

    # Fiyat hareketi
    change_pct = (close - df["close"].iloc[0]) / df["close"].iloc[0] * 100
    period_days = len(df)
    lines.append(
        f"{symbol} son {period_days} mumda {df['close'].iloc[0]:.4f}'den "
        f"{close:.4f}'e {'yükseldi' if change_pct > 0 else 'düştü'} "
        f"({'%+.2f' % change_pct})."
    )

    # RSI
    if not pd.isna(row.get("rsi")):
        r = row["rsi"]
        if r >= 70:
            comment = "aşırı alım bölgesinde"
        elif r <= 30:
            comment = "aşırı satım bölgesinde"
        elif r >= 60:
            comment = "güçlü momentum, henüz aşırı alım değil"
        elif r <= 40:
            comment = "zayıf momentum, henüz aşırı satım değil"
        else:
            comment = "nötr bölgede"
        lines.append(f"RSI {r:.1f} — {comment}.")

    # MACD
    if not pd.isna(row.get("macd")) and not pd.isna(prev.get("macd")):
        hist = row["macd_hist"]
        prev_hist = prev["macd_hist"]
        if hist > 0 and prev_hist <= 0:
            lines.append("MACD yeni pozitif kesişim yaptı — bullish sinyal.")
        elif hist < 0 and prev_hist >= 0:
            lines.append("MACD negatif kesişim yaptı — bearish sinyal.")
        elif hist > 0:
            lines.append(f"MACD pozitif bölgede (histogram: {hist:.4f}).")
        else:
            lines.append(f"MACD negatif bölgede (histogram: {hist:.4f}).")

    # EMA trend
    if not pd.isna(row.get("ema_21")) and not pd.isna(row.get("ema_55")):
        if close > row["ema_21"] > row["ema_55"]:
            lines.append("Fiyat EMA21 ve EMA55 üzerinde — kısa ve orta vadeli trend yukarı.")
        elif close < row["ema_21"] < row["ema_55"]:
            lines.append("Fiyat EMA21 ve EMA55 altında — kısa ve orta vadeli trend aşağı.")
        else:
            lines.append("EMA'lar karışık — net trend yok.")

    # Bollinger
    if not pd.isna(row.get("bb_upper")):
        pct = row.get("bb_pct", 0.5)
        if pct >= 0.9:
            lines.append("Bollinger üst bandına yakın — direnç bölgesi, aşırı alım riski.")
        elif pct <= 0.1:
            lines.append("Bollinger alt bandına yakın — destek bölgesi, aşırı satım riski.")
        width = row.get("bb_width", 0)
        if width < df["bb_width"].quantile(0.2):
            lines.append("Bollinger bantları sıkışmış — yakında kırılım bekleniyor.")

    # Hacim
    if not pd.isna(row.get("volume_sma20")) and row["volume_sma20"] != 0:
        vol_ratio = row["volume"] / row["volume_sma20"]
        if vol_ratio >= 1.5:
            lines.append(f"Hacim 20 günlük ortalamanın {vol_ratio:.1f}x üzerinde — güçlü ilgi.")
        elif vol_ratio <= 0.5:
            lines.append(f"Hacim ortalamanın altında ({vol_ratio:.1f}x) — düşük katılım.")

    # ADX (trend gücü)
    if not pd.isna(row.get("adx")):
        a = row["adx"]
        if a >= 40:
            lines.append(f"ADX {a:.1f} — çok güçlü trend.")
        elif a >= 25:
            lines.append(f"ADX {a:.1f} — trend var.")
        else:
            lines.append(f"ADX {a:.1f} — zayıf trend, yatay piyasa.")

    # Candlestick pattern
    patterns_found = []
    for p in ["bullish_engulfing", "bearish_engulfing", "hammer", "shooting_star", "doji"]:
        if row.get(p, 0) == 1:
            patterns_found.append(p.replace("_", " "))
    if patterns_found:
        lines.append(f"Son mumda formasyon: {', '.join(patterns_found)}.")

    # Pivot seviyeleri
    if not pd.isna(row.get("r1")) and not pd.isna(row.get("s1")):
        lines.append(
            f"Pivot seviyeleri — Direnç: R1={row['r1']:.4f}, R2={row['r2']:.4f} | "
            f"Destek: S1={row['s1']:.4f}, S2={row['s2']:.4f}."
        )

    # ATR (volatilite)
    if not pd.isna(row.get("atr")):
        atr_pct = row["atr"] / close * 100
        lines.append(f"ATR {row['atr']:.4f} (fiyatın %{atr_pct:.2f}'i) — {'yüksek' if atr_pct > 3 else 'normal'} volatilite.")

    return "\n".join(lines)
