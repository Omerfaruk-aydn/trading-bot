"""Piyasa Tarama Motoru — hızlı teknik + ML skorlama, en iyi sinyalleri listeler."""
from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import yfinance as yf
from loguru import logger

from data.indicators import compute_all
from ml.support_resistance import sr_signal_score

SignalFilter = Literal["buy", "sell", "all"]

# ── Sembol listeleri ──────────────────────────────────────────────────────────

BIST_SYMBOLS = [
    "THYAO.IS", "GARAN.IS", "AKBNK.IS", "ASELS.IS", "EREGL.IS",
    "KCHOL.IS", "SISE.IS", "YKBNK.IS", "BIMAS.IS", "ARCLK.IS",
    "TUPRS.IS", "PGSUS.IS", "FROTO.IS", "TOASO.IS", "SAHOL.IS",
    "HALKB.IS", "VAKBN.IS", "CIMSA.IS", "TCELL.IS", "ISCTR.IS",
    "DOHOL.IS", "EKGYO.IS", "ENKAI.IS", "MGROS.IS", "TAVHL.IS",
    "ULKER.IS", "VESTL.IS", "KORDS.IS", "PETKM.IS", "TTKOM.IS",
]

US_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "AMD", "NFLX", "INTC", "JPM", "BAC", "GS", "V", "MA",
    "JNJ", "PFE", "XOM", "CVX", "WMT", "COST", "DIS", "SBUX",
    "BA", "CAT", "UBER", "COIN", "PLTR", "NET", "CRWD",
]

CRYPTO_SYMBOLS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
    "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "LINK-USD",
    "LTC-USD", "ATOM-USD", "NEAR-USD", "FIL-USD", "APT-USD",
]

MARKET_SYMBOLS = {
    "bist":   BIST_SYMBOLS,
    "us":     US_SYMBOLS,
    "crypto": CRYPTO_SYMBOLS,
    "all":    BIST_SYMBOLS + US_SYMBOLS + CRYPTO_SYMBOLS,
}


@dataclass
class ScanResult:
    symbol: str
    price: float
    score: int
    signal: Literal["BUY", "SELL", "HOLD"]
    trend: str
    rsi: float
    adx: float
    macd_positive: bool
    sr_score: int
    ml_signal: int | None
    ml_conf: float | None
    ret_1m: float | None    # 1 aylık getiri %
    currency: str


def _fetch_ohlcv(symbol: str) -> pd.DataFrame | None:
    try:
        df = yf.download(symbol, period="1y", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 30:
            return None
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                      for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]].dropna()
    except Exception:
        return None


def _quick_score(symbol: str) -> ScanResult | None:
    df = _fetch_ohlcv(symbol)
    if df is None:
        return None

    try:
        ind = compute_all(df.copy())
    except Exception:
        return None

    last  = ind.iloc[-1]
    price = float(last["close"])
    currency = "TL" if symbol.endswith(".IS") else "USD"

    rsi     = float(last.get("rsi", 50))
    macd_h  = float(last.get("macd_hist", 0))
    adx     = float(last.get("adx", 0))
    bb_pct  = float(last.get("bb_pct", 0.5))
    stoch_k = float(last.get("stoch_k", 50))
    cmf     = float(last.get("cmf", 0))
    ema21   = float(last.get("ema_21", price))
    ema55   = float(last.get("ema_55", price))
    ema200  = float(df["close"].ewm(span=200, adjust=False).mean().iloc[-1])
    obv_trend = int(last.get("obv_trend", 0))

    # 1 aylık getiri
    ret_1m = None
    past = df[df.index <= df.index[-1] - pd.Timedelta(days=30)]
    if not past.empty:
        ret_1m = round((price / float(past["close"].iloc[-1]) - 1) * 100, 1)

    # Destek/Direnç skoru
    sr_val = 0
    try:
        sr_val, _ = sr_signal_score(df)
    except Exception:
        pass

    # ML sinyali
    ml_sig = ml_conf = None
    try:
        from ml.predictor import ml_signal as _ml_signal
        mt = "bist" if symbol.endswith(".IS") else ("crypto" if "-USD" in symbol else "us")
        ml_sig, ml_conf = _ml_signal(df, threshold=0.58, market=mt)
    except Exception:
        pass

    # Hızlı puanlama
    score = 0

    if price > ema21 > ema55 > ema200:
        score += 3
    elif price > ema200:
        score += 1
    elif price < ema21 < ema55:
        score -= 2
    else:
        score -= 1

    if rsi < 30:
        score += 2
    elif 40 <= rsi <= 60:
        score += 1
    elif rsi > 75:
        score -= 2

    if macd_h > 0:
        score += 1
    else:
        score -= 1

    if adx >= 25:
        score += 1
    elif adx < 15:
        score -= 1

    if bb_pct <= 0.10:
        score += 2
    elif bb_pct >= 0.90:
        score -= 1

    if stoch_k < 20:
        score += 1
    elif stoch_k > 80:
        score -= 1

    if cmf > 0.15:
        score += 1
    elif cmf < -0.15:
        score -= 1

    if obv_trend == 1:
        score += 1
    elif obv_trend == -1:
        score -= 1

    score += sr_val

    if ml_sig == 1 and ml_conf and ml_conf >= 0.65:
        score += 2
    elif ml_sig == 1:
        score += 1
    elif ml_conf and ml_conf < 0.40:
        score -= 1

    # Trend etiketi
    if price > ema21 > ema55 > ema200:
        trend = "↑↑ Güçlü"
    elif price > ema200:
        trend = "↑  Zayıf"
    elif price < ema21 < ema55:
        trend = "↓↓ Güçlü"
    else:
        trend = "↓  Zayıf"

    if score >= 4:
        signal = "BUY"
    elif score <= -3:
        signal = "SELL"
    else:
        signal = "HOLD"

    return ScanResult(
        symbol=symbol, price=price, score=score, signal=signal,
        trend=trend, rsi=rsi, adx=adx, macd_positive=(macd_h > 0),
        sr_score=sr_val, ml_signal=ml_sig, ml_conf=ml_conf,
        ret_1m=ret_1m, currency=currency,
    )


def scan_market(
    market: str = "bist",
    signal: SignalFilter = "buy",
    top_n: int = 15,
    workers: int = 6,
) -> list[ScanResult]:
    """
    Piyasayı tarar, en güçlü sinyalleri döner.

    Args:
        market:  "bist" | "us" | "crypto" | "all"
        signal:  "buy" | "sell" | "all"
        top_n:   Kaç sembol göster
        workers: Paralel thread sayısı

    Returns:
        ScanResult listesi, skora göre sıralı
    """
    symbols = MARKET_SYMBOLS.get(market, BIST_SYMBOLS)
    logger.info("Tarama başlıyor: {} ({} sembol)", market.upper(), len(symbols))

    results: list[ScanResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_quick_score, s): s for s in symbols}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            done += 1
            sym = futures[fut]
            try:
                r = fut.result()
                if r:
                    results.append(r)
                    logger.debug("[{}/{}] {} → {} (skor: {:+d})",
                                 done, len(symbols), sym, r.signal, r.score)
            except Exception as e:
                logger.debug("{} hata: {}", sym, e)

    # Filtrele
    if signal == "buy":
        results = [r for r in results if r.signal == "BUY"]
        results.sort(key=lambda r: r.score, reverse=True)
    elif signal == "sell":
        results = [r for r in results if r.signal == "SELL"]
        results.sort(key=lambda r: r.score)
    else:
        results.sort(key=lambda r: r.score, reverse=True)

    return results[:top_n]


def print_scan_results(results: list[ScanResult], market: str, signal: str) -> None:
    """Tarama sonuçlarını terminale yazar."""
    if not results:
        print(f"\nSonuç bulunamadı: {market.upper()} | {signal.upper()}")
        return

    signal_icon = {"buy": "[AL]", "sell": "[SAT]", "all": "[TÜMÜ]"}.get(signal, "")
    S  = "=" * 80
    s2 = "-" * 80

    print(f"\n{S}")
    print(f"  PIYASA TARAMASI — {market.upper()} {signal_icon}")
    print(f"  {len(results)} sonuç bulundu")
    print(S)
    print(f"  {'Sembol':<14} {'Fiyat':>10} {'Skor':>5} {'Sinyal':>6}  "
          f"{'Trend':<12} {'RSI':>5} {'ADX':>5} {'1A%':>6}  {'ML':>6}")
    print(s2)

    for r in results:
        sinyal_str = {"BUY": " AL  ", "SELL": " SAT ", "HOLD": "BEKLE"}.get(r.signal, r.signal)
        ml_str = f"{r.ml_conf:.0%}" if r.ml_conf is not None else "  -  "
        ret_str = f"{r.ret_1m:+.1f}%" if r.ret_1m is not None else "  -  "
        macd_icon = "+" if r.macd_positive else "-"
        print(
            f"  {r.symbol:<14} {r.price:>9.2f}{r.currency[0]}  {r.score:>+4d}  "
            f"[{sinyal_str}]  {r.trend:<12} {r.rsi:>5.1f} {r.adx:>5.1f} "
            f"{ret_str:>6}  {ml_str:>5} MACD:{macd_icon}"
        )

    print(S)
    print(f"  Skor: +4 ve üzeri = AL | -3 ve altı = SAT | Arası = BEKLE")
    print(S)


def run_scan(market: str = "bist", signal: str = "buy", top_n: int = 15) -> None:
    """CLI giriş noktası."""
    results = scan_market(market=market, signal=signal, top_n=top_n)
    print_scan_results(results, market=market, signal=signal)
