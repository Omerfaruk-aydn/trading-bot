"""Walk-Forward Kalibrasyon Backtest.

Amaç:
  "Bot %35 güven veriyor, bu ne anlama geliyor?" sorusunu cevaplayabilmek.
  Güven → gerçek başarı oranı (precision) tablosu oluşturur.

Çalışma mantığı:
  1. Sembol için 1-2 yıllık OHLCV indir.
  2. Kayan pencere: train_days'de indikatör hesapla, test_days geleceğe bak.
  3. Her test günü için teknik sinyal üret + güven hesapla.
  4. Gerçek fiyat hareketi ile karşılaştır (BUY sonrası fiyat yükseldiyse doğru?).
  5. Güven kategorilerine (0-25%, 25-50%, …) göre doğruluk oranı hesapla.

Kullanım:
    python -m backtest.walk_forward --symbol THYAO.IS --market bist
    python -m backtest.walk_forward --symbol BTC-USD  --market crypto
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from typing import NamedTuple

import pandas as pd
import yfinance as yf
from loguru import logger

from data.indicators import compute_all


class WalkForwardResult(NamedTuple):
    symbol:       str
    n_signals:    int
    calibration:  dict[str, dict]   # {"0-25%": {"n": 10, "precision": 0.45}, …}
    overall_precision: float
    sharpe_ratio: float


_BUCKETS = [
    (0.00, 0.35, "0-35%"),
    (0.35, 0.50, "35-50%"),
    (0.50, 0.65, "50-65%"),
    (0.65, 0.80, "65-80%"),
    (0.80, 1.01, "80-100%"),
]


def _bucket_name(conf: float) -> str:
    for lo, hi, label in _BUCKETS:
        if lo <= conf < hi:
            return label
    return "80-100%"


def _technical_score_simple(row: pd.Series, prev_row: pd.Series) -> tuple[int, float]:
    """Hızlı teknik skor — sadece walk-forward için basitleştirilmiş."""
    score = 0

    rsi = float(row.get("rsi", 50))
    if rsi < 30:
        score += 2
    elif rsi < 40:
        score += 1
    elif rsi > 75:
        score -= 2
    elif rsi > 65:
        score -= 1

    macd_h      = float(row.get("macd_hist", 0))
    macd_h_prev = float(prev_row.get("macd_hist", 0))
    if macd_h > 0 and macd_h_prev <= 0:
        score += 2
    elif macd_h > 0:
        score += 1
    elif macd_h < 0 and macd_h_prev >= 0:
        score -= 2
    elif macd_h < 0:
        score -= 1

    ema21  = float(row.get("ema_21", 0))
    ema55  = float(row.get("ema_55", 0))
    price  = float(row.get("close", 0))
    if price > ema21 > ema55 > 0:
        score += 2
    elif price > ema21 > 0:
        score += 1
    elif price < ema21 < ema55:
        score -= 2

    bb = float(row.get("bb_pct", 0.5))
    if bb <= 0.10:
        score += 1
    elif bb >= 0.90:
        score -= 1

    # Güven: eşikten yukarıya doğru 0.45-0.88
    threshold = 2
    conf = round(0.45 + max(abs(score) - threshold, 0) * 0.07, 3)
    conf = min(conf, 0.88)
    return score, conf


def run_walk_forward(
    symbol:      str,
    market:      str = "us",
    period:      str = "2y",
    train_days:  int = 60,
    test_days:   int = 15,
    hold_days:   int = 5,
) -> WalkForwardResult:
    """
    Walk-forward backtest çalıştır.

    Args:
        symbol:     Sembol adı (ör: "THYAO.IS", "BTC-USD")
        market:     "bist" | "us" | "crypto"
        period:     yfinance period (ör: "2y", "1y")
        train_days: Her pencere eğitim uzunluğu
        test_days:  Her pencere test uzunluğu
        hold_days:  Sinyal sonrası kaç gün tutacağız?

    Returns:
        WalkForwardResult
    """
    logger.info("Walk-forward başlıyor: {} | {} | period={}", symbol, market, period)

    # ── Veri indir ────────────────────────────────────────────────────────────
    df = yf.download(symbol, period=period, interval="1d",
                     progress=False, auto_adjust=True)
    if df.empty or len(df) < train_days + test_days + hold_days + 10:
        raise ValueError(f"Yeterli veri yok: {symbol} ({len(df)} satır)")

    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                  for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()

    # ── İndikatör hesapla ─────────────────────────────────────────────────────
    ind = compute_all(df.copy())

    # ── Walk-forward penceresi ────────────────────────────────────────────────
    bucket_hits:   dict[str, int] = defaultdict(int)
    bucket_total:  dict[str, int] = defaultdict(int)
    pnl_series:    list[float]    = []

    n = len(ind)
    step = max(test_days // 2, 1)

    for start in range(train_days, n - hold_days - 1, step):
        train_end  = start
        signal_idx = train_end
        future_idx = min(signal_idx + hold_days, n - 1)

        row      = ind.iloc[signal_idx]
        prev_row = ind.iloc[signal_idx - 1]

        score, conf = _technical_score_simple(row, prev_row)

        # BUY sinyali eşiği: 2
        if abs(score) < 2:
            continue

        signal = "buy" if score >= 2 else "sell"
        entry  = float(ind.iloc[signal_idx]["close"])
        exit_  = float(ind.iloc[future_idx]["close"])

        if entry <= 0:
            continue

        ret = (exit_ - entry) / entry
        correct = ret > 0 if signal == "buy" else ret < 0

        bname = _bucket_name(conf)
        bucket_total[bname] += 1
        if correct:
            bucket_hits[bname] += 1

        pnl = ret if signal == "buy" else -ret
        pnl_series.append(pnl)

    # ── Kalibrasyon tablosu ───────────────────────────────────────────────────
    calibration: dict[str, dict] = {}
    for _, _, label in _BUCKETS:
        total = bucket_total.get(label, 0)
        hits  = bucket_hits.get(label, 0)
        calibration[label] = {
            "n":         total,
            "precision": round(hits / total, 3) if total > 0 else None,
        }

    n_signals = sum(bucket_total.values())
    overall   = sum(bucket_hits.values()) / max(n_signals, 1)

    # Basit Sharpe (hold_days periyodu üstünde)
    if pnl_series:
        import statistics
        mu    = statistics.mean(pnl_series)
        sigma = statistics.stdev(pnl_series) if len(pnl_series) > 1 else 1e-9
        sharpe = mu / sigma * (252 / hold_days) ** 0.5 if sigma > 0 else 0.0
    else:
        sharpe = 0.0

    return WalkForwardResult(
        symbol=symbol,
        n_signals=n_signals,
        calibration=calibration,
        overall_precision=round(overall, 3),
        sharpe_ratio=round(sharpe, 2),
    )


def print_calibration(result: WalkForwardResult) -> None:
    """Kalibrasyon tablosunu terminale yaz."""
    S = "=" * 55
    print(f"\n{S}")
    print(f"  WALK-FORWARD KALİBRASYON — {result.symbol}")
    print(f"  Toplam sinyal: {result.n_signals} | "
          f"Genel doğruluk: {result.overall_precision:.1%} | "
          f"Sharpe: {result.sharpe_ratio:.2f}")
    print(S)
    print(f"  {'Güven Aralığı':<12} {'Sinyal':>8} {'Doğruluk':>10}  {'Durum'}")
    print("-" * 55)
    for bucket, stats in result.calibration.items():
        n   = stats["n"]
        prec = stats["precision"]
        if n == 0:
            status = "  -"
        elif prec is None:
            status = "  -"
        elif prec >= 0.60:
            status = "  ✓ Güvenilir"
        elif prec >= 0.50:
            status = "  ~ Zayıf"
        else:
            status = "  ✗ Kalibrasyon gerekli"
        prec_str = f"{prec:.1%}" if prec is not None else "  -"
        print(f"  {bucket:<12} {n:>8}  {prec_str:>9}  {status}")
    print(S)
    print("  Not: Güven eşiklerinizi 'Doğruluk' sütununa göre ayarlayın.")
    print(f"  Eğer %65+ güven → %55'ten az doğruysa eşiği yükselt.")
    print(S)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-Forward Kalibrasyon Backtest")
    parser.add_argument("--symbol", required=True, help="Sembol (ör: THYAO.IS, BTC-USD)")
    parser.add_argument("--market", default="us",
                        choices=["bist", "us", "crypto"], help="Piyasa türü")
    parser.add_argument("--period", default="2y", help="Veri periyodu (ör: 2y, 1y)")
    parser.add_argument("--hold",   type=int, default=5,  help="Tutma günü (varsayılan: 5)")
    args = parser.parse_args()

    try:
        result = run_walk_forward(
            symbol=args.symbol,
            market=args.market,
            period=args.period,
            hold_days=args.hold,
        )
        print_calibration(result)
    except Exception as e:
        logger.error("Backtest hatası: {}", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
