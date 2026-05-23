"""Backtest motoru — geçmiş verilerle strateji testi.

Kullanım:
    py agents/backtest.py --symbols THYAO.IS GARAN.IS --period 3mo
    py agents/backtest.py --symbols THYAO.IS --period 6mo --mode aggressive
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import yfinance as yf
from loguru import logger

from data.indicators import compute_all

Signal = str  # "buy" | "sell" | "hold"


@dataclass
class BTrade:
    symbol: str
    action: str
    shares: float
    price: float
    date: str
    reason: str
    pnl: float = 0.0


@dataclass
class BPosition:
    symbol: str
    shares: float
    entry_price: float
    entry_date: str
    stop_loss: float
    take_profit: float

    def pnl(self, price: float) -> float:
        return (price - self.entry_price) * self.shares

    def pnl_pct(self, price: float) -> float:
        return (price - self.entry_price) / self.entry_price * 100


# ── Risk mod profilleri ───────────────────────────────────────────────────────

RISK_MODES = {
    "conservative": dict(min_score=2,   max_pos_pct=0.10, stop=0.04, take=0.05),
    "normal":       dict(min_score=2,   max_pos_pct=0.15, stop=0.03, take=0.06),
    "aggressive":   dict(min_score=1,   max_pos_pct=0.20, stop=0.02, take=0.08),
    "scalping":     dict(min_score=1,   max_pos_pct=0.05, stop=0.01, take=0.02),
}


# ── Kural tabanlı sinyal (LLM olmadan hızlı backtest) ────────────────────────

def _signal_from_row(row: pd.Series, min_score: int = 2) -> tuple[Signal, str]:
    signals, reasons = [], []

    rsi = row.get("rsi", 50)
    if rsi < 35:
        signals.append(1); reasons.append(f"RSI={rsi:.0f} aşırı satım")
    elif rsi > 65:
        signals.append(-1); reasons.append(f"RSI={rsi:.0f} aşırı alım")

    macd = row.get("macd", 0)
    macd_sig = row.get("macd_signal", 0)
    if macd > macd_sig:
        signals.append(1); reasons.append("MACD pozitif")
    else:
        signals.append(-1); reasons.append("MACD negatif")

    close = row.get("close", 0)
    sma20 = row.get("sma20", close)
    if close > sma20 * 1.01:
        signals.append(1); reasons.append("SMA20 üstü")
    elif close < sma20 * 0.99:
        signals.append(-1); reasons.append("SMA20 altı")

    bb_low = row.get("bb_lower", 0)
    bb_high = row.get("bb_upper", float("inf"))
    if close <= bb_low:
        signals.append(1); reasons.append("BB alt bant")
    elif close >= bb_high:
        signals.append(-1); reasons.append("BB üst bant")

    score = sum(signals)
    reason = " | ".join(reasons)

    if score >= min_score:
        return "buy", reason
    elif score <= -min_score:
        return "sell", reason
    return "hold", reason


# ── Backtest motoru ───────────────────────────────────────────────────────────

def run_backtest(
    symbols: list[str],
    initial_capital: float = 100_000.0,
    period: str = "3mo",
    mode: str = "normal",
) -> dict:
    profile = RISK_MODES.get(mode, RISK_MODES["normal"])
    min_score = profile["min_score"]
    max_pos_pct = profile["max_pos_pct"]
    stop_pct = profile["stop"]
    take_pct = profile["take"]

    cash = initial_capital
    positions: dict[str, BPosition] = {}
    trades: list[BTrade] = []
    daily_values: list[tuple[str, float]] = []

    # Tüm sembollerin verisini çek
    data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = yf.download(sym, period=period, interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 20:
            logger.warning("{} için yeterli veri yok, atlanıyor.", sym)
            continue
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df = compute_all(df)
        data[sym] = df
        logger.info("{} | {} gün veri yüklendi", sym, len(df))

    if not data:
        logger.error("Hiç veri yüklenemedi.")
        return {}

    # Ortak tarih aralığı
    all_dates = sorted(set.intersection(*[set(df.index) for df in data.values()]))
    logger.info("Backtest başlıyor | {} gün | Mod: {} | Sermaye: {:,.0f} TL",
                len(all_dates), mode.upper(), initial_capital)

    for date in all_dates:
        date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)

        # Stop/TP kontrol
        for sym, pos in list(positions.items()):
            if sym not in data:
                continue
            df = data[sym]
            if date not in df.index:
                continue
            price = float(df.loc[date, "close"])
            if price <= pos.stop_loss:
                pnl = pos.pnl(price)
                cash += pos.shares * price
                trades.append(BTrade(sym, "sell", pos.shares, price, date_str,
                                     f"Stop-loss ({price:.2f})", pnl))
                del positions[sym]
            elif price >= pos.take_profit:
                pnl = pos.pnl(price)
                cash += pos.shares * price
                trades.append(BTrade(sym, "sell", pos.shares, price, date_str,
                                     f"Take-profit ({price:.2f})", pnl))
                del positions[sym]

        # Sinyal tara
        for sym, df in data.items():
            if date not in df.index:
                continue
            row = df.loc[date]
            price = float(row.get("close", 0))
            if price <= 0:
                continue

            signal, reason = _signal_from_row(row, min_score)

            if signal == "buy" and sym not in positions:
                max_spend = cash * max_pos_pct
                if max_spend >= price:
                    shares = max_spend / price
                    cash -= shares * price
                    positions[sym] = BPosition(
                        symbol=sym, shares=shares, entry_price=price,
                        entry_date=date_str,
                        stop_loss=price * (1 - stop_pct),
                        take_profit=price * (1 + take_pct),
                    )
                    trades.append(BTrade(sym, "buy", shares, price, date_str, reason))

            elif signal == "sell" and sym in positions:
                pos = positions[sym]
                pnl = pos.pnl(price)
                cash += pos.shares * price
                trades.append(BTrade(sym, "sell", pos.shares, price, date_str, reason, pnl))
                del positions[sym]

        # Günlük portföy değeri
        pos_value = sum(
            p.shares * float(data[s].loc[date, "close"])
            for s, p in positions.items()
            if s in data and date in data[s].index
        )
        daily_values.append((date_str, cash + pos_value))

    # Son açık pozisyonları kapat
    last_date = all_dates[-1]
    for sym, pos in list(positions.items()):
        if sym in data and last_date in data[sym].index:
            price = float(data[sym].loc[last_date, "close"])
            pnl = pos.pnl(price)
            cash += pos.shares * price
            trades.append(BTrade(sym, "sell", pos.shares, price,
                                 last_date.strftime("%Y-%m-%d"), "Backtest sonu", pnl))

    final_value = cash
    total_pnl = final_value - initial_capital
    pnl_pct = total_pnl / initial_capital * 100

    sell_trades = [t for t in trades if t.action == "sell"]
    winning = [t for t in sell_trades if t.pnl > 0]
    losing  = [t for t in sell_trades if t.pnl <= 0]
    win_rate = len(winning) / max(len(sell_trades), 1) * 100
    avg_win  = sum(t.pnl for t in winning) / max(len(winning), 1)
    avg_loss = sum(t.pnl for t in losing)  / max(len(losing), 1)

    # Max drawdown
    values = [v for _, v in daily_values]
    peak = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd:
            max_dd = dd

    result = {
        "period": period,
        "mode": mode,
        "symbols": symbols,
        "initial_capital": initial_capital,
        "final_value": final_value,
        "total_pnl": total_pnl,
        "pnl_pct": pnl_pct,
        "total_trades": len(sell_trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown": max_dd,
        "trades": trades,
        "daily_values": daily_values,
    }

    _print_report(result)
    return result


def _print_report(r: dict) -> None:
    logger.info("=" * 60)
    logger.info("BACKTEST RAPORU")
    logger.info("  Dönem       : {}", r["period"])
    logger.info("  Mod         : {}", r["mode"].upper())
    logger.info("  Semboller   : {}", ", ".join(r["symbols"]))
    logger.info("-" * 60)
    logger.info("  Başlangıç   : {:>12,.0f} TL", r["initial_capital"])
    logger.info("  Bitiş       : {:>12,.0f} TL", r["final_value"])
    logger.info("  P&L         : {:>+12,.0f} TL ({:+.2f}%)", r["total_pnl"], r["pnl_pct"])
    logger.info("-" * 60)
    logger.info("  Toplam işlem: {}", r["total_trades"])
    logger.info("  Kazanma oranı: %{:.1f}", r["win_rate"])
    logger.info("  Ort. kazanç : {:>+8,.0f} TL", r["avg_win"])
    logger.info("  Ort. kayıp  : {:>+8,.0f} TL", r["avg_loss"])
    logger.info("  Max drawdown: %{:.2f}", r["max_drawdown"])
    logger.info("=" * 60)

    if r["trades"]:
        logger.info("Son 10 işlem:")
        for t in r["trades"][-10:]:
            if t.action == "sell":
                logger.info(
                    "  {} {} @ {:.2f} | P&L: {:+,.0f} TL | {}",
                    t.symbol, t.date, t.price, t.pnl, t.reason,
                )


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Motoru")
    parser.add_argument("--symbols", nargs="*", default=["THYAO.IS", "GARAN.IS", "ASELS.IS"])
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--period", default="3mo",
                        choices=["1mo", "3mo", "6mo", "1y", "2y"],
                        help="Backtest dönemi")
    parser.add_argument("--mode", default="normal",
                        choices=["conservative", "normal", "aggressive", "scalping"])
    args = parser.parse_args()

    symbols = [s if s.endswith(".IS") else f"{s}.IS" for s in args.symbols]
    run_backtest(symbols, args.capital, args.period, args.mode)


if __name__ == "__main__":
    main()
