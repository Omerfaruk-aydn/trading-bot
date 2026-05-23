"""Sandbox Trader — 100,000 TL ile BIST'te gün sonu 120,000 TL hedefli paper trader.

Kullanım:
    py sandbox_trader.py                        # Varsayılan semboller, 100k TL
    py sandbox_trader.py --capital 50000        # Farklı sermaye
    py sandbox_trader.py --target-pct 15        # %15 hedef
    py sandbox_trader.py --symbols THYAO GARAN  # Belirli hisseler
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

import yfinance as yf
from loguru import logger

from agents.sentiment import analyze_batch, summarize_sentiment
from data.collectors.news_collector import fetch_all_feeds, filter_by_symbols
from data.indicators import compute_all

Signal = Literal["buy", "sell", "hold"]


# ── Veri yapıları ────────────────────────────────────────────────────────────

@dataclass
class Position:
    symbol: str
    shares: float
    entry_price: float
    entry_time: str
    stop_loss: float
    take_profit: float
    highest_price: float = 0.0   # trailing stop için yüksek su işareti
    trailing_pct: float = 0.0    # 0 = trailing stop devre dışı

    def __post_init__(self):
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price

    @property
    def cost(self) -> float:
        return self.shares * self.entry_price

    def pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.shares

    def pnl_pct(self, current_price: float) -> float:
        return (current_price - self.entry_price) / self.entry_price * 100

    def update_trailing_stop(self, current_price: float) -> None:
        """Fiyat yeni zirve yaptıysa trailing stop'u yukarı taşı."""
        if self.trailing_pct <= 0:
            return
        if current_price > self.highest_price:
            self.highest_price = current_price
            new_stop = current_price * (1 - self.trailing_pct)
            if new_stop > self.stop_loss:
                self.stop_loss = new_stop


@dataclass
class Trade:
    symbol: str
    action: Literal["buy", "sell"]
    shares: float
    price: float
    timestamp: str
    reason: str
    pnl: float = 0.0


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    daily_pnl: float = 0.0

    @property
    def total_value(self) -> float:
        pos_value = sum(
            p.shares * self._get_price(p.symbol)
            for p in self.positions.values()
        )
        return self.cash + pos_value

    def _get_price(self, symbol: str) -> float:
        try:
            t = yf.Ticker(symbol)
            return float(t.fast_info.last_price or 0)
        except Exception:
            return 0.0


# ── Teknik sinyal üretici ────────────────────────────────────────────────────

def _get_signal(symbol: str, sentiment_score: float = 0.0) -> tuple[Signal, float, str]:
    """
    OHLCV + indikatör verisi + sentiment'i birleştirip sinyal üretir.

    Returns:
        (signal, confidence, reason)
    """
    try:
        df = yf.download(symbol, period="60d", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 30:
            return "hold", 0.0, "Yetersiz veri"

        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df = compute_all(df)
        row = df.iloc[-1]

        signals: list[int] = []   # +1 bullish, -1 bearish
        reasons: list[str] = []

        # RSI
        rsi = row.get("rsi", 50)
        if rsi < 35:
            signals.append(1); reasons.append(f"RSI aşırı satım ({rsi:.0f})")
        elif rsi > 65:
            signals.append(-1); reasons.append(f"RSI aşırı alım ({rsi:.0f})")

        # MACD
        macd = row.get("macd", 0)
        macd_sig = row.get("macd_signal", 0)
        if macd > macd_sig:
            signals.append(1); reasons.append("MACD pozitif kesişim")
        else:
            signals.append(-1); reasons.append("MACD negatif kesişim")

        # Fiyat vs SMA20
        close = row.get("close", 0)
        sma20 = row.get("sma20", close)
        if close > sma20 * 1.01:
            signals.append(1); reasons.append("Fiyat SMA20 üzerinde")
        elif close < sma20 * 0.99:
            signals.append(-1); reasons.append("Fiyat SMA20 altında")

        # Bollinger Band
        bb_low = row.get("bb_lower", 0)
        bb_high = row.get("bb_upper", float("inf"))
        if close <= bb_low:
            signals.append(1); reasons.append("Bollinger alt bant")
        elif close >= bb_high:
            signals.append(-1); reasons.append("Bollinger üst bant")

        # Sentiment bonus
        if sentiment_score > 0.3:
            signals.append(1); reasons.append(f"Pozitif haber sentiment ({sentiment_score:.2f})")
        elif sentiment_score < -0.3:
            signals.append(-1); reasons.append(f"Negatif haber sentiment ({sentiment_score:.2f})")

        score = sum(signals)
        confidence = abs(score) / max(len(signals), 1)

        if score >= 2:
            return "buy", confidence, " | ".join(reasons)
        elif score <= -2:
            return "sell", confidence, " | ".join(reasons)
        else:
            return "hold", confidence, " | ".join(reasons)

    except Exception as e:
        logger.error("{} sinyal hatası: {}", symbol, e)
        return "hold", 0.0, str(e)


# ── Sandbox Trader ───────────────────────────────────────────────────────────

class SandboxTrader:
    """
    100,000 TL sermayeyle BIST hisselerinde gün sonu hedef trader.

    Strateji:
    - Her döngüde tüm sembolleri tara
    - Teknik sinyal + haber sentiment → al/sat kararı
    - Risk kuralları: tek pozisyon max %15 sermaye, stop %3, take %6
    - Günlük kayıp limiti: %5 → dur
    """

    def __init__(
        self,
        symbols: list[str],
        initial_capital: float = 100_000.0,
        target_pct: float = 20.0,
        max_position_pct: float = 15.0,
        stop_loss_pct: float = 3.0,
        take_profit_pct: float = 6.0,
        daily_loss_limit_pct: float = 5.0,
        scan_interval: int = 300,
    ):
        self.symbols = symbols
        self.target_pct = target_pct
        self.target_value = initial_capital * (1 + target_pct / 100)
        self.max_position_pct = max_position_pct / 100
        self.stop_loss_pct = stop_loss_pct / 100
        self.take_profit_pct = take_profit_pct / 100
        self.daily_loss_limit = initial_capital * daily_loss_limit_pct / 100
        self.scan_interval = scan_interval

        self.portfolio = Portfolio(cash=initial_capital)
        self.initial_capital = initial_capital
        self._running = False

        logger.info(
            "SandboxTrader başlatıldı | Sermaye: {:,.0f} TL | Hedef: {:,.0f} TL ({:+.0f}%)",
            initial_capital, self.target_value, target_pct,
        )

    # ── Haber sentiment ──────────────────────────────────────────────────────

    def _get_sentiment(self, symbol: str) -> float:
        try:
            news = fetch_all_feeds()
            sym_news = filter_by_symbols(news, [symbol])
            if not sym_news:
                return 0.0
            enriched = analyze_batch(sym_news)
            summary = summarize_sentiment(enriched)
            return summary["score"]
        except Exception as e:
            logger.debug("{} sentiment hatası: {}", symbol, e)
            return 0.0

    # ── Emir yönetimi ────────────────────────────────────────────────────────

    def _buy(self, symbol: str, price: float, reason: str) -> bool:
        max_spend = self.portfolio.cash * self.max_position_pct
        if max_spend < price:
            logger.debug("{} için yeterli nakit yok ({:.0f} TL)", symbol, max_spend)
            return False
        if symbol in self.portfolio.positions:
            logger.debug("{} zaten pozisyonda", symbol)
            return False

        shares = max_spend / price
        cost = shares * price

        stop = price * (1 - self.stop_loss_pct)
        tp   = price * (1 + self.take_profit_pct)

        self.portfolio.cash -= cost
        self.portfolio.positions[symbol] = Position(
            symbol=symbol, shares=shares, entry_price=price,
            entry_time=datetime.now(tz=timezone.utc).isoformat(),
            stop_loss=stop, take_profit=tp,
        )
        self.portfolio.trades.append(Trade(
            symbol=symbol, action="buy", shares=shares, price=price,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            reason=reason,
        ))
        logger.info(
            "ALIŞ | {} | {:.4f} adet @ {:.2f} TL | Maliyet: {:.0f} TL | SL: {:.2f} | TP: {:.2f}",
            symbol, shares, price, cost, stop, tp,
        )
        return True

    def _sell(self, symbol: str, price: float, reason: str) -> bool:
        pos = self.portfolio.positions.get(symbol)
        if not pos:
            return False

        proceeds = pos.shares * price
        pnl = pos.pnl(price)

        self.portfolio.cash += proceeds
        self.portfolio.daily_pnl += pnl
        del self.portfolio.positions[symbol]

        self.portfolio.trades.append(Trade(
            symbol=symbol, action="sell", shares=pos.shares, price=price,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            reason=reason, pnl=pnl,
        ))
        logger.info(
            "SATIŞ | {} | {:.4f} adet @ {:.2f} TL | P&L: {:+.0f} TL ({:+.1f}%)",
            symbol, pos.shares, price, pnl, pos.pnl_pct(price),
        )
        return True

    # ── Döngü ────────────────────────────────────────────────────────────────

    def _check_stops(self) -> None:
        for symbol, pos in list(self.portfolio.positions.items()):
            try:
                price = float(yf.Ticker(symbol).fast_info.last_price or 0)
                if price <= 0:
                    continue
                if price <= pos.stop_loss:
                    self._sell(symbol, price, f"Stop-loss tetiklendi ({price:.2f} <= {pos.stop_loss:.2f})")
                elif price >= pos.take_profit:
                    self._sell(symbol, price, f"Take-profit tetiklendi ({price:.2f} >= {pos.take_profit:.2f})")
            except Exception as e:
                logger.debug("{} stop kontrol hatası: {}", symbol, e)

    def _scan(self) -> None:
        total = self.portfolio.total_value
        logger.info(
            "TARAMA | Portföy: {:,.0f} TL | Nakit: {:,.0f} TL | Hedef: {:,.0f} TL | P&L: {:+,.0f} TL",
            total, self.portfolio.cash, self.target_value, total - self.initial_capital,
        )

        # Hedef aşıldıysa tüm pozisyonları kapat
        if total >= self.target_value:
            logger.success("HEDEF ULAŞILDI! {:,.0f} TL / {:,.0f} TL", total, self.target_value)
            self._close_all("Günlük hedef ulaşıldı")
            self._running = False
            return

        # Günlük kayıp limitini kontrol et
        if -self.portfolio.daily_pnl >= self.daily_loss_limit:
            logger.warning("Günlük kayıp limiti aşıldı! Sistem durduruluyor.")
            self._close_all("Günlük kayıp limiti")
            self._running = False
            return

        # Stop/TP kontrol
        self._check_stops()

        # Yeni sinyal tara
        for symbol in self.symbols:
            if not self._running:
                break
            sentiment = self._get_sentiment(symbol)
            signal, confidence, reason = _get_signal(symbol, sentiment)

            try:
                price = float(yf.Ticker(symbol).fast_info.last_price or 0)
            except Exception:
                continue
            if price <= 0:
                continue

            logger.debug("{} | Sinyal: {} ({:.0f}%) | {}", symbol, signal, confidence * 100, reason)

            if signal == "buy" and confidence >= 0.5 and self.portfolio.cash > price:
                self._buy(symbol, price, reason)
            elif signal == "sell" and symbol in self.portfolio.positions:
                self._sell(symbol, price, reason)

    def _close_all(self, reason: str) -> None:
        for symbol in list(self.portfolio.positions.keys()):
            try:
                price = float(yf.Ticker(symbol).fast_info.last_price or 0)
                if price > 0:
                    self._sell(symbol, price, reason)
            except Exception as e:
                logger.error("{} kapatma hatası: {}", symbol, e)

    def run(self) -> None:
        self._running = True
        logger.info("Trader başladı. Durdur: Ctrl+C")
        try:
            while self._running:
                self._scan()
                if self._running:
                    logger.info("{} sn bekleniyor...", self.scan_interval)
                    time.sleep(self.scan_interval)
        except KeyboardInterrupt:
            logger.info("Kullanıcı durdurdu.")
        finally:
            self._print_summary()

    def _print_summary(self) -> None:
        total = self.portfolio.total_value
        pnl = total - self.initial_capital
        pnl_pct = pnl / self.initial_capital * 100
        logger.info("=" * 60)
        logger.info("ÖZET")
        logger.info("  Başlangıç : {:>12,.0f} TL", self.initial_capital)
        logger.info("  Bitiş     : {:>12,.0f} TL", total)
        logger.info("  P&L       : {:>+12,.0f} TL ({:+.2f}%)", pnl, pnl_pct)
        logger.info("  Hedef     : {:>12,.0f} TL", self.target_value)
        logger.info("  Toplam işlem: {}", len(self.portfolio.trades))
        logger.info("=" * 60)
        if self.portfolio.trades:
            logger.info("İşlem geçmişi:")
            for t in self.portfolio.trades:
                logger.info(
                    "  {} {} {} adet @ {:.2f} TL | P&L: {:+.0f} TL | {}",
                    t.action.upper(), t.symbol, f"{t.shares:.4f}", t.price, t.pnl, t.reason,
                )


# ── CLI ──────────────────────────────────────────────────────────────────────

DEFAULT_SYMBOLS = [
    "THYAO.IS", "GARAN.IS", "ASELS.IS", "EREGL.IS",
    "AKBNK.IS", "YKBNK.IS", "KCHOL.IS", "SISE.IS",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="BIST Sandbox Trader")
    parser.add_argument("--capital", type=float, default=100_000, help="Başlangıç sermayesi (TL)")
    parser.add_argument("--target-pct", type=float, default=20.0, help="Hedef getiri %%")
    parser.add_argument("--symbols", nargs="*", default=None, help="BIST sembolleri (örn: THYAO GARAN)")
    parser.add_argument("--interval", type=int, default=300, help="Tarama aralığı (saniye)")
    parser.add_argument("--stop", type=float, default=3.0, help="Stop-loss %%")
    parser.add_argument("--take", type=float, default=6.0, help="Take-profit %%")
    args = parser.parse_args()

    symbols = [s if s.endswith(".IS") else f"{s}.IS" for s in (args.symbols or DEFAULT_SYMBOLS)]

    trader = SandboxTrader(
        symbols=symbols,
        initial_capital=args.capital,
        target_pct=args.target_pct,
        stop_loss_pct=args.stop,
        take_profit_pct=args.take,
        scan_interval=args.interval,
    )
    trader.run()


if __name__ == "__main__":
    main()
