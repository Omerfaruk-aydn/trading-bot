"""LLM Tabanlı Trading Agent — Fine-tuned Qwen 2.5-1.5B ile otonom multi-market trader.

Desteklenen piyasalar:
  • BIST (Borsa İstanbul) — .IS uzantılı hisseler
  • Kripto Spot          — BTC-USD, ETH-USD, SOL-USD …
  • Kripto Futures       — BTC-PERP, ETH-PERP … (Binance USDT-M)
  • VİOP                 — XU030-FUT, USDTRY-FUT … (Borsa İstanbul Vadeli)
  • NYSE/NASDAQ          — AAPL, MSFT, NVDA … (ABD hisseleri)

Kullanım:
    py agents/llm_trading_agent.py --lora lora_weights/ --capital 100000 --target-pct 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yfinance as yf
from loguru import logger

from agents.sandbox_trader import Portfolio, Position, Trade, Signal
from agents.sentiment import analyze_batch, summarize_sentiment
from data.collectors.news_collector import fetch_all_feeds, filter_by_symbols
from data.indicators import compute_all
from data.db.database import TradeDB
from data.realtime.binance_ws import build_price_stream
from data.realtime.binance_futures_ws import BinanceFuturesStream, FUTURES_MAP
from data.markets.viop import VIOP_CONTRACTS, is_viop, get_viop_price, calc_liquidation_price as _calc_liq
from data.markets.us_stocks import (
    is_us_symbol, is_us_market_open, is_us_tradeable,
    get_us_session, session_label,
    get_usdtry_rate, usd_to_tl, tl_to_usd,
)
from data.calendar.economic_calendar import EconomicCalendar
from data.sources.yahoo_news import fetch_yahoo_news
from data.sources.market_screener import get_dynamic_watchlist, print_market_overview
from utils.telegram_bot import TelegramBot

# ── Yeni modüller ─────────────────────────────────────────────────────────────
from ml.ensemble import (
    bayesian_ensemble, SignalInput,
    technical_to_signal, ml_to_signal,
    sentiment_to_signal, multiframe_to_signal, onchain_to_signal,
)
from agents.context_classifier import ContextClassifier
from strategies import get_strategy

# ── Model yükleyici ──────────────────────────────────────────────────────────

class QwenInference:
    """Fine-tuned Qwen 2.5-1.5B modelini yükler ve çıkarım yapar."""

    def __init__(self, lora_path: str, base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.lora_path = Path(lora_path)
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> bool:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel

            logger.info("Model yükleniyor: {} + {}", self.base_model, self.lora_path)

            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.lora_path), trust_remote_code=True
            )

            # CUDA varsa 4-bit, yoksa cpu fp32
            if torch.cuda.is_available():
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                base = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
                logger.info("GPU ile 4-bit yüklendi.")
            else:
                base = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    torch_dtype="auto",
                    device_map="cpu",
                    trust_remote_code=True,
                )
                logger.info("CPU ile yüklendi (yavaş olabilir).")

            self.model = PeftModel.from_pretrained(base, str(self.lora_path))
            self.model.eval()
            self._loaded = True
            logger.info("Model hazır.")
            return True

        except Exception as e:
            logger.error("Model yüklenemedi: {}", e)
            return False

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        if not self._loaded:
            raise RuntimeError("Model yüklenmedi.")

        import torch

        messages = [
            {
                "role": "system",
                "content": (
                    "Sen uzman bir Türk borsası (BIST) trading asistanısın. "
                    "Piyasa verilerini analiz edip JSON formatında al/sat/bekle kararı verirsin."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Yardımcı ─────────────────────────────────────────────────────────────────

CRYPTO_SYMBOLS = {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "DOT", "MATIC"}


def _is_crypto(symbol: str) -> bool:
    """
    Kripto spot mu? Format: XXX-USD veya XXX-USDT (Binance evreninden gelebilir).
    -PERP sonekli futures hariç.
    """
    if symbol.endswith("-PERP"):
        return False
    # XXX-USD veya XXX-USDT formatı
    if "-USD" in symbol or "-USDT" in symbol:
        return True
    # Klasik set kontrolü (geriye uyumluluk)
    base = symbol.split("-")[0].upper()
    return base in CRYPTO_SYMBOLS


def _is_futures(symbol: str) -> bool:
    """Kripto futures mu? (BTC-PERP, ETH-PERP …)"""
    return symbol.upper().endswith("-PERP")


def _market_type(symbol: str) -> str:
    """
    Piyasa türünü döndürür:
      'bist'    — BIST hisse senedi (.IS)
      'crypto'  — Kripto spot (-USD)
      'futures' — Binance Futures perpetual (-PERP)
      'viop'    — VİOP vadeli sözleşmesi (-FUT)
      'us'      — NYSE/NASDAQ hissesi
    """
    if symbol.upper().endswith(".IS"):
        return "bist"
    if _is_futures(symbol):
        return "futures"
    if is_viop(symbol):
        return "viop"
    if _is_crypto(symbol):
        return "crypto"
    return "us"


def _currency(symbol: str) -> str:
    mt = _market_type(symbol)
    return "TL" if mt in ("bist", "viop") else "USD"


# ── Futures pozisyon veri yapısı ──────────────────────────────────────────────

@dataclass
class FuturesPosition:
    """Kaldıraçlı vadeli sözleşme pozisyonu (Binance Futures veya VİOP)."""
    symbol: str
    side: str            # "long" | "short"
    contracts: float     # Sözleşme adedi / miktar
    entry_price: float
    entry_time: str
    leverage: int
    margin_used: float   # Kullanılan teminat (TL)
    liquidation_price: float
    stop_loss: float
    take_profit: float
    highest_price: float = 0.0
    trailing_pct: float = 0.0

    def __post_init__(self):
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price

    def unrealized_pnl(self, mark_price: float, multiplier: float = 1.0) -> float:
        """Gerçekleşmemiş kar/zarar (TL)."""
        if self.side == "long":
            raw = (mark_price - self.entry_price) * self.contracts * multiplier
        else:
            raw = (self.entry_price - mark_price) * self.contracts * multiplier
        # Kripto futures: USD → TL
        if _is_futures(self.symbol):
            return raw * get_usdtry_rate()
        return raw

    def pnl_pct(self, mark_price: float, multiplier: float = 1.0) -> float:
        if self.margin_used == 0:
            return 0.0
        return self.unrealized_pnl(mark_price, multiplier) / self.margin_used * 100

    def update_trailing_stop(self, price: float) -> None:
        if self.trailing_pct <= 0:
            return
        if self.side == "long" and price > self.highest_price:
            self.highest_price = price
            new_stop = price * (1 - self.trailing_pct)
            if new_stop > self.stop_loss:
                self.stop_loss = new_stop
        elif self.side == "short" and price < self.highest_price:
            self.highest_price = price
            new_stop = price * (1 + self.trailing_pct)
            if new_stop < self.stop_loss:
                self.stop_loss = new_stop


# ── Piyasa verisi toplayıcı ──────────────────────────────────────────────────

@dataclass
class MarketSnapshot:
    symbol: str
    price: float
    change_pct: float
    # Momentum
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    macd_hist_prev: float      # önceki histogram değeri (crossover tespiti)
    stoch_k: float
    stoch_d: float
    # Trend
    sma20: float
    ema21: float
    ema55: float
    adx: float                 # trend gücü (>25 = güçlü trend)
    weekly_trend: int          # 1=yukarı, -1=aşağı, 0=yatay (haftalık zaman dilimi)
    # Volatilite & Bantlar
    bb_upper: float
    bb_lower: float
    bb_pct: float              # Bollinger %B konumu (0-1)
    atr: float                 # Average True Range (stop loss için)
    # Hacim
    volume: float
    volume_ratio: float        # volume / volume_sma20 (1.0 = normal)
    obv_trend: int             # 1=yükselen, -1=düşen, 0=yatay
    cmf: float                 # Chaikin Money Flow (-1..+1)
    # Mum formasyonu
    candle_pattern: int        # 1=bullish, -1=bearish, 0=yok
    # Sentiment
    sentiment_score: float
    sentiment_label: str
    news_headlines: list[str]


def _yf_symbol(symbol: str) -> str:
    """
    Teknik analiz için yfinance'e gönderilecek sembolü döndürür.

    Futures ve VİOP sembolleri yfinance'de yoktur; bunlar için
    dayanak varlık sembolü kullanılır:
      BTC-PERP  → BTC-USD
      ETH-PERP  → ETH-USD
      XU030-FUT → XU030.IS   (VIOP_CONTRACTS tablosundan)
      USDTRY-FUT→ USDTRY=X
    """
    mt = _market_type(symbol)
    if mt == "futures":
        base = symbol.split("-")[0].upper()
        return f"{base}-USD"
    if mt == "viop":
        contract = VIOP_CONTRACTS.get(symbol)
        if contract:
            return contract.underlying_yf
    return symbol


def _safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if not (v != v) else default  # NaN kontrolü
    except Exception:
        return default


def _weekly_trend(yf_sym: str) -> int:
    """
    Haftalık trend yönü: 1=yukarı, -1=aşağı, 0=yatay.
    Haftalık SMA10 eğimine bakılır.
    """
    try:
        wdf = yf.download(yf_sym, period="52wk", interval="1wk", progress=False, auto_adjust=True)
        if wdf.empty or len(wdf) < 12:
            return 0
        wdf.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in wdf.columns]
        close = wdf["close"].dropna()
        sma10 = close.rolling(10).mean()
        if len(sma10.dropna()) < 3:
            return 0
        slope = sma10.iloc[-1] - sma10.iloc[-3]
        if slope > sma10.iloc[-1] * 0.01:
            return 1
        elif slope < -sma10.iloc[-1] * 0.01:
            return -1
        return 0
    except Exception:
        return 0


def _collect_multiframe_signals(
    yf_sym: str,
) -> dict[str, tuple[str, float]]:
    """
    15dk, 1s ve 4s zaman dilimlerinde basit teknik sinyal üretir.

    Returns:
        {"15m": ("buy"|"sell"|"hold", conf), "1h": ..., "4h": ...}
    """
    frames: dict[str, tuple[str, str]] = {
        "15m": ("5d",  "15m"),
        "1h":  ("30d", "1h"),
        "4h":  ("60d", "1h"),   # yfinance 4h desteklemiyor; 1h + rolling ile yaklaşım
    }
    results: dict[str, tuple[str, float]] = {}

    for label, (period, interval) in frames.items():
        try:
            df = yf.download(yf_sym, period=period, interval=interval,
                             progress=False, auto_adjust=True)
            if df.empty or len(df) < 20:
                continue
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                          for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].dropna()

            # 4h yaklaşımı: 1h'lık verileri 4 mum topla
            if label == "4h" and interval == "1h":
                df = df.resample("4h").agg({
                    "open": "first", "high": "max",
                    "low": "min",   "close": "last",
                    "volume": "sum",
                }).dropna()
                if len(df) < 15:
                    continue

            from data.indicators import compute_all
            ind = compute_all(df)
            row = ind.iloc[-1]

            score = 0
            rsi   = float(row.get("rsi", 50))
            macd_h = float(row.get("macd_hist", 0))
            ema21  = float(row.get("ema_21", row.get("close", 1)))
            price  = float(row.get("close", 1))

            if rsi < 35:    score += 1
            elif rsi > 70:  score -= 1
            if macd_h > 0:  score += 1
            else:           score -= 1
            if price > ema21 > 0: score += 1
            else:                 score -= 1

            if score >= 2:
                results[label] = ("buy",  round(0.50 + score * 0.08, 2))
            elif score <= -2:
                results[label] = ("sell", round(0.50 + abs(score) * 0.08, 2))
            else:
                results[label] = ("hold", 0.45)
        except Exception:
            pass

    return results


def _collect_snapshot(
    symbol: str,
    cached_news: list[dict] | None = None,
    price_override: float | None = None,
    return_df: bool = False,
) -> "MarketSnapshot | None | tuple[MarketSnapshot, object]":
    """
    Kapsamlı teknik göstergeler + sentiment + haftalık trend snapshot'ı.
    """
    try:
        yf_sym = _yf_symbol(symbol)
        df = yf.download(yf_sym, period="90d", interval="1d", progress=False,
                         auto_adjust=True, prepost=True)
        if df.empty or len(df) < 26:
            return (None, None) if return_df else None

        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df = compute_all(df)
        row = df.iloc[-1]
        prev = df.iloc[-2]

        hist_price = _safe_float(row.get("close"), 0)

        # Pre/after-market fiyatını dene (ABD hissesi, seans dışı)
        ext_price = 0.0
        mt_check = _market_type(symbol)
        if mt_check == "us" and not price_override:
            us_sess = get_us_session()
            if us_sess in ("premarket", "aftermarket"):
                try:
                    ticker_obj = yf.Ticker(yf_sym)
                    fi = ticker_obj.fast_info
                    if us_sess == "premarket":
                        ext_price = float(getattr(fi, "pre_market_price", 0) or 0)
                    else:
                        ext_price = float(getattr(fi, "post_market_price", 0) or 0)
                except Exception:
                    pass

        price = (
            price_override if (price_override and price_override > 0)
            else ext_price if ext_price > 0
            else hist_price
        )
        prev_price = _safe_float(prev.get("close"), price)
        change_pct = (price - prev_price) / prev_price * 100 if prev_price else 0.0

        # OBV trendi: son 5 günlük eğim
        obv_vals = df["obv"].dropna()
        if len(obv_vals) >= 5:
            obv_slope = obv_vals.iloc[-1] - obv_vals.iloc[-5]
            obv_trend = 1 if obv_slope > 0 else (-1 if obv_slope < 0 else 0)
        else:
            obv_trend = 0

        # Mum formasyonu (son mum)
        candle_pattern = 0
        if _safe_float(row.get("bullish_engulfing")) == 1 or _safe_float(row.get("hammer")) == 1:
            candle_pattern = 1
        elif _safe_float(row.get("bearish_engulfing")) == 1 or _safe_float(row.get("shooting_star")) == 1:
            candle_pattern = -1

        # Volume ratio
        vol_sma = _safe_float(row.get("volume_sma20"), 1)
        vol = _safe_float(row.get("volume"), 0)
        vol_ratio = vol / vol_sma if vol_sma > 0 else 1.0

        # Haftalık trend (ayrı indirme — cache ile hızlandırılabilir)
        wtrend = _weekly_trend(yf_sym)

        # ── Haber & Sentiment ─────────────────────────────────────────────────
        mt = _market_type(symbol)
        sentiment_score = 0.0
        sentiment_label = "neutral"
        headlines: list[str] = []

        try:
            if mt in ("us", "crypto", "futures"):
                # Yahoo Finance RSS — hisse başına güncel haber
                yf_news = fetch_yahoo_news(symbol, max_items=5)
                if yf_news:
                    headlines = [n["title"] for n in yf_news[:3] if n.get("title")]
                    # Basit sentiment: pozitif/negatif kelime sayımı
                    positive_kw = {"surge", "beat", "record", "growth", "up", "rise", "gain",
                                   "strong", "buy", "upgrade", "rally", "bull", "profit"}
                    negative_kw = {"fall", "drop", "miss", "loss", "cut", "down", "decline",
                                   "sell", "downgrade", "warn", "bear", "risk", "crash"}
                    pos = neg = 0
                    for h in headlines:
                        hl = h.lower()
                        pos += sum(1 for k in positive_kw if k in hl)
                        neg += sum(1 for k in negative_kw if k in hl)
                    total = pos + neg
                    if total > 0:
                        sentiment_score = round((pos - neg) / total, 2)
                        sentiment_label = "positive" if sentiment_score > 0.1 else (
                            "negative" if sentiment_score < -0.1 else "neutral"
                        )
            else:
                # BIST/VİOP: mevcut Türkçe RSS sistemi
                news = cached_news if cached_news is not None else fetch_all_feeds()
                sym_news = filter_by_symbols(news, [symbol])
                enriched = analyze_batch(sym_news)
                summary = summarize_sentiment(enriched)
                headlines = [
                    n.get("payload", {}).get("title", "")
                    for n in sym_news[:3]
                    if n.get("payload", {}).get("title")
                ]
                sentiment_score = summary["score"]
                sentiment_label = summary["overall"]
        except Exception:
            pass

        # sma20 yoksa sma_20 dene (compute_all'ın çıktısına göre)
        sma20_val = _safe_float(
            row.get("sma20") if not (row.get("sma20") != row.get("sma20")) else row.get("sma_20"),
            hist_price
        )

        snap_obj = MarketSnapshot(
            symbol=symbol,
            price=price,
            change_pct=round(change_pct, 2),
            rsi=_safe_float(row.get("rsi"), 50),
            macd=_safe_float(row.get("macd"), 0),
            macd_signal=_safe_float(row.get("macd_signal"), 0),
            macd_hist=_safe_float(row.get("macd_hist"), 0),
            macd_hist_prev=_safe_float(prev.get("macd_hist"), 0),
            stoch_k=_safe_float(row.get("stoch_k"), 50),
            stoch_d=_safe_float(row.get("stoch_d"), 50),
            sma20=sma20_val,
            ema21=_safe_float(row.get("ema_21"), hist_price),
            ema55=_safe_float(row.get("ema_55"), hist_price),
            adx=_safe_float(row.get("adx"), 0),
            weekly_trend=wtrend,
            bb_upper=_safe_float(row.get("bb_upper"), hist_price * 1.02),
            bb_lower=_safe_float(row.get("bb_lower"), hist_price * 0.98),
            bb_pct=_safe_float(row.get("bb_pct"), 0.5),
            atr=_safe_float(row.get("atr"), 0),
            volume=vol,
            volume_ratio=round(vol_ratio, 2),
            obv_trend=obv_trend,
            cmf=_safe_float(row.get("cmf"), 0),
            candle_pattern=candle_pattern,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            news_headlines=headlines,
        )

        # ML için ham OHLCV — sütun isimlerini büyük harfe çevir
        if return_df:
            ohlcv = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
            return snap_obj, ohlcv
        return snap_obj

    except Exception as e:
        logger.debug("{} snapshot hatası: {}", symbol, e)
        if return_df:
            return None, None
        return None


_MARKET_TYPE_LABELS = {
    "bist":    "BIST hisse senedi (Borsa İstanbul)",
    "crypto":  "Kripto para (spot)",
    "futures": "Kripto Futures (Binance USDT-M Perpetual, kaldıraçlı)",
    "viop":    "VİOP vadeli sözleşmesi (Borsa İstanbul Vadeli)",
    "us":      "NYSE/NASDAQ ABD hisse senedi",
}


def _build_prompt(
    snap: MarketSnapshot,
    portfolio: Portfolio,
    futures_positions: dict | None = None,
    leverage: int = 1,
) -> str:
    headlines_text = (
        "\n".join(f"  - {h}" for h in snap.news_headlines)
        if snap.news_headlines
        else "  - Haber bulunamadı"
    )

    cur = _currency(snap.symbol)
    mt = _market_type(snap.symbol)

    # Mevcut pozisyon metni
    fpos = (futures_positions or {}).get(snap.symbol)
    spot_pos = portfolio.positions.get(snap.symbol)
    if fpos:
        position_text = (
            f"Açık {fpos.side.upper()} pozisyon: {fpos.contracts:.4f} sözleşme @ "
            f"{fpos.entry_price:.4f} {cur} | Kaldıraç: {fpos.leverage}x | "
            f"Tasfiye fiyatı: {fpos.liquidation_price:.4f} "
            f"(SL: {fpos.stop_loss:.4f}, TP: {fpos.take_profit:.4f})"
        )
    elif spot_pos:
        position_text = (
            f"Açık pozisyon: {spot_pos.shares:.4f} adet @ {spot_pos.entry_price:.2f} {cur} "
            f"(SL: {spot_pos.stop_loss:.2f}, TP: {spot_pos.take_profit:.2f})"
        )
    else:
        position_text = "Açık pozisyon yok"

    leverage_note = f"\nKaldıraç: {leverage}x" if mt in ("futures", "viop") else ""
    action_hint = (
        '{"action": "buy" (long aç) | "sell" (short aç / long kapat) | "hold", "confidence": 0.0-1.0, "reason": "..."}'
        if mt in ("futures", "viop")
        else '{"action": "buy" | "sell" | "hold", "confidence": 0.0-1.0, "reason": "kısa açıklama"}'
    )

    wt_str = {1: "YUKARI", -1: "ASAGI", 0: "YATAY"}.get(snap.weekly_trend, "YATAY")
    return f"""Piyasa turu: {_MARKET_TYPE_LABELS.get(mt, mt)}
Sembol: {snap.symbol}{leverage_note}
Guncel fiyat: {snap.price:.4f} {cur} ({snap.change_pct:+.2f}%)
Haftalik trend: {wt_str}

Teknik gostergeler:
  RSI: {snap.rsi:.1f} | Stoch K/D: {snap.stoch_k:.0f}/{snap.stoch_d:.0f}
  MACD hist: {snap.macd_hist:+.4f} (onceki: {snap.macd_hist_prev:+.4f})
  EMA21: {snap.ema21:.4f} | EMA55: {snap.ema55:.4f} | ADX: {snap.adx:.1f}
  Bollinger %B: {snap.bb_pct:.0%} (ust: {snap.bb_upper:.4f} alt: {snap.bb_lower:.4f})
  ATR: {snap.atr:.4f} | Hacim orani: {snap.volume_ratio:.1f}x
  OBV trend: {'YUKARI' if snap.obv_trend==1 else 'ASAGI' if snap.obv_trend==-1 else 'YATAY'}
  CMF: {snap.cmf:+.3f}

Haber sentiment: {snap.sentiment_label} (skor: {snap.sentiment_score:+.2f})
Son haberler:
{headlines_text}

Portföy durumu:
  Nakit: {portfolio.cash:,.0f} TL
  {position_text}

Yukarıdaki verilere göre bu sembol için kararını JSON olarak ver:
{action_hint}"""


def _parse_decision(response: str) -> dict | None:
    try:
        match = re.search(r'\{[^{}]*"action"[^{}]*\}', response, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(response)
    except Exception:
        low = response.lower()
        if "buy" in low or "al" in low:
            return {"action": "buy", "confidence": 0.5, "reason": response[:100]}
        elif "sell" in low or "sat" in low:
            return {"action": "sell", "confidence": 0.5, "reason": response[:100]}
        return {"action": "hold", "confidence": 0.5, "reason": "parse edilemedi"}


def _technical_signal(snap: MarketSnapshot, ohlcv_df=None) -> tuple[str, float, str]:
    """
    Profesyonel çok katmanlı teknik sinyal motoru — 13 indikatör, ağırlıklı puanlama.
    """
    score = 0
    reasons: list[str] = []
    rsi = snap.rsi
    adx_val = snap.adx

    # 1. Haftalık trend yönü (±2)
    if snap.weekly_trend == 1:
        score += 2; reasons.append("Haftalık trend yukari")
    elif snap.weekly_trend == -1:
        score -= 2; reasons.append("Haftalık trend asagi")

    # 2. EMA hizalaması (±2 / ±1)
    if snap.ema21 > 0 and snap.ema55 > 0:
        if snap.price > snap.ema21 > snap.ema55:
            score += 2; reasons.append(f"EMA hizasi yukari ({snap.price/snap.ema21-1:+.1%})")
        elif snap.price < snap.ema21 < snap.ema55:
            score -= 2; reasons.append(f"EMA hizasi asagi ({snap.price/snap.ema21-1:+.1%})")
        elif snap.price > snap.ema21:
            score += 1; reasons.append("Fiyat EMA21 ustu")
        elif snap.price < snap.ema21:
            score -= 1; reasons.append("Fiyat EMA21 alti")

    # 3. MACD (taze kesisim ±2, mevcut durum ±1)
    h, hp = snap.macd_hist, snap.macd_hist_prev
    if h > 0 and hp <= 0:
        score += 2; reasons.append("MACD taze yukari kesisim")
    elif h < 0 and hp >= 0:
        score -= 2; reasons.append("MACD taze asagi kesisim")
    elif h > 0:
        score += 1; reasons.append(f"MACD pozitif ({h:+.4f})")
    elif h < 0:
        score -= 1; reasons.append(f"MACD negatif ({h:+.4f})")

    # 4. RSI (±2 asiri bolge, ±1 normal)
    if rsi < 25:
        score += 2; reasons.append(f"RSI asiri satim ({rsi:.0f})")
    elif rsi < 35:
        score += 1; reasons.append(f"RSI dusuk ({rsi:.0f})")
    elif 52 <= rsi <= 65:
        score += 1; reasons.append(f"RSI saglikli momentum ({rsi:.0f})")
    elif rsi > 82:
        score -= 2; reasons.append(f"RSI asiri alim ({rsi:.0f})")
    elif rsi > 72:
        score -= 1; reasons.append(f"RSI yuksek ({rsi:.0f})")

    # 5. Stochastic K-D kesisimi (±1)
    k, d = snap.stoch_k, snap.stoch_d
    if k < 20 and k > d:
        score += 1; reasons.append(f"Stoch oversold donus (K={k:.0f})")
    elif k > 80 and k < d:
        score -= 1; reasons.append(f"Stoch overbought donus (K={k:.0f})")

    # 6. Bollinger bantlari (±2 ekstrem, ±1 yakın)
    bb = snap.bb_pct
    if bb <= 0.05:
        score += 2; reasons.append(f"BB alt bant (%B={bb:.0%})")
    elif bb <= 0.20:
        score += 1; reasons.append(f"BB alt yakini (%B={bb:.0%})")
    elif bb >= 0.95:
        score -= 2; reasons.append(f"BB ust bant (%B={bb:.0%})")
    elif bb >= 0.80:
        score -= 1; reasons.append(f"BB ust yakini (%B={bb:.0%})")

    # 7. Hacim onayi (±1)
    vr = snap.volume_ratio
    if vr >= 1.5:
        score += 1; reasons.append(f"Guclu hacim ({vr:.1f}x)")
    elif vr <= 0.5:
        score -= 1; reasons.append(f"Zayif hacim ({vr:.1f}x)")

    # 8. OBV trendi (±1)
    if snap.obv_trend == 1:
        score += 1; reasons.append("OBV yukseliyor")
    elif snap.obv_trend == -1:
        score -= 1; reasons.append("OBV dusiyor")

    # 9. Chaikin Money Flow (±1)
    if snap.cmf > 0.15:
        score += 1; reasons.append(f"CMF pozitif ({snap.cmf:+.2f})")
    elif snap.cmf < -0.15:
        score -= 1; reasons.append(f"CMF negatif ({snap.cmf:+.2f})")

    # 10. Mum formasyonu (±1)
    if snap.candle_pattern == 1:
        score += 1; reasons.append("Bullish mum formasyonu")
    elif snap.candle_pattern == -1:
        score -= 1; reasons.append("Bearish mum formasyonu")

    # 11. Sentiment (±1)
    if snap.sentiment_score > 0.25:
        score += 1; reasons.append(f"Pozitif haber ({snap.sentiment_score:+.2f})")
    elif snap.sentiment_score < -0.25:
        score -= 1; reasons.append(f"Negatif haber ({snap.sentiment_score:+.2f})")

    # 12. Destek/Direnç seviyesi (±2)
    if ohlcv_df is not None:
        try:
            from ml.support_resistance import sr_signal_score
            sr_val, sr_reason = sr_signal_score(ohlcv_df)
            if sr_val != 0:
                score += sr_val
                reasons.append(sr_reason)
        except Exception:
            pass

    # 13. Fibonacci geri çekilme yakınlığı (±1)
    if ohlcv_df is not None and len(ohlcv_df) >= 30:
        try:
            from ml.support_resistance import fibonacci_levels
            fib = fibonacci_levels(ohlcv_df)
            price = snap.price
            for label, level in [("fib_38", fib["fib_38"]), ("fib_50", fib["fib_50"]), ("fib_61", fib["fib_61"])]:
                dist = abs(price - level) / price
                if dist <= 0.015 and price > fib["fib_50"]:
                    score += 1; reasons.append(f"Fibonacci destek {label}({level:.4g})"); break
                elif dist <= 0.015 and price <= fib["fib_50"]:
                    score -= 1; reasons.append(f"Fibonacci direnç {label}({level:.4g})"); break
        except Exception:
            pass

    # ADX filtresi: yatay piyasada esigi yukselt
    strong_trend = adx_val >= 22
    if adx_val > 0:
        reasons.append(f"ADX={adx_val:.0f}({'trend' if strong_trend else 'yatay'})")
    buy_threshold  = 2 if strong_trend else 4
    sell_threshold = -2 if strong_trend else -4

    # Hacim filtresi: cok dusuk hacimde alim zayiflatilir
    if snap.volume_ratio < 0.4 and score > 0:
        score = max(score - 1, 0)
        reasons.append("(hacim filtresi -1)")

    confidence = round(min(abs(score) / 12.0, 1.0), 2)
    reason_str = " | ".join(reasons) if reasons else "Notrl sinyal"

    logger.debug("{} Teknik skor: {:+d} (BUY>={} SELL<={}) | {}",
                 snap.symbol, score, buy_threshold, sell_threshold, reason_str)

    if score >= buy_threshold:
        return "buy", confidence, f"[Teknik] {reason_str}"
    elif score <= sell_threshold:
        return "sell", confidence, f"[Teknik] {reason_str}"
    return "hold", confidence, f"[Teknik-hold] {reason_str}"


def _technical_signal_with_ml(
    snap: "MarketSnapshot",
    ohlcv_df,
    ml_weight: float = 0.30,   # artık Bayesian ağırlığı için kullanılmıyor, geriye dönük uyumluluk
    market: str = "us",
    multiframe_signals: dict | None = None,
) -> tuple[str, float, str]:
    """
    Bayesian Ensemble — teknik + ML + sentiment + multi-timeframe sinyallerini
    log-odds yöntemiyle birleştir.

    Önceki basit ağırlıklı ortalama yerine her sinyal kaynağı
    güvenilirlik ağırlığıyla bağımsız kanıt olarak birleştirilir.
    Bu, ML'nin %50 civarında dolanmasının sinyali bastırmasını önler.
    """
    # ── 1. Teknik sinyal ──────────────────────────────────────────────────────
    t_action, t_conf, t_reason = _technical_signal(snap, ohlcv_df)

    # Teknik skoru ensemble girişine çevir (eşik=2, max=14)
    # Gerçek skoru bilmiyoruz ama conf'tan tersine hesaplayabiliriz
    # Bunun yerine doğrudan action+conf kullanıyoruz
    t_raw = max(t_conf, 0.50) if t_action != "hold" else 0.50
    tech_signal = SignalInput("technical", t_action, t_raw)  # type: ignore[arg-type]

    # ── 2. ML sinyali ─────────────────────────────────────────────────────────
    ml_sig_input: SignalInput | None = None
    ml_label = ""
    try:
        from ml.predictor import ml_signal
        ml_sig, ml_conf = ml_signal(ohlcv_df, threshold=0.60, market=market)
        if ml_conf > 0.0:
            ml_sig_input = ml_to_signal(ml_sig, ml_conf)
            ml_label = f"ML={'AL' if ml_sig == 1 else 'BEKLE'}({ml_conf:.0%})"
    except Exception:
        pass

    # ── 3. Sentiment sinyali ──────────────────────────────────────────────────
    sent_signal = sentiment_to_signal(snap.sentiment_score, snap.sentiment_label)

    # ── 4. Multi-timeframe ────────────────────────────────────────────────────
    mtf_signal: SignalInput | None = None
    if multiframe_signals:
        mtf_signal = multiframe_to_signal(multiframe_signals, t_action)

    # ── 5. Ensemble ───────────────────────────────────────────────────────────
    inputs: list[SignalInput] = [tech_signal]
    if ml_sig_input:
        inputs.append(ml_sig_input)
    if sent_signal.action != "hold":
        inputs.append(sent_signal)
    if mtf_signal:
        inputs.append(mtf_signal)

    final_action, final_conf, ens_summary = bayesian_ensemble(inputs)

    # Teknik sinyal hold → ensemble hold olarak kal (teknik çok zayıfsa geçme)
    if t_action == "hold" and final_action != "hold":
        # Ensemble BUY/SELL dedi ama teknik hold → yalnızca çok yüksek conf'ta geçir
        if final_conf < 0.72:
            final_action = "hold"
            final_conf   = 0.35

    # ── Reason birleştir ──────────────────────────────────────────────────────
    parts = [t_reason]
    if ml_label:
        parts.append(ml_label)
    if mtf_signal:
        parts.append(f"MTF={mtf_signal.action}")
    full_reason = " | ".join(parts)

    logger.debug("{} Ensemble: {} ({:.0%}) | {}", snap.symbol, final_action, final_conf, ens_summary)

    return final_action, final_conf, full_reason


# ── Ana Agent ────────────────────────────────────────────────────────────────

class LLMTradingAgent:
    """
    Fine-tuned Qwen modeli ile otonom BIST trader.

    Kullanım:
        agent = LLMTradingAgent(
            lora_path="lora_weights/",
            symbols=["THYAO.IS", "GARAN.IS"],
            initial_capital=100_000,
            target_pct=20.0,
        )
        agent.run()
    """

    # Risk mod profilleri
    RISK_MODES = {
        "conservative": dict(min_confidence=0.70, max_position_pct=0.10, stop_loss_pct=0.04, take_profit_pct=0.05),
        "normal":       dict(min_confidence=0.55, max_position_pct=0.15, stop_loss_pct=0.03, take_profit_pct=0.06),
        "aggressive":   dict(min_confidence=0.25, max_position_pct=0.20, stop_loss_pct=0.02, take_profit_pct=0.08),
        "scalping":     dict(min_confidence=0.50, max_position_pct=0.05, stop_loss_pct=0.01, take_profit_pct=0.02),
    }

    def __init__(
        self,
        lora_path: str,
        symbols: list[str],
        initial_capital: float = 100_000.0,
        target_pct: float = 20.0,
        mode: str = "normal",
        scan_interval: int | None = None,
        crypto_symbols: list[str] | None = None,
        futures_symbols: list[str] | None = None,
        viop_symbols: list[str] | None = None,
        us_symbols: list[str] | None = None,
        leverage: int = 20,
        universe_mode: bool = False,
        universe_top_n: int = 20,
    ):
        profile = self.RISK_MODES.get(mode, self.RISK_MODES["normal"])

        self.symbols = symbols                            # BIST
        self.crypto_symbols = crypto_symbols or []       # Kripto spot
        # Universe modunda futures verilmemişse varsayılan PERP'leri ekle
        _default_futures = ["BTC-PERP", "ETH-PERP", "SOL-PERP"]
        self.futures_symbols = futures_symbols or (_default_futures if universe_mode else [])
        self.viop_symbols = viop_symbols or []           # VİOP
        self.us_symbols = us_symbols or []               # NYSE/NASDAQ
        self.leverage = leverage

        # Dinamik evren modu: tüm borsaları tarar
        self._universe_mode = universe_mode
        self._universe_top_n = universe_top_n
        self._universe_cache: dict[str, list[str]] = {}
        self._universe_last_update: float = 0.0
        self._universe_ttl: float = 120.0  # 2 dakikada bir screener yenile

        self.all_symbols = (
            self.symbols
            + self.crypto_symbols
            + self.futures_symbols
            + self.viop_symbols
            + self.us_symbols
        )

        self.initial_capital = initial_capital
        self.target_value = initial_capital * (1 + target_pct / 100)
        self.max_position_pct = profile["max_position_pct"]
        self.stop_loss_pct = profile["stop_loss_pct"]
        self.take_profit_pct = profile["take_profit_pct"]
        self.daily_loss_limit = initial_capital * 0.05
        self.min_confidence = profile["min_confidence"]
        self.mode = mode

        self.portfolio = Portfolio(cash=initial_capital)
        self.futures_positions: dict[str, FuturesPosition] = {}

        self._news_cache: list[dict] = []
        self._news_last_fetch: float = 0.0
        self._news_ttl: int = 600  # 10 dakika
        self._running = False
        self._market_regime: dict = {}      # market → MarketRegime
        self._regime_last_fetch: float = 0.0
        self._regime_ttl: int = 900         # 15 dakikada bir yenile
        self._news_watcher: "NewsWatcher | None" = None  # type: ignore
        self._context_clf = ContextClassifier()
        # Korelasyon cache: {symbol: {"symbols": [...], "corr_matrix": df, "ts": float}}
        self._corr_cache: dict[str, object] = {}
        # Tarama aralığı: mod varsayılanları saniye cinsinden
        _default_intervals = {"conservative": 300, "normal": 180, "aggressive": 60, "scalping": 30}
        self._scan_interval: int = scan_interval or _default_intervals.get(mode, 180)
        self._start_time = datetime.now(tz=timezone.utc)

        self.model = QwenInference(lora_path)

        # ── Sistemler ────────────────────────────────────────────────────────
        self.db = TradeDB("logs/trades.db")
        self.telegram = TelegramBot()
        self.calendar = EconomicCalendar()

        # Kripto spot — WebSocket
        self._price_stream = build_price_stream(self.crypto_symbols)

        # Kripto futures — Futures WebSocket
        self._futures_stream: BinanceFuturesStream | None = None
        if self.futures_symbols:
            self._futures_stream = BinanceFuturesStream(self.futures_symbols)

        # Trailing stop oranı
        self._trailing_pct = self.stop_loss_pct

        logger.info(
            "LLM Agent başlatılıyor | Mod: {} | Sermaye: {:,.0f} TL | Hedef: {:,.0f} TL",
            mode.upper(), initial_capital, self.target_value,
        )
        logger.info(
            "Ayarlar | Min güven: {:.0%} | Max pozisyon: {:.0%} | SL: {:.0%} | TP: {:.0%} | Kaldıraç: {}x",
            self.min_confidence, self.max_position_pct,
            self.stop_loss_pct, self.take_profit_pct, leverage,
        )
        if self.viop_symbols:
            logger.info("VİOP sözleşmeleri: {}", self.viop_symbols)
        if self.us_symbols:
            logger.info("ABD hisseleri: {}", self.us_symbols)

    # ── Fiyat alma ────────────────────────────────────────────────────────────

    def _get_price(self, symbol: str) -> float:
        """Sembol türüne göre en güncel fiyatı döndürür."""
        mt = _market_type(symbol)

        if mt == "futures" and self._futures_stream:
            p = self._futures_stream.get_mark_price(symbol)
            if p and p > 0:
                return p

        if mt == "crypto" and self._price_stream:
            p = self._price_stream.get_price(symbol)
            if p and p > 0:
                return p

        if mt == "viop":
            p = get_viop_price(symbol)
            if p > 0:
                return p

        # Futures/VİOP sembollerini yfinance'in tanıyacağı dayanak varlığa çevir
        yf_sym = _yf_symbol(symbol)
        try:
            return float(yf.Ticker(yf_sym).fast_info.last_price or 0)
        except Exception:
            return 0.0

    # ── Vadeli işlem emir yönetimi ────────────────────────────────────────────

    def _buy_futures(
        self, symbol: str, price: float, reason: str, confidence: float, side: str = "long"
    ) -> bool:
        """Futures/VİOP pozisyon aç (long veya short)."""
        if symbol in self.futures_positions:
            logger.debug("{} futures pozisyonu zaten var.", symbol)
            return False

        mt = _market_type(symbol)
        max_margin_tl = self.portfolio.cash * self.max_position_pct

        if mt == "futures":
            # Binance USDT-M: margin USD cinsinden → TL'ye çevir
            rate = get_usdtry_rate()
            margin_usd = tl_to_usd(max_margin_tl)
            notional_usd = margin_usd * self.leverage
            contracts = notional_usd / price          # coin miktarı
            actual_margin_tl = usd_to_tl(margin_usd)
            liq = _calc_liq(price, side, self.leverage)
            multiplier = 1.0
        else:
            # VİOP: TL cinsinden margin
            contract_spec = VIOP_CONTRACTS.get(symbol)
            if not contract_spec:
                return False
            mp = contract_spec.margin_pct / 100
            multiplier = contract_spec.multiplier
            # contracts = margin_tl / (price × multiplier × margin_pct)
            contracts = max_margin_tl / (price * multiplier * mp)
            actual_margin_tl = contracts * price * multiplier * mp
            liq = _calc_liq(price, side, self.leverage)
            multiplier = contract_spec.multiplier

        if actual_margin_tl > self.portfolio.cash:
            logger.debug("{} için yeterli teminat yok.", symbol)
            return False

        if side == "long":
            sl = price * (1 - self.stop_loss_pct)
            tp = price * (1 + self.take_profit_pct)
        else:
            sl = price * (1 + self.stop_loss_pct)
            tp = price * (1 - self.take_profit_pct)

        self.portfolio.cash -= actual_margin_tl
        self.futures_positions[symbol] = FuturesPosition(
            symbol=symbol, side=side, contracts=contracts,
            entry_price=price,
            entry_time=datetime.now(tz=timezone.utc).isoformat(),
            leverage=self.leverage,
            margin_used=actual_margin_tl,
            liquidation_price=liq,
            stop_loss=sl, take_profit=tp,
            highest_price=price, trailing_pct=self._trailing_pct,
        )
        cur = _currency(symbol)
        logger.info(
            "{} {} ACIŞ | {:.4f} sözleşme @ {:.4f} {} | {}x kaldıraç | "
            "Teminat: {:,.0f} TL | Likidasyon: {:.4f} | Güven: {:.0%}",
            symbol, side.upper(), contracts, price, cur,
            self.leverage, actual_margin_tl, liq, confidence,
        )
        self.db.log_trade(symbol, f"futures_{side}", contracts, price, reason=reason, mode=self.mode)
        self.telegram.buy_alert(symbol, contracts, price, confidence, reason, cur)
        return True

    def _sell_futures(self, symbol: str, price: float, reason: str) -> bool:
        """Futures/VİOP pozisyonu kapat."""
        fpos = self.futures_positions.get(symbol)
        if not fpos:
            return False

        mt = _market_type(symbol)
        contract_spec = VIOP_CONTRACTS.get(symbol)
        multiplier = contract_spec.multiplier if contract_spec else 1.0

        pnl_tl = fpos.unrealized_pnl(price, multiplier)
        returned_tl = fpos.margin_used + pnl_tl

        self.portfolio.cash += returned_tl
        self.portfolio.daily_pnl += pnl_tl
        del self.futures_positions[symbol]

        self.portfolio.trades.append(Trade(
            symbol=symbol, action="sell", shares=fpos.contracts, price=price,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            reason=reason, pnl=pnl_tl,
        ))
        cur = _currency(symbol)
        logger.info(
            "{} {} KAPAT | {:.4f} sözleşme @ {:.4f} {} | P&L: {:+,.0f} TL ({:+.1f}%)",
            symbol, fpos.side.upper(), fpos.contracts, price, cur,
            pnl_tl, fpos.pnl_pct(price, multiplier),
        )
        self.db.log_trade(symbol, "futures_close", fpos.contracts, price,
                          pnl=pnl_tl, reason=reason, mode=self.mode)
        self.telegram.sell_alert(symbol, fpos.contracts, price, pnl_tl, reason, cur)
        return True

    def _check_futures_stops(self) -> None:
        """Futures/VİOP stop-loss ve take-profit kontrolü."""
        for symbol, fpos in list(self.futures_positions.items()):
            try:
                price = self._get_price(symbol)
                if price <= 0:
                    continue

                # Tasfiye kontrolü
                if fpos.side == "long" and price <= fpos.liquidation_price:
                    logger.warning("{} TAsfiye! Fiyat: {:.4f} | Likidasyon: {:.4f}",
                                   symbol, price, fpos.liquidation_price)
                    self._sell_futures(symbol, price, "Tasfiye (likidasyon)")
                    continue
                elif fpos.side == "short" and price >= fpos.liquidation_price:
                    logger.warning("{} TAsfiye! Fiyat: {:.4f} | Likidasyon: {:.4f}",
                                   symbol, price, fpos.liquidation_price)
                    self._sell_futures(symbol, price, "Tasfiye (likidasyon)")
                    continue

                fpos.update_trailing_stop(price)

                if fpos.side == "long":
                    if price <= fpos.stop_loss:
                        self._sell_futures(symbol, price, f"Stop-loss ({price:.4f} <= {fpos.stop_loss:.4f})")
                    elif price >= fpos.take_profit:
                        self._sell_futures(symbol, price, f"Take-profit ({price:.4f} >= {fpos.take_profit:.4f})")
                else:  # short
                    if price >= fpos.stop_loss:
                        self._sell_futures(symbol, price, f"Stop-loss ({price:.4f} >= {fpos.stop_loss:.4f})")
                    elif price <= fpos.take_profit:
                        self._sell_futures(symbol, price, f"Take-profit ({price:.4f} <= {fpos.take_profit:.4f})")
            except Exception as e:
                logger.debug("{} futures stop hatası: {}", symbol, e)

    # ── Kelly Criterion pozisyon büyüklüğü ───────────────────────────────────

    def _kelly_position(self, symbol: str) -> tuple[float, float | None]:
        """
        (max_spend, kelly_pct) döndürür.
        Yeterli geçmiş yoksa varsayılan max_position_pct kullanılır.
        """
        kelly = self.db.kelly_fraction(symbol, last_n=30, half_kelly=True)
        if kelly > 0:
            # Kelly oranını max_position_pct ile sınırla
            kelly_pct = min(kelly, self.max_position_pct * 2)
            max_spend = self.portfolio.cash * kelly_pct
            return max_spend, kelly_pct * 100
        return self.portfolio.cash * self.max_position_pct, None

    # ── Emir yönetimi ────────────────────────────────────────────────────────

    def _buy(self, symbol: str, price: float, reason: str, confidence: float,
             atr: float = 0.0) -> bool:
        if symbol in self.portfolio.positions:
            return False

        max_spend, kelly_pct = self._kelly_position(symbol)
        if max_spend < price:
            return False

        # ATR tabanlı stop loss — sabit %'den çok daha iyi
        if atr > 0:
            stop = price - 1.5 * atr        # 1.5 ATR mesafe
            tp   = price + 3.0 * atr        # 2:1 risk/ödül oranı
        else:
            stop = price * (1 - self.stop_loss_pct)
            tp   = price * (1 + self.take_profit_pct)

        # ATR tabanlı pozisyon büyüklüğü: risk_tl / stop_mesafe
        if atr > 0:
            risk_tl = self.portfolio.cash * 0.01  # toplam sermayenin %1'i riski
            atr_shares = risk_tl / (1.5 * atr)
            max_shares = max_spend / price
            shares = min(atr_shares, max_shares)
        else:
            shares = max_spend / price

        cost = shares * price
        if cost > self.portfolio.cash:
            shares = self.portfolio.cash * 0.99 / price
            cost = shares * price

        cur = _currency(symbol)

        self.portfolio.cash -= cost
        self.portfolio.positions[symbol] = Position(
            symbol=symbol, shares=shares, entry_price=price,
            entry_time=datetime.now(tz=timezone.utc).isoformat(),
            stop_loss=stop, take_profit=tp,
            highest_price=price, trailing_pct=self._trailing_pct,
        )
        self.portfolio.trades.append(Trade(
            symbol=symbol, action="buy", shares=shares, price=price,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            reason=f"[{confidence:.0%}] {reason}",
        ))

        kelly_str = f" | Kelly: %{kelly_pct:.1f}" if kelly_pct else ""
        logger.info(
            "ALIŞ | {} | {:.4f} adet @ {:.2f} {} | Güven: {:.0%}{} | {}",
            symbol, shares, price, cur, confidence, kelly_str, reason,
        )

        # DB + Telegram
        self.db.log_trade(symbol, "buy", shares, price, reason=reason, mode=self.mode)
        self.telegram.buy_alert(symbol, shares, price, confidence, reason, cur, kelly_pct)
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
        cur = _currency(symbol)
        logger.info(
            "SATIŞ | {} | {:.4f} adet @ {:.2f} {} | P&L: {:+,.0f} TL ({:+.1f}%)",
            symbol, pos.shares, price, cur, pnl, pos.pnl_pct(price),
        )

        # DB + Telegram
        self.db.log_trade(symbol, "sell", pos.shares, price, pnl=pnl, reason=reason, mode=self.mode)
        self.telegram.sell_alert(symbol, pos.shares, price, pnl, reason, cur)
        return True

    def _check_stops(self) -> None:
        for symbol, pos in list(self.portfolio.positions.items()):
            try:
                price = self._get_price(symbol)
                if price <= 0:
                    continue

                pos.update_trailing_stop(price)

                if price <= pos.stop_loss:
                    pnl = pos.pnl(price)
                    self.telegram.stop_alert(symbol, price, pnl, "Stop-loss")
                    self._sell(symbol, price, f"Stop-loss ({price:.4f} <= {pos.stop_loss:.4f})")
                elif price >= pos.take_profit:
                    pnl = pos.pnl(price)
                    self.telegram.stop_alert(symbol, price, pnl, "Take-profit")
                    self._sell(symbol, price, f"Take-profit ({price:.4f} >= {pos.take_profit:.4f})")
            except Exception as e:
                logger.debug("{} stop kontrol hatası: {}", symbol, e)

        self._check_futures_stops()

    def _close_all(self, reason: str) -> None:
        for symbol in list(self.portfolio.positions.keys()):
            try:
                price = self._get_price(symbol)
                if price > 0:
                    self._sell(symbol, price, reason)
            except Exception as e:
                logger.error("{} kapatma hatası: {}", symbol, e)
        for symbol in list(self.futures_positions.keys()):
            try:
                price = self._get_price(symbol)
                if price > 0:
                    self._sell_futures(symbol, price, reason)
            except Exception as e:
                logger.error("{} futures kapatma hatası: {}", symbol, e)

    # ── Toplam portföy değeri ─────────────────────────────────────────────────

    def _total_value(self) -> float:
        spot_value = sum(
            p.shares * self._get_price(s)
            for s, p in self.portfolio.positions.items()
        )
        futures_value = sum(
            fpos.margin_used + fpos.unrealized_pnl(
                self._get_price(sym),
                VIOP_CONTRACTS[sym].multiplier if sym in VIOP_CONTRACTS else 1.0,
            )
            for sym, fpos in self.futures_positions.items()
        )
        # US stock değerleri USD → TL çevir
        rate = get_usdtry_rate() if self.us_symbols else 1.0
        us_spot_tl = sum(
            p.shares * self._get_price(s) * (rate if is_us_symbol(s) else 1.0)
            for s, p in self.portfolio.positions.items()
            if is_us_symbol(s)
        )
        return self.portfolio.cash + spot_value + futures_value

    # ── Evren güncelleyici ────────────────────────────────────────────────────

    def _refresh_universe(self) -> list[str]:
        """
        Dinamik evren modunda tüm piyasaları tarar ve öncelikli sembol listesi döndürür.
        Mevcut pozisyonlar listede her zaman yer alır (stop/TP takibi için).
        """
        now = time.time()
        if now - self._universe_last_update < self._universe_ttl and self._universe_cache:
            return self._universe_cache.get("all_priority", self.all_symbols)

        logger.info("EVREN TARAMASI BAŞLADI — tüm piyasalar kontrol ediliyor...")

        include_us     = True
        include_crypto = bool(self.crypto_symbols) or self._universe_mode
        include_bist   = bool(self.symbols) or self._universe_mode

        try:
            watchlist = get_dynamic_watchlist(
                include_us=include_us,
                include_crypto=include_crypto,
                include_bist=include_bist,
                top_per_category=self._universe_top_n,
            )
            print_market_overview(watchlist)
            self._universe_cache = watchlist
            self._universe_last_update = now

            # Öncelik listesi: screener çıktıları
            priority = list(watchlist.get("all_priority", []))

            # Sabit listeler (kullanıcının belirlediği VİOP/futures/özel semboller)
            for sym in self.viop_symbols + self.futures_symbols:
                if sym not in priority:
                    priority.append(sym)

            # Açık pozisyonları her zaman ekle (stop takibi)
            for sym in list(self.portfolio.positions.keys()) + list(self.futures_positions.keys()):
                if sym not in priority:
                    priority.insert(0, sym)

            logger.info(
                "Evren güncellendi: {} sembol | ABD kazananlar: {} | Kripto kazananlar: {} | BIST kazananlar: {}",
                len(priority),
                ", ".join(watchlist.get("us_gainers", [])[:5]),
                ", ".join(watchlist.get("crypto_gainers", [])[:5]),
                ", ".join(watchlist.get("bist_gainers", [])[:5]),
            )
            return priority

        except Exception as e:
            logger.error("Evren tarama hatası: {} — mevcut listeye devam ediliyor.", e)
            return self.all_symbols

    # ── Ana döngü ────────────────────────────────────────────────────────────

    def _on_news_trigger(self, trigger) -> None:
        """NewsWatcher'dan gelen güçlü haber sinyalini logla ve Telegram'a ilet."""
        try:
            icon = "📈" if trigger.sentiment == "bullish" else "📉"
            msg = (f"{icon} Haber Sinyali: {trigger.symbol}\n"
                   f"{trigger.sentiment.upper()} (güven: {trigger.confidence:.0%})\n"
                   f"{trigger.title[:120]}")
            self.telegram.info(msg)
        except Exception:
            pass

    def _update_regimes(self) -> None:
        """Piyasa rejimlerini günceller (15 dakikada bir)."""
        now = time.time()
        if now - self._regime_last_fetch < self._regime_ttl:
            return
        self._regime_last_fetch = now
        try:
            from ml.market_regime import detect_market_regime
            for market in ("bist", "us", "crypto"):
                regime = detect_market_regime(market)
                if regime:
                    self._market_regime[market] = regime
                    logger.info("Rejim [{}]: {} | {}", market, regime.regime, regime.description)
        except Exception as e:
            logger.debug("Rejim güncellenemedi: {}", e)

    def _get_regime_threshold_multiplier(self, symbol: str) -> float:
        """Sembolün piyasasına göre sinyal eşiği çarpanını döner."""
        try:
            from ml.market_regime import REGIME_THRESHOLD_MULTIPLIER
            mt = _market_type(symbol)
            market = "bist" if mt == "bist" else ("crypto" if mt in ("crypto", "futures") else "us")
            regime = self._market_regime.get(market)
            if regime:
                return REGIME_THRESHOLD_MULTIPLIER.get(regime.regime, 1.0)
        except Exception:
            pass
        return 1.0

    def _check_correlation(self, symbol: str) -> bool:
        """
        Yeni pozisyon açmadan önce mevcut portföyle korelasyonu kontrol eder.

        Koşul: Herhangi bir açık pozisyonla 30g rolling korelasyon > 0.72 ise False döner.
        Bu, BTC + ETH + SOL'ün aynı anda açılmasını engeller (%90 korelasyon içerirler).

        Returns:
            True  → pozisyon açılabilir
            False → korelasyon çok yüksek, engelle
        """
        all_open = list(self.portfolio.positions.keys()) + list(self.futures_positions.keys())
        if not all_open:
            return True  # portföy boş, her zaman izin ver

        try:
            yf_sym   = _yf_symbol(symbol)
            # Mevcut pozisyonların yfinance sembollerine çevir
            existing = [_yf_symbol(s) for s in all_open if s != symbol]
            if not existing:
                return True

            all_syms = list(dict.fromkeys([yf_sym] + existing))  # tekrar kaldır
            if len(all_syms) < 2:
                return True

            raw = yf.download(all_syms, period="30d", interval="1d",
                              progress=False, auto_adjust=True)

            # Multi-sembol indirme 'Close' üst seviye sütun üretir
            if hasattr(raw.columns, "levels"):
                close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
            else:
                close = raw[["close"]] if "close" in raw.columns else raw

            if yf_sym not in close.columns or len(close) < 10:
                return True

            corr = close.pct_change().dropna().corr()
            if yf_sym not in corr.index:
                return True

            row = corr[yf_sym].drop(labels=[yf_sym], errors="ignore")
            high_corr = row[row.abs() > 0.72]
            if not high_corr.empty:
                corr_syms = high_corr.index.tolist()
                logger.warning(
                    "{} KORELASYON FİLTRESİ: {} ile yüksek korelasyon ({:.2f}) — pozisyon engellendi.",
                    symbol, corr_syms, float(high_corr.abs().max()),
                )
                return False
        except Exception as e:
            logger.debug("{} korelasyon kontrol hatası (geçiliyor): {}", symbol, e)

        return True

    def _scan(self) -> None:
        total = self._total_value()
        us_sess = get_us_session()

        logger.info(
            "TARAMA | Portföy: {:,.0f} TL | Nakit: {:,.0f} TL | Hedef: {:,.0f} TL | P&L: {:+,.0f} TL | ABD: {}",
            total, self.portfolio.cash, self.target_value, total - self.initial_capital,
            session_label(),
        )

        if total >= self.target_value:
            logger.success("HEDEF ULAŞILDI! {:,.0f} TL", total)
            self._close_all("Günlük hedef ulaşıldı")
            self._running = False
            return

        if -self.portfolio.daily_pnl >= self.daily_loss_limit:
            logger.warning("Günlük kayıp limiti aşıldı! Durduruluyor.")
            self._close_all("Günlük kayıp limiti")
            self._running = False
            return

        self._check_stops()

        # Ekonomik takvim kontrolü — yüksek etkili olay günlerinde yeni pozisyon açma
        skip_new_positions = False
        try:
            events = self.calendar.today_events()
            for ev in events:
                logger.warning("Ekonomik olay: {} [{}]", ev.title, ev.impact)
                if ev.impact == "high":
                    skip_new_positions = True
                    self.telegram.economic_event(ev.title, ev.impact,
                                                 ev.event_date.strftime("%d.%m.%Y"))
        except Exception:
            pass

        # Piyasa rejimi güncelle
        self._update_regimes()

        # Panik rejiminde yeni pozisyon açma
        any_panic = any(
            r.is_panic for r in self._market_regime.values()
        )
        if any_panic:
            logger.warning("PANİK REJİMİ: Yeni pozisyon açılmıyor, mevcut pozisyonlar korunuyor.")
            skip_new_positions = True

        # Acil haberlerden gelen semboller kuyruğunu işle (öncelikli)
        urgent_symbols: list[str] = []
        if self._news_watcher:
            for trigger in self._news_watcher.drain_urgent():
                if trigger.symbol not in urgent_symbols:
                    urgent_symbols.append(trigger.symbol)
                    logger.info("⚡ Acil tarama: {} ({})", trigger.symbol, trigger.sentiment)

        # Haberleri 10 dakikada bir çek, arada cache kullan
        now = time.time()
        if now - self._news_last_fetch >= self._news_ttl:
            try:
                self._news_cache = fetch_all_feeds()
                self._news_last_fetch = now
                logger.info("Haberler güncellendi ({} adet).", len(self._news_cache))
            except Exception:
                self._news_cache = []
        all_news = self._news_cache

        # Dinamik evren modu: screener ile sembol listesini güncelle
        if self._universe_mode:
            scan_symbols = self._refresh_universe()
        else:
            scan_symbols = self.all_symbols

        # Acil semboller en başa al (tekrar tarama önle)
        if urgent_symbols:
            scan_symbols = urgent_symbols + [s for s in scan_symbols if s not in urgent_symbols]

        for symbol in scan_symbols:
            if not self._running:
                break

            mt = _market_type(symbol)

            # ABD hisseleri — tüm seanslar (pre-market, regular, after-market)
            if mt == "us":
                us_session = get_us_session()
                if us_session == "closed":
                    logger.debug("{} — NYSE/NASDAQ kapalı seans, atlanıyor.", symbol)
                    continue
                # Pre/after-market'te düşük likidite — eşiği hafifçe yükselt
                _us_session_active = us_session

            # VİOP — ekonomik takvim kontrolü (skip_new_positions zaten var ama ek log)
            if mt == "viop" and skip_new_positions:
                logger.debug("{} — Yüksek etkili olay günü, VİOP pozisyon açılmıyor.", symbol)

            logger.debug("{} analiz ediliyor [{}]...", symbol, mt)
            # Futures/crypto için WebSocket mark fiyatını snapshot'a ilet
            ws_price = self._get_price(symbol) if mt in ("futures", "crypto") else None
            snap, ohlcv_df = _collect_snapshot(
                symbol, cached_news=all_news,
                price_override=ws_price, return_df=True,
            )
            if not snap or snap.price <= 0:
                continue

            # KAP insider sinyali — BIST ve VİOP hisse vadeli için
            insider_signal = "neutral"
            if mt in ("bist", "viop") and not symbol.endswith("-FUT"):
                try:
                    insider_signal = self.calendar.insider_signal(symbol)
                    if insider_signal != "neutral":
                        logger.info("{} KAP insider: {}", symbol, insider_signal.upper())
                except Exception:
                    pass

            # ── Multi-timeframe sinyali (BUY onay için) ──────────────────────
            # Sadece aktif BUY/SELL değerlendirmesi için ek zaman dilimleri çek.
            # Bu işlem yavaş olduğundan yalnızca teknik sinyal "hold" değilse çalışır.
            # İlk hızlı sinyal değerlendirmesi için önce teknik skoru al.
            _quick_action, _, _ = _technical_signal(snap, ohlcv_df)
            mtf_signals: dict | None = None
            if _quick_action != "hold":
                yf_sym_for_mtf = _yf_symbol(symbol)
                try:
                    mtf_signals = _collect_multiframe_signals(yf_sym_for_mtf)
                    if mtf_signals:
                        agree = sum(1 for a, _ in mtf_signals.values() if a == _quick_action)
                        logger.debug("{} MTF: {}/{} dilim {} yönünde | {}",
                                     symbol, agree, len(mtf_signals), _quick_action,
                                     {k: v[0] for k, v in mtf_signals.items()})
                except Exception:
                    mtf_signals = None

            # ── BIST dışı piyasalar: Ensemble teknik + ML + MTF ──────────────
            # Model BIST üzerine fine-tune edildi; US/crypto/futures için
            # Bayesian ensemble (teknik + ML + sentiment + multi-timeframe) kullan.
            if mt in ("us", "crypto", "futures"):
                _ml_market = "crypto" if mt in ("crypto", "futures") else "us"
                action, confidence, reason = _technical_signal_with_ml(
                    snap, ohlcv_df,
                    market=_ml_market,
                    multiframe_signals=mtf_signals,
                )

                # ── On-chain veri (kripto/futures için ekstra sinyal) ──────────
                if mt in ("crypto", "futures"):
                    try:
                        from data.sources.onchain import onchain_signal_cached
                        oc = onchain_signal_cached(symbol)
                        if oc["score"] != 0:
                            # On-chain sinyali ensemble'a ekle
                            oc_inp = onchain_to_signal(oc["score"])
                            if oc_inp.action != "hold":
                                # Yeniden ensemble — on-chain dahil
                                t_inp  = SignalInput("technical", action, max(confidence, 0.50))  # type: ignore[arg-type]
                                merged_action, merged_conf, _ = bayesian_ensemble(
                                    [t_inp, oc_inp]
                                )
                                if merged_action == action or merged_action == "hold":
                                    confidence = merged_conf
                                else:
                                    # On-chain karşı sinyal → conf düşür
                                    confidence = max(confidence - 0.08, 0.20)
                                reason += f" | OC:{oc['reason']}"
                    except Exception:
                        pass

                _in_portfolio = symbol in self.portfolio.positions or symbol in self.futures_positions
                if action != "hold" or _in_portfolio:
                    logger.info(
                        "{} [{}] Ensemble: {} ({:.0%}) | {}",
                        symbol, mt, action.upper(), confidence, reason,
                    )

            else:
                # ── BIST / VİOP: LLM bağlam sınıflandırıcı + strateji ────────
                # 1. LLM (Qwen) → regime sınıflandır (trend/range/panic)
                # 2. Deterministik strateji → karar ver
                # 3. Bayesian ensemble → strateji + ML + MTF birleştir
                ctx = self._context_clf.classify(snap, self.model)
                strategy = get_strategy(ctx.regime)
                s_action, s_conf, s_reason = strategy.decide(snap, ohlcv_df)

                logger.debug("{} Bağlam: {} ({}) | Strateji: {} → {} ({:.0%})",
                             symbol, ctx.regime, ctx.source, strategy.name,
                             s_action, s_conf)

                # Strateji "hold" ise teknik sinyale bak (güvenlik ağı)
                if s_action == "hold":
                    t_action, t_conf, t_reason = _technical_signal(snap, ohlcv_df)
                    if t_action != "hold":
                        s_action, s_conf, s_reason = t_action, t_conf, t_reason
                        logger.debug("{} Strateji=HOLD → teknik fallback: {} ({:.0%})",
                                     symbol, s_action, s_conf)

                # Bayesian ensemble: strateji + ML + MTF
                s_inp  = SignalInput("technical", s_action, max(s_conf, 0.50))  # type: ignore[arg-type]
                inputs: list[SignalInput] = [s_inp]

                try:
                    from ml.predictor import ml_signal
                    ml_sig, ml_conf = ml_signal(ohlcv_df, threshold=0.60, market="bist")
                    if ml_conf > 0.0:
                        inputs.append(ml_to_signal(ml_sig, ml_conf))
                except Exception:
                    pass

                if mtf_signals:
                    mtf_inp = multiframe_to_signal(mtf_signals, s_action)
                    if mtf_inp:
                        inputs.append(mtf_inp)

                action, confidence, _ = bayesian_ensemble(inputs)
                reason = s_reason + f" [Bağlam: {ctx.regime}/{ctx.source}]"

                if symbol in self.portfolio.positions or symbol in self.futures_positions:
                    logger.info("{} [{}] | Strateji: {} | {} ({:.0%}) | {}",
                                symbol, mt, strategy.name, action.upper(), confidence, reason)

            # KAP insider sinyali etkisi
            if insider_signal == "buy" and action == "buy":
                confidence = min(confidence + 0.10, 1.0)
                reason += " | KAP insider alım"
            elif insider_signal == "sell":
                action = "sell"
                reason += " | KAP insider satım"

            # US/crypto için karar zaten üstte loglandı; BIST'te buy/sell durumunu logla
            if mt not in ("us", "crypto", "futures") or action in ("buy", "sell"):
                if insider_signal != "neutral" or action in ("buy", "sell"):
                    logger.info(
                        "{} [{}] | Karar: {} | Güven: {:.0%} | {}",
                        symbol, mt, action.upper(), confidence, reason,
                    )

            try:
                self.db.log_decision(
                    symbol, action, confidence, reason,
                    price=snap.price, rsi=snap.rsi, macd=snap.macd,
                    sentiment=snap.sentiment_score,
                )
            except Exception:
                pass

            # ── Emir uygula ──────────────────────────────────────────────────
            is_futures_sym = mt in ("futures", "viop")

            # Pre/after-market'te güven eşiğini yükselt (düşük likidite / geniş spread)
            effective_min_conf = self.min_confidence
            if mt == "us":
                us_sess_now = get_us_session()
                if us_sess_now == "premarket":
                    effective_min_conf = min(self.min_confidence + 0.10, 0.90)
                    reason += " [pre-mkt]"
                elif us_sess_now == "aftermarket":
                    effective_min_conf = min(self.min_confidence + 0.15, 0.95)
                    reason += " [after-mkt]"

            in_spot     = symbol in self.portfolio.positions
            in_futures  = symbol in self.futures_positions

            if action == "buy" and confidence >= effective_min_conf and not skip_new_positions:
                # Korelasyon filtresi: portföyde yüksek korelasyonlu pozisyon varsa engelle
                if not in_spot and not in_futures:
                    if not self._check_correlation(symbol):
                        reason += " [korelasyon filtresi]"
                        action = "hold"
                if action == "buy":
                    if is_futures_sym:
                        self._buy_futures(symbol, snap.price, reason, confidence, side="long")
                    else:
                        self._buy(symbol, snap.price, reason, confidence, atr=snap.atr)

            elif action == "sell":
                if is_futures_sym:
                    if in_futures:
                        fpos = self.futures_positions[symbol]
                        if fpos.side == "long":
                            self._sell_futures(symbol, snap.price, reason)
                        # Short zaten açıksa → bekle, tekrar açma
                    elif confidence >= self.min_confidence and not skip_new_positions:
                        # Portföyde yoksa short aç (futures'a özgü)
                        if self._check_correlation(symbol):
                            self._buy_futures(symbol, snap.price, reason, confidence, side="short")
                elif in_spot:
                    # Sadece elimizde varsa sat
                    self._sell(symbol, snap.price, reason)
                # else: portföyde yok, sell/hold sinyali anlamsız → atla

            elif action == "hold":
                if not in_spot and not in_futures:
                    pass  # portföyde yok, hold sinyali anlamsız → atla

        # Portföy snapshot kaydet
        try:
            total_now = self._total_value()
            self.db.log_snapshot(
                total_now, self.portfolio.cash,
                total_now - self.initial_capital,
                len(self.portfolio.positions) + len(self.futures_positions),
            )
        except Exception:
            pass

    def run(self) -> None:
        if not self.model.load():
            logger.error("Model yüklenemedi, çıkılıyor.")
            return

        if self._price_stream:
            self._price_stream.start()
            logger.info("Kripto spot WebSocket başlatıldı.")

        if self._futures_stream:
            self._futures_stream.start()
            logger.info("Kripto Futures WebSocket başlatıldı.")

        # Haber tetikleyici başlat
        try:
            from agents.news_trigger import NewsWatcher
            watch_syms = list(self.all_symbols)[:40]  # ilk 40 sembol izle
            self._news_watcher = NewsWatcher(
                symbols=watch_syms,
                on_trigger=self._on_news_trigger,
                interval=120,
                min_confidence=0.68,
            )
            self._news_watcher.start()
        except Exception as e:
            logger.debug("NewsWatcher başlatılamadı: {}", e)

        self._running = True
        logger.info("LLM Agent çalışıyor. Durdur: Ctrl+C")
        self.telegram.info("🤖 Trading Agent başlatıldı\n"
                           f"Sermaye: {self.initial_capital:,.0f} TL\n"
                           f"Hedef: {self.target_value:,.0f} TL\n"
                           f"Mod: {self.mode.upper()}")

        try:
            while self._running:
                self._scan()
                if self._running:
                    logger.info("Sonraki tarama {} saniye sonra...", self._scan_interval)
                    time.sleep(self._scan_interval)
        except KeyboardInterrupt:
            logger.info("Kullanıcı durdurdu.")
        finally:
            if self._price_stream:
                self._price_stream.stop()
            if self._futures_stream:
                self._futures_stream.stop()
            if self._news_watcher:
                self._news_watcher.stop()
            self._print_summary()

    def _print_summary(self) -> None:
        total = self._total_value()
        pnl = total - self.initial_capital
        pnl_pct = pnl / self.initial_capital * 100
        elapsed = datetime.now(tz=timezone.utc) - self._start_time
        hours, rem = divmod(int(elapsed.total_seconds()), 3600)
        mins = rem // 60

        winning = [t for t in self.portfolio.trades if t.action == "sell" and t.pnl > 0]
        losing  = [t for t in self.portfolio.trades if t.action == "sell" and t.pnl < 0]
        win_rate = len(winning) / max(len(winning) + len(losing), 1) * 100

        futures_note = (
            f"  Futures poz : {len(self.futures_positions)} açık\n"
            if self.futures_positions else ""
        )
        piyasalar = []
        if self.symbols:       piyasalar.append(f"BIST({len(self.symbols)})")
        if self.crypto_symbols: piyasalar.append(f"Kripto({len(self.crypto_symbols)})")
        if self.futures_symbols: piyasalar.append(f"Futures({len(self.futures_symbols)})")
        if self.viop_symbols:   piyasalar.append(f"VİOP({len(self.viop_symbols)})")
        if self.us_symbols:     piyasalar.append(f"US({len(self.us_symbols)})")

        lines = [
            "=" * 60,
            f"  GÜNLÜK RAPOR — {datetime.now().strftime('%d.%m.%Y %H:%M')}",
            f"  Mod         : {self.mode.upper()} | Kaldıraç: {self.leverage}x",
            f"  Piyasalar   : {', '.join(piyasalar) or 'yok'}",
            f"  Süre        : {hours}s {mins}dk",
            "=" * 60,
            f"  Başlangıç   : {self.initial_capital:>12,.0f} TL",
            f"  Bitiş       : {total:>12,.0f} TL",
            f"  P&L         : {pnl:>+12,.0f} TL ({pnl_pct:+.2f}%)",
            f"  Hedef       : {self.target_value:>12,.0f} TL",
            f"  Hedef durumu: {'✓ ULAŞILDI' if total >= self.target_value else f'% {total/self.target_value*100:.1f} tamamlandı'}",
            "-" * 60,
            f"  Toplam işlem: {len(self.portfolio.trades)}",
            f"  Kazanan     : {len(winning)} | Kaybeden: {len(losing)} | Oran: %{win_rate:.0f}",
            f"{futures_note}={'=' * 60}",
        ]

        for line in lines:
            logger.info(line)

        if self.portfolio.trades:
            logger.info("İşlem geçmişi:")
            for t in self.portfolio.trades:
                logger.info(
                    "  {} {} {:.2f} adet @ {:.2f} | P&L: {:+,.0f} TL | {}",
                    t.action.upper(), t.symbol, t.shares, t.price, t.pnl, t.reason,
                )

        # Telegram günlük özet
        self.telegram.daily_summary(
            total=total,
            initial=self.initial_capital,
            pnl=pnl,
            pnl_pct=pnl_pct,
            target=self.target_value,
            win_trades=len(winning),
            loss_trades=len(losing),
            open_positions=len(self.portfolio.positions),
        )

        if total >= self.target_value:
            self.telegram.target_reached(total, pnl, pnl_pct)

        self.db.close()

        # Raporu dosyaya kaydet
        from pathlib import Path
        report_path = Path("logs") / f"rapor_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Rapor kaydedildi: {}", report_path)


# ── CLI ──────────────────────────────────────────────────────────────────────

DEFAULT_SYMBOLS = [
    "THYAO.IS", "GARAN.IS", "ASELS.IS", "EREGL.IS",
    "AKBNK.IS", "YKBNK.IS", "KCHOL.IS", "SISE.IS",
]


DEFAULT_CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD"]


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM BIST Trading Agent")
    parser.add_argument("--lora", required=True, help="LoRA adapter klasörü (örn: lora_weights/)")
    parser.add_argument("--capital", type=float, default=100_000, help="Başlangıç sermayesi (TL)")
    parser.add_argument("--target-pct", type=float, default=20.0, help="Hedef getiri %%")
    parser.add_argument("--symbols", nargs="*", default=None, help="BIST sembolleri")
    parser.add_argument("--interval", type=int, default=None, help="Tarama aralığı (saniye, mod varsayılanını ezer)")
    parser.add_argument("--mode", default="normal",
                        choices=["conservative", "normal", "aggressive", "scalping"],
                        help="Risk modu")
    parser.add_argument("--crypto", action="store_true", help="BTC/ETH/SOL kripto ekle")
    parser.add_argument("--crypto-symbols", nargs="*", default=None, help="Özel kripto sembolleri (örn: BTC-USD)")
    args = parser.parse_args()

    crypto = None
    if args.crypto:
        crypto = args.crypto_symbols or DEFAULT_CRYPTO

    if args.crypto and not args.symbols:
        symbols = []
    else:
        symbols = [s if s.endswith(".IS") else f"{s}.IS" for s in (args.symbols or DEFAULT_SYMBOLS)]

    agent = LLMTradingAgent(
        lora_path=args.lora,
        symbols=symbols,
        initial_capital=args.capital,
        target_pct=args.target_pct,
        mode=args.mode,
        scan_interval=args.interval,
        crypto_symbols=crypto,
    )
    agent.run()


if __name__ == "__main__":
    main()
