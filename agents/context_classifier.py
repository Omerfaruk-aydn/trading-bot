"""LLM Bağlam Sınıflandırıcı — Qwen'i strateji seçici olarak kullanır.

Qwen'den istenilen şey:
  ❌ "Bu sembol için AL mı SAT mı?" (sayısal akıl yürütme → 1.5B model kötü)
  ✅ "Piyasa trend mi, range mi, panik mi?" (bağlam sınıflandırma → 1.5B model iyi)

Sonra deterministik strateji (strategies/) karar verir.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from loguru import logger

if TYPE_CHECKING:
    pass

RegimeLabel = Literal["trend_up", "trend_down", "range", "panic"]
NewsTone    = Literal["bullish", "bearish", "neutral"]


@dataclass
class MarketContext:
    regime:      RegimeLabel
    news_tone:   NewsTone
    confidence:  float          # 0.0–1.0 (Qwen'in kendi kararından veya fallback'ten)
    source:      str            # "llm" | "fallback"

    @property
    def is_panic(self) -> bool:
        return self.regime == "panic"

    @property
    def is_trending(self) -> bool:
        return self.regime in ("trend_up", "trend_down")

    @property
    def is_range(self) -> bool:
        return self.regime == "range"


_SYSTEM = (
    "Sen piyasa bağlamı sınıflandırıcısısın. "
    "Verilen teknik verilere bakarak piyasa rejimini sınıflandır. "
    "YALNIZCA istenen JSON formatında cevap ver, başka hiçbir şey yazma."
)

_PROMPT_TEMPLATE = """\
Şu verilere bak ve piyasa bağlamını sınıflandır:

Son 5 günlük fiyat değişimi: {price_change:+.1f}%
Volatilite (ATR/fiyat): {atr_pct:.1f}%
ADX (trend gücü): {adx:.0f}
RSI: {rsi:.0f}
EMA durumu: {ema_status}
Son haberler: {headlines}

Sadece bu JSON formatında cevap ver (başka hiçbir şey):
{{"regime": "trend_up" veya "trend_down" veya "range" veya "panic", "news_tone": "bullish" veya "bearish" veya "neutral", "confidence": 0.0-1.0}}"""


class ContextClassifier:
    """
    Qwen modeline küçük, sınıflandırma odaklı soru sorar.

    Kullanım:
        clf = ContextClassifier()
        ctx = clf.classify(snap, qwen_model)
        strategy = get_strategy(ctx.regime)
    """

    def classify(self, snap, qwen_model=None) -> MarketContext:
        """
        MarketSnapshot + opsiyonel Qwen modeli → MarketContext.
        Model yoksa ya da hata verirse deterministik fallback çalışır.
        """
        if qwen_model is not None:
            try:
                ctx = self._classify_with_llm(snap, qwen_model)
                if ctx is not None:
                    return ctx
            except Exception as e:
                logger.debug("ContextClassifier LLM hatası: {}", e)

        return self._fallback(snap)

    def _classify_with_llm(self, snap, qwen_model) -> MarketContext | None:
        """Qwen'e kısa JSON sorusu sor; parse başarısızsa None döner."""
        ema_status = self._ema_status(snap)
        headlines  = self._headlines(snap)
        atr_pct    = snap.atr / snap.price * 100 if snap.price > 0 else 0.0

        prompt = _PROMPT_TEMPLATE.format(
            price_change=snap.change_pct,
            atr_pct=atr_pct,
            adx=snap.adx,
            rsi=snap.rsi,
            ema_status=ema_status,
            headlines=headlines,
        )

        # Kısa yanıt yeterli — 80 token
        response = qwen_model.generate(prompt, max_new_tokens=80)
        return self._parse_response(response)

    @staticmethod
    def _parse_response(text: str) -> MarketContext | None:
        try:
            match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group())

            regime    = data.get("regime", "")
            news_tone = data.get("news_tone", "neutral")
            conf      = float(data.get("confidence", 0.6))

            if regime not in ("trend_up", "trend_down", "range", "panic"):
                return None
            if news_tone not in ("bullish", "bearish", "neutral"):
                news_tone = "neutral"

            logger.debug("LLM bağlam: {} | {} | conf={:.0%}", regime, news_tone, conf)
            return MarketContext(
                regime=regime,
                news_tone=news_tone,
                confidence=round(conf, 2),
                source="llm",
            )
        except Exception:
            return None

    @staticmethod
    def _fallback(snap) -> MarketContext:
        """ADX + ATR + EMA tabanlı deterministik sınıflandırma."""
        atr_pct = snap.atr / snap.price * 100 if snap.price > 0 else 0.0

        # Panik: ATR/fiyat %4'ün üstünde veya RSI çok aşırı
        if atr_pct > 4.0 or snap.rsi > 90 or snap.rsi < 10:
            regime: RegimeLabel = "panic"

        elif snap.adx >= 25:
            regime = "trend_up" if snap.price > snap.ema21 else "trend_down"

        else:
            regime = "range"

        # Haber tonu
        s = snap.sentiment_score
        if s > 0.20:
            news_tone: NewsTone = "bullish"
        elif s < -0.20:
            news_tone = "bearish"
        else:
            news_tone = "neutral"

        logger.debug("Fallback bağlam: {} | {}", regime, news_tone)
        return MarketContext(
            regime=regime,
            news_tone=news_tone,
            confidence=0.65,
            source="fallback",
        )

    @staticmethod
    def _ema_status(snap) -> str:
        if snap.price > snap.ema21 > snap.ema55:
            return "fiyat EMA21 ve EMA55 üstünde (yukarı hizalı)"
        if snap.price < snap.ema21 < snap.ema55:
            return "fiyat EMA21 ve EMA55 altında (aşağı hizalı)"
        return "EMA'lar karışık"

    @staticmethod
    def _headlines(snap) -> str:
        heads = snap.news_headlines[:2] if snap.news_headlines else []
        return " | ".join(h[:80] for h in heads) if heads else "haber yok"
