"""Bayesian Ensemble — çoklu sinyal kaynaklarını log-odds yöntemiyle birleştirir.

Neden log-odds?
  Basit ağırlıklı ortalama (0.30 * 50% + 0.70 * 40% = 43%) güven sıkıştırır.
  Log-odds birleştirmede bağımsız kanıtlar doğru birikir:
    log-odds(prior) + Σ weight_i * log-odds(p_i) → posterior
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

SignalSource = Literal["technical", "ml", "sentiment", "regime", "onchain", "multiframe"]

# Tarihsel güvenilirlik priorleri — walk-forward backtest ile güncellenebilir.
# 0.50 = rastgele (hiç katkı yok), 1.0 = mükemmel (tam katkı).
# Finansal sinyaller için gerçekçi aralık: 0.52–0.68.
_SOURCE_RELIABILITY: dict[str, float] = {
    "technical":  0.62,   # teknik analiz: birincil kaynak
    "ml":         0.55,   # XGBoost: ikincil onay
    "sentiment":  0.52,   # haber sentiment: zayıf sinyaller
    "regime":     0.64,   # piyasa rejimi: bağlam filtresi
    "onchain":    0.58,   # on-chain: kripto için orta güç
    "multiframe": 0.66,   # çoklu zaman dilimi: güçlü onay
}

# Ağırlık çarpanı: (reliability - 0.50) * _WEIGHT_SCALE = etkin ağırlık.
# 2.0 çok küçük (signaller neredeyse prior'u değiştirmiyor),
# 6.0 çok büyük (korelasyonlu sinyaller aşırı yığılıyor).
# 4.0 dengeli: 3 uyumlu sinyal %55-65 güven üretir.
_WEIGHT_SCALE = 4.0

# Bu aralıkta ML model belirsiz → sinyal olarak kullanma
_ML_UNCERTAIN_LOW  = 0.38
_ML_UNCERTAIN_HIGH = 0.62

# Posterior BUY/SELL eşikleri
# 0.57: güçlü bir teknik sinyal + ML onayı bu eşiği aşar
# 0.43: ters yön için simetrik
_BUY_THRESHOLD  = 0.57
_SELL_THRESHOLD = 0.43


@dataclass
class SignalInput:
    source: SignalSource
    action: Literal["buy", "sell", "hold"]
    raw_confidence: float           # 0.0–1.0 (buy yönü olasılığı)
    weight_override: float | None = field(default=None)  # None → reliability tablosu


def _log_odds(p: float) -> float:
    p = max(0.01, min(0.99, p))
    return math.log(p / (1.0 - p))


def _from_log_odds(lo: float) -> float:
    return 1.0 / (1.0 + math.exp(-lo))


def bayesian_ensemble(
    signals: list[SignalInput],
    prior: float = 0.50,
) -> tuple[str, float, str]:
    """
    Birden fazla sinyal kaynağını Bayesian log-odds ile birleştirir.

    Args:
        signals: SignalInput listesi
        prior:   Başlangıç tarafsız olasılık (varsayılan 0.50)

    Returns:
        (action, confidence, debug_summary)
        action:     "buy" | "sell" | "hold"
        confidence: 0.0–1.0
    """
    if not signals:
        return "hold", 0.35, "sinyal yok"

    combined_lo = _log_odds(prior)
    parts: list[str] = []

    for sig in signals:
        if sig.action == "hold":
            continue

        # ML belirsizlik filtresi
        if sig.source == "ml":
            if _ML_UNCERTAIN_LOW <= sig.raw_confidence <= _ML_UNCERTAIN_HIGH:
                continue

        reliability = sig.weight_override if sig.weight_override is not None \
            else _SOURCE_RELIABILITY.get(sig.source, 0.50)

        # Güvenilirlik → katkı ağırlığı.
        # 0.50 reliability → weight=0 (prior'u değiştirmiyor)
        # 0.62 reliability → weight=0.48 (dengeli katkı)
        weight = max(0.0, (reliability - 0.50) * _WEIGHT_SCALE)

        # BUY sinyalinde "al" olasılığı, SELL sinyalinde tersle
        p = max(sig.raw_confidence, 0.50)  # en az 0.50 (sinyalin tarafından emin)
        if sig.action == "sell":
            p = 1.0 - p

        contribution = _log_odds(p) * weight
        combined_lo += contribution
        parts.append(f"{sig.source}={sig.action}({sig.raw_confidence:.0%})")

    final_prob = _from_log_odds(combined_lo)
    summary = " | ".join(parts) if parts else "yalnızca prior"

    if final_prob >= _BUY_THRESHOLD:
        action = "buy"
        confidence = round(final_prob, 3)
    elif final_prob <= _SELL_THRESHOLD:
        action = "sell"
        confidence = round(1.0 - final_prob, 3)
    else:
        action = "hold"
        # Hold güveni: merkeze uzaklık → 0.30–0.50 arası
        confidence = round(0.30 + abs(final_prob - 0.50) * 0.40, 3)

    return action, confidence, summary


# ── Dönüştürücüler ────────────────────────────────────────────────────────────

def technical_to_signal(
    action: str,
    score: int,
    threshold: int,
    max_score: int = 14,
) -> SignalInput:
    """
    Teknik skor → SignalInput.
    Eşiği aştıktan sonraki güç 0.55–0.92 aralığına map edilir.
    """
    if action == "hold":
        return SignalInput("technical", "hold", 0.50)

    excess     = max(abs(score) - abs(threshold), 0)
    extra_range = max(max_score - abs(threshold), 1)
    raw_conf   = round(0.55 + (excess / extra_range) * 0.37, 3)
    raw_conf   = min(raw_conf, 0.92)
    return SignalInput("technical", action, raw_conf)  # type: ignore[arg-type]


def ml_to_signal(ml_signal: int, ml_conf: float) -> SignalInput:
    """XGBoost ML çıktısı → SignalInput."""
    action = "buy" if ml_signal == 1 else "sell"
    return SignalInput("ml", action, round(ml_conf, 4))  # type: ignore[arg-type]


def sentiment_to_signal(sentiment_score: float, sentiment_label: str) -> SignalInput:
    """Haber sentiment skoru → SignalInput."""
    if sentiment_label in ("positive", "bullish") and sentiment_score > 0.15:
        conf = round(0.55 + min(sentiment_score * 0.30, 0.25), 3)
        return SignalInput("sentiment", "buy", conf)
    if sentiment_label in ("negative", "bearish") and sentiment_score < -0.15:
        conf = round(0.55 + min(abs(sentiment_score) * 0.30, 0.25), 3)
        return SignalInput("sentiment", "sell", conf)
    return SignalInput("sentiment", "hold", 0.50)


def multiframe_to_signal(
    frame_signals: dict[str, tuple[str, float]],
    primary_action: str,
) -> SignalInput | None:
    """
    Zaman dilimi sinyal sözlüğü → SignalInput.
    En az 2 dilim primary_action'ı desteklemiyorsa None döner.
    """
    if not frame_signals:
        return None
    agree = sum(1 for act, _ in frame_signals.values() if act == primary_action)
    total = max(len(frame_signals), 1)
    if agree < 2:
        return None
    agreement_ratio = agree / total
    conf = round(0.55 + agreement_ratio * 0.25, 3)
    return SignalInput("multiframe", primary_action, conf)  # type: ignore[arg-type]


def onchain_to_signal(onchain_score: int) -> SignalInput:
    """On-chain puan (-3 ile +3 arası) → SignalInput."""
    if onchain_score >= 1:
        conf = round(0.55 + min(onchain_score * 0.08, 0.25), 3)
        return SignalInput("onchain", "buy", conf)
    if onchain_score <= -1:
        conf = round(0.55 + min(abs(onchain_score) * 0.08, 0.25), 3)
        return SignalInput("onchain", "sell", conf)
    return SignalInput("onchain", "hold", 0.50)
