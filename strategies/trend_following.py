"""Trend Takip Stratejisi — trend_up / trend_down rejimleri için.

Mantık:
  EMA hizalaması en önemli sinyal.
  MACD ivmesi ikincil onay.
  ADX gücü tüm puanları amplifiye eder.
  RSI: sadece aşırı bölgelerde filtre (200'e yakın aşırı alım).
"""
from __future__ import annotations

import pandas as pd

from strategies.base import BaseStrategy


class TrendFollowingStrategy(BaseStrategy):
    name = "trend_following"

    def decide(self, snap, ohlcv_df: pd.DataFrame | None) -> tuple[str, float, str]:
        score   = 0
        reasons: list[str] = []

        # 1. EMA hizalaması (ağırlık: 3)
        if snap.price > snap.ema21 > snap.ema55:
            score += 3
            reasons.append("EMA hizası ^")
        elif snap.price < snap.ema21 < snap.ema55:
            score -= 3
            reasons.append("EMA hizası v")
        elif snap.price > snap.ema21:
            score += 1
            reasons.append("Fiyat > EMA21")
        elif snap.price < snap.ema21:
            score -= 1
            reasons.append("Fiyat < EMA21")

        # 2. MACD ivmesi (ağırlık: 2)
        h, hp = snap.macd_hist, snap.macd_hist_prev
        if h > 0 and h > hp:
            score += 2
            reasons.append("MACD ivmesi ^")
        elif h > 0:
            score += 1
            reasons.append(f"MACD+ ({h:+.4f})")
        elif h < 0 and h < hp:
            score -= 2
            reasons.append("MACD ivmesi v")
        elif h < 0:
            score -= 1
            reasons.append(f"MACD- ({h:+.4f})")

        # 3. Haftalık trend onayı (ağırlık: 2)
        if snap.weekly_trend == 1:
            score += 2
            reasons.append("Haftalık trend ^")
        elif snap.weekly_trend == -1:
            score -= 2
            reasons.append("Haftalık trend v")

        # 4. Hacim onayı (ağırlık: 1)
        if snap.volume_ratio >= 1.5:
            score += 1
            reasons.append(f"Hacim {snap.volume_ratio:.1f}x")
        elif snap.volume_ratio < 0.5:
            score -= 1
            reasons.append("Düşük hacim")

        # 5. RSI filtresi — sadece aşırı bölgelerde
        if snap.rsi > 82:
            score -= 2
            reasons.append(f"RSI aşırı alım ({snap.rsi:.0f})")
        elif snap.rsi < 22:
            score += 2
            reasons.append(f"RSI aşırı satım ({snap.rsi:.0f})")

        # 6. ADX amplifikatörü: güçlü trendde puanları artır
        if snap.adx >= 35:
            score = int(score * 1.4)
            reasons.append(f"ADX güçlü ({snap.adx:.0f})")
        elif snap.adx >= 25:
            score = int(score * 1.15)

        # Trend modunda eşikler düşürülür (sinyal daha kolay tetikler)
        buy_threshold  = 3
        sell_threshold = -3

        reason_str = " | ".join(reasons) or "Nötr"

        if score >= buy_threshold:
            conf = self._clamp_conf(0.52 + (score - buy_threshold) * 0.06)
            return "buy", conf, f"[TrendFollow] {reason_str}"
        if score <= sell_threshold:
            conf = self._clamp_conf(0.52 + (abs(score) - abs(sell_threshold)) * 0.06)
            return "sell", conf, f"[TrendFollow] {reason_str}"
        return "hold", 0.35, f"[TrendFollow-hold] {reason_str}"
