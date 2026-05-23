"""Ortalamaya Dönüş Stratejisi — range (yatay) piyasalar için.

Mantık:
  Bollinger bantları fiyat ekstremlerini gösterir.
  RSI + Stochastic aşırı bölge onayı.
  MACD çok fazla ağırlık taşımaz (trend yok).
  VWAP referans seviyesi (varsa).
"""
from __future__ import annotations

import pandas as pd

from strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    name = "mean_reversion"

    def decide(self, snap, ohlcv_df: pd.DataFrame | None) -> tuple[str, float, str]:
        score   = 0
        reasons: list[str] = []

        # 1. Bollinger bantları (ağırlık: 3)
        bb = snap.bb_pct
        if bb <= 0.05:
            score += 3
            reasons.append(f"BB alt aşımı (%B={bb:.0%})")
        elif bb <= 0.15:
            score += 2
            reasons.append(f"BB alt yakın (%B={bb:.0%})")
        elif bb >= 0.95:
            score -= 3
            reasons.append(f"BB üst aşımı (%B={bb:.0%})")
        elif bb >= 0.85:
            score -= 2
            reasons.append(f"BB üst yakın (%B={bb:.0%})")

        # 2. RSI (ağırlık: 2)
        rsi = snap.rsi
        if rsi < 28:
            score += 2
            reasons.append(f"RSI aşırı satım ({rsi:.0f})")
        elif rsi < 38:
            score += 1
            reasons.append(f"RSI düşük ({rsi:.0f})")
        elif rsi > 72:
            score -= 2
            reasons.append(f"RSI aşırı alım ({rsi:.0f})")
        elif rsi > 62:
            score -= 1
            reasons.append(f"RSI yüksek ({rsi:.0f})")

        # 3. Stochastic dönüş sinyali (ağırlık: 1)
        k, d = snap.stoch_k, snap.stoch_d
        if k < 20 and k > d:
            score += 1
            reasons.append(f"Stoch oversold dönüş (K={k:.0f})")
        elif k > 80 and k < d:
            score -= 1
            reasons.append(f"Stoch overbought dönüş (K={k:.0f})")

        # 4. CMF hacim akışı (ağırlık: 1)
        if snap.cmf > 0.15:
            score += 1
            reasons.append(f"CMF pozitif ({snap.cmf:+.2f})")
        elif snap.cmf < -0.15:
            score -= 1
            reasons.append(f"CMF negatif ({snap.cmf:+.2f})")

        # 5. VWAP pozisyonu (ohlcv_df'den hesapla, ağırlık: 1)
        if ohlcv_df is not None and "vwap" in ohlcv_df.columns:
            try:
                vwap_val = float(ohlcv_df["vwap"].iloc[-1])
                if vwap_val > 0:
                    if snap.price < vwap_val * 0.98:
                        score += 1
                        reasons.append(f"VWAP altı ({snap.price/vwap_val-1:+.1%})")
                    elif snap.price > vwap_val * 1.02:
                        score -= 1
                        reasons.append(f"VWAP üstü ({snap.price/vwap_val-1:+.1%})")
            except Exception:
                pass

        # 6. ADX filtresi: yatay piyasada trend başladıysa stratejiden çık
        if snap.adx >= 30:
            score = int(score * 0.6)  # trend başladı, range stratejisi güvenilmez
            reasons.append(f"ADX yüksek-uyarı ({snap.adx:.0f})")

        # Yatay piyasada eşikler biraz daha yüksek (gürültü filtresi)
        buy_threshold  = 4
        sell_threshold = -4

        reason_str = " | ".join(reasons) or "Nötr"

        if score >= buy_threshold:
            conf = self._clamp_conf(0.50 + (score - buy_threshold) * 0.05)
            return "buy", conf, f"[MeanRev] {reason_str}"
        if score <= sell_threshold:
            conf = self._clamp_conf(0.50 + (abs(score) - abs(sell_threshold)) * 0.05)
            return "sell", conf, f"[MeanRev] {reason_str}"
        return "hold", 0.32, f"[MeanRev-hold] {reason_str}"
