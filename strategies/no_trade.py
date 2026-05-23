"""Panik Rejimi Stratejisi — yeni pozisyon açmaz, mevcut pozisyonları korur."""
from __future__ import annotations

import pandas as pd

from strategies.base import BaseStrategy


class NoTradeStrategy(BaseStrategy):
    name = "no_trade"

    def decide(self, snap, ohlcv_df: pd.DataFrame | None) -> tuple[str, float, str]:
        return "hold", 0.20, "[Panik] Yeni pozisyon açılmıyor"
