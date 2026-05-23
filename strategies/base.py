"""Strateji temel sınıfı."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class BaseStrategy(ABC):
    name: str = "base"

    def __init__(self, regime: str = "") -> None:
        self.regime = regime

    @abstractmethod
    def decide(
        self,
        snap,           # MarketSnapshot — tip döngüsü önlemek için Any
        ohlcv_df: "pd.DataFrame | None",
    ) -> tuple[str, float, str]:
        """
        Returns:
            (action, confidence, reason)
            action: "buy" | "sell" | "hold"
        """

    def _clamp_conf(self, conf: float) -> float:
        return round(max(0.20, min(0.95, conf)), 3)
