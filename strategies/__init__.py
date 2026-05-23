"""Deterministik strateji modülleri.

Mimari:
    LLM → regime sınıflandır → strateji seç → kural tabanlı karar

Her strateji bir `decide(snap, ohlcv_df)` metodu döner:
    (action, confidence, reason)
"""
from __future__ import annotations

from strategies.base import BaseStrategy
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.no_trade import NoTradeStrategy

_REGISTRY: dict[str, type[BaseStrategy]] = {
    "trend_up":   TrendFollowingStrategy,
    "trend_down": TrendFollowingStrategy,
    "range":      MeanReversionStrategy,
    "panic":      NoTradeStrategy,
}


def get_strategy(regime: str) -> BaseStrategy:
    """Rejim adından uygun strateji instance döner."""
    cls = _REGISTRY.get(regime, MeanReversionStrategy)
    return cls(regime=regime)


__all__ = [
    "BaseStrategy",
    "TrendFollowingStrategy",
    "MeanReversionStrategy",
    "NoTradeStrategy",
    "get_strategy",
]
