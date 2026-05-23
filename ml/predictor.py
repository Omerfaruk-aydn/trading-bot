"""
Canlı sinyal tahmincisi.

Tek hisse veya DataFrame alır, ML modeliyle AL olasılığı döndürür.
Bot'un teknik analiz skoruyla birleştirilebilir.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import pandas as pd
from loguru import logger

from ml.features import compute_features, FEATURE_COLS
from ml.trainer import load_model


class SignalPredictor:
    """
    Tek instance olarak kullan (singleton pattern).

    Kullanım:
        predictor = SignalPredictor(market="us")
        signal, confidence = predictor.predict(ohlcv_df)
    """

    def __init__(self, market: str = "us"):
        self._market = market
        self._model  = None
        self._meta   = None
        self._load()

    def _load(self):
        try:
            self._model, self._meta = load_model(self._market)
            trained_at = self._meta.get("trained_at", "?")[:19]
            n_train    = self._meta.get("n_train", "?")
            logger.info("ML modeli yüklendi [{market}] (eğitim: {at}, {n} örnek)",
                        market=self._market, at=trained_at, n=n_train)
        except FileNotFoundError as e:
            logger.warning("{}", e)
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def predict(
        self,
        ohlcv_df: pd.DataFrame,
        threshold: float = 0.60,
    ) -> tuple[int, float]:
        """
        DataFrame'den AL/BEKLE sinyali üretir.

        Args:
            ohlcv_df:  Son N günlük OHLCV verisi (en az 200 satır önerilir)
            threshold: AL kararı için minimum olasılık (varsayılan 0.55)

        Returns:
            (signal, confidence)
            signal: 1=AL, 0=BEKLE
            confidence: 0.0-1.0 arası olasılık
        """
        if not self.available:
            return 0, 0.0

        try:
            feats = compute_features(ohlcv_df)
            last_row = feats[FEATURE_COLS].iloc[[-1]]

            if last_row.isna().any(axis=1).iloc[0]:
                return 0, 0.0

            proba = float(self._model.predict_proba(last_row)[0][1])
            signal = 1 if proba >= threshold else 0
            return signal, round(proba, 4)

        except Exception as e:
            logger.debug("ML tahmin hatası: {}", e)
            return 0, 0.0

    def predict_batch(
        self,
        symbol_dfs: dict[str, pd.DataFrame],
        threshold: float = 0.60,
    ) -> dict[str, tuple[int, float]]:
        """
        Birden fazla sembol için toplu tahmin.

        Returns:
            {symbol: (signal, confidence)}
        """
        results = {}
        for sym, df in symbol_dfs.items():
            results[sym] = self.predict(df, threshold)
        return results


# ── Modül seviyesinde singleton'lar (piyasa başına) ───────────────────────────

_predictors: dict[str, SignalPredictor] = {}


def get_predictor(market: str = "us") -> SignalPredictor:
    """Piyasaya özel predictor instance döndürür (lazy init)."""
    if market not in _predictors:
        _predictors[market] = SignalPredictor(market=market)
    return _predictors[market]


def ml_signal(ohlcv_df: pd.DataFrame, threshold: float = 0.60, market: str = "us") -> tuple[int, float]:
    """
    Kısa yol fonksiyonu — agent'tan direkt çağrılabilir.

    Returns: (signal, confidence)  →  (1, 0.72) gibi
    """
    return get_predictor(market).predict(ohlcv_df, threshold)
