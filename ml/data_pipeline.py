"""
Veri boru hattı — hisse verisi indir, özellik hesapla, etiket üret.

Kullanım:
    X, y = build_dataset(symbols, period="2y", horizon=5, buy_thr=0.03)
"""

from __future__ import annotations

import warnings
import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from ml.features import compute_features, FEATURE_COLS
from ml.labeler import generate_labels, label_stats

logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def _download_one(symbol: str, period: str) -> Optional[pd.DataFrame]:
    """Tek bir sembol için OHLCV verisi indirir."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(
                symbol,
                period=period,
                progress=False,
                auto_adjust=True,
                threads=False,
            )
        if df is None or len(df) < 60:
            return None
        # MultiIndex varsa düzelt
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(df.columns):
            return None
        return df
    except Exception:
        return None


def build_single(
    symbol: str,
    period: str = "2y",
    horizon: int = 5,
    buy_thr: float = 0.03,
) -> Optional[tuple[pd.DataFrame, pd.Series]]:
    """
    Tek sembol için özellik + etiket üretir.
    Returns: (X_df, y_series) veya None
    """
    df = _download_one(symbol, period)
    if df is None:
        return None

    try:
        feats = compute_features(df)
        labels = generate_labels(df["Close"], horizon=horizon, buy_thr=buy_thr)

        combined = feats.copy()
        combined["_label"] = labels

        # NaN satırları at (başlangıç warmup + son horizon gün)
        combined = combined.dropna()
        if len(combined) < 20:
            return None

        X = combined[FEATURE_COLS]
        y = combined["_label"]

        # Sembol sütunu ekle (çoklu hisse eğitimi için)
        X = X.copy()
        X["_symbol"] = symbol
        return X, y
    except Exception as e:
        logger.debug("Özellik hatası {}: {}", symbol, e)
        return None


def build_dataset(
    symbols: list[str],
    period: str = "2y",
    horizon: int = 5,
    buy_thr: float = 0.03,
    max_symbols: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Birden fazla sembol için eğitim dataset'i oluşturur.

    Args:
        symbols:     Hisse senedi sembollerinin listesi
        period:      yfinance dönemi (örn. "2y", "5y", "max")
        horizon:     Kaç gün ilerisi hedef alınsın
        buy_thr:     AL eşiği (varsayılan %3)
        max_symbols: Test için sembol sayısını sınırla

    Returns:
        X: özellik DataFrame (sembol sütunu olmadan)
        y: etiket Serisi (0/1)
    """
    if max_symbols:
        symbols = symbols[:max_symbols]

    all_X, all_y = [], []
    ok, fail = 0, 0

    logger.info("Veri indiriliyor: {} sembol, {} dönem...", len(symbols), period)

    for i, sym in enumerate(symbols, 1):
        result = build_single(sym, period, horizon, buy_thr)
        if result is None:
            fail += 1
        else:
            X, y = result
            all_X.append(X)
            all_y.append(y)
            ok += 1

        if i % 50 == 0:
            logger.info("  {}/{} sembol işlendi (başarılı={}, hata={})", i, len(symbols), ok, fail)

    if not all_X:
        raise RuntimeError("Hiçbir sembolden veri alınamadı.")

    X_all = pd.concat(all_X, axis=0)
    y_all = pd.concat(all_y, axis=0)

    # Sembol sütununu çıkar (eğitimde kullanılmaz)
    X_all = X_all.drop(columns=["_symbol"])

    stats = label_stats(y_all)
    logger.info(
        "Dataset hazır: {} örnek | {} sembol | AL oranı: {:.1f}%",
        len(X_all), ok, stats["buy_rate"] * 100,
    )

    return X_all, y_all
