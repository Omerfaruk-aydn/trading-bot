"""
Etiket üretici — ileriye dönük getiriye göre AL/BEKLE sinyali oluşturur.

Strateji:
  - Etiket = 1 (AL): Sonraki `horizon` günde kapanış fiyatı >= %`buy_thr` yükselirse
  - Etiket = 0 (BEKLE/SAT): Diğer tüm durumlar

Not: Etiketler her zaman ileriye bakar → eğitimde son `horizon` gün kullanılamaz.
"""

from __future__ import annotations
import pandas as pd


def generate_labels(
    close: pd.Series,
    horizon: int = 5,
    buy_thr: float = 0.03,
) -> pd.Series:
    """
    Her gün için `horizon` gün sonraki kapanışa göre etiket üretir.

    Args:
        close:    Kapanış fiyatı serisi
        horizon:  Kaç gün ilerisi bakılsın (varsayılan 5 iş günü = 1 hafta)
        buy_thr:  AL eşiği (varsayılan %3 = 0.03)

    Returns:
        0/1 etiket serisi (son horizon satır NaN)
    """
    fwd_return = close.shift(-horizon) / close - 1
    labels = (fwd_return >= buy_thr).astype(float)
    labels[fwd_return.isna()] = float("nan")
    return labels


def label_stats(labels: pd.Series) -> dict:
    """Etiket dağılımı istatistiklerini döndürür."""
    valid = labels.dropna()
    n_buy  = int((valid == 1).sum())
    n_hold = int((valid == 0).sum())
    total  = len(valid)
    return {
        "total": total,
        "buy":   n_buy,
        "hold":  n_hold,
        "buy_rate": round(n_buy / total, 4) if total else 0,
    }
