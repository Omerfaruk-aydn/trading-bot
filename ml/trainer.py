"""
XGBoost sinyal modeli eğitici.

Özellikler:
- Zaman serisi doğru bölme (data leakage yok)
- Erken durdurma (overfitting önleme)
- Sınıf dengesizliği düzeltme
- Tam metrik raporu (AUC, precision, recall, F1)
- Özellik önemi loglama
- Model kaydetme / yükleme
"""

from __future__ import annotations

import os
import json
import joblib
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb

from ml.features import FEATURE_COLS

MODEL_DIR = Path(__file__).parent / "models"


def model_paths(market: str = "us") -> tuple[Path, Path]:
    name = f"signal_model_{market}"
    return MODEL_DIR / f"{name}.pkl", MODEL_DIR / f"{name}_meta.json"


# Geriye dönük uyumluluk
MODEL_PATH = MODEL_DIR / "signal_model_us.pkl"
META_PATH  = MODEL_DIR / "signal_model_us_meta.json"

# XGBoost varsayılan hiperparametreler
DEFAULT_PARAMS = {
    "n_estimators":      1000,
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  5,
    "gamma":             0.1,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "objective":         "binary:logistic",
    "eval_metric":       "auc",
    "random_state":      42,
    "n_jobs":            -1,
    "tree_method":       "hist",
}


def time_split(
    X: pd.DataFrame,
    y: pd.Series,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
):
    """
    Zamansal bölme — gelecek veri eğitime sızmaz.
    Sıra: TRAIN | VAL | TEST
    """
    n = len(X)
    test_start = int(n * (1 - test_frac))
    val_start  = int(n * (1 - test_frac - val_frac))

    X_train = X.iloc[:val_start]
    y_train = y.iloc[:val_start]
    X_val   = X.iloc[val_start:test_start]
    y_val   = y.iloc[val_start:test_start]
    X_test  = X.iloc[test_start:]
    y_test  = y.iloc[test_start:]

    logger.info(
        "Veri bölme: train={} | val={} | test={}",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _buy_rate(y: pd.Series) -> float:
    return float(y.mean())


def train(
    X: pd.DataFrame,
    y: pd.Series,
    params: Optional[dict] = None,
    early_stopping_rounds: int = 50,
    save: bool = True,
    market: str = "us",
) -> xgb.XGBClassifier:
    """
    Modeli eğitir, değerlendirir ve kaydeder.

    Args:
        X:                    Özellik matrisi (FEATURE_COLS sütunları)
        y:                    Etiketler (0/1)
        params:               XGBoost parametreleri (None = varsayılan)
        early_stopping_rounds: Doğrulama AUC düzelmedikçe kaç tur beklensin
        save:                 Modeli diske kaydet

    Returns:
        Eğitilmiş XGBClassifier
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    p = {**DEFAULT_PARAMS, **(params or {})}
    p["early_stopping_rounds"] = early_stopping_rounds

    X_train, X_val, X_test, y_train, y_val, y_test = time_split(X, y)

    # Sınıf dengesizliği: AL örnekleri az → ağırlık ver
    pos_rate = _buy_rate(y_train)
    neg_rate = 1 - pos_rate
    scale_pw = neg_rate / pos_rate if pos_rate > 0 else 1.0
    logger.info(
        "Eğitim seti AL oranı: {:.1f}% | scale_pos_weight: {:.2f}",
        pos_rate * 100, scale_pw,
    )
    p["scale_pos_weight"] = scale_pw

    model = xgb.XGBClassifier(**p)
    model.fit(
        X_train[FEATURE_COLS], y_train,
        eval_set=[(X_val[FEATURE_COLS], y_val)],
        verbose=False,
    )

    logger.info("En iyi iterasyon: {}", model.best_iteration)

    # ── Değerlendirme ─────────────────────────────────────────────────────────
    _evaluate(model, X_val, y_val, "Doğrulama")
    _evaluate(model, X_test, y_test, "Test")

    # ── Özellik Önemi ─────────────────────────────────────────────────────────
    _log_feature_importance(model, top_n=15)

    if save:
        _save(model, X_train, y_train, p, market=market)

    return model


def _evaluate(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    label: str,
    threshold: float = 0.5,
):
    """Seti değerlendirir ve metrikleri loglar."""
    proba = model.predict_proba(X[FEATURE_COLS])[:, 1]
    pred  = (proba >= threshold).astype(int)

    auc  = roc_auc_score(y, proba)
    prec = precision_score(y, pred, zero_division=0)
    rec  = recall_score(y, pred, zero_division=0)
    f1   = f1_score(y, pred, zero_division=0)
    cm   = confusion_matrix(y, pred)

    logger.info(
        "── {} Metrikleri ──\n"
        "  AUC: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f}\n"
        "  Confusion Matrix:\n"
        "    TN={} FP={}\n"
        "    FN={} TP={}",
        label, auc, prec, rec, f1,
        cm[0][0], cm[0][1],
        cm[1][0], cm[1][1],
    )
    return {"auc": auc, "precision": prec, "recall": rec, "f1": f1}


def _log_feature_importance(model: xgb.XGBClassifier, top_n: int = 15):
    """En önemli özellikleri loglar."""
    importance = model.feature_importances_
    feat_imp = sorted(
        zip(FEATURE_COLS, importance),
        key=lambda x: x[1], reverse=True,
    )
    lines = [f"  {name:<25} {imp:.4f}" for name, imp in feat_imp[:top_n]]
    logger.info("── Top {} Özellik Önemi ──\n{}", top_n, "\n".join(lines))


def _save(model: xgb.XGBClassifier, X_train, y_train, params: dict, market: str = "us"):
    """Modeli ve meta verisini kaydeder."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    m_path, meta_path = model_paths(market)
    joblib.dump(model, m_path)

    meta = {
        "trained_at":    datetime.now().isoformat(),
        "market":        market,
        "n_train":       len(X_train),
        "buy_rate":      round(_buy_rate(y_train), 4),
        "best_iter":     int(model.best_iteration),
        "feature_cols":  FEATURE_COLS,
        "params":        {k: v for k, v in params.items()
                          if k not in ("eval_metric", "objective")},
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    logger.info("Model kaydedildi: {}", m_path)
    logger.info("Meta kaydedildi: {}", meta_path)


def load_model(market: str = "us") -> tuple[xgb.XGBClassifier, dict]:
    """Kaydedilmiş modeli ve meta verisini yükler."""
    m_path, meta_path = model_paths(market)
    if not m_path.exists():
        raise FileNotFoundError(
            f"Model bulunamadı: {m_path}\n"
            f"Önce 'py cli.py train-ml --market {market}' komutuyla modeli eğitin."
        )
    model = joblib.load(m_path)
    meta  = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return model, meta
