"""Destek/Direnç seviye tespiti — pivot point tabanlı algoritmik analiz."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _find_pivots(df: pd.DataFrame, window: int = 10) -> tuple[list[float], list[float]]:
    highs = df["high"].values
    lows = df["low"].values
    n = len(highs)

    pivot_highs: list[float] = []
    pivot_lows: list[float] = []

    for i in range(window, n - window):
        hi_window = highs[i - window : i + window + 1]
        lo_window = lows[i - window : i + window + 1]
        if highs[i] == hi_window.max() and list(hi_window).count(highs[i]) == 1:
            pivot_highs.append(float(highs[i]))
        if lows[i] == lo_window.min() and list(lo_window).count(lows[i]) == 1:
            pivot_lows.append(float(lows[i]))

    return pivot_highs, pivot_lows


def _cluster_levels(levels: list[float], tolerance: float = 0.015) -> list[float]:
    if not levels:
        return []
    sorted_lvls = sorted(levels)
    used = [False] * len(sorted_lvls)
    clustered: list[float] = []

    for i, lvl in enumerate(sorted_lvls):
        if used[i]:
            continue
        group = [lvl]
        for j in range(i + 1, len(sorted_lvls)):
            if not used[j] and (sorted_lvls[j] / lvl - 1) <= tolerance:
                group.append(sorted_lvls[j])
                used[j] = True
        clustered.append(float(np.mean(group)))

    return clustered


def find_sr_levels(
    df: pd.DataFrame,
    window: int = 10,
    n_levels: int = 5,
    tolerance: float = 0.015,
) -> tuple[list[float], list[float]]:
    """
    Destek ve direnç seviyelerini döndürür.

    Returns:
        (supports, resistances)
        supports: güncel fiyatın altındaki seviyeler, en yakından en uzağa
        resistances: güncel fiyatın üstündeki seviyeler, en yakından en uzağa
    """
    if len(df) < window * 2 + 5:
        return [], []

    current = float(df["close"].iloc[-1])
    pivot_highs, pivot_lows = _find_pivots(df, window)

    all_pivots = pivot_highs + pivot_lows
    clustered = _cluster_levels(all_pivots, tolerance)

    supports = sorted(
        [lvl for lvl in clustered if lvl < current * 0.999],
        reverse=True,
    )[:n_levels]

    resistances = sorted(
        [lvl for lvl in clustered if lvl > current * 1.001],
    )[:n_levels]

    return supports, resistances


def sr_signal_score(df: pd.DataFrame) -> tuple[int, str]:
    """
    Destek/Direnç yakınlık skoru.

    Returns:
        (score, reason)
        +2 : Güçlü destek bölgesi (<%1.5 uzakta)
        +1 : Destek yakını (<%4 uzakta)
        -1 : Direnç yakını (<%4 uzakta)
        -2 : Güçlü direnç bölgesi (<%1.5 uzakta)
    """
    if df is None or len(df) < 25:
        return 0, ""

    try:
        current = float(df["close"].iloc[-1])
        supports, resistances = find_sr_levels(df)

        if supports:
            nearest_sup = supports[0]
            dist = (current - nearest_sup) / current
            if 0 <= dist <= 0.015:
                return 2, f"Destek bölgesi ({nearest_sup:.4g}, %{dist*100:.1f} uzakta)"
            elif 0 < dist <= 0.04:
                return 1, f"Destek yakını ({nearest_sup:.4g})"

        if resistances:
            nearest_res = resistances[0]
            dist = (nearest_res - current) / current
            if 0 <= dist <= 0.015:
                return -2, f"Direnç bölgesi ({nearest_res:.4g}, %{dist*100:.1f} uzakta)"
            elif 0 < dist <= 0.04:
                return -1, f"Direnç yakını ({nearest_res:.4g})"

    except Exception:
        pass

    return 0, ""


def fibonacci_levels(df: pd.DataFrame, lookback: int = 60) -> dict[str, float]:
    """
    Son N mumdan Fibonacci geri çekilme seviyelerini hesaplar.

    Returns:
        {
          "swing_high": ..., "swing_low": ...,
          "fib_23": ..., "fib_38": ..., "fib_50": ..., "fib_61": ..., "fib_78": ...
        }
    """
    window = df.tail(lookback)
    high = float(window["high"].max())
    low = float(window["low"].min())
    diff = high - low
    return {
        "swing_high": high,
        "swing_low": low,
        "fib_23": high - diff * 0.236,
        "fib_38": high - diff * 0.382,
        "fib_50": high - diff * 0.500,
        "fib_61": high - diff * 0.618,
        "fib_78": high - diff * 0.786,
    }
