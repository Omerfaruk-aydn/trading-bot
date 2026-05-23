"""Büyük ölçekli dataset pipeline — 150.000+ eğitim örneği.

Her örnek şunları kapsar:
- Multi-timeframe analiz (1m → 1W uyumu)
- Piyasa rejimi (bull/bear/sideways/volatile)
- Korelasyon bağlamı (DXY, VIX, BTC dominansı)
- İndikatör çakışması ve çelişkileri
- Risk/ödül hesaplamaları
- Gerçek fiyat hareketi etiketi (lookahead ile)
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from loguru import logger

from config.symbols import (
    BIST_SYMBOLS, COMMODITY_SYMBOLS, CRYPTO_TOP200,
    FOREX_PAIRS, INDEX_SYMBOLS, SP500_SYMBOLS,
    TIMEFRAMES_CRYPTO, TIMEFRAMES_STOCK,
)
from data.indicators import compute_all, generate_summary, prepare_df
from data.storage import get_ohlcv, get_recent_news, init_db
from finetune.dataset_builder import (
    DATASETS_DIR, SYSTEM_PROMPTS,
    save_jsonl, split_dataset,
    _nan, _signal_from_indicators,
)

LARGE_DATASETS_DIR = DATASETS_DIR.parent / "datasets_large"
LARGE_DATASETS_DIR.mkdir(exist_ok=True)

# ── Piyasa Rejimi Tespiti ─────────────────────────────────────────────────────

def detect_market_regime(df_ind) -> str:
    """
    Piyasa rejimini tespit eder.
    Returns: bull | bear | sideways | volatile
    """
    if len(df_ind) < 50:
        return "unknown"

    close = df_ind["close"]
    sma50 = df_ind.get("sma_50", close.rolling(50).mean())
    atr_val = df_ind.get("atr", (df_ind["high"] - df_ind["low"]).rolling(14).mean())
    adx_val = df_ind.get("adx")

    last = df_ind.iloc[-1]
    close_now = last["close"]
    sma50_now = sma50.iloc[-1] if not _nan(sma50.iloc[-1]) else close_now
    atr_now = atr_val.iloc[-1] if not _nan(atr_val.iloc[-1]) else close_now * 0.02

    # Volatilite — son 14 günlük ATR'nin uzun dönem ortalamasına oranı
    atr_ratio = atr_now / (atr_val.mean() if atr_val.mean() > 0 else atr_now)

    # ADX trend gücü
    adx_now = last.get("adx", 20)
    adx_now = 20 if _nan(adx_now) else adx_now

    # Fiyat pozisyonu
    price_above_sma = close_now > sma50_now

    # 20 günlük getiri
    ret_20 = (close_now - close.iloc[-min(20, len(close))]) / close.iloc[-min(20, len(close))] * 100

    if atr_ratio > 1.8:
        return "volatile"
    elif adx_now >= 25 and price_above_sma and ret_20 > 2:
        return "bull"
    elif adx_now >= 25 and not price_above_sma and ret_20 < -2:
        return "bear"
    else:
        return "sideways"


# ── Multi-Timeframe Analiz ────────────────────────────────────────────────────

def get_mtf_context(symbol: str, timeframes: list[str]) -> dict[str, str]:
    """
    Birden fazla zaman diliminde özet üretir.

    Returns:
        {"1h": "özet...", "4h": "özet...", "1d": "özet..."}
    """
    context: dict[str, str] = {}
    for tf in timeframes:
        candles = get_ohlcv(symbol, tf, 200)
        if len(candles) < 60:
            continue
        df = prepare_df(candles)
        df_ind = compute_all(df)
        context[tf] = generate_summary(df_ind, symbol)
    return context


def mtf_alignment(mtf_context: dict[str, str]) -> tuple[str, float]:
    """
    Tüm timeframe'lerin trend yönü uyumunu değerlendirir.

    Returns:
        (alignment: "aligned_bull" | "aligned_bear" | "mixed", score: 0-1)
    """
    bull_count = sum(1 for s in mtf_context.values() if "yukarı" in s or "bullish" in s or "yüksel" in s)
    bear_count = sum(1 for s in mtf_context.values() if "aşağı" in s or "bearish" in s or "düş" in s)
    total = len(mtf_context)

    if total == 0:
        return "mixed", 0.5

    if bull_count >= total * 0.67:
        return "aligned_bull", round(bull_count / total, 2)
    elif bear_count >= total * 0.67:
        return "aligned_bear", round(bear_count / total, 2)
    else:
        return "mixed", 0.5


# ── Gerçek Fiyat Etiketi (Lookahead) ─────────────────────────────────────────

def _future_label(df_ind, idx: int, forward_bars: int = 10) -> tuple[str, float]:
    """
    İndikatör satırından N mum ilerisinin fiyatına göre gerçek etiket üretir.
    NOT: Sadece eğitim verisi için kullanılır — canlı sistemde lookahead YASAK.

    Returns:
        (signal: bullish|bearish|neutral, magnitude: yüzde değişim)
    """
    rows = df_ind["close"]
    future_idx = idx + forward_bars

    if future_idx >= len(rows):
        return "neutral", 0.0

    now_price = rows.iloc[idx]
    future_price = rows.iloc[future_idx]
    atr_now = df_ind["atr"].iloc[idx] if "atr" in df_ind.columns else now_price * 0.01
    if _nan(atr_now) or atr_now == 0:
        atr_now = now_price * 0.01

    pct_change = (future_price - now_price) / now_price * 100

    # ATR cinsinden kaç ATR hareket etti
    atr_moves = abs(future_price - now_price) / atr_now

    if pct_change > 0.5 and atr_moves > 0.5:
        return "bullish", round(pct_change, 3)
    elif pct_change < -0.5 and atr_moves > 0.5:
        return "bearish", round(pct_change, 3)
    else:
        return "neutral", round(pct_change, 3)


# ── Kapsamlı Teknik Ajan Örnekleri ───────────────────────────────────────────

def build_technical_large(
    symbol: str,
    primary_tf: str = "1h",
    context_tfs: list[str] | None = None,
    step: int = 5,
    forward_bars: int = 12,
) -> list[dict]:
    """
    Tek sembol için zengin teknik ajan örnekleri üretir.

    Her örnek:
    - Ana TF indikatörleri + özet
    - Multi-TF bağlam
    - Piyasa rejimi
    - Gerçek fiyat etiketi (lookahead)
    """
    context_tfs = context_tfs or ["4h", "1d"]

    candles = get_ohlcv(symbol, primary_tf, 1000)
    if len(candles) < 100:
        return []

    df = prepare_df(candles)
    df_ind = compute_all(df)
    rows = df_ind.to_dict("records")

    # MTF bağlam (bir kez hesapla, hafızada tut)
    mtf_ctx = get_mtf_context(symbol, context_tfs)
    mtf_align, mtf_score = mtf_alignment(mtf_ctx)
    regime = detect_market_regime(df_ind)

    examples: list[dict] = []

    for i in range(80, len(rows) - forward_bars - 1, step):
        row = rows[i]
        prev_row = rows[i - 1]

        # Gerçek etiket
        true_signal, magnitude = _future_label(df_ind, i, forward_bars)

        # İndikatör sinyali (kural tabanlı)
        ind_signal, ind_conf, key_points = _signal_from_indicators(row, prev_row)

        # İkisi uyuşuyorsa güven artar
        if true_signal == ind_signal and true_signal != "neutral":
            confidence = min(0.92, ind_conf + 0.10)
        elif true_signal != ind_signal and true_signal != "neutral":
            confidence = max(0.35, ind_conf - 0.15)
        else:
            confidence = ind_conf

        # Özet
        mini_df = df_ind.iloc[max(0, i - 49): i + 1]
        summary = generate_summary(mini_df, symbol)

        close = row.get("close", 0)
        atr_val = row.get("atr", close * 0.015)
        if _nan(atr_val):
            atr_val = close * 0.015

        if true_signal == "bullish":
            sl = round(close - 1.5 * atr_val, 6)
            tp = round(close + 3.0 * atr_val, 6)
        elif true_signal == "bearish":
            sl = round(close + 1.5 * atr_val, 6)
            tp = round(close - 3.0 * atr_val, 6)
        else:
            sl = tp = None

        output = {
            "signal": true_signal,
            "confidence": round(confidence, 2),
            "reasoning": _build_rich_reasoning(
                true_signal, key_points, regime, mtf_align, magnitude
            ),
            "key_points": key_points[:6],
            "market_regime": regime,
            "mtf_alignment": mtf_align,
            "mtf_score": mtf_score,
            "price_change_forward": magnitude,
            "stop_loss": sl,
            "take_profit": tp,
        }

        # İnsan mesajı — zengin bağlam
        mtf_section = ""
        for tf, ctx in mtf_ctx.items():
            mtf_section += f"\n[{tf} Özeti]\n{ctx}\n"

        human_msg = (
            f"Sembol: {symbol}\n"
            f"Ana Zaman Dilimi: {primary_tf}\n"
            f"Piyasa Rejimi: {regime}\n"
            f"MTF Uyumu: {mtf_align} (skor: {mtf_score})\n\n"
            f"[{primary_tf} Teknik Özet]\n{summary}\n"
            f"{mtf_section}"
        )

        examples.append({
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPTS["technical"]},
                {"from": "human", "value": human_msg},
                {"from": "gpt", "value": json.dumps(output, ensure_ascii=False)},
            ]
        })

    return examples


# ── Piyasa Bilgisi Örnekleri (Genel Finansal Eğitim) ─────────────────────────

def build_market_knowledge_examples() -> list[dict]:
    """
    İndikatörler, piyasa kavramları, risk yönetimi hakkında
    soru-cevap formatında eğitim örnekleri.
    """
    qa_pairs = [
        # İndikatörler
        ("RSI 30'un altına düştüğünde ne anlama gelir?",
         '{"signal": "bullish", "confidence": 0.65, "reasoning": "RSI 30 altı aşırı satım bölgesidir. Fiyat düşüşünün hız kazandığını gösterir ancak dönüş garantisi vermez. Konfirmasyon için hacim artışı ve destek kırılımının olmaması gerekir.", "key_points": ["RSI < 30 aşırı satım", "Dönüş sinyali değil, olasılık", "Hacim konfirmasyonu şart", "Destek seviyesi kontrol edilmeli"]}'),
        ("MACD golden cross nedir?",
         '{"signal": "bullish", "confidence": 0.70, "reasoning": "MACD hattı sinyal hattını aşağıdan yukarıya keser. Kısa vadeli momentum uzun vadeli momentumu geçmiştir. Genellikle trend dönüşünün erken sinyalidir.", "key_points": ["MACD > sinyal hattı", "Momentum değişimi", "Hacimle desteklenirse güçlenir", "Yatay piyasada yanıltıcı olabilir"]}'),
        ("Bollinger Band sıkışması ne anlama gelir?",
         '{"signal": "neutral", "confidence": 0.60, "reasoning": "Volatilite düşmüş, fiyat dar aralıkta hareket ediyor. Yakında güçlü bir kırılım beklenir. Yön belirsiz — momentum indikatörleriyle teyit gerekir.", "key_points": ["BB genişliği minimumda", "Volatilite sıkışması", "Kırılım yakın", "Yön belirsiz"]}'),
        ("Death cross ne anlama gelir?",
         '{"signal": "bearish", "confidence": 0.68, "reasoning": "50 gunluk SMA, 200 gunluk SMAyi asagi kesiyor. Uzun vadeli trend degisiminin gecikmeli sinyali. Genellikle bear market baslangicindan gorulur.", "key_points": ["SMA50 < SMA200", "Uzun vadeli bearish sinyal", "Gecikmeli gosterge", "Dip avi icin erken olabilir"]}'),
        # Risk yönetimi
        ("Tek bir trade'de hesabın yüzde kaçı riske edilmeli?",
         '{"decision": "approve", "confidence": 0.95, "reasoning": "Profesyonel risk yonetiminde standart kural hesabin maksimum yuzde 1-2sini tek bir tradede riske etmektir. Yuzde 2 ile 50 ardisik kayipta hesabin yuzde 36si korunur.", "risk_reward_ratio": 2.0, "position_size_pct": 2.0, "key_risks": ["Sermayeyi tek tradede tuketme riski", "Ardisik kayiplar"]}'),
        ("Stop-loss neden zorunludur?",
         '{"decision": "approve", "confidence": 0.99, "reasoning": "Stop-loss olmayan pozisyon sonsuz kayip riski tasir. Piyasa her zaman beklentinin tersine hareket edebilir. Stop-loss, yanlis oldugumda ne kadar kaybederim sorusunu onceden yanitlar.", "risk_reward_ratio": 0.0, "position_size_pct": 0.0, "key_risks": ["Sinirsiz kayip riski", "Duygusal karar verme", "Hesap silinebilir"]}'),
        # Piyasa rejimleri
        ("Bull market'te nasıl işlem yapılır?",
         '{"action": "buy", "confidence": 0.72, "reasoning": "Bull markette trend takip stratejileri calisir. Dipleri al, direnclerden kismi kar al. Shorta girme trende karsi gitme.", "position_size_pct": 2.0, "stop_loss": null, "take_profit_1": null, "take_profit_2": null, "invalidation": "Fiyat 200 gunluk SMAni altina kalici kirarilirsa"}'),
        ("VIX 30'un üzerine çıktığında ne yapmalı?",
         '{"action": "hold", "confidence": 0.78, "reasoning": "VIX 30 üzeri aşırı korku bölgesidir. Piyasada panik var. Kısa vadeli dip fırsatı olabilir ama önce stabilizasyon bekle. Pozisyon büyüklüğünü yarıya indir.", "position_size_pct": 1.0, "stop_loss": null, "take_profit_1": null, "take_profit_2": null, "invalidation": "VIX 40 üzerine çıkarsa"}'),
        # Korelasyonlar
        ("DXY (Dolar endeksi) yükselince altın ne yapar?",
         '{"signal": "bearish", "confidence": 0.73, "reasoning": "DXY ile altın arasında güçlü negatif korelasyon vardır. Dolar güçlenince dolar cinsinden fiyatlanan altın pahalılaşır, talep düşer.", "key_points": ["DXY-Altın negatif korelasyon", "Tarihsel korelasyon ~-0.7", "FED kararları belirleyici", "Kısa vadede kırılabilir"]}'),
        ("Bitcoin ve altın arasındaki korelasyon nedir?",
         '{"signal": "neutral", "confidence": 0.55, "reasoning": "BTC ve altın arasındaki korelasyon dönemsel değişir. Kriz dönemlerinde her ikisi de değer kaybedebilir (risk-off). Normal dönemde düşük korelasyon. BTC dijital altın narratifi güçlendikçe korelasyon artabilir.", "key_points": ["Dönemsel korelasyon", "Kriz anında ayrışabilir", "Risk-off dönemde ikisi düşer", "Uzun vadede güçleniyor"]}'),
        # Makro
        ("FED faiz artırımı piyasaları nasıl etkiler?",
         '{"signal": "bearish", "confidence": 0.70, "reasoning": "Faiz artışı borçlanma maliyetini artırır, büyüme hisselerini baskılar, tahvil cazibesini artırır. Kısa vadede hisse senetleri ve kripto düşer, dolar güçlenir.", "key_points": ["Büyüme hisseleri düşer", "Dolar güçlenir", "Tahvil getirileri yükselir", "Kripto baskı altında", "6-12 ay gecikmeyle ekonomiye yansır"]}'),
        ("Enflasyon düşerken hangi varlıklar yükselir?",
         '{"signal": "bullish", "confidence": 0.68, "reasoning": "Enflasyon düşüşü faiz indirim beklentisi yaratır. Büyüme hisseleri, teknoloji ve kripto olumlu etkilenir. Tahvil fiyatları yükselir.", "key_points": ["Büyüme hisseleri yükselir", "Tahvil rallisi", "Kripto toparlanır", "Faiz indirimi beklentisi kritik"]}'),
    ]

    examples = []
    for question, answer_json in qa_pairs:
        try:
            answer_dict = json.loads(answer_json)
        except Exception:
            continue

        # Hangi ajan türü?
        if "action" in answer_dict:
            system = SYSTEM_PROMPTS["decision"]
        elif "decision" in answer_dict:
            system = SYSTEM_PROMPTS["risk"]
        else:
            system = SYSTEM_PROMPTS["technical"]

        examples.append({
            "conversations": [
                {"from": "system", "value": system},
                {"from": "human", "value": question},
                {"from": "gpt", "value": json.dumps(answer_dict, ensure_ascii=False)},
            ]
        })

    return examples


# ── Ana Pipeline ───────────────────────────────────────────────────────────────

def build_large_dataset(
    crypto_symbols: list[str] | None = None,
    stock_symbols: list[str] | None = None,
    target_examples: int = 150_000,
    step: int = 3,
) -> dict[str, int]:
    """
    150.000+ eğitim örneği üretir.

    Args:
        crypto_symbols: None ise DB'deki tüm kripto semboller
        stock_symbols: None ise DB'deki tüm hisse semboller
        target_examples: Hedef toplam örnek sayısı
        step: Kaç mumda bir örnek (küçük = daha fazla örnek)

    Returns:
        Her ajan türü için üretilen örnek sayısı
    """
    init_db()
    random.seed(42)

    all_examples: list[dict] = []

    # 1. Genel finansal bilgi örnekleri
    knowledge = build_market_knowledge_examples()
    all_examples.extend(knowledge)
    logger.info("Finansal bilgi örnekleri: {}", len(knowledge))

    # 2. Kripto teknik örnekleri
    crypto_syms = crypto_symbols or _get_available_symbols("binance")
    logger.info("Kullanılabilir kripto sembol: {}", len(crypto_syms))

    crypto_examples: list[dict] = []
    for i, symbol in enumerate(crypto_syms):
        for tf in ["1h", "4h"]:
            examples = build_technical_large(
                symbol, primary_tf=tf,
                context_tfs=["4h", "1d"] if tf == "1h" else ["1d"],
                step=step,
            )
            crypto_examples.extend(examples)
            if (i + 1) % 10 == 0:
                logger.info("Kripto ilerleme: {}/{} sembol, {} örnek",
                            i + 1, len(crypto_syms), len(crypto_examples))

            if len(all_examples) + len(crypto_examples) >= target_examples:
                break
        if len(all_examples) + len(crypto_examples) >= target_examples:
            break

    all_examples.extend(crypto_examples)
    logger.info("Kripto teknik örnekler: {}", len(crypto_examples))

    # 3. Hisse teknik örnekleri — min 10k kota, hedeften bağımsız
    STOCK_MIN = 10_000
    stock_syms = stock_symbols or _get_available_symbols("yfinance")
    logger.info("Kullanılabilir hisse sembol: {}", len(stock_syms))

    stock_examples: list[dict] = []
    for i, symbol in enumerate(stock_syms):
        examples = build_technical_large(
            symbol, primary_tf="1d",
            context_tfs=["1wk"],
            step=step,
        )
        stock_examples.extend(examples)
        if len(stock_examples) >= STOCK_MIN:
            break

    all_examples.extend(stock_examples)
    logger.info("Hisse teknik örnekler: {}", len(stock_examples))

    # 4. Risk ve karar örnekleri (sentetik — her zaman üret)
    from finetune.dataset_builder import build_risk_examples, build_decision_examples, build_news_examples
    risk_ex = build_risk_examples(n_synthetic=5000)
    decision_ex = build_decision_examples(n_synthetic=3000)
    news_ex = build_news_examples(limit=2000)

    all_examples.extend(risk_ex)
    all_examples.extend(decision_ex)
    all_examples.extend(news_ex)

    logger.info("Risk örnekleri: {}", len(risk_ex))
    logger.info("Karar örnekleri: {}", len(decision_ex))
    logger.info("Haber örnekleri: {}", len(news_ex))

    # Karıştır ve böl
    random.shuffle(all_examples)
    train, val = split_dataset(all_examples, train_ratio=0.90)

    save_jsonl(train, LARGE_DATASETS_DIR / "large_train.jsonl")
    save_jsonl(val, LARGE_DATASETS_DIR / "large_val.jsonl")

    counts = {
        "knowledge": len(knowledge),
        "crypto_technical": len(crypto_examples),
        "stock_technical": len(stock_examples),
        "risk": len(risk_ex),
        "decision": len(decision_ex),
        "news": len(news_ex),
        "total": len(all_examples),
        "train": len(train),
        "val": len(val),
    }

    logger.info("\n{'='*60}")
    logger.info("BÜYÜK DATASET TAMAMLANDI")
    logger.info("Toplam: {:,} örnek", len(all_examples))
    logger.info("Train: {:,} | Val: {:,}", len(train), len(val))
    logger.info("Çıktı: {}", LARGE_DATASETS_DIR)

    return counts


# ── Yardımcılar ───────────────────────────────────────────────────────────────

def _get_available_symbols(source_prefix: str) -> list[str]:
    """DB'de verisi olan sembolleri döner."""
    from data.storage import get_session, MarketData
    with get_session() as session:
        rows = (
            session.query(MarketData.symbol)
            .filter(MarketData.source.like(f"{source_prefix}%"))
            .filter(MarketData.data_type == "ohlcv")
            .distinct()
            .all()
        )
    return [r[0] for r in rows]


def _build_rich_reasoning(
    signal: str,
    key_points: list[str],
    regime: str,
    mtf_align: str,
    magnitude: float,
) -> str:
    direction = {"bullish": "yükseliş", "bearish": "düşüş", "neutral": "yatay"}.get(signal, "nötr")
    pts = "; ".join(key_points[:3]) if key_points else "karma sinyaller"
    regime_tr = {"bull": "boğa", "bear": "ayı", "sideways": "yatay", "volatile": "volatil"}.get(regime, regime)
    align_tr = {
        "aligned_bull": "tüm TF'ler yükseliş",
        "aligned_bear": "tüm TF'ler düşüş",
        "mixed": "TF'ler karışık",
    }.get(mtf_align, mtf_align)
    return (
        f"Piyasa rejiimi {regime_tr}, {align_tr}. "
        f"Teknik göstergeler {direction} yönünde: {pts}. "
        f"İleri {abs(magnitude):.2f}% {'yükseliş' if magnitude > 0 else 'düşüş'} gözlemlendi."
    )
