"""Dataset Pipeline — piyasa verisi, indikatörler ve haberleri
fine-tune için JSONL eğitim örneklerine dönüştürür.

Format: ShareGPT (system / human / gpt turları)
Her örnek bir uzman ajanın gerçekleştirmesi gereken analizi temsil eder.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from loguru import logger

from data.indicators import compute_all, generate_summary, prepare_df
from data.storage import get_ohlcv, get_recent_news, init_db

DATASETS_DIR = Path(__file__).parent / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)

AgentType = Literal["technical", "news", "risk", "decision"]

# ── Sistem Promptları (kısa, model için optimize) ─────────────────────────────

SYSTEM_PROMPTS: dict[str, str] = {
    "technical": (
        "Sen uzman bir teknik analistsin. Verilen fiyat ve indikatör verisini analiz et. "
        "Yanıtını YALNIZCA şu JSON formatında ver:\n"
        '{"signal": "bullish|bearish|neutral", "confidence": 0.0-1.0, '
        '"reasoning": "gerekçe", "key_points": ["nokta1", "nokta2"], '
        '"stop_loss": fiyat_veya_null, "take_profit": fiyat_veya_null}'
    ),
    "news": (
        "Sen uzman bir finansal haber analistisin. Verilen haberi analiz et ve "
        "hangi varlıkları nasıl etkilediğini değerlendir. "
        "Yanıtını YALNIZCA şu JSON formatında ver:\n"
        '{"signal": "bullish|bearish|neutral", "confidence": 0.0-1.0, '
        '"reasoning": "gerekçe", "affected_symbols": ["sembol"], '
        '"impact_magnitude": 1-10, "impact_duration": "short|medium|long", '
        '"key_points": ["nokta1", "nokta2"]}'
    ),
    "risk": (
        "Sen bir risk yöneticisisin. Verilen trade önerisini değerlendir. "
        "Yanıtını YALNIZCA şu JSON formatında ver:\n"
        '{"decision": "approve|reject", "confidence": 0.0-1.0, '
        '"risk_reward_ratio": sayı, "position_size_pct": 0.0-10.0, '
        '"reasoning": "gerekçe", "key_risks": ["risk1", "risk2"]}'
    ),
    "decision": (
        "Sen baş trader'sın. 5 uzman analistten görüş aldın. "
        "Tüm görüşleri değerlendirip final karar ver. "
        "Yanıtını YALNIZCA şu JSON formatında ver:\n"
        '{"action": "buy|sell|hold", "confidence": 0.0-1.0, '
        '"position_size_pct": 0.0-10.0, "stop_loss": fiyat, '
        '"take_profit_1": fiyat, "take_profit_2": fiyat, '
        '"reasoning": "gerekçe", "invalidation": "iptal_senaryosu"}'
    ),
}

# ── Teknik Ajan Eğitim Örnekleri ─────────────────────────────────────────────

def _signal_from_indicators(row, prev_row) -> tuple[str, float, list[str]]:
    """İndikatör değerlerinden kural tabanlı sinyal üretir (etiketleme için)."""
    bullish_score = 0
    bearish_score = 0
    points: list[str] = []

    rsi_val = row.get("rsi", 50)
    if not _nan(rsi_val):
        if rsi_val < 35:
            bullish_score += 2
            points.append(f"RSI {rsi_val:.1f} aşırı satım bölgesinde")
        elif rsi_val > 65:
            bearish_score += 2
            points.append(f"RSI {rsi_val:.1f} aşırı alım bölgesinde")

    macd_hist = row.get("macd_hist")
    prev_hist = prev_row.get("macd_hist")
    if not _nan(macd_hist) and not _nan(prev_hist):
        if macd_hist > 0 and prev_hist <= 0:
            bullish_score += 3
            points.append("MACD yeni bullish kesişim")
        elif macd_hist < 0 and prev_hist >= 0:
            bearish_score += 3
            points.append("MACD yeni bearish kesişim")
        elif macd_hist > 0:
            bullish_score += 1
            points.append("MACD pozitif bölgede")
        else:
            bearish_score += 1
            points.append("MACD negatif bölgede")

    close = row.get("close", 0)
    ema21 = row.get("ema_21")
    ema55 = row.get("ema_55")
    if not _nan(ema21) and not _nan(ema55) and close:
        if close > ema21 > ema55:
            bullish_score += 2
            points.append("Fiyat EMA21 ve EMA55 üzerinde")
        elif close < ema21 < ema55:
            bearish_score += 2
            points.append("Fiyat EMA21 ve EMA55 altında")

    bb_pct = row.get("bb_pct")
    if not _nan(bb_pct):
        if bb_pct < 0.15:
            bullish_score += 1
            points.append("Bollinger alt bandına yakın (destek)")
        elif bb_pct > 0.85:
            bearish_score += 1
            points.append("Bollinger üst bandına yakın (direnç)")

    if row.get("bullish_engulfing", 0) == 1:
        bullish_score += 2
        points.append("Bullish engulfing formasyonu")
    if row.get("bearish_engulfing", 0) == 1:
        bearish_score += 2
        points.append("Bearish engulfing formasyonu")
    if row.get("hammer", 0) == 1:
        bullish_score += 1
        points.append("Hammer formasyonu")

    total = bullish_score + bearish_score
    if total == 0:
        return "neutral", 0.5, points or ["Sinyal yok"]

    if bullish_score > bearish_score:
        conf = min(0.95, 0.5 + (bullish_score - bearish_score) / (total + 2) * 0.5)
        return "bullish", round(conf, 2), points
    elif bearish_score > bullish_score:
        conf = min(0.95, 0.5 + (bearish_score - bullish_score) / (total + 2) * 0.5)
        return "bearish", round(conf, 2), points
    else:
        return "neutral", 0.5, points


def build_technical_examples(
    symbol: str,
    interval: str = "1h",
    limit: int = 500,
    step: int = 10,
) -> list[dict]:
    """
    DB'deki OHLCV verisiyle teknik ajan eğitim örnekleri üretir.

    Args:
        symbol: İşlem sembolü
        interval: Zaman dilimi
        limit: Kaç mum kullanılacak
        step: Kaç mumda bir örnek üretilecek (veri çeşitliliği için)

    Returns:
        ShareGPT formatında dict listesi
    """
    candles = get_ohlcv(symbol, interval, limit)
    if len(candles) < 60:
        logger.warning("Yetersiz veri: {} {} ({} mum)", symbol, interval, len(candles))
        return []

    df = prepare_df(candles)
    df_ind = compute_all(df)
    rows = df_ind.to_dict("records")

    examples: list[dict] = []

    for i in range(55, len(rows) - 1, step):
        row = rows[i]
        prev_row = rows[i - 1]

        # Etiket (kural tabanlı — gerçek veriden türetildi)
        signal, confidence, key_points = _signal_from_indicators(row, prev_row)

        # Son 50 mumun özeti için mini-df
        mini_df = df_ind.iloc[max(0, i - 49): i + 1]
        summary = generate_summary(mini_df, symbol)

        close = row.get("close", 0)
        atr_val = row.get("atr", close * 0.01)

        if signal == "bullish":
            stop_loss = round(close - 1.5 * atr_val, 4)
            take_profit = round(close + 3 * atr_val, 4)
        elif signal == "bearish":
            stop_loss = round(close + 1.5 * atr_val, 4)
            take_profit = round(close - 3 * atr_val, 4)
        else:
            stop_loss = None
            take_profit = None

        output = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": _build_reasoning(signal, key_points, summary),
            "key_points": key_points[:5],
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

        example = {
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPTS["technical"]},
                {"from": "human", "value": f"Sembol: {symbol}\nZaman dilimi: {interval}\n\nTeknik Analiz Özeti:\n{summary}"},
                {"from": "gpt", "value": json.dumps(output, ensure_ascii=False)},
            ]
        }
        examples.append(example)

    logger.info("Teknik ajan: {} örnek üretildi — {} {}", len(examples), symbol, interval)
    return examples


# ── Haber Ajanı Eğitim Örnekleri ─────────────────────────────────────────────

_NEWS_SIGNAL_KEYWORDS = {
    "bullish": [
        "rekor", "yükseldi", "artış", "büyüme", "kâr", "anlaşma", "ihracat",
        "yatırım", "pozitif", "güçlü", "aşıldı", "rally", "kazanım",
    ],
    "bearish": [
        "düştü", "kayıp", "zarar", "iflas", "kriz", "geriledi", "faiz artışı",
        "enflasyon", "resesyon", "uyarı", "risk", "baskı", "sert",
    ],
}


def _news_signal(title: str, summary: str) -> tuple[str, float]:
    """Haber metninden kural tabanlı sinyal tahmini."""
    text = (title + " " + summary).lower()
    b = sum(1 for kw in _NEWS_SIGNAL_KEYWORDS["bullish"] if kw in text)
    be = sum(1 for kw in _NEWS_SIGNAL_KEYWORDS["bearish"] if kw in text)

    if b > be and b >= 2:
        return "bullish", min(0.85, 0.5 + b * 0.07)
    elif be > b and be >= 2:
        return "bearish", min(0.85, 0.5 + be * 0.07)
    return "neutral", 0.55


def build_news_examples(limit: int = 200) -> list[dict]:
    """
    DB'deki haberlerden haber ajanı eğitim örnekleri üretir.

    Returns:
        ShareGPT formatında dict listesi
    """
    news_items = get_recent_news(limit=limit)
    if not news_items:
        logger.warning("DB'de haber yok.")
        return []

    examples: list[dict] = []

    for item in news_items:
        title = item.get("title", "")
        summary = item.get("summary", "")
        source = item.get("source", "")
        symbol = item.get("symbol")

        if not title:
            continue

        signal, confidence = _news_signal(title, summary)

        affected = [symbol] if symbol else []

        # Etki büyüklüğü tahmini
        text = (title + summary).lower()
        magnitude = 3
        for strong_kw in ["rekor", "iflas", "kriz", "anlaşma", "birleşme", "fed", "tcmb"]:
            if strong_kw in text:
                magnitude = min(10, magnitude + 2)

        output = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": f"Haber analizi: '{title[:100]}' — {signal} etki tespit edildi.",
            "affected_symbols": affected,
            "impact_magnitude": magnitude,
            "impact_duration": "short" if magnitude <= 4 else ("medium" if magnitude <= 7 else "long"),
            "key_points": [title[:120], f"Kaynak: {source}"],
        }

        human_msg = f"Haber Başlığı: {title}\n\nHaber Özeti:\n{summary[:500] if summary else '(özet yok)'}"

        example = {
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPTS["news"]},
                {"from": "human", "value": human_msg},
                {"from": "gpt", "value": json.dumps(output, ensure_ascii=False)},
            ]
        }
        examples.append(example)

    logger.info("Haber ajanı: {} örnek üretildi.", len(examples))
    return examples


# ── Risk Ajanı Eğitim Örnekleri ───────────────────────────────────────────────

def build_risk_examples(n_synthetic: int = 200) -> list[dict]:
    """
    Sentetik trade önerileriyle risk ajanı eğitim örnekleri üretir.

    Gerçek oran: yaklaşık %60 approve, %40 reject (konservatif)
    """
    examples: list[dict] = []
    random.seed(42)

    scenarios = [
        # (rr_ratio, position_pct, stop_distance_pct, signal_agreement, expected)
        (3.0, 2.0, 1.5, 4, "approve"),
        (2.5, 1.5, 1.2, 5, "approve"),
        (1.5, 2.0, 2.0, 3, "reject"),   # RR çok düşük
        (2.0, 8.0, 1.0, 4, "reject"),   # pozisyon çok büyük
        (4.0, 1.0, 0.8, 5, "approve"),
        (1.8, 2.0, 3.0, 3, "reject"),
        (2.2, 2.0, 1.5, 2, "reject"),   # ajan uyuşmazlığı
        (3.5, 1.5, 1.2, 4, "approve"),
        (2.0, 2.0, 5.0, 3, "reject"),   # stop çok uzak
        (2.8, 2.0, 1.5, 5, "approve"),
    ]

    for i in range(n_synthetic):
        sc = scenarios[i % len(scenarios)]
        rr, pos_pct, stop_pct, agreement, expected = sc

        # Küçük rastgele gürültü ekle
        rr = round(rr + random.uniform(-0.3, 0.3), 2)
        pos_pct = round(pos_pct + random.uniform(-0.3, 0.3), 2)

        is_approve = expected == "approve"
        confidence = round(random.uniform(0.65, 0.90) if is_approve else random.uniform(0.60, 0.85), 2)

        if is_approve:
            risks = ["Piyasa volatilitesi yüksek olabilir", "Makro belirsizlik devam ediyor"]
            reasoning = (
                f"Risk/ödül oranı {rr}:1 kabul edilebilir. "
                f"Pozisyon büyüklüğü %{pos_pct} kurallara uygun. "
                f"{agreement}/5 ajan aynı yönde. Trade onaylandı."
            )
        else:
            risks = _reject_reasons(rr, pos_pct, stop_pct, agreement)
            reasoning = f"Trade reddedildi: {'; '.join(risks)}."

        output = {
            "decision": "approve" if is_approve else "reject",
            "confidence": confidence,
            "risk_reward_ratio": rr,
            "position_size_pct": pos_pct if is_approve else 0.0,
            "reasoning": reasoning,
            "key_risks": risks,
        }

        human_msg = (
            f"Trade Önerisi:\n"
            f"- Sembol: {'BTC/USDT' if i % 3 == 0 else ('ETH/USDT' if i % 3 == 1 else 'SOL/USDT')}\n"
            f"- Yön: {'AL' if i % 2 == 0 else 'SAT'}\n"
            f"- Risk/Ödül Oranı: {rr}:1\n"
            f"- Önerilen Pozisyon: %{pos_pct}\n"
            f"- Stop-Loss Mesafesi: %{stop_pct}\n"
            f"- Ajan Uyuşması: {agreement}/5\n"
        )

        example = {
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPTS["risk"]},
                {"from": "human", "value": human_msg},
                {"from": "gpt", "value": json.dumps(output, ensure_ascii=False)},
            ]
        }
        examples.append(example)

    logger.info("Risk ajanı: {} sentetik örnek üretildi.", len(examples))
    return examples


def _reject_reasons(rr: float, pos_pct: float, stop_pct: float, agreement: int) -> list[str]:
    reasons = []
    if rr < 2.0:
        reasons.append(f"Risk/ödül oranı {rr}:1 yetersiz (minimum 2:1)")
    if pos_pct > 5.0:
        reasons.append(f"Pozisyon büyüklüğü %{pos_pct} çok yüksek (max %5)")
    if stop_pct > 3.0:
        reasons.append(f"Stop-loss mesafesi %{stop_pct} çok geniş")
    if agreement < 3:
        reasons.append(f"Ajan uyuşması düşük ({agreement}/5)")
    if not reasons:
        reasons.append("Genel risk profili kabul edilemez")
    return reasons


# ── Karar Ajanı Eğitim Örnekleri ─────────────────────────────────────────────

def build_decision_examples(n_synthetic: int = 150) -> list[dict]:
    """
    5 ajanın çıktısından karar ajanı eğitim örnekleri üretir (sentetik).
    """
    examples: list[dict] = []
    random.seed(7)

    actions = ["buy", "sell", "hold"]
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "THYAO.IS", "GARAN.IS"]

    for i in range(n_synthetic):
        symbol = symbols[i % len(symbols)]

        # 5 ajanın sentetik sinyalleri
        tech_signal = random.choice(["bullish", "bearish", "neutral"])
        news_signal = random.choice(["bullish", "bearish", "neutral"])
        macro_signal = random.choice(["bullish", "bearish", "neutral"])
        sent_signal = random.choice(["bullish", "bearish", "neutral"])
        risk_decision = random.choice(["approve", "approve", "reject"])  # %67 approve

        signals = [tech_signal, news_signal, macro_signal, sent_signal]
        bull_count = signals.count("bullish")
        bear_count = signals.count("bearish")

        # Karar kuralı: Risk reject → hold | 3+ aynı yön → al/sat | diğer → hold
        if risk_decision == "reject":
            action = "hold"
            confidence = round(random.uniform(0.75, 0.90), 2)
            reasoning = "Risk yöneticisi trade'i reddetti. Bekle."
        elif bull_count >= 3:
            action = "buy"
            confidence = round(random.uniform(0.65, 0.88), 2)
            reasoning = f"{bull_count}/4 ajan bullish. Teknik ve haber ajanı hemfikir."
        elif bear_count >= 3:
            action = "sell"
            confidence = round(random.uniform(0.65, 0.88), 2)
            reasoning = f"{bear_count}/4 ajan bearish. Risk yöneticisi onayladı."
        else:
            action = "hold"
            confidence = round(random.uniform(0.60, 0.80), 2)
            reasoning = "Ajanlar arasında uyuşmazlık var. Belirsizlikte bekle."

        base_price = random.uniform(20000, 90000) if "USDT" in symbol else random.uniform(50, 500)
        atr_est = base_price * 0.015

        output = {
            "action": action,
            "confidence": confidence,
            "position_size_pct": round(random.uniform(1.0, 2.5), 1) if action != "hold" else 0.0,
            "stop_loss": round(base_price - 1.5 * atr_est, 2) if action == "buy" else round(base_price + 1.5 * atr_est, 2) if action == "sell" else None,
            "take_profit_1": round(base_price + 2 * atr_est, 2) if action == "buy" else round(base_price - 2 * atr_est, 2) if action == "sell" else None,
            "take_profit_2": round(base_price + 4 * atr_est, 2) if action == "buy" else round(base_price - 4 * atr_est, 2) if action == "sell" else None,
            "reasoning": reasoning,
            "invalidation": f"Fiyat stop-loss seviyesini kırarsa pozisyon kapatılır.",
        }

        human_msg = (
            f"Sembol: {symbol}\n\n"
            f"Ajan Görüşleri:\n"
            f"- Teknik Ajan: {tech_signal} (güven: {random.uniform(0.6,0.9):.2f})\n"
            f"- Haber Ajanı: {news_signal} (güven: {random.uniform(0.6,0.9):.2f})\n"
            f"- Makro Ajan: {macro_signal} (güven: {random.uniform(0.5,0.85):.2f})\n"
            f"- Sentiment Ajanı: {sent_signal} (güven: {random.uniform(0.5,0.8):.2f})\n"
            f"- Risk Yöneticisi: {risk_decision} (güven: {random.uniform(0.65,0.95):.2f})\n"
        )

        example = {
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPTS["decision"]},
                {"from": "human", "value": human_msg},
                {"from": "gpt", "value": json.dumps(output, ensure_ascii=False)},
            ]
        }
        examples.append(example)

    logger.info("Karar ajanı: {} sentetik örnek üretildi.", len(examples))
    return examples


# ── Dataset Kaydetme ve Bölme ──────────────────────────────────────────────────

def save_jsonl(examples: list[dict], path: Path) -> None:
    """Örnekleri JSONL formatında kaydeder."""
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info("Kaydedildi: {} ({} örnek)", path, len(examples))


def split_dataset(
    examples: list[dict],
    train_ratio: float = 0.85,
) -> tuple[list[dict], list[dict]]:
    """Train/validation bölmesi (karıştırılmış)."""
    shuffled = examples.copy()
    random.shuffle(shuffled)
    split = int(len(shuffled) * train_ratio)
    return shuffled[:split], shuffled[split:]


# ── Ana Pipeline ───────────────────────────────────────────────────────────────

def build_all_datasets(
    symbols: list[str] | None = None,
    intervals: list[str] | None = None,
) -> dict[str, int]:
    """
    Tüm ajan türleri için dataset oluşturur ve JSONL olarak kaydeder.

    Returns:
        Her ajan türü için üretilen örnek sayısı
    """
    init_db()
    symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    intervals = intervals or ["1h", "4h", "1d"]

    all_technical: list[dict] = []
    for symbol in symbols:
        for interval in intervals:
            ex = build_technical_examples(symbol, interval, limit=500, step=8)
            all_technical.extend(ex)

    all_news = build_news_examples(limit=500)
    all_risk = build_risk_examples(n_synthetic=300)
    all_decision = build_decision_examples(n_synthetic=200)

    # Tüm örnekleri birleştir (karma eğitim için)
    combined = all_technical + all_news + all_risk + all_decision
    random.shuffle(combined)

    # Ayrı ajan datasetleri
    datasets = {
        "technical": all_technical,
        "news": all_news,
        "risk": all_risk,
        "decision": all_decision,
        "combined": combined,
    }

    counts: dict[str, int] = {}
    for name, examples in datasets.items():
        if not examples:
            logger.warning("{} dataset boş, atlandı.", name)
            continue
        train, val = split_dataset(examples)
        save_jsonl(train, DATASETS_DIR / f"{name}_train.jsonl")
        save_jsonl(val, DATASETS_DIR / f"{name}_val.jsonl")
        counts[name] = len(examples)

    logger.info("Dataset pipeline tamamlandı. Toplam: {} örnek", sum(counts.values()))
    return counts


# ── Yardımcılar ───────────────────────────────────────────────────────────────

def _nan(val) -> bool:
    import math
    try:
        return val is None or math.isnan(float(val))
    except (TypeError, ValueError):
        return True


def _build_reasoning(signal: str, key_points: list[str], summary: str) -> str:
    direction = {"bullish": "yükseliş", "bearish": "düşüş", "neutral": "yatay"}.get(signal, "nötr")
    points_str = "; ".join(key_points[:3]) if key_points else "indikatörler karışık"
    return f"Teknik göstergeler {direction} yönünde: {points_str}."
