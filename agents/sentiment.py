"""Haber sentiment analizi — keyword tabanlı + LLM destekli sınıflandırma."""

import re
from dataclasses import dataclass, field
from typing import Literal

from loguru import logger

SentimentLabel = Literal["bullish", "bearish", "neutral"]


@dataclass
class SentimentResult:
    label: SentimentLabel
    score: float           # -1.0 (çok bearish) → +1.0 (çok bullish)
    confidence: float      # 0.0 → 1.0
    matched_keywords: list[str] = field(default_factory=list)
    method: str = "keyword"


# ── Keyword sözlükleri ───────────────────────────────────────────────────────

_BULLISH_KEYWORDS = [
    # Türkçe
    "artış", "yükseliş", "rekor", "büyüme", "kâr", "kazanç", "olumlu",
    "güçlü", "toparlanma", "ivme", "beklenti aştı", "üzerinde",
    "temettü", "hisse geri alım", "ortaklık", "anlaşma", "ihracat",
    "sipariş", "kapasite artışı", "faiz indirimi", "teşvik",
    # İngilizce
    "growth", "profit", "beat", "strong", "rally", "surge", "jump",
    "record", "upgrade", "buy", "bullish", "outperform", "positive",
    "recovery", "partnership", "deal", "contract", "expansion",
]

_BEARISH_KEYWORDS = [
    # Türkçe
    "düşüş", "gerileme", "zarar", "kayıp", "olumsuz", "zayıf",
    "endişe", "risk", "beklenti altında", "altında", "kriz", "iflas",
    "soruşturma", "ceza", "vergi", "daralma", "durgunluk", "enflasyon",
    "faiz artışı", "döviz krizi", "devalüasyon", "işten çıkarma",
    # İngilizce
    "loss", "decline", "fall", "drop", "weak", "miss", "below",
    "downgrade", "sell", "bearish", "underperform", "negative",
    "recession", "inflation", "layoff", "fine", "lawsuit", "fraud",
]

_STRONG_BULLISH = {"rekor", "record", "beat", "beklenti aştı", "surge", "jump"}
_STRONG_BEARISH = {"iflas", "fraud", "soruşturma", "lawsuit", "kriz", "crisis"}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def analyze_keyword(text: str) -> SentimentResult:
    """Keyword sayımına dayalı hızlı sentiment analizi."""
    norm = _normalize(text)

    bull_matches = [kw for kw in _BULLISH_KEYWORDS if kw in norm]
    bear_matches = [kw for kw in _BEARISH_KEYWORDS if kw in norm]

    # Güçlü sinyal bonusu
    bull_score = sum(2.0 if kw in _STRONG_BULLISH else 1.0 for kw in bull_matches)
    bear_score = sum(2.0 if kw in _STRONG_BEARISH else 1.0 for kw in bear_matches)

    total = bull_score + bear_score
    if total == 0:
        return SentimentResult("neutral", 0.0, 0.5, [], "keyword")

    net = (bull_score - bear_score) / max(total, 1)
    confidence = min(total / 5.0, 1.0)  # 5+ eşleşme → yüksek güven

    if net > 0.15:
        label: SentimentLabel = "bullish"
    elif net < -0.15:
        label = "bearish"
    else:
        label = "neutral"

    return SentimentResult(
        label=label,
        score=round(net, 3),
        confidence=round(confidence, 3),
        matched_keywords=bull_matches + bear_matches,
        method="keyword",
    )


class QwenSentimentAnalyzer:
    """
    Fine-tune edilmis Qwen sentiment modeli (lora_weights_sentiment/).
    Model yoksa otomatik keyword fallback kullanir.
    """

    _instance: "QwenSentimentAnalyzer | None" = None
    LORA_PATH = "lora_weights_sentiment"

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._load()

    @classmethod
    def get(cls) -> "QwenSentimentAnalyzer":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self) -> None:
        from pathlib import Path
        lora = Path(self.LORA_PATH)
        if not lora.exists():
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel

            self._tokenizer = AutoTokenizer.from_pretrained(str(lora), trust_remote_code=True)
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            base = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-1.5B-Instruct",
                quantization_config=bnb,
                device_map="auto",
                trust_remote_code=True,
            )
            self._model = PeftModel.from_pretrained(base, str(lora))
            self._model.eval()
            # do_sample=False ile uyumsuz parametreleri temizle
            self._model.generation_config.temperature = None
            self._model.generation_config.top_p = None
            self._model.generation_config.top_k = None
            self._loaded = True
            logger.info("Sentiment modeli yuklendi: {}", lora)
        except Exception as e:
            logger.debug("Sentiment model yuklenemedi: {}", e)

    def analyze(self, text: str) -> SentimentResult:
        """Metni Qwen ile analiz eder, model yoksa keyword fallback."""
        if not self._loaded:
            return analyze_keyword(text)

        try:
            import torch

            SYSTEM = (
                "Sen uzman bir finansal haber sentiment analistsin. "
                "Haberi analiz et. Sadece tek kelime yaz: 'olumlu', 'olumsuz' veya 'notr'."
            )
            messages = [
                {"role": "system",    "content": SYSTEM},
                {"role": "user",      "content": f"Haber: {text[:300]}"},
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()

            label_tr_map = {"olumlu": "bullish", "olumsuz": "bearish", "notr": "neutral"}
            if "olumlu" in raw:
                lbl, score = "bullish", 0.85
            elif "olumsuz" in raw:
                lbl, score = "bearish", -0.85
            else:
                lbl, score = "neutral", 0.0

            return SentimentResult(label=lbl, score=score, confidence=0.9, method="qwen")  # type: ignore

        except Exception as e:
            logger.debug("Qwen sentiment hatasi: {}", e)
            return analyze_keyword(text)


def analyze(text: str) -> SentimentResult:
    """
    Tek metin analizi — Qwen modeli varsa kullanır, yoksa keyword fallback.
    Bu fonksiyonu kullan: hem eğitilmiş hem eğitilmemiş durumda çalışır.
    """
    return QwenSentimentAnalyzer.get().analyze(text)


def analyze_batch(news_list: list[dict]) -> list[dict]:
    """
    Haber listesine sentiment ekle.

    Args:
        news_list: fetch_all_feeds() çıktısı — {"payload": {"title":..., "summary":...}, ...}

    Returns:
        Her habere "sentiment" alanı eklenmiş liste
    """
    results = []
    for item in news_list:
        payload = item.get("payload", {})
        text = f"{payload.get('title', '')} {payload.get('summary', '')}"
        sentiment = analyze(text)
        enriched = {
            **item,
            "sentiment": {
                "label":    sentiment.label,
                "score":    sentiment.score,
                "confidence": sentiment.confidence,
                "keywords": sentiment.matched_keywords,
            },
        }
        results.append(enriched)
    return results


def filter_actionable(
    news_list: list[dict],
    min_confidence: float = 0.4,
    labels: list[SentimentLabel] | None = None,
) -> list[dict]:
    """
    Yalnızca aksiyon alınabilir haberleri döner.

    Args:
        news_list: analyze_batch() çıktısı
        min_confidence: Minimum güven eşiği
        labels: ["bullish"] veya ["bearish"] veya ikisi birden
    """
    labels = labels or ["bullish", "bearish"]
    return [
        item for item in news_list
        if item.get("sentiment", {}).get("label") in labels
        and item.get("sentiment", {}).get("confidence", 0) >= min_confidence
    ]


def summarize_sentiment(news_list: list[dict]) -> dict:
    """
    Haber listesinden genel piyasa sentiment özeti üretir.

    Returns:
        {"overall": "bullish"|"bearish"|"neutral", "score": float,
         "bullish_count": int, "bearish_count": int, "neutral_count": int}
    """
    if not news_list:
        return {"overall": "neutral", "score": 0.0,
                "bullish_count": 0, "bearish_count": 0, "neutral_count": 0}

    counts = {"bullish": 0, "bearish": 0, "neutral": 0}
    scores = []

    for item in news_list:
        s = item.get("sentiment", {})
        label = s.get("label", "neutral")
        counts[label] = counts.get(label, 0) + 1
        scores.append(s.get("score", 0.0))

    avg_score = sum(scores) / len(scores) if scores else 0.0

    if avg_score > 0.1:
        overall: SentimentLabel = "bullish"
    elif avg_score < -0.1:
        overall = "bearish"
    else:
        overall = "neutral"

    return {
        "overall": overall,
        "score": round(avg_score, 3),
        "bullish_count": counts["bullish"],
        "bearish_count": counts["bearish"],
        "neutral_count": counts["neutral"],
    }
