"""
Haber Sentiment Dataset Builder

Üç piyasadan (BIST, US, Kripto) haber başlıkları toplar,
fiyat hareketiyle otomatik etiketler ve Qwen fine-tune formatına dönüştürür.

Etiketleme mantığı:
  Sonraki gün getiri >= +1.5%  → "olumlu"
  Sonraki gün getiri <= -1.5%  → "olumsuz"
  Arasında                     → "notr"
  Fiyat verisi yoksa keyword fallback.

Çalıştır:
    py -3.11 finetune/news_sentiment_builder.py --symbols all --out finetune/datasets/sentiment_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import yfinance as yf
from loguru import logger

from agents.sentiment import analyze_keyword

# ── Sembol listeleri ──────────────────────────────────────────────────────────

BIST_SYMS = [
    "THYAO.IS", "GARAN.IS", "AKBNK.IS", "ASELS.IS", "EREGL.IS",
    "KCHOL.IS", "SISE.IS", "YKBNK.IS", "BIMAS.IS", "ARCLK.IS",
    "TUPRS.IS", "PGSUS.IS", "FROTO.IS", "TOASO.IS", "SAHOL.IS",
    "HALKB.IS", "VAKBN.IS", "KOZAL.IS", "CIMSA.IS", "TCELL.IS",
    "DOHOL.IS", "EKGYO.IS", "ENKAI.IS", "MGROS.IS", "TAVHL.IS",
    "ULKER.IS", "VESTL.IS", "KORDS.IS", "PETKM.IS", "TTKOM.IS",
]

US_SYMS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "AMD", "NFLX", "INTC", "JPM", "BAC", "GS", "V", "MA",
    "JNJ", "PFE", "MRNA", "XOM", "CVX", "WMT", "COST", "DIS",
    "SBUX", "BA", "CAT", "GE", "F", "GM", "UBER",
]

CRYPTO_SYMS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
    "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "MATIC-USD",
    "LINK-USD", "UNI-USD", "LTC-USD", "ATOM-USD", "NEAR-USD",
]

# ── Sistem promptu ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Sen uzman bir finansal haber sentiment analistsin. "
    "Sana verilen haber basligini veya ozetini analiz et ve "
    "yatirimc etkisi acisindan siniflandir. "
    "Yanit olarak sadece tek kelime yaz: 'olumlu', 'olumsuz' veya 'notr'."
)

# Cevap normalizasyonu
LABEL_MAP = {
    "bullish":  "olumlu",
    "bearish":  "olumsuz",
    "neutral":  "notr",
    "olumlu":   "olumlu",
    "olumsuz":  "olumsuz",
    "notr":     "notr",
}


# ── Veri çekici ──────────────────────────────────────────────────────────────

def _fetch_price_change(symbol: str) -> float | None:
    """Dünden bugüne fiyat değişimi (%) döndürür."""
    try:
        df = yf.download(symbol, period="5d", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 2:
            return None
        close = df["Close"] if "Close" in df.columns else df[("Close", symbol)]
        close = close.dropna()
        if len(close) < 2:
            return None
        return float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
    except Exception:
        return None


def _price_label(pct: float | None, threshold: float = 1.5) -> str | None:
    """Fiyat değişiminden etiket üretir."""
    if pct is None:
        return None
    if pct >= threshold:
        return "olumlu"
    if pct <= -threshold:
        return "olumsuz"
    return "notr"


def _keyword_label(text: str) -> str:
    result = analyze_keyword(text)
    return LABEL_MAP.get(result.label, "notr")


def _fetch_news_for_symbol(symbol: str, max_items: int = 15) -> list[dict]:
    """Yahoo Finance RSS + yfinance news endpoint."""
    items: list[dict] = []

    # Yahoo RSS
    try:
        from data.sources.yahoo_news import fetch_yahoo_news
        rss = fetch_yahoo_news(symbol, max_items=max_items)
        for it in rss:
            title = it.get("title", "").strip()
            desc  = it.get("description", "").strip()
            if title:
                items.append({"text": f"{title}. {desc}".strip(), "source": "rss"})
    except Exception:
        pass

    # yfinance news
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news or []
        for n in news[:max_items]:
            title = n.get("title", "").strip()
            if title and not any(i["text"].startswith(title) for i in items):
                items.append({"text": title, "source": "yf"})
    except Exception:
        pass

    return items[:max_items]


# ── Sentetik veri ─────────────────────────────────────────────────────────────

SYNTHETIC_EXAMPLES: list[tuple[str, str]] = [
    # Olumlu
    ("Sirket karini beklentilerin yuzde 20 uzerinde acikladi", "olumlu"),
    ("Merkez bankasi faiz indirdi, piyasalar yukseldi", "olumlu"),
    ("Rekor temettü odeme duyuruldu", "olumlu"),
    ("Güçlü satis rakamları açıklandı, analistler hedef fiyatı yükseltti", "olumlu"),
    ("Stratejik ortaklık anlaşması imzalandı", "olumlu"),
    ("Hisse geri alım programı duyuruldu", "olumlu"),
    ("Ihracat rakamlari beklentilerin uzerinde geldi", "olumlu"),
    ("Analist notu yukseltildi, AL tavsiyesi verildi", "olumlu"),
    ("FDA onayı alındı, ürün piyasaya sürülüyor", "olumlu"),
    ("Q3 geliri yıllık bazda yüzde 35 arttı", "olumlu"),
    ("Stock beats earnings estimates, revenue surges", "olumlu"),
    ("Company announces record quarterly profit", "olumlu"),
    ("Deal signed worth billions, shares rally", "olumlu"),
    ("Strong guidance for next quarter issued", "olumlu"),
    ("Bitcoin breaks all-time high, bull run continues", "olumlu"),
    ("ETF approval boosts crypto market sentiment", "olumlu"),
    # Olumsuz
    ("Sirket iflasini acikladi", "olumsuz"),
    ("Zararda kalan sirket 10 bin calisan cikariyor", "olumsuz"),
    ("Beklentilerin altında kalan kar açıklandı", "olumsuz"),
    ("Soruşturma başlatıldı, hisse senetleri çakıldı", "olumsuz"),
    ("Faiz artışı beklentilerinin ardından piyasalar geriledi", "olumsuz"),
    ("Döviz krizinin etkisiyle satışlar düştü", "olumsuz"),
    ("Vergi cezası kesildi, şirket açıklama yaptı", "olumsuz"),
    ("Rekor enflasyon rakamları açıklandı", "olumsuz"),
    ("Analist notu düşürüldü, SAT tavsiyesi verildi", "olumsuz"),
    ("Ürün geri çağırma kararı alındı", "olumsuz"),
    ("Company misses revenue targets, stock falls sharply", "olumsuz"),
    ("SEC launches investigation into accounting practices", "olumsuz"),
    ("Massive layoffs announced amid declining sales", "olumsuz"),
    ("Crypto exchange hacked, millions stolen", "olumsuz"),
    ("Regulatory crackdown on crypto markets", "olumsuz"),
    ("Bitcoin plunges 20% on recession fears", "olumsuz"),
    # Nötr
    ("Yönetim kurulu toplantısı yapıldı", "notr"),
    ("Şirket genel kurul tarihini açıkladı", "notr"),
    ("Piyasalar bugün karma bir seyir izledi", "notr"),
    ("Analistler hisse senedi için beklentilerini korudu", "notr"),
    ("Şirket yeni CFO atadı", "notr"),
    ("Borsa bugün saat 18:00'de kapanacak", "notr"),
    ("Markets trade sideways ahead of Fed decision", "notr"),
    ("Company maintains full-year guidance unchanged", "notr"),
    ("CEO to present at industry conference next week", "notr"),
    ("Crypto markets consolidate after recent moves", "notr"),
]


def _build_example(text: str, label: str) -> dict:
    """ShareGPT formatında örnek üretir."""
    text = text.strip()
    if len(text) > 300:
        text = text[:300] + "..."
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human",  "value": f"Haber: {text}"},
            {"from": "gpt",    "value": label},
        ]
    }


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

def build_dataset(
    markets: list[str],
    out_path: Path,
    val_split: float = 0.15,
    threshold: float = 1.5,
) -> None:
    """
    Haber sentiment dataseti oluşturur ve JSONL olarak kaydeder.

    Args:
        markets: ["bist", "us", "crypto"] kombinasyonu
        out_path: Çıktı dosya yolu (train) — val otomatik türetilir
        val_split: Validasyon oranı
        threshold: Etiket için fiyat değişim eşiği (%)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    val_path = out_path.with_name(out_path.stem.replace("train", "val") + ".jsonl")

    all_syms: list[str] = []
    if "bist"   in markets: all_syms += BIST_SYMS
    if "us"     in markets: all_syms += US_SYMS
    if "crypto" in markets: all_syms += CRYPTO_SYMS

    examples: list[dict] = []
    label_counts = {"olumlu": 0, "olumsuz": 0, "notr": 0}

    logger.info("Haber toplaniyor: {} sembol", len(all_syms))

    for i, sym in enumerate(all_syms):
        logger.info("[{}/{}] {}", i + 1, len(all_syms), sym)
        news = _fetch_news_for_symbol(sym, max_items=15)
        price_pct = _fetch_price_change(sym)

        for item in news:
            text = item["text"]
            if not text or len(text) < 10:
                continue

            # Fiyat değişiminden etiket (birincil)
            label = _price_label(price_pct, threshold)
            # Kelime tabanlı fallback
            if label is None:
                label = _keyword_label(text)

            ex = _build_example(text, label)
            examples.append(ex)
            label_counts[label] = label_counts.get(label, 0) + 1

        time.sleep(0.3)  # rate limit

    # Sentetik örnekler ekle (çeşitlilik ve sınıf dengesi için)
    logger.info("Sentetik ornekler ekleniyor: {}", len(SYNTHETIC_EXAMPLES))
    for text, label in SYNTHETIC_EXAMPLES:
        for _ in range(3):  # her örneği 3 kez ekle (augmentasyon)
            examples.append(_build_example(text, label))
            label_counts[label] = label_counts.get(label, 0) + 3

    logger.info(
        "Toplam ornek: {} | Olumlu: {} | Olumsuz: {} | Notr: {}",
        len(examples), label_counts["olumlu"], label_counts["olumsuz"], label_counts["notr"],
    )

    # Karıştır ve böl
    random.shuffle(examples)
    split_idx = int(len(examples) * (1 - val_split))
    train_ex = examples[:split_idx]
    val_ex   = examples[split_idx:]

    def _write(path: Path, data: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        logger.success("{} yazildi ({} ornek)", path, len(data))

    _write(out_path, train_ex)
    _write(val_path, val_ex)

    logger.success(
        "Dataset hazir! Train: {} | Val: {} ornek",
        len(train_ex), len(val_ex),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Haber Sentiment Dataset Builder")
    parser.add_argument(
        "--symbols", default="all",
        choices=["all", "bist", "us", "crypto"],
        help="Hangi piyasa (varsayilan: all)"
    )
    parser.add_argument(
        "--out", default="finetune/datasets/sentiment_train.jsonl",
        help="Cikti JSONL dosya yolu"
    )
    parser.add_argument(
        "--threshold", type=float, default=1.5,
        help="Etiket icin fiyat degisim esigi %% (varsayilan: 1.5)"
    )
    args = parser.parse_args()

    markets = ["bist", "us", "crypto"] if args.symbols == "all" else [args.symbols]
    build_dataset(markets=markets, out_path=Path(args.out), threshold=args.threshold)


if __name__ == "__main__":
    main()
