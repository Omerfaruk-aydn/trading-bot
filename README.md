# Trading Bot — AI Destekli Multi-Market Trader

```
╔══════════════════════════════════════════════════════════╗
║         TRADING BOT — AI Destekli Multi-Market Trader    ║
║    BIST · Kripto · Futures · VİOP · NYSE/NASDAQ · LLM    ║
╚══════════════════════════════════════════════════════════╝
```

Qwen 2.5-1.5B tabanlı yerel LLM, XGBoost ML sinyal motoru ve 13+ teknik indikatörü birleştiren otonom trading botu. Kağıt trading (simülasyon) modunda çalışır — gerçek para transferi yapmaz.

---

## İçindekiler

1. [Özellikler](#özellikler)
2. [Mimari](#mimari)
3. [Gereksinimler](#gereksinimler)
4. [Kurulum](#kurulum)
5. [Yapılandırma](#yapılandırma)
6. [LLM Kurulumu (Qwen)](#llm-kurulumu-qwen)
7. [Kullanım — CLI](#kullanım--cli)
8. [ML Modeli Eğitimi](#ml-modeli-eğitimi)
9. [Backtest](#backtest)
10. [Proje Yapısı](#proje-yapısı)
11. [Sinyal Motoru](#sinyal-motoru)
12. [Risk Yönetimi](#risk-yönetimi)
13. [Performans Gerçekçiliği](#performans-gerçekçiliği)

---

## Özellikler

### Desteklenen Piyasalar
| Piyasa | Sembol Formatı | Örnek |
|---|---|---|
| Borsa İstanbul (BIST) | `TICKER.IS` | `THYAO.IS`, `GARAN.IS` |
| Kripto Spot | `COIN-USD` | `BTC-USD`, `ETH-USD` |
| Kripto Futures (Binance) | `COIN-PERP` | `BTC-PERP`, `ETH-PERP` |
| VİOP Vadeli | `CONTRACT-FUT` | `XU030-FUT`, `USDTRY-FUT` |
| NYSE / NASDAQ | `TICKER` | `AAPL`, `NVDA`, `MSFT` |

### Sinyal Kaynakları
- **13 Teknik İndikatör** — RSI, MACD, EMA hizası, Bollinger Bantları, ADX, Stochastic, CMF, OBV, ATR, Fibonacci, Destek/Direnç, Mum Formasyonları, Haftalık Trend
- **XGBoost ML Modeli** — piyasa başına (BIST/US/Kripto) ayrı eğitilmiş tahmin modeli
- **Bayesian Ensemble** — tüm sinyalleri log-odds birleştirmesiyle ağırlıklandırır
- **Çoklu Zaman Dilimi (MTF)** — 15dk + 1s + 4s + 1g uyum onayı
- **Haber Sentiment** — RSS beslemelerinden gerçek zamanlı Türkçe/İngilizce analiz
- **On-Chain Veri (Kripto)** — Binance API: funding rate, open interest, long/short oranı
- **Piyasa Rejim Dedektörü** — Trend/Yatay/Panik otomatik tespiti
- **KAP İçeriden Öğrenme** — BIST için KAP bildirimleri
- **Ekonomik Takvim** — Yüksek etkili olaylarda pozisyon engeli

### LLM Kullanım Mimarisi
Qwen 2.5-1.5B modeli **strateji seçici** olarak kullanılır (karar verici değil):
```
Qwen → "Piyasa trend mi, range mi, panik mi?"
         ↓
Deterministik strateji (TrendFollowing / MeanReversion / NoTrade)
         ↓
Bayesian Ensemble → final güven skoru
```

### Diğer Özellikler
- Telegram bildirimleri (al/sat/stop/hedef)
- SQLite işlem kaydı ve portföy snapshot
- Dinamik evren modu (piyasaları otomatik tarar, en hareketlileri seçer)
- Kelly Criterion + ATR tabanlı dinamik pozisyon boyutu
- Trailing stop-loss
- Korelasyon filtresi (>0.72 korelasyonlu çift pozisyon engeli)
- Gerçek zamanlı WebSocket fiyat akışı (Binance)
- Walk-forward kalibrasyon backtest

---

## Mimari

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI (cli.py)                        │
│  start │ analyze │ scan │ backtest │ train-ml │ chat │ ...  │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │   LLMTradingAgent       │
          │  (llm_trading_agent.py) │
          └──┬───────┬──────┬──────┘
             │       │      │
    ┌─────────▼──┐ ┌──▼──────────┐ ┌──────────────────┐
    │  Snapshot  │ │  Ensemble   │ │  Pozisyon Yönt.  │
    │ Collector  │ │  (Bayesian) │ │  Buy/Sell/Stop   │
    └─────────┬──┘ └──┬──────────┘ └──────────────────┘
              │        │
    ┌──────────▼──┐  ┌─▼──────────────────────────────────┐
    │  Teknik    │  │  Sinyal Kaynakları                  │
    │ İndikatör  │  │  ┌──────────┐ ┌──────┐ ┌────────┐  │
    │ (13 adet)  │  │  │ XGBoost  │ │ MTF  │ │On-Chain│  │
    └────────────┘  │  │   ML     │ │ 15m  │ │Binance │  │
                    │  │ Modeli   │ │  1h  │ │ API    │  │
    ┌────────────┐  │  └──────────┘ │  4h  │ └────────┘  │
    │  Qwen LLM  │  │  ┌──────────┐ └──────┘             │
    │  Bağlam    │  │  │Sentiment │ ┌──────────────────┐  │
    │  Sınıf.    │  │  │  Analiz  │ │  Piyasa Rejimi   │  │
    └────────────┘  │  └──────────┘ │  Dedektörü       │  │
                    │               └──────────────────┘  │
                    └────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │              Veri Katmanı                           │
    │  yfinance │ Binance WS │ RSS │ KAP │ EconomicCal.  │
    └─────────────────────────────────────────────────────┘
```

---

## Gereksinimler

### Yazılım
- Python **3.11** veya üstü
- CUDA destekli GPU (opsiyonel ama önerilir — CPU'da Qwen çok yavaş)
- Windows 10/11 veya Linux

### Donanım
| Senaryo | Minimum | Önerilen |
|---|---|---|
| Sadece teknik analiz (ML, LLM yok) | 4 GB RAM | 8 GB RAM |
| XGBoost ML + teknik | 8 GB RAM | 16 GB RAM |
| Qwen 2.5-1.5B (CPU) | 8 GB RAM | 16 GB RAM |
| Qwen 2.5-1.5B (GPU, 4-bit) | 4 GB VRAM | 6 GB VRAM |

### Python Kütüphaneleri
```
yfinance>=0.2.40      # Piyasa verisi
pandas>=2.0           # Veri işleme
numpy                 # Sayısal hesaplama
xgboost>=2.0          # ML modeli
scikit-learn          # Özellik mühendisliği
transformers>=4.40    # Qwen LLM
peft>=0.10            # LoRA adapter
bitsandbytes          # 4-bit quantization (GPU)
torch>=2.2            # PyTorch
loguru                # Loglama
rich                  # Terminal UI
requests              # HTTP / Binance API
websockets            # Binance WebSocket
python-dotenv         # .env konfigürasyon
```

---

## Kurulum

```bash
# 1. Repoyu klon
git clone <repo-url>
cd trading_bot

# 2. Sanal ortam oluştur
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. Loglar klasörünü oluştur
mkdir logs
```

---

## Yapılandırma

### `.env` Dosyası

Proje kökündeki `.env.example` dosyasını `.env` olarak kopyalayın:

```bash
copy .env.example .env   # Windows
cp .env.example .env     # Linux
```

`.env` içeriği:

```env
# ── Telegram (opsiyonel — bildirim almak için) ────────────────
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# ── Binance (sadece WebSocket fiyat akışı için — API key GEREKMİYOR) ──
BINANCE_API_KEY=
BINANCE_API_SECRET=

# ── Loglama ───────────────────────────────────────────────────
LOG_LEVEL=INFO

# ── Paper Trading (her zaman true kalmalı — gerçek işlem YOK) ─
PAPER_TRADING=true
```

> **Önemli:** `PAPER_TRADING=true` olduğu sürece bot gerçek para transferi yapmaz.  
> Binance API anahtarı olmadan on-chain veri (funding rate vb.) çalışır çünkü Binance'in kamuya açık endpoint'leri kullanılır.

### Telegram Bildirimi Kurulumu (Opsiyonel)

1. [@BotFather](https://t.me/BotFather) ile yeni bot oluştur → `BOT_TOKEN` al
2. Bota `/start` gönder, ardından `https://api.telegram.org/bot<TOKEN>/getUpdates` ile `chat_id` bul
3. `.env` dosyasına ekle

---

## LLM Kurulumu (Qwen)

Bot, **Qwen 2.5-1.5B-Instruct** modelini kullanır. İki seçenek:

### Seçenek A: Ham Model (Qwen yorum olmadan)
```bash
# Qwen indirmeden sadece teknik analiz + ML ile başlatabilirsiniz
# analyze komutuna --no-qwen ekleyin
analyze --symbol THYAO --no-qwen
```

### Seçenek B: LoRA Fine-Tune Edilmiş Model (Tam özellik)

**Kaggle / HuggingFace'den LoRA adaptörünü indirin ve `lora_weights/` klasörüne koyun:**

```
lora_weights/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── generation_config.json
```

### Seçenek C: Kendi Modelinizi Fine-Tune Edin

```bash
# 1. Sentiment veri seti oluştur
train-sentiment --symbols all

# Bu komut şunları yapar:
#   - Bloomberg HT, Reuters, Dünya Gazetesi, CNBC-e RSS'lerinden haber toplar
#   - Qwen 2.5-1.5B üzerinde LoRA ile fine-tune yapar (3 epoch)
#   - Adapter'ı lora_weights/ klasörüne kaydeder
```

---

## Kullanım — CLI

Botu başlatın:

```bash
python cli.py
```

Karşılama ekranı gelir. Komutları aşağıdaki gibi yazabilirsiniz:

---

### `start` — Trading Agent Başlat

```bash
# Sadece BIST (8 büyük hisse, 100.000 TL)
start

# Kripto futures, 10x kaldıraç
start --futures --leverage 10 --capital 10000

# BIST + Kripto spot
start --symbols THYAO GARAN ASELS --crypto

# Sadece ABD hisseleri
start --us --capital 50000 --mode conservative

# Tüm piyasalar — dinamik evren modu
start --universe --capital 100000 --top-n 20

# Agresif mod, 20x futures, 5000 TL
start --futures --leverage 20 --capital 5000 --mode aggressive
```

**Risk Modları:**
| Mod | Min Güven | Maks Pozisyon | Stop Loss | Take Profit |
|---|---|---|---|---|
| `conservative` | %70 | %10 | %4 | %5 |
| `normal` | %55 | %15 | %3 | %6 |
| `aggressive` | %25 | %20 | %2 | %8 |
| `scalping` | %50 | %5 | %1 | %2 |

---

### `analyze` — Derin Teknik Analiz

```bash
# BIST hissesi (2 yıllık grafik + Qwen yorum)
analyze --symbol THYAO

# Kripto
analyze --symbol BTC-USD

# ABD hissesi
analyze --symbol AAPL --period 1y

# Qwen yorumu olmadan (daha hızlı)
analyze --symbol GARAN --no-qwen
```

**Çıktı içeriği:**
- 2 yıllık fiyat grafiği (ASCII)
- 13 teknik indikatör analizi (bullish/bearish sıralanmış)
- Destek/Direnç seviyeleri
- Fibonacci geri çekilme noktaları
- ML sinyali ve güven skoru
- Piyasa rejimi
- Qwen 2.5 yorumu

---

### `scan` — Piyasa Taraması

```bash
# BIST'te en güçlü AL sinyalleri (varsayılan)
scan

# Kripto'da SAT sinyalleri
scan --market crypto --signal sell

# Tüm piyasalar, tüm sinyaller, top 20
scan --market all --signal all --top 20

# ABD hisseleri
scan --market us --signal buy --top 10
```

**Tarama kriterleri:** 13 indikatörden puanlama (+4 ve üzeri = AL, -3 ve altı = SAT)

---

### `status` — Portföy Durumu

```bash
status
```

Açık pozisyonlar, P&L, hedef ilerleme, futures/VİOP tablolarını gösterir.

---

### `movers` — Piyasa Hareketlileri

```bash
movers
```

Tüm piyasalarda (ABD, Kripto, BIST) o gün en çok kazanan, kaybeden ve en aktif hisseleri gösterir.

---

### `train-ml` — ML Modeli Eğit

```bash
# ABD piyasası (varsayılan)
train-ml

# BIST için 5 yıllık veri, 3 günlük hedef
train-ml --market bist --period 5y --horizon 3

# Kripto, hızlı test (ilk 50 sembol)
train-ml --market crypto --max-sym 50

# Tüm piyasalar
train-ml --market all --period 2y
```

Model `ml/models/signal_model_{market}.pkl` dosyasına kaydedilir.

---

### `ml-predict` — ML Sinyal Göster

```bash
# Varsayılan semboller
ml-predict

# Özel semboller
ml-predict --symbols AAPL MSFT NVDA AMZN --threshold 0.60
```

---

### `backtest` — Geçmiş Testi

```bash
backtest

backtest --symbols THYAO GARAN AKBNK --period 6mo --mode aggressive
```

---

### `chat` — AI ile Sohbet

```bash
chat

chat --symbol THYAO.IS
```

Qwen modeli ile interaktif piyasa analizi.

---

## ML Modeli Eğitimi

### Özellik Seti (FEATURE_COLS)
Modelde kullanılan teknik özellikler:

| Kategori | Özellikler |
|---|---|
| Momentum | RSI, Stochastic K/D, CCI, Williams %R |
| Trend | EMA 9/21/55, SMA 20/50/200, ADX, MACD hist |
| Volatilite | ATR, Bollinger %B, Bollinger genişliği |
| Hacim | OBV, CMF, Hacim oranı (vol/vol_sma20) |
| Destek/Direnç | Pivot P/R1/R2/S1/S2, SR score |
| Pattern | Doji, Hammer, Engulfing formasyonları |

### Etiketleme Yöntemi
- `1 (AL)` = N gün sonra fiyat ≥ +%3
- `0 (BEKLE)` = diğer tüm durumlar

### Model Performansı (Tipik)
- AUC-ROC: 0.58–0.65
- Precision (AL sinyali): 0.52–0.58

> Not: Finansal piyasalarda AUC > 0.65 güvenilmez veya overfitting işareti kabul edilir.

---

## Backtest

### Walk-Forward Kalibrasyon Backtest

```bash
# Komut satırından doğrudan çalıştır
python -m backtest.walk_forward --symbol THYAO.IS --market bist
python -m backtest.walk_forward --symbol BTC-USD  --market crypto --period 2y
python -m backtest.walk_forward --symbol AAPL     --market us --hold 3
```

**Çıktı örneği:**
```
=======================================================
  WALK-FORWARD KALİBRASYON — THYAO.IS
  Toplam sinyal: 62 | Genel doğruluk: 56.5% | Sharpe: 0.87
=======================================================
  Güven Aralığı   Sinyal   Doğruluk  Durum
-------------------------------------------------------
  0-35%                0          -    -
  35-50%              18      50.0%    ~ Zayıf
  50-65%              31      58.1%    ~ Zayıf
  65-80%              11      63.6%    ✓ Güvenilir
  80-100%              2      50.0%    ✗ Kalibrasyon gerekli
=======================================================
```

Bu çıktı, `min_confidence` eşiğini hangi değere ayarlamanız gerektiğini söyler.

---

## Proje Yapısı

```
trading_bot/
│
├── cli.py                          # Ana CLI giriş noktası
├── main.py                         # Alternatif giriş noktası
├── .env                            # Konfigürasyon (git'e dahil DEĞİL)
├── .env.example                    # Örnek konfigürasyon
│
├── agents/                         # Trading ajanları
│   ├── llm_trading_agent.py        # Ana otonom agent
│   ├── analyzer.py                 # Tek sembol derin analiz
│   ├── scanner.py                  # Çok sembol tarama
│   ├── sentiment.py                # Haber sentiment (Qwen tabanlı)
│   ├── news_trigger.py             # Gerçek zamanlı haber tetikleyici
│   ├── context_classifier.py       # LLM bağlam sınıflandırıcı (YENİ)
│   ├── sandbox_trader.py           # Kağıt trading portföy simülasyonu
│   ├── backtest.py                 # Backtest runner
│   └── chat.py                     # Interaktif sohbet
│
├── ml/                             # Machine Learning
│   ├── ensemble.py                 # Bayesian log-odds ensemble (YENİ)
│   ├── predictor.py                # Canlı sinyal tahmincisi
│   ├── trainer.py                  # XGBoost eğitim
│   ├── features.py                 # Özellik mühendisliği
│   ├── data_pipeline.py            # ML veri hattı
│   ├── labeler.py                  # Etiketleme (AL/BEKLE)
│   ├── market_regime.py            # Piyasa rejim dedektörü
│   ├── support_resistance.py       # Destek/Direnç & Fibonacci
│   └── models/                     # Eğitilmiş modeller (.pkl)
│
├── strategies/                     # Deterministik stratejiler (YENİ)
│   ├── base.py                     # Temel strateji sınıfı
│   ├── trend_following.py          # Trend piyasası stratejisi
│   ├── mean_reversion.py           # Yatay piyasa stratejisi
│   └── no_trade.py                 # Panik rejimi (hiç işlem açma)
│
├── data/                           # Veri katmanı
│   ├── indicators.py               # 13+ teknik indikatör (saf pandas)
│   ├── collectors/                 # Veri toplayıcılar
│   │   ├── news_collector.py       # RSS haber beslemeleri
│   │   └── ...
│   ├── sources/                    # Harici veri kaynakları
│   │   ├── yahoo_news.py           # Yahoo Finance RSS
│   │   ├── market_screener.py      # Piyasa tarama
│   │   ├── onchain.py              # Binance on-chain API (YENİ)
│   │   └── stock_universe.py       # Sembol evren listeleri
│   ├── realtime/                   # Gerçek zamanlı akışlar
│   │   ├── binance_ws.py           # Kripto spot WebSocket
│   │   └── binance_futures_ws.py   # Futures mark price WebSocket
│   ├── markets/                    # Piyasaya özel modüller
│   │   ├── us_stocks.py            # NYSE/NASDAQ seans, USD/TL kur
│   │   └── viop.py                 # VİOP sözleşme özellikleri
│   ├── calendar/
│   │   └── economic_calendar.py    # Ekonomik takvim + KAP insider
│   └── db/
│       └── database.py             # SQLite işlem kaydı
│
├── backtest/                       # Backtest araçları
│   └── walk_forward.py             # Walk-forward kalibrasyon (YENİ)
│
├── finetune/                       # LLM fine-tuning
│   ├── sentiment_train.py          # Qwen LoRA sentiment eğitimi
│   ├── news_sentiment_builder.py   # Sentiment veri seti oluşturucu
│   └── ...
│
├── config/                         # Merkezi konfigürasyon
│   ├── settings.py                 # .env yükleyici
│   └── symbols.py                  # Sembol listeleri
│
├── utils/
│   └── telegram_bot.py             # Telegram bildirim gönderici
│
└── lora_weights/                   # Qwen LoRA adaptörü (git'te YOK)
    ├── adapter_model.safetensors
    ├── adapter_config.json
    └── tokenizer.json
```

---

## Sinyal Motoru

### Bayesian Ensemble Çalışma Prensibi

Geleneksel ağırlıklı ortalama yerine **log-odds birleştirmesi** kullanılır:

```
posterior_log_odds = prior_log_odds + Σ (reliability_i - 0.5) × 4 × log_odds(signal_i)
```

Bu yöntemin avantajları:
- ML modeli %50 civarında dolanırken ensemble'ı bozmaz (filtre uygulanır)
- Her sinyal kaynağının geçmiş güvenilirliği ağırlık belirler
- Birden fazla bağımsız sinyal matematiksel olarak doğru birikir

**Sinyal kaynağı güvenilirlik priorleri:**
| Kaynak | Güvenilirlik | Katkı Ağırlığı |
|---|---|---|
| Teknik (13 indikatör) | %62 | 0.48 |
| Çoklu zaman dilimi | %66 | 0.64 |
| Piyasa rejimi | %64 | 0.56 |
| XGBoost ML | %55 | 0.20 |
| On-chain (kripto) | %58 | 0.32 |
| Sentiment | %52 | 0.08 |

### Örnek Güven Hesabı

```
Senaryo: Güçlü BUY sinyali
  Teknik skor = 7/14  → conf = 0.67
  ML sinyali = AL, %72 → filtreden geçer (> %62)
  MTF: 15m=AL, 1h=AL, 4h=HOLD → 2/3 uyum

Sonuç (Bayesian): BUY @ %64 güven
  (Eski yöntemle: %22-28 geliyordu)
```

---

## Risk Yönetimi

### Otomatik Koruma Mekanizmaları

| Mekanizma | Tetikleyici | Eylem |
|---|---|---|
| Günlük kayıp limiti | Günlük P&L < -%5 | Tüm pozisyonları kapat, durdur |
| Panik rejimi | ATR > 4x normal | Yeni pozisyon açma |
| Yüksek etkili olay | Ekonomik takvim | Yeni pozisyon açma |
| Korelasyon filtresi | İki sembol > %72 korelasyon | İkincisini engelle |
| Likidasyon kontrolü | Futures mark price | Anında kapat |
| Günlük hedef | Portföy = hedef değer | Tüm pozisyonları kapat, durdur |

### Pozisyon Boyutu

**Spot:** Kelly Criterion (geçmiş işlem verisi bazlı) ile ATR tabanlı boyutlandırma:
```
risk_tl   = toplam_sermaye × %1         # her işlemde risk
sl_mesafe = ATR × 1.5                   # stop-loss mesafesi
hisse     = risk_tl / sl_mesafe         # kaç adet al
```

**Futures:** Margin = sermaye × `max_position_pct`, kaldıraçla büyütülür.

### Stop-Loss Türleri
- **ATR tabanlı:** 1.5 ATR mesafe (dinamik, volatiliteye uyum sağlar)
- **Trailing stop:** En yüksek fiyattan yüzde geri çekilme
- **Sabit yüzde:** Fallback (ATR hesaplanamadığında)

---

## Performans Gerçekçiliği

### Walk-Forward Backtest Sonuçları (1 Yıl, Günlük Veri)

| Senaryo | BTC-PERP | ETH-PERP | SOL-PERP |
|---|---|---|---|
| 20x leverage, sabit %2 SL | -%45 | -%44 | -%71 |
| 10x leverage, ATR-SL | -%10 | -%7 | -%18 |
| 5x leverage, ATR-SL (muhafazakâr) | **+%5** | -%3 | **+%7** |

### Neden Yüksek Kaldıraç Riskli?

```
20x leverage + %2 SL = Her yanlış tahminde -%40 margin kaybı
Win rate %31-45 ile pozitif EV olsa da pozisyon küçüldükçe
toparlanma giderek zorlaşır (bileşik etki).
```

### Gerçekçi Beklenti

- Teknik sinyaller günlük veride **%52-56** doğruluk oranı üretir
- Bu oran profesyonel trading için yeterince iyi ancak **yüksek kaldıraç tolere edemez**
- Önerilen kullanım: **5x veya altı kaldıraç**, seçici sinyaller (günde 1-3 işlem)
- Yıllık hedef: **+%5 ile +%20** aralığı (5x kaldıraçla, iyi piyasa koşullarında)

### Bu Bot Neye Göre Tasarlandı?

- **Öğrenme ve araştırma** — algoritmik trading kavramları pratikte görme
- **Kağıt trading** — gerçek para riski olmadan strateji test etme
- **Piyasa tarama** — `scan` komutu ile en güçlü sinyalleri hızlıca listele
- **Derin analiz** — `analyze` komutu ile tek hisse kapsamlı rapor

> **Uyarı:** Bu bot kağıt trading (simülasyon) aracıdır. Gerçek parayla kullanılması durumunda tüm finansal sorumluluk kullanıcıya aittir. Geçmiş performans gelecek sonuçları garanti etmez.

---

## Hızlı Başlangıç (5 Dakika)

```bash
# 1. Kurulum
git clone <repo-url> && cd trading_bot
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# 2. .env oluştur (Telegram olmadan da çalışır)
copy .env.example .env

# 3. ML modelini eğit (lora_weights olmadan çalışan özellik)
python cli.py
> train-ml --market us --period 1y --max-sym 50

# 4. Piyasayı tara (LLM gerekmez)
> scan --market bist --signal buy
> scan --market crypto --signal all

# 5. Tek hisse analizi (--no-qwen ile LLM olmadan)
> analyze --symbol THYAO --no-qwen
> analyze --symbol BTC-USD --no-qwen

# 6. Backtest
> backtest --symbols THYAO GARAN --period 3mo
```

---

## Geliştirme Notları

### Yeni Strateji Ekleme

```python
# strategies/my_strategy.py
from strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    name = "my_strategy"

    def decide(self, snap, ohlcv_df):
        # ... puanlama mantığı ...
        return "buy", 0.65, "Sebep açıklaması"

# strategies/__init__.py içine ekle:
_REGISTRY["my_regime"] = MyStrategy
```

### Yeni İndikatör Ekleme

```python
# data/indicators.py içine fonksiyon ekle
def my_indicator(df: pd.DataFrame) -> pd.Series:
    ...

# compute_all() içine ekle:
result["my_indicator"] = my_indicator(df)
```

### Walk-Forward Kalibrasyon

```bash
# Güven eşiklerini veriyle kalibre et
python -m backtest.walk_forward --symbol THYAO.IS --market bist --period 2y
python -m backtest.walk_forward --symbol BTC-USD  --market crypto --hold 3
```

---
