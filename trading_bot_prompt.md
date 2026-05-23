# AGENTIC MULTI-MARKET TRADING SYSTEM — TAM PROJE PROMPTU

## PROJE TANIMI

Sen kıdemli bir quant developer'sın. BIST, kripto ve forex piyasalarında çalışacak, çok ajanlı (multi-agent) bir LLM tabanlı trading sistemi inşa edeceksin. Sistem haber okur, teknik analiz yapar, makro veriyi yorumlar, risk hesaplar ve karar verir. Karar her zaman gerekçeli olmalı.

**Önemli:** Bu sistem önce simülasyon (paper trading) modunda çalışacak. Gerçek para ile bağlantı, kullanıcı açıkça onayladıktan sonra ayrı bir modüle aktarılacak.

---

## TEKNİK YIĞIN (TECH STACK)

- **Dil:** Python 3.11+
- **LLM Framework:** LangChain veya LangGraph (multi-agent için LangGraph tercih)
- **LLM Provider:** Yerel için Ollama (Llama 3.1, Qwen 2.5), bulut için OpenAI/Anthropic API (opsiyonel)
- **Veri:**
  - Kripto: `python-binance`, `ccxt`
  - BIST/Hisse: `yfinance`, KAP scraper (özel)
  - Forex: `oandapyV20` veya `yfinance`
  - Haber: `feedparser` (RSS), `newspaper3k`, `tweepy` (X API)
- **Teknik Analiz:** `pandas-ta` (birincil), `TA-Lib` (opsiyonel)
- **Veritabanı:** SQLite (geliştirme), PostgreSQL (üretim)
- **Backtest:** `backtesting.py` veya `vectorbt`
- **Arayüz:** Streamlit (dashboard), FastAPI (backend)
- **Orkestrasyon:** APScheduler (periyodik görevler)
- **Logging:** `loguru`
- **Test:** pytest

---

## KLASÖR YAPISI

```
trading_bot/
├── config/
│   ├── settings.py          # API anahtarları, ayarlar
│   └── prompts.py            # Ajan promptları
├── data/
│   ├── collectors/
│   │   ├── crypto_collector.py
│   │   ├── stock_collector.py
│   │   ├── forex_collector.py
│   │   ├── news_collector.py
│   │   └── kap_scraper.py
│   ├── indicators.py         # Teknik indikatörler
│   └── storage.py            # DB işlemleri
├── agents/
│   ├── base_agent.py
│   ├── news_agent.py         # Haber analizi
│   ├── technical_agent.py    # Teknik analiz
│   ├── macro_agent.py        # Makroekonomik analiz
│   ├── risk_agent.py         # Risk değerlendirmesi
│   ├── sentiment_agent.py    # Sosyal medya sentiment
│   └── decision_agent.py     # Final karar
├── orchestrator/
│   ├── workflow.py           # LangGraph workflow
│   └── scheduler.py          # Periyodik çalıştırma
├── backtest/
│   ├── engine.py
│   └── metrics.py            # Sharpe, drawdown vs
├── execution/
│   ├── paper_trader.py       # Simülasyon
│   └── live_trader.py        # GERÇEK PARA (varsayılan KAPALI)
├── dashboard/
│   └── app.py                # Streamlit
├── tests/
├── logs/
├── .env.example
├── requirements.txt
└── main.py
```

---

## AŞAMA AŞAMA UYGULAMA

### AŞAMA 1: VERİ TOPLAMA KATMANI

**Gereksinimler:**

1. **Crypto Collector** (`data/collectors/crypto_collector.py`)
   - Binance public API'den OHLCV verisi çek (1m, 5m, 15m, 1h, 4h, 1d)
   - Çoklu sembol desteği (BTCUSDT, ETHUSDT, SOLUSDT)
   - Rate limit yönetimi (Binance: 1200 req/min)
   - Cache mekanizması (aynı veriyi tekrar çekmesin)

2. **Stock Collector** (`data/collectors/stock_collector.py`)
   - yfinance ile BIST hisseleri (THYAO.IS, GARAN.IS formatında)
   - Hem günlük hem dakikalık veri
   - Şirket bilgisi (P/E, market cap, sektör)

3. **KAP Scraper** (`data/collectors/kap_scraper.py`)
   - kap.org.tr/tr/bildirim-sorgu üzerinden şirket açıklamaları
   - HTML parsing (BeautifulSoup)
   - Tarih, şirket, başlık, içerik, kategori
   - Yeni açıklamaları tespit eden delta mekanizması
   - **Etik not:** robots.txt'ye uy, makul gecikme bırak (2-3 sn arası)

4. **Forex Collector** (`data/collectors/forex_collector.py`)
   - OANDA practice account veya yfinance ile (EURUSD=X, USDTRY=X)
   - Spread ve pip değeri hesaplama

5. **News Collector** (`data/collectors/news_collector.py`)
   - RSS feed'leri: Bloomberg HT, Dünya, Reuters TR, CNBC-e
   - Anahtar kelime filtreleme
   - Duplicate tespiti (URL hash)
   - Zaman damgası mutlaka UTC

**Çıktı formatı (tüm collector'lar için ortak):**
```python
{
    "source": "binance",
    "symbol": "BTCUSDT",
    "timestamp": "2024-XX-XX UTC",
    "data_type": "ohlcv" | "news" | "fundamental",
    "payload": {...}
}
```

---

### AŞAMA 2: TEKNİK İNDİKATÖRLER

**Dosya:** `data/indicators.py`

`pandas-ta` kullanarak şu indikatörleri hesapla:

**Trend:**
- SMA (20, 50, 200)
- EMA (9, 21, 55)
- MACD (12, 26, 9)
- ADX (14)
- Ichimoku Cloud

**Momentum:**
- RSI (14)
- Stochastic (14, 3, 3)
- CCI (20)
- Williams %R

**Volatilite:**
- Bollinger Bands (20, 2)
- ATR (14)
- Keltner Channels

**Hacim:**
- OBV
- Volume MA
- VWAP
- Chaikin Money Flow

**Pattern Detection (özel fonksiyonlar):**
- Support/Resistance seviyeleri (pivot points)
- Trend line tespiti
- Candlestick patterns (engulfing, doji, hammer)
- Divergence (RSI/MACD ile fiyat arasında)

**Çıktı:** Hesaplanan tüm indikatörler DataFrame'e eklenir VE LLM için **doğal dile çevirilmiş özet** üretilir:

```
"THYAO son 14 günde 42.50'den 45.30'a yükseldi (+%6.5).
RSI 68'de — aşırı alım sınırında ama henüz değil.
MACD pozitif kesişim yaptı 3 gün önce.
Hacim son 5 günde ortalamanın 1.8 katı.
Fiyat 50 günlük EMA'nın üzerinde, trend yukarı.
Bollinger üst bandı yakın — direnç olabilir."
```

---

### AŞAMA 3: AJAN MİMARİSİ

**Genel kurallar:**

- Her ajan bir `BaseAgent` sınıfından miras alır
- Her ajan ayrı bir LLM prompt'u ve görev tanımı vardır
- Her ajan structured output döner (Pydantic model)
- Her ajan kararının gerekçesini açıklar
- Her ajan güven skoru (0-1) verir

**Base Agent (`agents/base_agent.py`):**

```python
class AgentOutput(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float  # 0-1
    reasoning: str
    key_points: list[str]
    timestamp: datetime

class BaseAgent:
    def __init__(self, llm, system_prompt: str):
        self.llm = llm
        self.system_prompt = system_prompt
    
    def analyze(self, context: dict) -> AgentOutput:
        ...
```

---

**3.1 News Agent**

**Görev:** Toplanan haberleri okur, hangi sembolü etkilediğini, etki yönünü ve büyüklüğünü değerlendirir.

**System prompt özü:**
```
Sen kıdemli bir finansal haber analistisin. Verilen haberi oku ve:
1. Hangi finansal enstrümanları etkiler?
2. Etki yönü (pozitif/negatif/nötr)?
3. Etki büyüklüğü (1-10)?
4. Etki süresi (kısa/orta/uzun vadeli)?
5. Önemli sayılar, isimler, olaylar nelerdir?

Spekülasyondan kaçın. Belirsiz haberlerde "yetersiz veri" de.
Hype, FUD, ve gerçek haberi ayır.
```

---

**3.2 Technical Agent**

**Görev:** İndikatör özetini okur ve teknik analiz yapar.

**System prompt özü:**
```
Sen 20 yıllık deneyimli bir teknik analistsin. 
Verilen fiyat ve indikatör verisini analiz et:

1. Mevcut trend ne (kısa, orta, uzun)?
2. Önemli destek/direnç seviyeleri?
3. Hangi sinyaller var (bullish/bearish)?
4. Hangi sinyaller çelişiyor?
5. Olası senaryolar?
6. Stop-loss ve take-profit seviyeleri öner?

Kesin tahmin yapma — olasılıklar ver. 
Tek bir indikatöre dayanma — konfirmasyon ara.
```

---

**3.3 Macro Agent**

**Görev:** Genel ekonomik durum, FED, ECB, TCMB kararları, enflasyon, faiz, jeopolitik durumu yorumlar.

**System prompt özü:**
```
Sen makroekonomistsin. Verilen makro veriyi yorumla:

1. Para politikası yönü?
2. Risk iştahı (risk-on / risk-off)?
3. Hangi varlık sınıfları olumlu/olumsuz etkilenir?
4. Önemli takvim olayları?
5. Korelasyonlar (DXY, altın, tahvil getirileri)?
```

---

**3.4 Sentiment Agent**

**Görev:** Sosyal medya (X, Reddit, Türkçe finansal forumlar) verisini analiz eder.

**System prompt özü:**
```
Sosyal medyada hangi semboller konuşuluyor?
Hype mi, gerçek ilgi mi?
Sentiment skoru ve hacim trendi?
Manipülasyon sinyalleri (pump & dump, organize gruplar)?
```

---

**3.5 Risk Agent**

**Görev:** Önerilen trade'in risk profilini değerlendirir.

**System prompt özü:**
```
Sen risk yöneticisisin. Verilen trade önerisini analiz et:

1. Pozisyon büyüklüğü uygun mu? (max %2 hesap riski)
2. Stop-loss mesafesi mantıklı mı?
3. Risk/Ödül oranı? (min 1:2 olmalı)
4. Korelasyon riski var mı? (zaten benzer pozisyon var mı)
5. Likidite riski?
6. Olağandışı durum (siyah kuğu) ihtimali?

Eğer risk yüksekse trade'i REDDET. Konservatif ol.
```

---

**3.6 Decision Agent (Orchestrator)**

**Görev:** Tüm ajanların çıktısını alır, ağırlıklandırır, final karar verir.

**System prompt özü:**
```
Sen baş trader'sın. 5 farklı analistten görüş aldın:
- News Analyst: {news_output}
- Technical Analyst: {technical_output}
- Macro Analyst: {macro_output}
- Sentiment Analyst: {sentiment_output}
- Risk Manager: {risk_output}

Görevin:
1. Çelişen görüşleri tespit et
2. Hangisine daha çok güvenmek lazım, neden?
3. Final karar: AL / SAT / BEKLE
4. Pozisyon büyüklüğü (% olarak)
5. Stop-loss seviyesi
6. Take-profit seviyesi (1, 2, 3 hedef)
7. Beklenen süre
8. Bu kararı iptal edecek senaryolar

KURAL: Risk Manager REDDET diyorsa, sen de BEKLE de.
KURAL: 5 ajandan en az 3'ü aynı yöndeyse karar ver.
KURAL: Belirsizlik varsa BEKLE — fırsat kaçırmak, para kaybetmekten iyidir.
```

---

### AŞAMA 4: ORKESTRATOR (LangGraph Workflow)

**Dosya:** `orchestrator/workflow.py`

LangGraph ile şu akışı kur:

```
START
  ↓
[Veri Toplama] (paralel)
  ├─ Fiyat verisi
  ├─ Haberler
  ├─ Makro veri
  └─ Sosyal medya
  ↓
[Ön İşleme]
  ├─ İndikatör hesapla
  └─ Doğal dil özetleri üret
  ↓
[Ajan Analizi] (paralel)
  ├─ News Agent
  ├─ Technical Agent
  ├─ Macro Agent
  └─ Sentiment Agent
  ↓
[Risk Değerlendirmesi]
  └─ Risk Agent
  ↓
[Karar Birleştirme]
  └─ Decision Agent
  ↓
[Onay Kapısı]
  ├─ Paper mode → otomatik uygula
  └─ Live mode → KULLANICI ONAYI BEKLE
  ↓
END
```

**Önemli:** Her node'da hata yakalama (try-except), her node loglanmalı, retry mekanizması olmalı.

---

### AŞAMA 5: BACKTEST MOTORU

**Dosya:** `backtest/engine.py`

Sistemi çalıştırmadan ÖNCE backtest zorunlu.

**Gereksinimler:**

1. Geçmiş veri ile sistemi koştur (en az 2 yıl)
2. Her trade için:
   - Giriş zamanı
   - Çıkış zamanı
   - Giriş fiyatı
   - Çıkış fiyatı
   - Komisyon (gerçekçi: spot %0.1, vadeli %0.05)
   - Slipaj (gerçekçi: %0.05)
   - Kar/zarar

3. Performans metrikleri:
   - Toplam getiri
   - Yıllık getiri
   - Sharpe ratio
   - Sortino ratio
   - Maximum drawdown
   - Win rate
   - Profit factor
   - Average win / Average loss
   - Calmar ratio

4. Görselleştirme:
   - Equity curve
   - Drawdown grafiği
   - Trade dağılımı

**KRİTİK:** Backtest sonucu **Sharpe < 1** ise sistem üretime geçmez.

**KRİTİK:** Lookahead bias'tan kaçın. Modelin geleceği görmediğine emin ol.

**KRİTİK:** Out-of-sample test yap. Eğitim verisi farklı, test verisi farklı dönemden.

---

### AŞAMA 6: PAPER TRADING

**Dosya:** `execution/paper_trader.py`

Sanal hesap ile gerçek zamanlı çalışsın:

- Başlangıç bakiyesi: 10.000 USD (örnek)
- Her sinyalde sanal trade aç
- Gerçek piyasa fiyatıyla çalış
- Komisyon ve slipaj simüle et
- Günlük/haftalık rapor üret

**Çalışma süresi:** Minimum 3 ay, ideal 6 ay.

Bu aşamada karar veren AJAN, kullanıcı sadece izleyici.

---

### AŞAMA 7: LIVE TRADING (DİKKAT)

**Dosya:** `execution/live_trader.py`

**Varsayılan olarak KAPALI olmalı.** Aktivasyon için config'de açık flag, ortam değişkeninde onay, ve kullanıcının elle yazması gereken bir confirm cümlesi olsun.

**Zorunlu güvenlik özellikleri:**

1. **Kill switch:** Tek tuşla tüm pozisyonları kapat
2. **Daily loss limit:** Günde max %2 kayıp → sistem durur
3. **Max position size:** Hesabın max %10'u tek pozisyonda
4. **Cooldown:** Bir trade kapandıktan sonra min 30 dk bekle
5. **Sanity checks:** 
   - Fiyat son fiyattan %5'ten fazla farklıysa işlem yapma
   - API gecikmesi 5sn'den fazlaysa işlem yapma
   - Hesap bakiyesi beklenenden farklıysa dur
6. **2FA:** Her live trade için kullanıcı onayı (en başta)
7. **Audit log:** Her işlem zaman damgalı, değiştirilemez logla

---

### AŞAMA 8: DASHBOARD

**Dosya:** `dashboard/app.py`

Streamlit ile:

- Anlık pozisyonlar
- Geçmiş trade'ler
- P&L grafiği
- Ajan kararları ve gerekçeleri (her trade için)
- Backtest sonuçları
- Sistem sağlık göstergeleri
- Manuel müdahale butonu (acil durdurma)

---

## KRİTİK KURALLAR (HİÇBİRİNİ ATLAMA)

1. **Yapay zeka her zaman gerekçe sunmalı.** "Al" der ama nedenini söylemezse, sistem hatalıdır.

2. **Çelişen sinyaller varsa BEKLE.** "Bilmiyorum" demek, yanlış karar vermekten iyidir.

3. **Her şey loglanır.** Hata ayıklama için tüm girdiler, çıktılar, kararlar kalıcı kaydedilmeli.

4. **Backtest geçmeden production'a geçilmez.** Sharpe < 1 ise sistem hazır değildir.

5. **Paper trading 3+ ay yapılmadan gerçek para BAĞLANMAZ.**

6. **Position sizing kuralı:** Tek bir trade'de hesabın %2'sinden fazlasını riske atma.

7. **Stop-loss ZORUNLU.** Her pozisyonun stop-loss'u olmalı, istisnasız.

8. **Korelasyon kontrolü:** Aynı anda korelasyonlu varlıklara aynı yönde girme.

9. **API rate limit'lere uy.** Sistem ban yememeli.

10. **Modelin güvenini sorgula:** LLM çok eminmiş gibi konuşsa bile, asla %100 emin olamaz.

---

## REQUIREMENTS.TXT (TAM LİSTE)

```
# Core
python-dotenv==1.0.1
pydantic==2.6.1
loguru==0.7.2

# LLM
langchain==0.1.16
langchain-community==0.0.32
langgraph==0.0.40
ollama==0.1.7
openai==1.14.0
anthropic==0.20.0

# Data — Crypto
python-binance==1.0.19
ccxt==4.2.50

# Data — Stocks/Forex
yfinance==0.2.36
oandapyV20==0.7.2

# Data — News & Scraping
feedparser==6.0.11
newspaper3k==0.2.8
beautifulsoup4==4.12.3
requests==2.31.0
tweepy==4.14.0

# Technical Analysis
pandas==2.2.0
numpy==1.26.4
pandas-ta==0.3.14b0

# Backtest
backtesting==0.3.3
vectorbt==0.26.1

# Database
sqlalchemy==2.0.28
psycopg2-binary==2.9.9

# Dashboard
streamlit==1.32.0
plotly==5.20.0

# Scheduling
apscheduler==3.10.4

# Testing
pytest==8.1.1
pytest-asyncio==0.23.5
```

---

## GELİŞTİRME SIRASI (BU SIRAYI TAKİP ET)

**Hafta 1-2:** AŞAMA 1 (Veri toplama)
- Tek bir sembol için tek bir collector çalıştır
- Veriyi DB'ye yaz
- Test et

**Hafta 3:** AŞAMA 2 (İndikatörler)
- pandas-ta entegrasyonu
- Doğal dil özet üretimi
- Test et

**Hafta 4-5:** AŞAMA 3 (Ajanlar)
- Tek tek her ajanı kur
- Promptları iyileştir
- Yerel LLM ile test (Ollama + Llama 3.1)

**Hafta 6:** AŞAMA 4 (Orkestratör)
- LangGraph workflow
- End-to-end pipeline

**Hafta 7-8:** AŞAMA 5 (Backtest)
- 2 yıllık veri ile backtest
- Metrikleri hesapla
- Parametre optimizasyonu

**Hafta 9-20:** AŞAMA 6 (Paper Trading)
- 3 ay minimum kağıt trading
- Performans takibi
- İyileştirme döngüsü

**Hafta 21+:** AŞAMA 7 (Live — opsiyonel)
- Çok küçük miktarla başla
- Her hata kritik, derhal durdur

---

## YASAL VE FİNANSAL UYARI

Bu sistem yatırım tavsiyesi değildir. Algoritmik trading risk içerir. Geçmiş performans gelecek garanti etmez. Sermayenin tamamını kaybetme ihtimali vardır. Kendi araştırmanı yap, profesyonel danışmana başvur.

---

## KOD KALİTESİ STANDARTLARI

- Her fonksiyon docstring içermeli
- Type hint zorunlu
- Black ile formatla
- Ruff ile lint et
- Test coverage min %70
- Hassas veri (.env dosyasına): API key, secret, parola
- .env asla commit edilmez
- Logger seviyeleri: DEBUG (geliştirme), INFO (üretim), ERROR (her zaman)

---

BU PROMPT'U AI CODING TOOL'A VER (Cursor, Claude Code, Windsurf vs). HER AŞAMAYI TAMAMLAMADAN BİR SONRAKİNE GEÇMESİN. HER AŞAMADAN SONRA SEN TEST ET, ONAY VER, SONRA DEVAM ETSİN.
