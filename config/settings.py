"""Central configuration — loads from environment variables via .env file."""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Repo kökündeki .env dosyasını yükle
load_dotenv(Path(__file__).parent.parent / ".env")


# ── Dizinler ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ── Loglama ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG")

# ── Veritabanı ────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/trading_bot.db")

# ── LLM ───────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_LLM: Literal["ollama", "openai", "anthropic"] = "ollama"  # tip: yerel LLM tercih

# ── Binance ───────────────────────────────────────────────────────────────────
BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
BINANCE_TESTNET: bool = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

# İzlenen kripto sembolleri
CRYPTO_SYMBOLS: list[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# Desteklenen timeframe'ler
CRYPTO_INTERVALS: list[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]

# Binance rate limit: 1200 req/dk → güvenli aralık
BINANCE_REQUEST_INTERVAL_SECONDS: float = 0.1

# ── OANDA (Forex) ─────────────────────────────────────────────────────────────
OANDA_API_KEY: str = os.getenv("OANDA_API_KEY", "")
OANDA_ACCOUNT_ID: str = os.getenv("OANDA_ACCOUNT_ID", "")
OANDA_ENVIRONMENT: Literal["practice", "live"] = os.getenv("OANDA_ENVIRONMENT", "practice")  # type: ignore[assignment]

FOREX_SYMBOLS: list[str] = ["EURUSD=X", "USDTRY=X", "GBPUSD=X"]

# ── Hisse / BIST ──────────────────────────────────────────────────────────────
STOCK_SYMBOLS: list[str] = ["THYAO.IS", "GARAN.IS", "ASELS.IS", "EREGL.IS"]

# ── KAP Scraper ───────────────────────────────────────────────────────────────
KAP_BASE_URL: str = "https://www.kap.org.tr/tr/bildirim-sorgu"
KAP_REQUEST_DELAY_SECONDS: float = 2.5  # robots.txt'e saygı

# ── Haber RSS ─────────────────────────────────────────────────────────────────
NEWS_RSS_FEEDS: list[dict] = [
    {"name": "Bloomberg HT",      "url": "https://www.bloomberght.com/rss"},
    {"name": "Dünya Gazetesi",    "url": "https://www.dunya.com/rss"},
    {"name": "Reuters Business",  "url": "https://feeds.reuters.com/reuters/businessNews"},
    {"name": "Investing.com TR",  "url": "https://tr.investing.com/rss/news.rss"},
    {"name": "Ekonomim",          "url": "https://www.ekonomim.com/rss"},
    {"name": "Para Analiz",       "url": "https://www.paraanaliz.com/feed/"},
    {"name": "CNBC-e",            "url": "https://www.cnbce.com/rss"},
]

# TCMB veri URL'leri (RSS değil, JSON API)
TCMB_URLS: dict[str, str] = {
    "faiz": "https://evds2.tcmb.gov.tr/service/evds/series=TP.DK.USD.A&type=json",
    "kur":  "https://www.tcmb.gov.tr/kurlar/today.xml",
}

# ── Twitter / X ───────────────────────────────────────────────────────────────
TWITTER_BEARER_TOKEN: str = os.getenv("TWITTER_BEARER_TOKEN", "")

# ── Trading Modu ──────────────────────────────────────────────────────────────
PAPER_TRADING: bool = os.getenv("PAPER_TRADING", "true").lower() == "true"
_LIVE_CONFIRM_PHRASE = "I CONFIRM LIVE TRADING WITH REAL MONEY"
LIVE_TRADING_ENABLED: bool = (
    not PAPER_TRADING
    and os.getenv("LIVE_TRADING_CONFIRM", "") == _LIVE_CONFIRM_PHRASE
)

# ── Paper Trading ─────────────────────────────────────────────────────────────
PAPER_INITIAL_BALANCE_USD: float = float(os.getenv("PAPER_INITIAL_BALANCE", "10000"))

# ── Risk Kuralları ────────────────────────────────────────────────────────────
MAX_POSITION_RISK_PCT: float = 0.02   # tek trade'de max %2 hesap riski
MAX_POSITION_SIZE_PCT: float = 0.10   # tek pozisyonda max %10 hesap
DAILY_LOSS_LIMIT_PCT: float = 0.02    # günlük max %2 kayıp → sistem durur
MIN_RISK_REWARD_RATIO: float = 2.0    # min 1:2
COOLDOWN_AFTER_TRADE_MINUTES: int = 30

# ── Veri Cache ────────────────────────────────────────────────────────────────
CACHE_TTL_SECONDS: int = 60  # aynı veriyi tekrar çekme süresi
