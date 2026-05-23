"""Trading Bot — giriş noktası.

Kullanım:
    python main.py --mode collect          # Veri topla ve DB'ye yaz
    python main.py --mode collect-crypto   # Yalnızca kripto
    python main.py --mode collect-stocks   # Yalnızca hisse
    python main.py --mode collect-news     # Yalnızca haberler
    python main.py --mode status           # Sistem durumu
"""

import argparse
import sys

from loguru import logger

from config.settings import LOG_LEVEL, LOGS_DIR, PAPER_TRADING, LIVE_TRADING_ENABLED
from data.storage import init_db, save_collector_output


def _setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:MM:SS}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )
    logger.add(
        LOGS_DIR / "trading_bot_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="00:00",
        retention="30 days",
        encoding="utf-8",
    )


def _collect_all() -> None:
    """Tüm kaynaklardan veri toplar ve DB'ye kaydeder."""
    from data.collectors import crypto_collector, stock_collector, forex_collector, news_collector

    logger.info("=== Veri toplama başlıyor ===")

    collectors = [
        ("Kripto", crypto_collector.collect_all),
        ("Hisse", stock_collector.collect_all),
        ("Forex", forex_collector.collect_all),
        ("Haberler", news_collector.fetch_all_feeds),
    ]

    for name, fn in collectors:
        try:
            logger.info("{} toplanıyor...", name)
            items = fn()
            for item in items:
                save_collector_output(item)
            logger.info("{}: {} kayıt işlendi.", name, len(items))
        except Exception as exc:
            logger.error("{} hatası: {}", name, exc)

    logger.info("=== Veri toplama tamamlandı ===")


def _collect_crypto() -> None:
    from data.collectors import crypto_collector
    items = crypto_collector.collect_all()
    for item in items:
        save_collector_output(item)
    logger.info("Kripto: {} kayıt kaydedildi.", len(items))


def _collect_stocks() -> None:
    from data.collectors import stock_collector
    items = stock_collector.collect_all()
    for item in items:
        save_collector_output(item)
    logger.info("Hisse: {} kayıt kaydedildi.", len(items))


def _collect_news() -> None:
    from data.collectors import news_collector
    items = news_collector.fetch_all_feeds()
    for item in items:
        save_collector_output(item)
    logger.info("Haber: {} kayıt kaydedildi.", len(items))


def _show_status() -> None:
    mode = "CANLI (LIVE)" if LIVE_TRADING_ENABLED else ("Paper Trading" if PAPER_TRADING else "TANIMLI DEĞİL")
    logger.info("Trading modu: {}", mode)
    logger.info("DB: {}", __import__("config.settings", fromlist=["DATABASE_URL"]).DATABASE_URL)

    if LIVE_TRADING_ENABLED:
        logger.warning("!!! CANLI TRADING AKTİF — Gerçek para riski var !!!")


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(description="Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["collect", "collect-crypto", "collect-stocks", "collect-news", "status"],
        default="status",
        help="Çalışma modu",
    )
    args = parser.parse_args()

    init_db()

    dispatch = {
        "collect": _collect_all,
        "collect-crypto": _collect_crypto,
        "collect-stocks": _collect_stocks,
        "collect-news": _collect_news,
        "status": _show_status,
    }
    dispatch[args.mode]()


if __name__ == "__main__":
    main()
