"""RSS feed'lerinden haber toplayan modül — duplicate tespiti ve UTC timestamp içerir."""

import hashlib
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
from loguru import logger

from config.settings import NEWS_RSS_FEEDS


def _url_hash(url: str) -> str:
    """URL'den benzersiz kimlik üretir — duplicate tespitinde kullanılır."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _parse_date(entry: Any) -> str:
    """feedparser entry'sinden UTC timestamp üretir."""
    for field in ("published", "updated"):
        raw = getattr(entry, field, None)
        if raw:
            try:
                dt = parsedate_to_datetime(raw)
                return dt.astimezone(timezone.utc).isoformat()
            except Exception:
                pass
    return datetime.now(tz=timezone.utc).isoformat()


def _extract_text(entry: Any) -> str:
    """Summary veya content alanından ham metni çeker."""
    if hasattr(entry, "summary"):
        return entry.summary
    if hasattr(entry, "content") and entry.content:
        return entry.content[0].get("value", "")
    return ""


def fetch_feed(feed_config: dict, keyword_filter: list[str] | None = None) -> list[dict]:
    """
    Tek bir RSS feed'ini çeker ve parse eder.

    Args:
        feed_config: {"name": str, "url": str} formatında feed tanımı
        keyword_filter: Varsa sadece bu kelimeleri içeren haberleri döner

    Returns:
        CollectorOutput listesi
    """
    name = feed_config["name"]
    url = feed_config["url"]
    logger.info("RSS çekiliyor: {} — {}", name, url)

    try:
        parsed = feedparser.parse(url)
    except Exception as exc:
        logger.error("RSS parse hatası: {} — {}", name, exc)
        return []

    if parsed.bozo and parsed.bozo_exception:
        logger.warning("RSS uyarısı: {} — {}", name, parsed.bozo_exception)

    results: list[dict] = []
    seen_hashes: set[str] = set()

    for entry in parsed.entries:
        link = getattr(entry, "link", "") or ""
        uid = _url_hash(link)

        if uid in seen_hashes:
            logger.debug("Duplicate atlandı: {}", link)
            continue
        seen_hashes.add(uid)

        title = getattr(entry, "title", "")
        text = _extract_text(entry)
        combined = f"{title} {text}".lower()

        if keyword_filter and not any(kw.lower() in combined for kw in keyword_filter):
            continue

        output = {
            "source": name,
            "symbol": None,  # News Agent sembol eşleştirmeyi yapacak
            "timestamp": _parse_date(entry),
            "data_type": "news",
            "payload": {
                "uid": uid,
                "title": title,
                "url": link,
                "summary": text,
                "tags": [t.get("term", "") for t in getattr(entry, "tags", [])],
            },
        }
        results.append(output)

    logger.info("{} haber alındı: {}", len(results), name)
    return results


def fetch_all_feeds(
    feeds: list[dict] | None = None,
    keyword_filter: list[str] | None = None,
) -> list[dict]:
    """
    Tüm tanımlı RSS feed'lerini çeker.

    Args:
        feeds: Feed listesi; None ise settings'ten alır
        keyword_filter: Anahtar kelime filtresi (opsiyonel)

    Returns:
        Tüm feed'lerden birleştirilmiş CollectorOutput listesi
    """
    feeds = feeds or NEWS_RSS_FEEDS
    all_news: list[dict] = []
    global_seen: set[str] = set()

    for feed_config in feeds:
        items = fetch_feed(feed_config, keyword_filter)
        for item in items:
            uid = item["payload"]["uid"]
            if uid not in global_seen:
                global_seen.add(uid)
                all_news.append(item)
            else:
                logger.debug("Cross-feed duplicate: {}", uid)

    # En yeni haberler başta olsun
    all_news.sort(key=lambda x: x["timestamp"], reverse=True)
    logger.info("Toplam {} benzersiz haber toplandı.", len(all_news))
    return all_news


def filter_by_symbols(news_list: list[dict], symbols: list[str]) -> list[dict]:
    """
    Haber listesini sembol adlarına göre filtreler (başlık ve özet içinde arar).

    Args:
        news_list: fetch_all_feeds çıktısı
        symbols: Aranacak sembol/şirket ismi listesi

    Returns:
        Eşleşen haberler
    """
    filtered = []
    for item in news_list:
        text = f"{item['payload']['title']} {item['payload']['summary']}".lower()
        for sym in symbols:
            # THYAO.IS → THYAO, BTCUSDT → BTC gibi kısaltmayı da ara
            short = sym.replace(".IS", "").replace("USDT", "").replace("=X", "").replace("-USD", "").replace("-USDT", "")
            if short.lower() in text or sym.lower() in text:
                filtered.append({**item, "symbol": sym})
                break
    return filtered
