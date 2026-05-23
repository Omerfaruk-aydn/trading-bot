"""Yahoo Finance RSS haber kaynağı — hisse başına ücretsiz, API key gerektirmez."""

from __future__ import annotations

import time
import urllib.request
import xml.etree.ElementTree as ET
from loguru import logger

_cache: dict[str, tuple[list[dict], float]] = {}
_TTL = 600  # 10 dakika


def fetch_yahoo_news(symbol: str, max_items: int = 5) -> list[dict]:
    """
    Yahoo Finance RSS'ten hisse bazlı haberler çeker.
    symbol: AAPL, MSFT, BTC-USD vb.
    """
    now = time.time()
    if symbol in _cache:
        items, ts = _cache[symbol]
        if now - ts < _TTL:
            return items

    url = f"https://finance.yahoo.com/rss/headline?s={symbol}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = resp.read()
        root = ET.fromstring(raw)
        items = []
        for item in root.iter("item"):
            title = item.findtext("title", "")
            desc  = item.findtext("description", "")
            pub   = item.findtext("pubDate", "")
            if title:
                items.append({"title": title, "description": desc, "published": pub})
            if len(items) >= max_items:
                break
        _cache[symbol] = (items, now)
        return items
    except Exception as e:
        logger.debug("Yahoo haber hatası ({}}): {}", symbol, e)
        cached = _cache.get(symbol, ([], 0))
        return cached[0]
