"""Anlık Haber Tetikleyici — RSS polling + sentiment → işlem sinyali."""
from __future__ import annotations

import hashlib
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from loguru import logger

from agents.sentiment import analyze as analyze_sentiment
from data.sources.yahoo_news import fetch_yahoo_news


@dataclass
class NewsTrigger:
    symbol: str
    title: str
    sentiment: str       # "bullish" | "bearish"
    score: float
    confidence: float
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        icon = "📈" if self.sentiment == "bullish" else "📉"
        return (f"{icon} [{self.symbol}] {self.sentiment.upper()} "
                f"(conf={self.confidence:.0%}) — {self.title[:60]}")


class NewsWatcher:
    """
    Arka planda çalışan haber izleyici.

    Her `interval` saniyede bir RSS beslemelerini tarar.
    Daha önce görülmemiş bir haber güçlü sentiment içeriyorsa
    `on_trigger` callback'ini çağırır ve sembolü `urgent_queue`'ya ekler.

    Kullanım:
        watcher = NewsWatcher(symbols, on_trigger=agent._on_news_trigger)
        watcher.start()
        ...
        watcher.stop()
    """

    def __init__(
        self,
        symbols: list[str],
        on_trigger: Callable[[NewsTrigger], None] | None = None,
        interval: int = 120,
        min_confidence: float = 0.65,
        max_seen: int = 2000,
    ):
        self.symbols = list(symbols)
        self.on_trigger = on_trigger
        self.interval = interval
        self.min_confidence = min_confidence
        self.max_seen = max_seen

        self._seen: set[str] = set()
        self._thread: threading.Thread | None = None
        self._running = False

        # Dışarıdan okunabilir kuyruk — acil taranacak semboller
        self.urgent_queue: queue.Queue[NewsTrigger] = queue.Queue()

    # ── Yaşam döngüsü ─────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="NewsWatcher")
        self._thread.start()
        logger.info("NewsWatcher başlatıldı: {} sembol | {}s aralık | min_conf={:.0%}",
                    len(self.symbols), self.interval, self.min_confidence)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def add_symbols(self, symbols: list[str]) -> None:
        for s in symbols:
            if s not in self.symbols:
                self.symbols.append(s)

    # ── İç döngü ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            for sym in list(self.symbols):
                if not self._running:
                    break
                try:
                    self._check(sym)
                except Exception as e:
                    logger.debug("NewsWatcher hata [{}]: {}", sym, e)
                time.sleep(0.5)  # semboller arası küçük bekleme

            if self._running:
                time.sleep(self.interval)

    def _check(self, symbol: str) -> None:
        news = fetch_yahoo_news(symbol, max_items=5)
        for item in news:
            title = item.get("title", "").strip()
            if not title:
                continue

            key = hashlib.md5(f"{symbol}:{title}".encode()).hexdigest()
            if key in self._seen:
                continue

            # Görülen haberler setini büyümeden koru
            if len(self._seen) >= self.max_seen:
                self._seen.pop() if hasattr(self._seen, "pop") else self._seen.clear()
            self._seen.add(key)

            sent = analyze_sentiment(title)

            if sent.label not in ("bullish", "bearish"):
                continue
            if sent.confidence < self.min_confidence:
                continue

            trigger = NewsTrigger(
                symbol=symbol,
                title=title,
                sentiment=sent.label,
                score=sent.score,
                confidence=sent.confidence,
            )

            logger.info("📰 Haber tetikleyici: {}", trigger)
            self.urgent_queue.put(trigger)

            if self.on_trigger:
                try:
                    self.on_trigger(trigger)
                except Exception as e:
                    logger.debug("on_trigger hatası: {}", e)

    # ── Yardımcı ──────────────────────────────────────────────────────────────

    def drain_urgent(self) -> list[NewsTrigger]:
        """Kuyruktaki tüm acil tetikleyicileri çek ve temizle."""
        triggers: list[NewsTrigger] = []
        while not self.urgent_queue.empty():
            try:
                triggers.append(self.urgent_queue.get_nowait())
            except queue.Empty:
                break
        return triggers
