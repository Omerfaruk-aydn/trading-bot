"""Ekonomik takvim — TCMB, enflasyon, önemli olaylar ve KAP insider takibi.

Özellikler:
  - Bugün yüksek etkili olay var mı? (pozisyon açmayı engelle)
  - KAP'tan büyük hissedar işlemleri (insider al/sat)
  - Sektör bazlı event filtresi
"""

from __future__ import annotations

import re
import time
from datetime import datetime, date
from functools import lru_cache

import requests
from loguru import logger


# ── Sabit Türkiye ekonomik takvimi ────────────────────────────────────────────
# TCMB toplantı tarihleri 2025 (resmi açıklanan)
TCMB_DATES_2025 = {
    date(2025, 1, 23), date(2025, 3, 6), date(2025, 4, 17),
    date(2025, 5, 22), date(2025, 6, 19), date(2025, 7, 24),
    date(2025, 8, 21), date(2025, 9, 18), date(2025, 10, 23),
    date(2025, 11, 20), date(2025, 12, 25),
}

TCMB_DATES_2026 = {
    date(2026, 1, 22), date(2026, 2, 26), date(2026, 3, 19),
    date(2026, 4, 23), date(2026, 5, 21), date(2026, 6, 18),
    date(2026, 7, 23), date(2026, 8, 20), date(2026, 9, 17),
    date(2026, 10, 22), date(2026, 11, 19), date(2026, 12, 24),
}

ALL_TCMB_DATES = TCMB_DATES_2025 | TCMB_DATES_2026

# TÜİK enflasyon açıklama günleri (her ayın 3. iş günü civarı, yaklaşık)
TUIK_CPI_MONTHS = list(range(1, 13))  # her ay


class EconomicEvent:
    def __init__(self, title: str, impact: str, event_date: date, source: str = ""):
        self.title = title
        self.impact = impact        # "high" | "medium" | "low"
        self.event_date = event_date
        self.source = source

    def __repr__(self):
        return f"EconomicEvent({self.title!r}, {self.impact}, {self.event_date})"


class EconomicCalendar:
    """
    Ekonomik olay takibi.

    Kullanım:
        cal = EconomicCalendar()
        if cal.is_high_impact_today():
            # pozisyon açma
        events = cal.today_events()
        insider = cal.get_kap_insider("THYAO")
    """

    def __init__(self):
        self._cache: dict[str, list] = {}
        self._cache_time: float = 0
        self._cache_ttl = 3600  # 1 saat

    # ── Yüksek etkili olay kontrolü ──────────────────────────────────────────

    def is_high_impact_today(self) -> bool:
        """Bugün TCMB toplantısı veya yüksek etkili olay varsa True."""
        today = date.today()
        if today in ALL_TCMB_DATES:
            logger.warning("TCMB toplantı günü! Yeni pozisyon açmak riskli.")
            return True
        events = self.today_events()
        return any(e.impact == "high" for e in events)

    def today_events(self) -> list[EconomicEvent]:
        """Bugünkü tüm önemli olayları listele."""
        today = date.today()
        events: list[EconomicEvent] = []

        # TCMB toplantısı
        if today in ALL_TCMB_DATES:
            events.append(EconomicEvent(
                "TCMB Para Politikası Kurulu Toplantısı", "high", today, "TCMB"
            ))

        # TÜİK CPI (her ayın 3. ile 5. günleri arası açıklanır)
        if 3 <= today.day <= 6:
            events.append(EconomicEvent(
                f"TÜİK TÜFE Enflasyon Verisi ({today.strftime('%B %Y')})", "high", today, "TÜİK"
            ))

        # Cuma günü pozisyon taşıma riski
        if today.weekday() == 4:
            events.append(EconomicEvent(
                "Hafta sonu kapanışı — açık pozisyon riski", "medium", today, "sistem"
            ))

        # Dış kaynaklı olaylar (opsiyonel)
        try:
            investing_events = self._fetch_investing_calendar()
            events.extend([e for e in investing_events if e.event_date == today])
        except Exception:
            pass

        return events

    def should_skip_trading(self, symbol: str = "") -> tuple[bool, str]:
        """
        İşlem yapılmamalı mı? (True, neden) döndürür.
        Sembol verilirse sektör spesifik kontrol de yapar.
        """
        today = date.today()

        # Hafta sonu
        if today.weekday() >= 5:
            return True, "Hafta sonu — piyasa kapalı"

        # TCMB toplantısı
        if today in ALL_TCMB_DATES:
            return True, "TCMB toplantı günü — volatilite riski"

        # TÜİK enflasyon günü (banka hisseleri için kritik)
        if 3 <= today.day <= 6:
            bank_symbols = {"GARAN", "AKBNK", "YKBNK", "ISCTR", "HALKB", "VAKBN"}
            sym_base = symbol.replace(".IS", "").upper()
            if sym_base in bank_symbols:
                return True, "Enflasyon açıklama dönemi — bankacılık sektörü riskli"

        return False, ""

    # ── Investing.com takvim (opsiyonel) ─────────────────────────────────────

    def _fetch_investing_calendar(self) -> list[EconomicEvent]:
        """Investing.com ekonomik takviminden yüksek etkili Türkiye olayları."""
        now = time.time()
        cache_key = "investing"
        if cache_key in self._cache and now - self._cache_time < self._cache_ttl:
            return self._cache[cache_key]

        events = []
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "X-Requested-With": "XMLHttpRequest",
            }
            today_str = date.today().strftime("%Y-%m-%d")
            r = requests.get(
                "https://tr.investing.com/economic-calendar/Service/getCalendarFilteredData",
                params={
                    "country[]": "56",      # Türkiye kodu
                    "importance[]": "3",    # sadece yüksek etkili
                    "dateFrom": today_str,
                    "dateTo": today_str,
                },
                headers=headers,
                timeout=8,
            )
            if r.ok:
                data = r.json()
                # HTML parse — basit regex
                for match in re.finditer(
                    r'data-event-datetime="([^"]+)"[^>]*>.*?<td[^>]*>([^<]+)</td>',
                    data.get("data", ""),
                    re.DOTALL,
                ):
                    dt_str, title = match.group(1), match.group(2).strip()
                    try:
                        ev_date = datetime.strptime(dt_str[:10], "%Y-%m-%d").date()
                        events.append(EconomicEvent(title, "high", ev_date, "Investing.com"))
                    except Exception:
                        pass
        except Exception as e:
            logger.debug("Investing.com takvim çekilemedi: {}", e)

        self._cache[cache_key] = events
        self._cache_time = now
        return events

    # ── KAP Insider İşlem Takibi ─────────────────────────────────────────────

    def get_kap_insider(self, symbol: str, days: int = 7) -> list[dict]:
        """
        KAP'tan son N günün büyük hissedar bildirimlerini çeker.
        Sadece BIST hisseleri için çalışır.
        """
        sym = symbol.replace(".IS", "").upper()
        cache_key = f"kap_{sym}"
        now = time.time()
        if cache_key in self._cache and now - self._cache_time < self._cache_ttl:
            return self._cache[cache_key]

        results = []
        try:
            r = requests.get(
                "https://www.kap.org.tr/tr/api/disclosures",
                params={"memberCode": sym, "year": date.today().year},
                timeout=8,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if r.ok:
                data = r.json()
                for item in data.get("data", [])[:20]:
                    title = item.get("disclosureType", "")
                    # Büyük hissedar/yönetici işlemi
                    if any(k in title.lower() for k in ["büyük", "yönetici", "insider", "pay"]):
                        results.append({
                            "date": item.get("disclosureDate", ""),
                            "title": title,
                            "summary": item.get("title", ""),
                        })
        except Exception as e:
            logger.debug("KAP çekilemedi ({}): {}", sym, e)

        self._cache[cache_key] = results
        return results

    def insider_signal(self, symbol: str) -> str:
        """
        'buy' — yönetici/büyük hissedar alım bildirimi
        'sell' — satım bildirimi
        'neutral' — bildirim yok veya belirsiz
        """
        insiders = self.get_kap_insider(symbol)
        if not insiders:
            return "neutral"

        buy_keywords  = ["alım", "satın alma", "edinim"]
        sell_keywords = ["satım", "elden çıkarma", "azaltma"]

        for item in insiders:
            text = (item.get("title", "") + item.get("summary", "")).lower()
            if any(k in text for k in buy_keywords):
                return "buy"
            if any(k in text for k in sell_keywords):
                return "sell"
        return "neutral"
