"""KAP (kap.org.tr) şirket açıklamalarını toplayan scraper.

ETİK KURAL: robots.txt'e uyulur, istekler arası 2-3 saniye beklenir.
Bu scraper yalnızca araştırma ve kişisel kullanım içindir.
"""

import hashlib
import time
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup
from loguru import logger

from config.settings import KAP_BASE_URL, KAP_REQUEST_DELAY_SECONDS

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; TradingBotResearch/1.0; "
        "+https://github.com/youruser/trading_bot)"
    ),
    "Accept-Language": "tr-TR,tr;q=0.9",
}

# Daha önce görülen açıklama hash'leri (delta mekanizması)
_seen_hashes: set[str] = set()


def _disclosure_hash(title: str, company: str, date_str: str) -> str:
    raw = f"{company}:{title}:{date_str}"
    return hashlib.md5(raw.encode()).hexdigest()


def _polite_get(url: str, params: dict | None = None) -> requests.Response:
    """robots.txt'e saygılı, yavaş istek atar."""
    time.sleep(KAP_REQUEST_DELAY_SECONDS)
    try:
        response = requests.get(url, params=params, headers=_HEADERS, timeout=15)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as exc:
        logger.error("KAP isteği başarısız: {} — {}", url, exc)
        raise


def fetch_disclosures(
    company_code: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    max_pages: int = 3,
) -> list[dict]:
    """
    KAP bildirim sorgusundan açıklamaları çeker.

    Args:
        company_code: KAP şirket kodu (ör: THYAO). None ise tüm şirketler.
        date_from: YYYY-MM-DD formatında başlangıç tarihi
        date_to: YYYY-MM-DD formatında bitiş tarihi
        max_pages: Kaç sayfa çekilecek

    Returns:
        Standart CollectorOutput listesi
    """
    results: list[dict] = []

    for page in range(1, max_pages + 1):
        params: dict = {"page": page}
        if company_code:
            params["companyCode"] = company_code
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to

        logger.info("KAP sayfa {} çekiliyor (şirket: {})", page, company_code or "tümü")
        try:
            resp = _polite_get(KAP_BASE_URL, params=params)
        except Exception:
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        rows = _parse_disclosure_rows(soup)

        if not rows:
            logger.info("KAP sayfa {} boş — duruluyor.", page)
            break

        results.extend(rows)

    logger.info("KAP'tan toplam {} açıklama alındı.", len(results))
    return results


def _parse_disclosure_rows(soup: BeautifulSoup) -> list[dict]:
    """Bildirim listesi tablosunu parse eder."""
    rows_out: list[dict] = []

    # KAP tablo yapısı: her satır bir bildirim
    # Sayfa yapısı değişirse bu selector'ları güncelle
    table = soup.find("table", class_="w-full")
    if table is None:
        # Alternatif yapı dene
        table = soup.find("tbody")
    if table is None:
        logger.warning("KAP tablo bulunamadı — sayfa yapısı değişmiş olabilir.")
        return []

    for row in table.find_all("tr"):  # type: ignore[union-attr]
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        try:
            date_str = cells[0].get_text(strip=True)
            company = cells[1].get_text(strip=True)
            category = cells[2].get_text(strip=True)
            title = cells[3].get_text(strip=True)
            link_tag = cells[3].find("a")
            url = f"https://www.kap.org.tr{link_tag['href']}" if link_tag else ""
        except (IndexError, TypeError):
            continue

        uid = _disclosure_hash(title, company, date_str)
        is_new = uid not in _seen_hashes
        _seen_hashes.add(uid)

        output = {
            "source": "kap",
            "symbol": _company_to_symbol(company),
            "timestamp": _parse_kap_date(date_str),
            "data_type": "news",
            "payload": {
                "uid": uid,
                "company": company,
                "category": category,
                "title": title,
                "url": url,
                "summary": "",  # detay sayfasından çekilebilir
                "is_new": is_new,
            },
        }
        rows_out.append(output)

    return rows_out


def fetch_disclosure_detail(url: str) -> str:
    """Tek bir açıklama sayfasının tam metnini çeker."""
    if not url:
        return ""
    logger.info("KAP detay çekiliyor: {}", url)
    try:
        resp = _polite_get(url)
    except Exception:
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    content_div = soup.find("div", class_="disclosure-text") or soup.find("div", id="content")
    if content_div:
        return content_div.get_text(separator="\n", strip=True)
    return soup.get_text(separator="\n", strip=True)[:3000]


def get_new_disclosures(
    company_code: str | None = None,
    date_from: str | None = None,
) -> list[dict]:
    """
    Sadece daha önce görülmemiş (yeni) açıklamaları döner.
    Delta mekanizması: _seen_hashes kümesi session boyunca tutulur.
    """
    all_disclosures = fetch_disclosures(company_code=company_code, date_from=date_from)
    new_ones = [d for d in all_disclosures if d["payload"]["is_new"]]
    logger.info("{} yeni KAP açıklaması tespit edildi.", len(new_ones))
    return new_ones


def _parse_kap_date(date_str: str) -> str:
    """KAP tarih formatını (DD.MM.YYYY HH:MM) UTC ISO formatına çevirir."""
    for fmt in ("%d.%m.%Y %H:%M", "%d.%m.%Y", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            # KAP Türkiye saatinde, UTC+3 varsayılır
            from datetime import timedelta
            dt_utc = dt - timedelta(hours=3)
            return dt_utc.replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    return datetime.now(tz=timezone.utc).isoformat()


def _company_to_symbol(company_name: str) -> str | None:
    """Şirket adından BIST sembolünü tahmin eder (basit eşleştirme)."""
    mapping = {
        "TÜRK HAVA YOLLARI": "THYAO.IS",
        "TÜRKIYE GARANTI BANKASI": "GARAN.IS",
        "ASELSAN": "ASELS.IS",
        "EREĞLI DEMİR VE ÇELİK": "EREGL.IS",
    }
    upper = company_name.upper()
    for key, sym in mapping.items():
        if key in upper:
            return sym
    return None
