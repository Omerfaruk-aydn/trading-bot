"""
Hızlı piyasa tarayıcı — tüm borsalarda en çok hareket eden hisseleri bulur.

Kaynaklar (ücretsiz, API key gerektirmez):
  • Yahoo Finance screener JSON (gün kazananları/kaybedenler/hacimliler)
  • Yahoo Finance pre/after-market fiyatları (extended hours)
  • Binance 24hr ticker API (kripto tüm çiftler)
  • yfinance batch download (BIST toplu tarama)
  • NASDAQ FTP tam evren (~5,000-6,000 hisse)
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from loguru import logger

from data.sources.stock_universe import (
    get_bist_symbols,
    get_crypto_universe,
    get_full_us_universe,
    get_us_universe,
)
from data.markets.us_stocks import get_us_session

_cache: dict[str, tuple[object, float]] = {}
_TTL = 120  # 2 dakika (movers çok sık değişir ama API'yi çok sık çağırma)


@dataclass
class QuickQuote:
    symbol: str
    price: float
    change_pct: float   # 24h değişim %
    volume_usd: float   # 24h hacim (USD veya yerel para)
    market: str         # "us" | "crypto" | "bist" | "futures"


def _cached_call(key: str, fn) -> object:
    now = time.time()
    if key in _cache:
        val, ts = _cache[key]
        if now - ts < _TTL:
            return val
    val = fn()
    _cache[key] = (val, now)
    return val


# ── Kripto: Binance 24hr ticker ──────────────────────────────────────────────

def screen_crypto(
    top_n: int = 30,
    min_volume_usd: float = 5_000_000,
    quote: str = "USDT",
) -> dict[str, list[QuickQuote]]:
    """
    Binance'ten tüm USDT çiftlerinin 24h istatistiklerini çeker.
    Tek API çağrısı ile 300+ çift, ~0.5 saniye.
    """
    def _fetch():
        url = "https://api.binance.com/api/v3/ticker/24hr"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
        except Exception as e:
            logger.warning("Binance screener hatası: {}", e)
            return {"gainers": [], "losers": [], "most_active": []}

        quotes = []
        for item in data:
            sym = item.get("symbol", "")
            if not sym.endswith(quote):
                continue
            try:
                vol = float(item.get("quoteVolume", 0))
                if vol < min_volume_usd:
                    continue
                chg = float(item.get("priceChangePercent", 0))
                price = float(item.get("lastPrice", 0))
                base = sym[:-len(quote)]
                yf_sym = f"{base}-USD"
                quotes.append(QuickQuote(
                    symbol=yf_sym,
                    price=price,
                    change_pct=chg,
                    volume_usd=vol,
                    market="crypto",
                ))
            except Exception:
                continue

        gainers     = sorted(quotes, key=lambda q: q.change_pct, reverse=True)[:top_n]
        losers      = sorted(quotes, key=lambda q: q.change_pct)[:top_n]
        most_active = sorted(quotes, key=lambda q: q.volume_usd, reverse=True)[:top_n]

        logger.info(
            "Kripto screener: {} çift tarandı | En çok yükselen: {} | En çok düşen: {}",
            len(quotes),
            gainers[0].symbol if gainers else "-",
            losers[0].symbol  if losers  else "-",
        )
        return {"gainers": gainers, "losers": losers, "most_active": most_active}

    return _cached_call("crypto_screen", _fetch)


# ── ABD: Yahoo Finance Screener ──────────────────────────────────────────────

def _yf_predefined_screen(scr_id: str, count: int = 25) -> list[QuickQuote]:
    """Yahoo Finance önceden tanımlı tarayıcısını kullanır."""
    url = (
        "https://query2.finance.yahoo.com/v1/finance/screener/predefined/saved"
        f"?formatted=false&scrIds={scr_id}&count={count}"
    )
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        quotes_data = (
            data.get("finance", {})
                .get("result", [{}])[0]
                .get("quotes", [])
        )
        results = []
        for q in quotes_data:
            sym = q.get("symbol", "")
            price = float(q.get("regularMarketPrice", 0))
            chg = float(q.get("regularMarketChangePercent", 0))
            vol = float(q.get("regularMarketVolume", 0)) * price
            results.append(QuickQuote(
                symbol=sym,
                price=price,
                change_pct=chg,
                volume_usd=vol,
                market="us",
            ))
        return results
    except Exception as e:
        logger.debug("Yahoo screener ({}) hatası: {}", scr_id, e)
        return []


def _yf_quote_batch(symbols: list[str], session: str) -> list[QuickQuote]:
    """
    Yahoo Finance v7 quote API ile toplu fiyat sorgular.
    Pre/after-market saatlerinde genişletilmiş seans fiyatlarını da döndürür.
    100 sembol başına 1 API çağrısı — çok hızlı.
    """
    results = []
    batch_size = 100
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        syms_str = "%2C".join(batch)
        url = (
            "https://query2.finance.yahoo.com/v7/finance/quote"
            f"?symbols={syms_str}&fields=regularMarketPrice,regularMarketChangePercent,"
            "regularMarketVolume,preMarketPrice,preMarketChangePercent,"
            "postMarketPrice,postMarketChangePercent"
        )
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
            quotes = data.get("quoteResponse", {}).get("result", [])
            for q in quotes:
                sym = q.get("symbol", "")
                reg_price = float(q.get("regularMarketPrice", 0) or 0)
                reg_chg   = float(q.get("regularMarketChangePercent", 0) or 0)
                reg_vol   = float(q.get("regularMarketVolume", 0) or 0)

                # Extended hours fiyat seçimi
                if session == "premarket":
                    ext_price = float(q.get("preMarketPrice", 0) or 0)
                    ext_chg   = float(q.get("preMarketChangePercent", 0) or 0)
                elif session == "aftermarket":
                    ext_price = float(q.get("postMarketPrice", 0) or 0)
                    ext_chg   = float(q.get("postMarketChangePercent", 0) or 0)
                else:
                    ext_price = 0.0
                    ext_chg   = 0.0

                # Extended hours fiyatı varsa kullan, yoksa regular
                price = ext_price if ext_price > 0 else reg_price
                chg   = ext_chg   if ext_price > 0 else reg_chg
                vol   = reg_vol * reg_price

                if price > 0:
                    results.append(QuickQuote(sym, price, chg, vol, "us"))
        except Exception as e:
            logger.debug("Yahoo quote batch hatası ({}...): {}", batch[0], e)
    return results


def screen_us_extended(top_n: int = 30) -> dict[str, list[QuickQuote]]:
    """
    Pre-market veya after-market saatlerinde TÜM NYSE+NASDAQ hisselerini tarar.
    NASDAQ FTP'den tam evren (~5,000 hisse) → batch Yahoo quote API ile sorgular.
    Sonuçları gün değişimine ve hacme göre sıralar.
    """
    session = get_us_session()
    cache_key = f"us_extended_{session}"

    def _fetch():
        logger.info("Extended hours tarama başlıyor ({})...", session.upper())
        # Tam evren listesi
        all_syms = get_full_us_universe()
        logger.info("Toplam {} hisse sorgulanıyor...", len(all_syms))

        quotes = _yf_quote_batch(all_syms, session)

        # Sıfır fiyatlı ve micro-cap spam filtresi
        valid = [q for q in quotes if q.price > 0.5 and q.volume_usd > 100_000]

        gainers     = sorted(valid, key=lambda q: q.change_pct, reverse=True)[:top_n]
        losers      = sorted(valid, key=lambda q: q.change_pct)[:top_n]
        most_active = sorted(valid, key=lambda q: q.volume_usd, reverse=True)[:top_n]

        logger.info(
            "Extended tarama ({}) tamamlandı: {} geçerli hisse | "
            "En çok yükselen: {} ({:+.2f}%) | En çok düşen: {} ({:+.2f}%)",
            session.upper(), len(valid),
            gainers[0].symbol if gainers else "-",
            gainers[0].change_pct if gainers else 0,
            losers[0].symbol if losers else "-",
            losers[0].change_pct if losers else 0,
        )
        return {"gainers": gainers, "losers": losers, "most_active": most_active,
                "session": session, "total_scanned": len(valid)}

    return _cached_call(cache_key, _fetch)


def screen_us(top_n: int = 25) -> dict[str, list[QuickQuote]]:
    """
    ABD piyasasını tarar. Seans durumuna göre farklı kaynak kullanır:
    • Regular seans  → Yahoo Finance screener (hızlı, önceden filtreli)
    • Pre/After-market → Tam evren batch taraması (~5,000 hisse)
    • Kapalı         → Son bilinen verileri döndürür
    """
    session = get_us_session()

    if session in ("premarket", "aftermarket"):
        logger.info("ABD piyasası {} seansında — tam evren taranıyor...", session.upper())
        return screen_us_extended(top_n)

    def _fetch():
        gainers     = _yf_predefined_screen("day_gainers",              top_n)
        losers      = _yf_predefined_screen("day_losers",               top_n)
        most_active = _yf_predefined_screen("most_actives",             top_n)
        growth      = _yf_predefined_screen("growth_technology_stocks", top_n)

        if not gainers and not most_active:
            logger.warning("Yahoo screener çalışmıyor, fallback batch taraması kullanılıyor.")
            return _fallback_us_screen(top_n)

        logger.info(
            "ABD screener (regular): kazananlar={} kaybeder={} aktif={}",
            len(gainers), len(losers), len(most_active),
        )
        return {
            "gainers": gainers,
            "losers": losers,
            "most_active": most_active,
            "growth": growth,
            "session": session,
        }

    return _cached_call("us_screen_regular", _fetch)


def _fallback_us_screen(top_n: int) -> dict[str, list[QuickQuote]]:
    """
    Yahoo screener çalışmazsa: S&P 500 + NASDAQ 100'den batch download ile
    en hareketli hisseleri bul.
    """
    import yfinance as yf

    symbols = get_us_universe()[:200]  # İlk 200 ile sınırla (performans)
    try:
        df = yf.download(symbols, period="2d", progress=False, threads=True, auto_adjust=True)
        closes = df["Close"] if "Close" in df else df.get("Adj Close")
        if closes is None or closes.empty:
            return {"gainers": [], "losers": [], "most_active": []}

        chg = closes.pct_change().iloc[-1] * 100
        vol = df.get("Volume", df.get("Volume"))

        quotes = []
        for sym in closes.columns:
            try:
                c = float(closes[sym].iloc[-1])
                ch = float(chg[sym])
                v = float(vol[sym].iloc[-1]) * c if vol is not None else 0
                if c > 0:
                    quotes.append(QuickQuote(sym, c, ch, v, "us"))
            except Exception:
                continue

        gainers     = sorted(quotes, key=lambda q: q.change_pct, reverse=True)[:top_n]
        losers      = sorted(quotes, key=lambda q: q.change_pct)[:top_n]
        most_active = sorted(quotes, key=lambda q: q.volume_usd, reverse=True)[:top_n]
        return {"gainers": gainers, "losers": losers, "most_active": most_active}
    except Exception as e:
        logger.warning("Fallback ABD tarama hatası: {}", e)
        return {"gainers": [], "losers": [], "most_active": []}


# ── BIST: yfinance toplu indirme ──────────────────────────────────────────────

def screen_bist(top_n: int = 20) -> dict[str, list[QuickQuote]]:
    """
    BIST sembollerini batch yfinance ile tarar.
    2 günlük günlük veriyle % değişim ve hacim hesaplar.
    """
    def _fetch():
        import yfinance as yf

        symbols = get_bist_symbols()
        try:
            import warnings, logging
            logging.getLogger("yfinance").setLevel(logging.CRITICAL)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = yf.download(symbols, period="2d", progress=False, threads=True, auto_adjust=True)
            closes = df["Close"] if "Close" in df.columns.get_level_values(0) else None
            if closes is None:
                return {"gainers": [], "losers": [], "most_active": []}

            chg = closes.pct_change().iloc[-1] * 100
            vol_df = df.get("Volume")

            quotes = []
            for sym in closes.columns:
                try:
                    c  = float(closes[sym].iloc[-1])
                    ch = float(chg[sym])
                    v  = float(vol_df[sym].iloc[-1]) * c if vol_df is not None else 0
                    if c > 0 and not (ch != ch):   # NaN check
                        quotes.append(QuickQuote(sym, c, ch, v, "bist"))
                except Exception:
                    continue

            gainers     = sorted(quotes, key=lambda q: q.change_pct, reverse=True)[:top_n]
            losers      = sorted(quotes, key=lambda q: q.change_pct)[:top_n]
            most_active = sorted(quotes, key=lambda q: q.volume_usd, reverse=True)[:top_n]

            logger.info(
                "BIST screener: {} hisse tarandı | En çok yükselen: {} (+{:.1f}%)",
                len(quotes),
                gainers[0].symbol if gainers else "-",
                gainers[0].change_pct if gainers else 0,
            )
            return {"gainers": gainers, "losers": losers, "most_active": most_active}
        except Exception as e:
            logger.warning("BIST screener hatası: {}", e)
            return {"gainers": [], "losers": [], "most_active": []}

    return _cached_call("bist_screen", _fetch)


# ── Ana fonksiyon: dinamik izleme listesi ────────────────────────────────────

def get_dynamic_watchlist(
    include_us: bool = True,
    include_crypto: bool = True,
    include_bist: bool = True,
    top_per_category: int = 15,
    min_crypto_volume: float = 5_000_000,
) -> dict[str, list[str]]:
    """
    Tüm piyasaları tarar ve kategorilere göre en ilgi çekici sembolleri döndürür.

    Dönüş:
        {
          "us_gainers":      [...],  # Gün kazananları
          "us_losers":       [...],  # Gün kaybedenler
          "us_most_active":  [...],  # En yüksek hacim
          "us_growth":       [...],  # Büyüme hisseleri
          "crypto_gainers":  [...],
          "crypto_losers":   [...],
          "crypto_most_active": [...],
          "bist_gainers":    [...],
          "bist_losers":     [...],
          "bist_most_active":[...],
          "all_priority":    [...],  # Tüm kategorilerin öncelikli birleşimi
        }
    """
    result: dict[str, list[str]] = {}
    all_priority: list[str] = []

    def _syms(quotes: list[QuickQuote]) -> list[str]:
        return [q.symbol for q in quotes]

    if include_us:
        us = screen_us(top_per_category)
        result["us_gainers"]     = _syms(us.get("gainers", []))
        result["us_losers"]      = _syms(us.get("losers",  []))
        result["us_most_active"] = _syms(us.get("most_active", []))
        result["us_growth"]      = _syms(us.get("growth", []))
        # Öncelik: gün kazananları ve en aktifler
        for sym in result["us_gainers"] + result["us_most_active"]:
            if sym not in all_priority:
                all_priority.append(sym)

    if include_crypto:
        crypto = screen_crypto(top_per_category, min_crypto_volume)
        result["crypto_gainers"]     = _syms(crypto.get("gainers", []))
        result["crypto_losers"]      = _syms(crypto.get("losers",  []))
        result["crypto_most_active"] = _syms(crypto.get("most_active", []))
        for sym in result["crypto_gainers"] + result["crypto_most_active"]:
            if sym not in all_priority:
                all_priority.append(sym)

    if include_bist:
        bist = screen_bist(top_per_category)
        result["bist_gainers"]     = _syms(bist.get("gainers", []))
        result["bist_losers"]      = _syms(bist.get("losers",  []))
        result["bist_most_active"] = _syms(bist.get("most_active", []))
        for sym in result["bist_gainers"] + result["bist_most_active"]:
            if sym not in all_priority:
                all_priority.append(sym)

    result["all_priority"] = all_priority
    return result


def print_market_overview(watchlist: dict[str, list[str]]) -> None:
    """Konsolda piyasa özetini gösterir."""
    from loguru import logger

    sections = [
        ("ABD Kazananlar",   "us_gainers"),
        ("ABD Kaybedenler",  "us_losers"),
        ("ABD En Aktif",     "us_most_active"),
        ("Kripto Kazananlar","crypto_gainers"),
        ("Kripto Kaybedenler","crypto_losers"),
        ("Kripto En Aktif",  "crypto_most_active"),
        ("BIST Kazananlar",  "bist_gainers"),
        ("BIST Kaybedenler", "bist_losers"),
        ("BIST En Aktif",    "bist_most_active"),
    ]
    for label, key in sections:
        syms = watchlist.get(key, [])
        if syms:
            logger.info("{}: {}", label, ", ".join(syms[:10]))
