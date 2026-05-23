"""Tüm borsalardaki hisse evrenini yönetir — ücretsiz kaynaklar."""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.parse
from loguru import logger

_cache: dict[str, tuple[list, float]] = {}
_TTL_LONG  = 3600 * 12   # 12 saat (sembol listeleri çok sık değişmez)
_TTL_SHORT = 60           # 1 dakika (24h istatistikleri)


def _cached(key: str, ttl: float, fn):
    now = time.time()
    if key in _cache:
        val, ts = _cache[key]
        if now - ts < ttl:
            return val
    val = fn()
    _cache[key] = (val, now)
    return val


# ── ABD: TAM EVREN — NYSE + NASDAQ (tüm listelenmiş hisseler) ────────────────

def get_full_us_universe() -> list[str]:
    """
    NYSE + NASDAQ'taki TÜM hisseleri çeker (~5,000-6,000 sembol).
    Kaynaklar (sırayla denenir):
      1. SEC EDGAR company_tickers.json — ~10,000 kayıtlı şirket (en güvenilir)
      2. NASDAQ Trader FTP txt dosyaları — ~6,300 listelenen hisse
      3. Fallback: S&P 500 + NASDAQ 100 statik liste
    ETF, warrant, preferred share, test sembolleri filtrelenir.
    """
    def _fetch():
        # ── Kaynak 1: SEC EDGAR (en güvenilir) ──────────────────────────────
        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            req = urllib.request.Request(url, headers={"User-Agent": "trading-bot/1.0 contact@example.com"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read())

            skip_keywords = ["ETF", "FUND", "TRUST", "WARRANT", "RIGHT", "UNIT",
                             "PREFERRED", "NOTE", "BOND", "DEPOSITARY"]
            symbols = []
            for entry in data.values():
                sym  = str(entry.get("ticker", "")).strip().upper()
                name = str(entry.get("title",  "")).strip().upper()
                if not sym or not sym.replace("-", "").isalpha():
                    continue
                if any(c in sym for c in ["+", "^", "$", "~", ".", "/"]):
                    continue
                if not (1 <= len(sym) <= 5):
                    continue
                if any(kw in name for kw in skip_keywords):
                    continue
                symbols.append(sym)

            result = sorted(set(symbols))
            if len(result) > 1000:
                logger.info("Tam ABD evreni yüklendi (SEC EDGAR): {} hisse", len(result))
                return result
        except Exception as e:
            logger.warning("SEC EDGAR hatası: {}", e)

        # ── Kaynak 2: NASDAQ Trader FTP ─────────────────────────────────────
        all_symbols = []
        urls = [
            "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
            "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        ]
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=20) as r:
                    lines = r.read().decode("utf-8", errors="ignore").splitlines()
                is_nasdaq = "nasdaqlisted" in url
                for line in lines[1:]:
                    parts = line.split("|")
                    if len(parts) < 3:
                        continue
                    if is_nasdaq:
                        sym  = parts[0].strip()
                        name = parts[1].strip().upper()
                        test = parts[3].strip() if len(parts) > 3 else ""
                        fin  = parts[4].strip() if len(parts) > 4 else ""
                        etf  = parts[5].strip() if len(parts) > 5 else ""
                        if test == "Y" or etf == "Y" or fin == "D":
                            continue
                    else:
                        sym  = parts[0].strip()
                        name = parts[1].strip().upper()
                        etf  = parts[4].strip() if len(parts) > 4 else ""
                        test = parts[6].strip() if len(parts) > 6 else ""
                        if test == "Y" or etf == "Y":
                            continue
                    if not sym or not sym.replace("-", "").isalpha():
                        continue
                    if any(c in sym for c in ["+", "^", "$", "~", "."]):
                        continue
                    if not (1 <= len(sym) <= 5):
                        continue
                    skip_kw = [" ETF", " FUND", "TRUST", "WARRANT", "RIGHTS", " UNIT"]
                    if any(kw in name for kw in skip_kw):
                        continue
                    all_symbols.append(sym)
            except Exception as e:
                logger.warning("NASDAQ FTP ({}) hatası: {}", url, e)

        result = sorted(set(all_symbols))
        if result:
            logger.info("Tam ABD evreni yüklendi (NASDAQ FTP): {} hisse", len(result))
            return result

        # ── Kaynak 3: Fallback statik liste ─────────────────────────────────
        logger.warning("Tüm kaynaklar başarısız — S&P 500 + NASDAQ 100 fallback kullanılıyor.")
        return sorted(set(_SP500_FALLBACK + _NASDAQ100_FALLBACK))

    return _cached("full_us_universe", _TTL_LONG, _fetch)


# ── ABD: S&P 500 ─────────────────────────────────────────────────────────────

def get_sp500_symbols() -> list[str]:
    """Wikipedia'dan S&P 500 sembollerini çeker (~500 hisse)."""
    def _fetch():
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                html = r.read().decode("utf-8", errors="ignore")
            # İlk tablodaki <td> hücrelerinden sembolleri çek
            import re
            # wikitable'daki ilk sütun (Symbol)
            rows = re.findall(r'<tr[^>]*>.*?</tr>', html, re.DOTALL)
            symbols = []
            for row in rows[1:]:  # başlık satırını atla
                cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
                if cells:
                    sym = re.sub(r'<[^>]+>', '', cells[0]).strip()
                    sym = sym.replace('.', '-')  # BRK.B → BRK-B (yfinance formatı)
                    if sym and 1 <= len(sym) <= 5 and sym.replace('-', '').isalpha():
                        symbols.append(sym)
            logger.info("S&P 500 sembolleri yüklendi: {} adet", len(symbols))
            return symbols
        except Exception as e:
            logger.warning("S&P 500 çekilemedi: {}", e)
            return _SP500_FALLBACK

    return _cached("sp500", _TTL_LONG, _fetch)


def get_nasdaq100_symbols() -> list[str]:
    """Wikipedia'dan NASDAQ 100 sembollerini çeker."""
    def _fetch():
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                html = r.read().decode("utf-8", errors="ignore")
            import re
            rows = re.findall(r'<tr[^>]*>.*?</tr>', html, re.DOTALL)
            symbols = []
            for row in rows[1:]:
                cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
                if len(cells) >= 2:
                    sym = re.sub(r'<[^>]+>', '', cells[1]).strip()
                    if sym and 1 <= len(sym) <= 5 and sym.replace('-', '').isalpha():
                        symbols.append(sym)
            logger.info("NASDAQ 100 sembolleri yüklendi: {} adet", len(symbols))
            return symbols if symbols else _NASDAQ100_FALLBACK
        except Exception as e:
            logger.warning("NASDAQ 100 çekilemedi: {}", e)
            return _NASDAQ100_FALLBACK

    return _cached("nasdaq100", _TTL_LONG, _fetch)


def get_us_universe() -> list[str]:
    """S&P 500 + NASDAQ 100 birleşimi (tekrar edenleri kaldırır)."""
    sp = set(get_sp500_symbols())
    nq = set(get_nasdaq100_symbols())
    combined = sorted(sp | nq)
    logger.info("ABD hisse evreni: {} benzersiz sembol", len(combined))
    return combined


# ── BIST: Borsa İstanbul ─────────────────────────────────────────────────────

BIST_ALL_SYMBOLS: list[str] = [
    # BIST 30
    "THYAO.IS", "GARAN.IS", "ASELS.IS", "EREGL.IS", "AKBNK.IS",
    "YKBNK.IS", "KCHOL.IS", "SISE.IS",  "BIMAS.IS", "TUPRS.IS",
    "SAHOL.IS", "TCELL.IS", "FROTO.IS", "TOASO.IS", "ARCLK.IS",
    "KRDMD.IS", "PETKM.IS", "TTKOM.IS", "KONTR.IS", "ENKAI.IS",
    "MGROS.IS", "HEKTS.IS", "DOAS.IS",  "VESTL.IS", "LOGO.IS",
    "EKGYO.IS", "ISGYO.IS", "TAVHL.IS",
    # BIST 50 ek
    "CIMSA.IS", "ULKER.IS", "OTKAR.IS", "AGHOL.IS", "TSKB.IS",
    "AEFES.IS", "CCOLA.IS", "TURSG.IS", "BRISA.IS", "PGSUS.IS",
    "ODAS.IS",  "ENJSA.IS", "AKSEN.IS", "ZOREN.IS", "SKBNK.IS",
    "VAKBN.IS", "HALKB.IS", "ISCTR.IS",
    # BIST 100 ek
    "MAVI.IS",  "NETAS.IS", "ALARK.IS", "DEVA.IS",  "SELEC.IS",
    "TKFEN.IS", "BERA.IS",  "DYOBY.IS", "NUHCM.IS", "KARTN.IS",
    "KLMSN.IS", "GOLTS.IS", "KENT.IS",  "GENIL.IS", "ASUZU.IS",
    "BJKAS.IS", "FENER.IS", "GSRAY.IS", "TRGYO.IS", "ISGSY.IS",
    "BVSAN.IS", "ALKIM.IS", "CEMTS.IS", "CEMAS.IS", "GOODY.IS",
    "BFREN.IS", "ARSAN.IS",
    # Ek BIST hisseleri
    "GSDHO.IS", "PRKME.IS", "ISDMR.IS", "CANTE.IS", "KARSN.IS",
]


def _fetch_bist_dynamic() -> list[str]:
    """
    Birden fazla kaynaktan BIST sembollerini dinamik çekmeyi dener.
    Başarılı olan ilk kaynaktan döner.
    """
    import re

    # Kaynak 1: IsYatirim screener (JSON API)
    try:
        url = "https://www.isyatirim.com.tr/_Layouts/15/IsYatirim/Components/OneDesign/Screener/ScreenerPageData.ashx?culture=tr-TR&Type=hisse"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"})
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read())
        symbols = []
        for item in (data.get("data") or data if isinstance(data, list) else []):
            code = (item.get("kod") or item.get("sembol") or item.get("symbol") or "").strip().upper()
            if code and 2 <= len(code) <= 6 and code.replace(".", "").isalpha():
                symbols.append(f"{code}.IS" if not code.endswith(".IS") else code)
        if len(symbols) > 100:
            logger.info("BIST evreni IsYatirim'dan yüklendi: {} sembol", len(symbols))
            return sorted(set(symbols))
    except Exception:
        pass

    # Kaynak 2: KAP yeni endpoint
    for kap_url in [
        "https://www.kap.org.tr/tr/api/memberEquity",
        "https://www.kap.org.tr/en/api/memberEquity",
        "https://www.kap.org.tr/tr/api/member/equity",
    ]:
        try:
            req = urllib.request.Request(kap_url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
            symbols = []
            for item in (data if isinstance(data, list) else data.get("data", [])):
                code = (item.get("stampCode") or item.get("memberCode") or item.get("code") or "").strip().upper()
                if code and 2 <= len(code) <= 6 and code.isalpha():
                    symbols.append(f"{code}.IS")
            if len(symbols) > 100:
                logger.info("BIST evreni KAP'tan yüklendi: {} sembol", len(symbols))
                return sorted(set(symbols))
        except Exception:
            pass

    return []


# Kapsamlı BIST statik listesi — BIST TÜM (~400 hisse)
BIST_ALL_SYMBOLS: list[str] = [
    # ── BIST 30 ──────────────────────────────────────────────────────────────
    "THYAO.IS", "GARAN.IS", "ASELS.IS", "EREGL.IS", "AKBNK.IS",
    "YKBNK.IS", "KCHOL.IS", "SISE.IS",  "BIMAS.IS", "TUPRS.IS",
    "SAHOL.IS", "TCELL.IS", "FROTO.IS", "TOASO.IS", "ARCLK.IS",
    "KRDMD.IS", "PETKM.IS", "TTKOM.IS", "KONTR.IS", "ENKAI.IS",
    "MGROS.IS", "HEKTS.IS", "DOAS.IS",  "VESTL.IS", "LOGO.IS",
    "EKGYO.IS", "ISGYO.IS", "TAVHL.IS",
    # ── BIST 50 ──────────────────────────────────────────────────────────────
    "CIMSA.IS", "ULKER.IS", "OTKAR.IS", "AGHOL.IS", "TSKB.IS",
    "AEFES.IS", "CCOLA.IS", "TURSG.IS", "BRISA.IS", "PGSUS.IS",
    "ODAS.IS",  "ENJSA.IS", "AKSEN.IS", "ZOREN.IS", "SKBNK.IS",
    "VAKBN.IS", "HALKB.IS", "ISCTR.IS",
    # ── BIST 100 ─────────────────────────────────────────────────────────────
    "MAVI.IS",  "NETAS.IS", "ALARK.IS", "DEVA.IS",  "SELEC.IS",
    "TKFEN.IS", "BERA.IS",  "DYOBY.IS", "NUHCM.IS", "KARTN.IS",
    "KLMSN.IS", "GOLTS.IS", "KENT.IS",  "GENIL.IS", "ASUZU.IS",
    "BJKAS.IS", "FENER.IS", "GSRAY.IS", "TRGYO.IS", "ISGSY.IS",
    "BVSAN.IS", "ALKIM.IS", "CEMTS.IS", "CEMAS.IS", "GOODY.IS",
    "BFREN.IS", "ARSAN.IS",
    # ── Bankacılık ───────────────────────────────────────────────────────────
    "ALBRK.IS", "FIBABANKA.IS", "QNBFB.IS", "QNBFL.IS", "ODEABANK.IS",
    "ICBCT.IS", "BURCE.IS", "BURVA.IS",
    # ── Sigorta ──────────────────────────────────────────────────────────────
    "AKGRT.IS", "ANSGR.IS", "ANHYT.IS", "ASYLK.IS", "GUSGF.IS",
    "RAYEN.IS", "RAYSG.IS", "TUREX.IS",
    # ── Gayrimenkul / GYO ────────────────────────────────────────────────────
    "ALGYO.IS", "AVGYO.IS", "DRGYO.IS", "HLGYO.IS", "HURGZ.IS",
    "ISFIN.IS", "KRGYO.IS", "MRGYO.IS", "NUGYO.IS", "OZKGY.IS",
    "PEGYO.IS", "RYGYO.IS", "SNGYO.IS", "VKGYO.IS", "YGGYO.IS",
    "ZKGYO.IS", "AKSGY.IS", "AKFGY.IS", "AKMGY.IS",
    # ── Enerji ───────────────────────────────────────────────────────────────
    "AKENR.IS", "AKFYE.IS", "AKPAZ.IS", "AYES.IS",  "AYDEM.IS",
    "BMELK.IS", "CANTE.IS", "GESAN.IS", "GWIND.IS", "HUNER.IS",
    "KAPLM.IS", "KARYE.IS", "MAGEN.IS", "MEGAP.IS", "ONRYT.IS",
    "OZRDN.IS", "PRKAB.IS", "PRKME.IS", "SAYAS.IS", "SRVGY.IS",
    "TATEN.IS", "TUREX.IS", "TURGG.IS", "YASAS.IS",
    # ── Teknoloji / Yazılım ──────────────────────────────────────────────────
    "ARDYZ.IS", "ARENA.IS", "DGATE.IS", "KAREL.IS", "KFEIN.IS",
    "LINK.IS",  "NETAŞ.IS", "PAPIL.IS", "SMART.IS", "UBIT.IS",
    "UYUM.IS",  "VERTU.IS", "FORTE.IS", "INDES.IS", "KRONT.IS",
    # ── Otomotiv ─────────────────────────────────────────────────────────────
    "ATEKS.IS", "KARSN.IS", "MUTLU.IS", "PARSN.IS", "PETUN.IS",
    "TMSN.IS",  "ORMGE.IS",
    # ── Gıda / İçecek ────────────────────────────────────────────────────────
    "BANVT.IS", "CKOMD.IS", "ERSU.IS",  "FRIGO.IS", "KERVT.IS",
    "KNFRT.IS", "KONTR.IS", "MERKO.IS", "PENGD.IS", "PNSUT.IS",
    "SKTAS.IS", "TATGD.IS", "TUBORG.IS","ULUUN.IS", "VANGD.IS",
    "YILDIZ.IS","YKSLN.IS",
    # ── Kimya / İlaç ─────────────────────────────────────────────────────────
    "BIOTEK.IS","BMEKS.IS", "ECZYT.IS", "EGPRO.IS", "EREGL.IS",
    "GEDIK.IS", "GENTS.IS", "HEKTS.IS", "KLBMO.IS", "KRPLAS.IS",
    "KUTPO.IS", "LIOY.IS",  "MEGES.IS", "ORCAY.IS", "PKENT.IS",
    "SERVE.IS", "SILVR.IS", "SRVGY.IS", "ULUSE.IS", "ULUUN.IS",
    # ── Tekstil ──────────────────────────────────────────────────────────────
    "ALTIN.IS", "ARSAN.IS", "BISAS.IS", "BOSSA.IS", "DMSAS.IS",
    "ESCOM.IS", "GENTS.IS", "INTEM.IS", "KARYE.IS", "KRVGD.IS",
    "LUKSK.IS", "MRSHL.IS", "NTHOL.IS", "SKTAS.IS", "SNPAM.IS",
    "TIRE.IS",  "YATAS.IS",
    # ── İnşaat / Çimento ─────────────────────────────────────────────────────
    "ADANA.IS", "ADNAC.IS", "AFYON.IS", "AKCNS.IS", "ASGYO.IS",
    "BOLUC.IS", "BUCIM.IS", "FENIS.IS", "KONYA.IS", "MRDIN.IS",
    "SANKO.IS", "TRCAS.IS", "UNYEC.IS", "USAK.IS",
    # ── Demir-Çelik / Metal ──────────────────────────────────────────────────
    "CELHA.IS", "CEMTS.IS", "DMSAS.IS", "DYTTO.IS", "ISDMR.IS",
    "IZMDC.IS", "KRDMA.IS", "KRDMB.IS", "NIBAS.IS", "SARKY.IS",
    "TURSG.IS", "ULUSE.IS",
    # ── Holding ──────────────────────────────────────────────────────────────
    "AGYO.IS",  "DOHOL.IS", "GSDHO.IS", "HATEK.IS", "HLGYO.IS",
    "IHEVA.IS", "IHLAS.IS", "KRTEK.IS", "NTHOL.IS", "OSMEN.IS",
    "PKART.IS", "RYSAS.IS", "SANEL.IS", "SNPAM.IS", "TATGD.IS",
    "ULAS.IS",  "ULKER.IS", "VKING.IS", "YATAS.IS",
    # ── Ulaşım / Lojistik ────────────────────────────────────────────────────
    "CLEBI.IS", "DOAS.IS",  "DURDO.IS", "FILO.IS",  "GSDDE.IS",
    "KARSN.IS", "METUR.IS", "RYSAS.IS", "SAMAT.IS", "SASA.IS",
    "ULUSE.IS", "ULAS.IS",
    # ── Medya / Eğlence ──────────────────────────────────────────────────────
    "DOGAN.IS", "HURDA.IS", "HURGZ.IS", "KIPA.IS",  "MARTI.IS",
    "NTTUR.IS", "RYSAS.IS", "TKNSA.IS",
    # ── Ek BIST hisseleri ────────────────────────────────────────────────────
    "GSDHO.IS", "PRKME.IS", "ISDMR.IS", "KARSN.IS",
    "ACSEL.IS", "ADEL.IS",  "AGYO.IS",  "AHGAZ.IS", "AKALIN.IS",
    "AKBO.IS",  "AKCNS.IS", "AKFGY.IS", "AKFYE.IS", "AKTIF.IS",
    "ALFAS.IS", "ALTNS.IS", "ALTNY.IS", "ALVES.IS", "ANACM.IS",
    "ANHYT.IS", "ANSGR.IS", "ARASE.IS", "ARFEN.IS", "ARGE.IS",
    "ARTMS.IS", "ASELS.IS", "ASYLK.IS", "ATAKP.IS", "ATLAS.IS",
    "ATSYH.IS", "AVHOL.IS", "AVOD.IS",  "AYCES.IS", "AYGAZ.IS",
    "BASGZ.IS", "BEYAZ.IS", "BMELK.IS", "BORLS.IS", "BOSSA.IS",
    "BRKO.IS",  "BRKSN.IS", "BRYAT.IS", "BTCIM.IS", "BUCIM.IS",
    "BURCE.IS", "BURVA.IS", "BVSAN.IS", "CMBTN.IS", "CMENT.IS",
    "CONSE.IS", "COSMO.IS", "DAGHL.IS", "DENGE.IS", "DERHL.IS",
    "DERIM.IS", "DESPC.IS", "DEVA.IS",  "DGATE.IS", "DGKLB.IS",
    "DITAS.IS", "DMRGD.IS", "DNISI.IS", "DOBUR.IS", "DOCO.IS",
    "DOGUB.IS", "DOHOL.IS", "DOKTA.IS", "DURDO.IS", "DYOBY.IS",
    "ECZYT.IS", "EGEEN.IS", "EGGUB.IS", "EGPRO.IS", "EKGYO.IS",
    "EKIZ.IS",  "ELITE.IS", "EMKEL.IS", "EMNIS.IS", "ENJSA.IS",
    "ENKAI.IS", "EPLAS.IS", "ERSU.IS",  "ESCOM.IS", "ESEN.IS",
    "ETILR.IS", "ETYAT.IS", "EUPWR.IS", "EUREN.IS", "EYGYO.IS",
    "FADE.IS",  "FENER.IS", "FLAP.IS",  "FMIZP.IS", "FONET.IS",
    "FORTE.IS", "FRIGO.IS", "FROTO.IS", "FZLGY.IS", "GARFA.IS",
    "GEDIK.IS", "GEDZA.IS", "GENIL.IS", "GESAN.IS", "GLRYH.IS",
    "GLYHO.IS", "GOLTS.IS", "GOODY.IS", "GSRAY.IS", "GWIND.IS",
    "HATEK.IS", "HDFGS.IS", "HEDEF.IS", "HEKTS.IS", "HKTM.IS",
    "HLGYO.IS", "HRKET.IS", "HTTBT.IS", "HUNER.IS", "HURGZ.IS",
    "ICBCT.IS", "IHEVA.IS", "IHLAS.IS", "IHYAY.IS", "IMASM.IS",
    "INDES.IS", "INTEM.IS", "IPEKE.IS", "ISATR.IS", "ISFIN.IS",
    "ISGSY.IS", "ISGYO.IS", "ISKPL.IS", "ISYAT.IS", "ITTFH.IS",
    "IZFAS.IS", "IZMDC.IS", "JANTS.IS", "KAPLM.IS", "KAREL.IS",
    "KARTN.IS", "KATMR.IS", "KAYSE.IS", "KBORU.IS", "KCHOL.IS",
    "KENT.IS",  "KERVT.IS", "KFEIN.IS", "KGYO.IS",  "KLBMO.IS",
    "KNFRT.IS", "KONYA.IS", "KORDS.IS", "KRGYO.IS", "KRPLAS.IS",
    "KRSAN.IS", "KRTEK.IS", "KTLEV.IS", "KUTPO.IS", "LIDER.IS",
    "LIOY.IS",  "LINK.IS",  "LKMNH.IS", "LUKSK.IS", "MAALT.IS",
    "MARTI.IS", "MAVI.IS",  "MEGAP.IS", "MEGES.IS", "MERKO.IS",
    "METRO.IS", "MGROS.IS", "MNDRS.IS", "MRSHL.IS", "MTRKS.IS",
    "MUTLU.IS", "NETAS.IS", "NIBAS.IS", "NTGAZ.IS", "NTTUR.IS",
    "NUHCM.IS", "NUGYO.IS", "OBAMS.IS", "ODAS.IS",  "ONRYT.IS",
    "ORCAY.IS", "OSEN.IS",  "OTKAR.IS", "OYAYO.IS", "OZKGY.IS",
    "OZRDN.IS", "PAPIL.IS", "PARSN.IS", "PASEU.IS", "PEGYO.IS",
    "PENGD.IS", "PETUN.IS", "PKART.IS", "PKENT.IS", "PLTUR.IS",
    "PNSUT.IS", "POLHO.IS", "PRKAB.IS", "PRZMA.IS", "RAYEN.IS",
    "RAYSG.IS", "RGYAS.IS", "RYGYO.IS", "RYSAS.IS", "SAMAT.IS",
    "SANEL.IS", "SANKO.IS", "SARKY.IS", "SASA.IS",  "SAYAS.IS",
    "SDTTR.IS", "SEKFK.IS", "SEKUR.IS", "SELEC.IS", "SERVE.IS",
    "SILVR.IS", "SISE.IS",  "SKBNK.IS", "SKTAS.IS", "SMART.IS",
    "SNGYO.IS", "TATGD.IS", "TATEN.IS", "TKNSA.IS", "TKFEN.IS",
    "TMSN.IS",  "TOASO.IS", "TRCAS.IS", "TRGYO.IS", "TSKB.IS",
    "TUBORG.IS","TUCLK.IS", "TUGBS.IS", "TURGG.IS", "TURSG.IS",
    "ULUSE.IS", "ULUUN.IS", "UNYEC.IS", "USAK.IS",  "VAKFN.IS",
    "VKGYO.IS", "VKING.IS", "YATAS.IS", "YGGYO.IS", "YILDIZ.IS",
    "YKGYO.IS", "YKSLN.IS", "YUNSA.IS", "ZKGYO.IS", "ZOREN.IS",
]


def get_bist_symbols() -> list[str]:
    """
    Tüm BIST hisselerini döndürür.
    Önce dinamik kaynakları dener (IsYatirim, KAP), başarısız olursa
    kapsamlı statik listeye döner (~400 sembol).
    """
    def _fetch():
        syms = _fetch_bist_dynamic()
        if len(syms) > 100:
            return syms
        logger.info("Dinamik BIST listesi alınamadı, statik liste kullanılıyor ({} sembol)", len(BIST_ALL_SYMBOLS))
        return sorted(set(BIST_ALL_SYMBOLS))

    return _cached("bist", _TTL_LONG, _fetch)


# ── Kripto: Binance ───────────────────────────────────────────────────────────

def get_crypto_universe(quote: str = "USDT", min_volume_usd: float = 1_000_000) -> list[str]:
    """
    Binance'ten tüm USDT spot çiftlerini çeker.
    min_volume_usd: 24h hacim filtresi (varsayılan $1M)
    Döndürdüğü format: BTC-USD, ETH-USD ... (yfinance uyumlu)
    """
    def _fetch():
        url = f"https://api.binance.com/api/v3/ticker/24hr"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
            symbols = []
            for item in data:
                sym = item.get("symbol", "")
                if not sym.endswith(quote):
                    continue
                vol = float(item.get("quoteVolume", 0))
                if vol < min_volume_usd:
                    continue
                base = sym[:-len(quote)]
                # yfinance formatına çevir: BTCUSDT → BTC-USD
                yf_sym = f"{base}-USD"
                symbols.append(yf_sym)
            logger.info("Kripto evreni (Binance USDT, >$1M hacim): {} çift", len(symbols))
            return sorted(symbols)
        except Exception as e:
            logger.warning("Binance evren çekilemedi: {}", e)
            return ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
                    "ADA-USD", "AVAX-USD", "DOGE-USD", "DOT-USD", "MATIC-USD"]

    return _cached(f"crypto_universe_{quote}", _TTL_LONG, _fetch)


# ── Fallback listeleri ────────────────────────────────────────────────────────

_SP500_FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
    "NFLX", "INTC", "JPM", "BAC", "WFC", "GS", "MS", "V", "MA",
    "JNJ", "PFE", "UNH", "ABBV", "MRK", "BMY", "LLY", "AMGN",
    "XOM", "CVX", "COP", "SLB", "EOG", "WMT", "HD", "COST", "TGT",
    "NKE", "SBUX", "MCD", "DIS", "CMCSA", "T", "VZ", "TMUS",
    "BA", "CAT", "GE", "HON", "MMM", "RTX", "LMT", "NOC", "GD",
]

_NASDAQ100_FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "GOOGL", "GOOG",
    "AVGO", "COST", "NFLX", "ASML", "AMD", "ADBE", "QCOM", "INTC",
    "CSCO", "TXN", "INTU", "AMAT", "ISRG", "BKNG", "AMGN", "MU",
    "PANW", "ADI", "LRCX", "MRVL", "KLAC", "CDNS", "SNPS", "FTNT",
    "CRWD", "MELI", "PYPL", "ABNB", "DDOG", "WDAY", "TEAM", "ZS",
    "OKTA", "SPLK", "VEEV", "DOCU", "ANSS", "IDXX", "ILMN", "BIIB",
]
