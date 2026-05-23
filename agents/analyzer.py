"""Derin Yatırım Analizi — 2 yıllık günlük + haftalık grafik, teknik + ML + Qwen yorumu."""
from __future__ import annotations

import textwrap
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from loguru import logger

from data.indicators import compute_all
from ml.support_resistance import find_sr_levels, fibonacci_levels, sr_signal_score
from agents.sentiment import analyze as analyze_sentiment
from data.sources.yahoo_news import fetch_yahoo_news


# ── Veri çekici ───────────────────────────────────────────────────────────────

def _fetch(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        return df
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]].dropna()


def _pct(a: float, b: float) -> str:
    if b == 0:
        return "—"
    return f"{(a / b - 1) * 100:+.1f}%"


# ── Trend belirleyici ─────────────────────────────────────────────────────────

def _trend_label(df: pd.DataFrame) -> str:
    close = df["close"]
    ema50  = close.ewm(span=50,  adjust=False).mean().iloc[-1]
    ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
    price  = close.iloc[-1]
    if price > ema50 > ema200:
        return "GÜÇLÜ YUKARI TREND"
    elif price > ema200:
        return "YUKARI TREND (zayıf)"
    elif price < ema50 < ema200:
        return "GÜÇLÜ ASAGI TREND"
    elif price < ema200:
        return "ASAGI TREND (zayıf)"
    return "YATAY / KARARSIZ"


# ── Ana analiz fonksiyonu ─────────────────────────────────────────────────────

def deep_analyze(symbol: str, period: str = "2y") -> dict:
    """
    Sembol için kapsamlı teknik + ML analizi yapar.

    Returns:
        Analiz özeti dict — 'report_text' anahtarında insan okunur rapor var.
    """
    sym = symbol.upper()
    # Bilinen ABD hisseleri ve kripto için .IS ekleme
    _US_SYMS = {
        "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","TSLA","AMD","NFLX",
        "INTC","JPM","BAC","GS","V","MA","JNJ","PFE","MRNA","XOM","CVX",
        "WMT","COST","DIS","SBUX","BA","CAT","GE","F","GM","UBER","PYPL",
        "COIN","HOOD","SOFI","PLTR","SNOW","NET","CRWD","DDOG","MDB","RBLX",
        "SHOP","SQ","ROKU","ZM","DOCU","TWLO","OKTA","DKNG","ABNB","DASH",
        "LYFT","SPOT","PINS","SNAP","BABA","JD","PDD","NIO","XPEV","LI",
        "SPY","QQQ","IWM","GLD","SLV","TLT","HYG","VIX",
    }
    if "-" in sym or "." in sym:
        pass  # BTC-USD, THYAO.IS gibi — olduğu gibi bırak
    elif sym in _US_SYMS:
        pass  # ABD hissesi — .IS ekleme
    else:
        sym = sym + ".IS"  # varsayılan: BIST

    logger.info("Derin analiz başlıyor: {} | Periyot: {}", sym, period)

    # ── 1. Veri çek ───────────────────────────────────────────────────────────
    daily  = _fetch(sym, period, "1d")
    weekly = _fetch(sym, period, "1wk")

    if daily.empty or len(daily) < 30:
        return {"error": f"{sym} için yeterli veri bulunamadı."}

    # ── 2. İndikatörler ───────────────────────────────────────────────────────
    daily_ind  = compute_all(daily.copy())
    weekly_ind = compute_all(weekly.copy()) if len(weekly) >= 20 else None

    last   = daily_ind.iloc[-1]
    price  = float(last["close"])

    # Temel indikatörler
    rsi      = float(last.get("rsi", 0))
    macd_val = float(last.get("macd", 0))
    macd_sig = float(last.get("macd_signal", 0))
    macd_h   = float(last.get("macd_hist", 0))
    adx_val  = float(last.get("adx", 0))
    bb_pct   = float(last.get("bb_pct", 0.5))
    ema21    = float(last.get("ema_21", 0))
    ema55    = float(last.get("ema_55", 0))
    ema200   = float(daily["close"].ewm(span=200, adjust=False).mean().iloc[-1])
    sma50    = float(daily["close"].rolling(50).mean().iloc[-1])
    vol_ratio = float(last.get("volume_ratio", 1.0))
    stoch_k  = float(last.get("stoch_k", 50))
    stoch_d  = float(last.get("stoch_d", 50))
    cmf      = float(last.get("cmf", 0))
    obv_trend = int(last.get("obv_trend", 0))
    atr      = float(last.get("atr", 0))

    # Haftalık RSI
    weekly_rsi = None
    weekly_macd_h = None
    if weekly_ind is not None and len(weekly_ind) > 0:
        weekly_rsi    = float(weekly_ind["rsi"].iloc[-1]) if "rsi" in weekly_ind else None
        weekly_macd_h = float(weekly_ind["macd_hist"].iloc[-1]) if "macd_hist" in weekly_ind else None

    # ── 3. Fiyat performansı ──────────────────────────────────────────────────
    def _ago(days: int) -> float | None:
        sub = daily[daily.index <= daily.index[-1]]
        sub = sub.iloc[:-1] if len(sub) > 1 else sub
        target = daily.index[-1] - pd.Timedelta(days=days)
        past = daily[daily.index <= target]
        return float(past["close"].iloc[-1]) if not past.empty else None

    p1m  = _ago(30)
    p3m  = _ago(90)
    p6m  = _ago(180)
    p1y  = _ago(365)
    high52 = float(daily["high"].tail(252).max())
    low52  = float(daily["low"].tail(252).min())

    # ── 4. Destek / Direnç ────────────────────────────────────────────────────
    supports, resistances = find_sr_levels(daily, window=10, n_levels=4)
    fib = fibonacci_levels(daily, lookback=120)
    sr_score_val, sr_reason = sr_signal_score(daily)

    # ── 5. ML sinyali ─────────────────────────────────────────────────────────
    ml_sig = ml_conf = None
    try:
        from ml.predictor import ml_signal as _ml_signal
        market = "bist" if sym.endswith(".IS") else ("crypto" if "-USD" in sym or "-PERP" in sym else "us")
        ml_sig, ml_conf = _ml_signal(daily, threshold=0.60, market=market)
    except Exception:
        pass

    # ── 6. Haberler + Sentiment ───────────────────────────────────────────────
    news_items: list[dict] = []
    news_sentiment_score = 0.0
    try:
        raw_news = fetch_yahoo_news(sym, max_items=10)
        for item in raw_news:
            title = item.get("title", "")
            desc  = item.get("description", "")
            pub   = item.get("published", "")
            text  = f"{title} {desc}"
            sent  = analyze_sentiment(text)
            news_items.append({
                "title":     title,
                "published": pub,
                "label":     sent.label,
                "score":     sent.score,
                "keywords":  sent.matched_keywords[:3],
            })
        if news_items:
            news_sentiment_score = sum(n["score"] for n in news_items) / len(news_items)
    except Exception:
        pass

    # ── 7. Hacim trendi ───────────────────────────────────────────────────────
    vol_30d_avg = float(daily["volume"].tail(30).mean())
    vol_90d_avg = float(daily["volume"].tail(90).mean())
    vol_trend_label = (
        "Artıyor" if vol_30d_avg > vol_90d_avg * 1.1 else
        "Azalıyor" if vol_30d_avg < vol_90d_avg * 0.9 else
        "Stabil"
    )

    # ── 8. Trend ──────────────────────────────────────────────────────────────
    trend = _trend_label(daily)

    # ── 9. Genel skor (teknik puanlama) ──────────────────────────────────────
    score = 0
    bullets_bull: list[str] = []
    bullets_bear: list[str] = []

    def _bull(msg: str, pts: int = 1):
        nonlocal score; score += pts; bullets_bull.append(f"✅ {msg} (+{pts})")

    def _bear(msg: str, pts: int = 1):
        nonlocal score; score -= pts; bullets_bear.append(f"❌ {msg} (-{pts})")

    # Trend (EMA)
    if price > ema21 > ema55 > ema200:
        _bull("EMA hizası tam yukari (21>55>200)", 3)
    elif price > ema200:
        _bull("Fiyat EMA200 üstünde", 1)
    elif price < ema21 < ema55:
        _bear("EMA hizası tam asagi (21<55)", 2)
    else:
        _bear("Fiyat EMA200 altında", 1)

    # RSI günlük
    if rsi < 30:
        _bull(f"RSI aşırı satım bölgesi ({rsi:.0f}) — potansiyel dönüş", 2)
    elif 40 <= rsi <= 60:
        _bull(f"RSI sağlıklı bölge ({rsi:.0f})", 1)
    elif rsi > 75:
        _bear(f"RSI aşırı alım ({rsi:.0f}) — düzeltme riski", 2)

    # RSI haftalık
    if weekly_rsi:
        if weekly_rsi < 40:
            _bull(f"Haftalık RSI düşük ({weekly_rsi:.0f}) — uzun vadede cazip", 2)
        elif weekly_rsi > 70:
            _bear(f"Haftalık RSI yüksek ({weekly_rsi:.0f})", 1)

    # MACD
    if macd_h > 0 and float(daily_ind["macd_hist"].iloc[-2]) <= 0:
        _bull("MACD taze yukarı kesişim", 2)
    elif macd_h > 0:
        _bull(f"MACD pozitif momentum", 1)
    elif macd_h < 0:
        _bear("MACD negatif momentum", 1)

    # ADX (trend gücü)
    if adx_val >= 25:
        _bull(f"Güçlü trend (ADX={adx_val:.0f})", 1)
    elif adx_val < 15:
        _bear(f"Zayıf/yatay piyasa (ADX={adx_val:.0f})", 1)

    # Bollinger
    if bb_pct <= 0.10:
        _bull(f"Bollinger alt bant — aşırı satım (%B={bb_pct:.0%})", 2)
    elif bb_pct >= 0.90:
        _bear(f"Bollinger üst bant — aşırı alım (%B={bb_pct:.0%})", 1)

    # Stochastic
    if stoch_k < 20 and stoch_k > stoch_d:
        _bull(f"Stochastic oversold dönüşü (K={stoch_k:.0f})", 1)
    elif stoch_k > 80:
        _bear(f"Stochastic overbought (K={stoch_k:.0f})", 1)

    # Destek/Direnç
    if sr_score_val > 0:
        _bull(sr_reason, sr_score_val)
    elif sr_score_val < 0:
        _bear(sr_reason, abs(sr_score_val))

    # Hacim
    if vol_ratio >= 1.5:
        _bull(f"Güçlü hacim ({vol_ratio:.1f}x ortalama)", 1)
    elif vol_trend_label == "Artıyor":
        _bull("Hacim trendi artıyor (alıcı ilgisi)", 1)
    elif vol_trend_label == "Azalıyor":
        _bear("Hacim trendi azalıyor", 1)

    # OBV
    if obv_trend == 1:
        _bull("OBV yükseliyor (birikim)", 1)
    elif obv_trend == -1:
        _bear("OBV düşüyor (dağıtım)", 1)

    # CMF
    if cmf > 0.15:
        _bull(f"Para girişi güçlü (CMF={cmf:+.2f})", 1)
    elif cmf < -0.15:
        _bear(f"Para çıkışı var (CMF={cmf:+.2f})", 1)

    # ML sinyali
    if ml_sig is not None:
        if ml_sig == 1 and ml_conf > 0.65:
            _bull(f"ML modeli AL sinyali ({ml_conf:.0%} güven)", 2)
        elif ml_sig == 1:
            _bull(f"ML modeli AL eğilimi ({ml_conf:.0%})", 1)
        elif ml_conf < 0.40:
            _bear(f"ML modeli BEKLE ({ml_conf:.0%})", 1)

    # Fiyat 52h pozisyonu
    dist_from_high = (price - high52) / high52
    dist_from_low  = (price - low52) / low52
    if dist_from_high > -0.10:
        _bear(f"52 hafta zirvesine yakın ({dist_from_high:+.0%})", 1)
    elif dist_from_low < 0.20:
        _bull(f"52 hafta dibine yakın ({dist_from_low:+.0%}) — deger firsati olabilir", 1)

    # Haber sentimenti (±2)
    if news_sentiment_score > 0.25:
        _bull(f"Haberler genel olumlu (skor: {news_sentiment_score:+.2f})", 2)
    elif news_sentiment_score > 0.10:
        _bull(f"Haberler hafif olumlu (skor: {news_sentiment_score:+.2f})", 1)
    elif news_sentiment_score < -0.25:
        _bear(f"Haberler genel olumsuz (skor: {news_sentiment_score:+.2f})", 2)
    elif news_sentiment_score < -0.10:
        _bear(f"Haberler hafif olumsuz (skor: {news_sentiment_score:+.2f})", 1)

    # Genel yorum
    max_score = 25
    score_pct = max(0, min(100, int((score + max_score) / (max_score * 2) * 100)))

    if score >= 6:
        verdict_short = "GÜÇLÜ AL"
        verdict_long  = "GÜÇLÜ AL"
    elif score >= 3:
        verdict_short = "AL / İZLE"
        verdict_long  = "ORTA VADELİ AL"
    elif score >= 0:
        verdict_short = "BEKLE / İZLE"
        verdict_long  = "NÖTR — Daha iyi giriş noktası bekle"
    elif score >= -3:
        verdict_short = "ZAYIF"
        verdict_long  = "KAÇIN — Düşüş riski var"
    else:
        verdict_short = "SAT / KAÇIN"
        verdict_long  = "KAÇIN — Güçlü düşüş sinyali"

    # Risk hesabı: 2x ATR stop-loss, %25 hedef
    risk_3m_sl = round(price - atr * 2.0, 2) if atr > 0 else round(price * 0.92, 2)
    risk_1y_tp = round(price * 1.25, 2)

    # ── 9. Rapor metni (tamamen ASCII — tum terminallerde calisiyor) ─────────
    sup_str  = " | ".join([f"{s:.4g}" for s in supports[:3]])    or "Bulunamadi"
    res_str  = " | ".join([f"{r:.4g}" for r in resistances[:3]]) or "Bulunamadi"
    bull_str = "\n  ".join(bullets_bull) if bullets_bull else "Yok"
    bear_str = "\n  ".join(bullets_bear) if bullets_bear else "Yok"
    currency = "TL" if sym.endswith(".IS") else "USD"

    S  = "=" * 66
    s2 = "-" * 66
    ml_sinyal  = "AL"    if ml_sig == 1 else ("BEKLE" if ml_sig == 0 else "-")
    ml_guven   = f"{ml_conf:.0%}" if ml_conf is not None else "-"
    now_str    = datetime.now(tz=timezone.utc).strftime("%d.%m.%Y %H:%M") + " UTC"
    sl_str     = f"{risk_3m_sl} {currency}" if risk_3m_sl else "-"
    tp_str     = f"{risk_1y_tp} {currency}"
    rsi_lbl    = (">70 asiri alim" if rsi > 70 else ("<30 asiri satim" if rsi < 30 else "normal bolge"))
    macd_lbl   = "yukari (+)" if macd_h > 0 else "asagi (-)"
    adx_lbl    = "guclu trend" if adx_val >= 25 else "zayif/yatay"
    bb_lbl     = "alt bant" if bb_pct < 0.2 else ("ust bant" if bb_pct > 0.8 else "orta bolge")
    ema200_lbl = "USTUNDE" if price > ema200 else "ALTINDA"
    cmf_lbl    = "para girisi" if cmf > 0 else "para cikisi"
    obv_lbl    = "Yukseliyor" if obv_trend == 1 else ("Dusuyor" if obv_trend == -1 else "Notr")

    lines = [
        S,
        f"  DERIN ANALIZ -- {sym}",
        f"  Tarih: {now_str}",
        S,
        "",
        "FIYAT BILGISI",
        s2,
        f"  Guncel Fiyat  : {price:.4g} {currency}",
        f"  52H Tepe      : {high52:.4g}  ({_pct(price, high52)} zirveye uzaklik)",
        f"  52H Dip       : {low52:.4g}  ({_pct(price, low52)} dipten yukselis)",
        f"  Getiri (1 ay) : {_pct(price, p1m) if p1m else '-'}",
        f"  Getiri (3 ay) : {_pct(price, p3m) if p3m else '-'}",
        f"  Getiri (6 ay) : {_pct(price, p6m) if p6m else '-'}",
        f"  Getiri (1 yil): {_pct(price, p1y) if p1y else '-'}",
        "",
        "TEKNIK INDIKTORLER (Gunluk + Haftalik)",
        s2,
        f"  Trend         : {trend}",
        f"  RSI (14)      : {rsi:.1f}   [{rsi_lbl}]",
        f"  RSI Haftalik  : {f'{weekly_rsi:.1f}' if weekly_rsi else '-'}",
        f"  MACD Hist.    : {macd_h:+.4f}  [{macd_lbl}]",
        f"  ADX           : {adx_val:.1f}   [{adx_lbl}]",
        f"  Bollinger %B  : {bb_pct:.0%}    [{bb_lbl}]",
        f"  EMA 21 / 55   : {ema21:.4g} / {ema55:.4g}",
        f"  EMA 200       : {ema200:.4g}  [{ema200_lbl}]",
        f"  Stoch K/D     : {stoch_k:.0f} / {stoch_d:.0f}",
        f"  CMF           : {cmf:+.2f}  [{cmf_lbl}]",
        f"  Hacim Trendi  : {vol_trend_label}",
        f"  OBV Trendi    : {obv_lbl}",
        "",
        "DESTEK / DIRENC VE FIBONACCI",
        s2,
        f"  Destek        : {sup_str}",
        f"  Direnc        : {res_str}",
        f"  Fib %38.2     : {fib['fib_38']:.4g}",
        f"  Fib %50.0     : {fib['fib_50']:.4g}",
        f"  Fib %61.8     : {fib['fib_61']:.4g}",
        f"  Swing Yuksek  : {fib['swing_high']:.4g}",
        f"  Swing Dusuk   : {fib['swing_low']:.4g}",
        "",
        "ML MODEL (XGBoost)",
        s2,
        f"  Sinyal        : {ml_sinyal}",
        f"  Guven         : {ml_guven}",
    ]

    # Haber + Sentiment bölümü
    lines += ["", "HABERLER VE SENTIMENT", s2]
    if news_items:
        icon_map  = {"bullish": "[+]", "bearish": "[-]", "neutral": "[=]"}
        label_map = {"bullish": "OLUMLU ", "bearish": "OLUMSUZ", "neutral": "NOTR   "}
        for n in news_items[:8]:
            icon  = icon_map.get(n["label"], "[=]")
            lbl   = label_map.get(n["label"], "NOTR   ")
            title = n["title"][:58] + ("..." if len(n["title"]) > 58 else "")
            kws   = ", ".join(n["keywords"]) if n["keywords"] else ""
            lines.append(f"  {icon} {lbl} | {title}")
            if kws:
                lines.append(f"              Anahtar: {kws}")
        overall_lbl = (
            "OLUMLU " if news_sentiment_score > 0.10 else
            "OLUMSUZ" if news_sentiment_score < -0.10 else
            "NOTR   "
        )
        lines.append(f"  Genel Sentiment: {overall_lbl}  (skor: {news_sentiment_score:+.2f})")
    else:
        lines.append("  Haber bulunamadi.")

    lines += [
        "",
        "BOGA SINYALLERI [Al lehine]",
        s2,
        f"  {bull_str}",
        "",
        "AYI SINYALLERI [Sat lehine]",
        s2,
        f"  {bear_str}",
        "",
        S,
        f"  GENEL SKOR: {score:+d} / +{max_score}    Guc: %{score_pct}",
        S,
        f"  KISA VADE (3 ay) : {verdict_short}",
        f"  UZUN VADE (1 yil): {verdict_long}",
        f"  Stop-Loss        : {sl_str}",
        f"  Hedef Fiyat (1y) : {tp_str}",
        S,
    ]
    report = "\n".join(lines)

    return {
        "symbol":        sym,
        "price":         price,
        "trend":         trend,
        "score":         score,
        "score_pct":     score_pct,
        "verdict_short": verdict_short,
        "verdict_long":  verdict_long,
        "report_text":   report,
        "daily_df":      daily,
        "weekly_df":     weekly,
        "supports":      supports,
        "resistances":   resistances,
        "ml_sig":        ml_sig,
        "ml_conf":       ml_conf,
        "_bullets_bull": bullets_bull,
        "_bullets_bear": bullets_bear,
    }


# ── Qwen yorumu ───────────────────────────────────────────────────────────────

def qwen_commentary(result: dict, lora_path: str) -> str:
    """Analiz verilerinden şablon tabanlı Türkçe yatırım yorumu üretir."""
    sym     = result.get("symbol", "")
    score   = result.get("score", 0)
    trend   = result.get("trend", "")
    vs      = result.get("verdict_short", "")
    vl      = result.get("verdict_long", "")
    price   = result.get("price", 0)
    bulls   = result.get("_bullets_bull", [])
    bears   = result.get("_bullets_bear", [])
    ml_sig  = result.get("ml_sig")
    ml_conf = result.get("ml_conf")

    trend_desc = {
        "GÜÇLÜ YUKARI TREND":   "güçlü yükseliş trendinde",
        "YUKARI TREND (zayıf)": "zayıf yükseliş eğiliminde",
        "GÜÇLÜ ASAGI TREND":    "güçlü düşüş baskısı altında",
        "ASAGI TREND (zayıf)":  "kısa vadeli satış baskısı altında",
        "YATAY / KARARSIZ":     "yatay / kararsız seyirde",
    }.get(trend, "belirsiz seyirde")

    en_bull = bulls[0].replace("✅ ", "").split(" (+")[0].lower() if bulls else None
    en_bear = bears[0].replace("❌ ", "").split(" (-")[0].lower() if bears else None

    lines = [f"{sym} şu an {trend_desc}."]

    if en_bull and en_bear:
        lines.append(f"Olumlu tarafta {en_bull} öne çıkarken, olumsuz tarafta {en_bear} baskı oluşturuyor.")
    elif en_bull:
        lines.append(f"Öne çıkan güç: {en_bull}.")
    elif en_bear:
        lines.append(f"Öne çıkan risk: {en_bear}.")

    if ml_sig == 1 and ml_conf and ml_conf >= 0.60:
        lines.append(f"ML modeli {ml_conf:.0%} güvenle AL sinyali veriyor.")
    elif ml_sig == 0 or (ml_conf and ml_conf < 0.50):
        lines.append("ML modeli henüz net sinyal vermiyor, bekle diyor.")

    if score >= 6:
        lines.append(f"Genel tablo güçlü: {vs} — uzun vadede {vl}.")
    elif score >= 3:
        lines.append(f"Genel tablo karma ama olumlu: {vs}. Uzun vadede {vl}.")
    elif score >= 0:
        lines.append(f"Genel tablo nötr: {vs}. Daha net sinyal beklenmeli.")
    elif score >= -3:
        lines.append(f"Genel tablo zayıf: {vs}. Dikkatli olunmalı.")
    else:
        lines.append(f"Genel tablo olumsuz: {vs}. Pozisyon açmaktan kaçınılmalı.")

    return " ".join(lines)


# ── CLI yardımcısı ────────────────────────────────────────────────────────────

def run_analyze(symbol: str, period: str = "2y", lora_path: str | None = None) -> None:
    """Terminale analiz raporunu yaz."""
    print(f"\nAnaliz ediliyor: {symbol} ({period}) ...")
    result = deep_analyze(symbol, period)

    if "error" in result:
        print(f"\n HATA: {result['error']}")
        return

    print("\n" + result["report_text"])

    if lora_path:
        print("\n⏳ Qwen yorum üretiyor...")
        commentary = qwen_commentary(result, lora_path)
        if commentary:
            print("\n" + "━" * 66)
            print("QWEN YORUM")
            print("━" * 66)
            print(textwrap.fill(commentary, width=66))
