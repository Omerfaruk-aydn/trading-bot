"""Tüm piyasalar için sembol listeleri."""

# ── Kripto — Binance Top 200 USDT çiftleri ────────────────────────────────────
CRYPTO_TOP200 = [
    # Top 20 (market cap)
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "SHIBUSDT", "DOTUSDT",
    "LINKUSDT", "TRXUSDT", "MATICUSDT", "LTCUSDT", "UNIUSDT",
    "ATOMUSDT", "ETCUSDT", "XLMUSDT", "BCHUSDT", "APTUSDT",
    # 21-60
    "FILUSDT", "NEARUSDT", "VETUSDT", "ALGOUSDT", "ICPUSDT",
    "FLOWUSDT", "EOSUSDT", "AAVEUSDT", "GRTUSDT", "MKRUSDT",
    "XTZUSDT", "THETAUSDT", "AXSUSDT", "SANDUSDT", "MANAUSDT",
    "HBARUSDT", "EGLDUSDT", "KLAYUSDT", "RUNEUSDT", "ENJUSDT",
    "CAKEUSDT", "CHZUSDT", "KSMUSDT", "WAVESUSDT", "BATUSDT",
    "ZECUSDT", "DASHUSDT", "COMPUSDT", "YFIUSDT", "SNXUSDT",
    "SUSHIUSDT", "1INCHUSDT", "CRVUSDT", "BALUSDT", "RENUSDT",
    "UMAUSDT", "BANDUSDT", "STORJUSDT", "ANKRUSDT", "CELRUSDT",
    # 61-120
    "IOTAUSDT", "ZILUSDT", "ONTUSDT", "ICXUSDT", "QTUMUSDT",
    "ZENUSDT", "SCUSDT", "DGBUSDT", "RVNUSDT", "XVGUSDT",
    "SXPUSDT", "BLZUSDT", "FETUSDT", "OCEANUSDT", "CTSIUSDT",
    "HARDUSDT", "DOCKUSDT", "LRCUSDT", "KNCUSDT", "COTIUSDT",
    "STMXUSDT", "MDTUSDT", "DREPUSDT", "MBLUSDT", "TKOUSDT",
    "PAXGUSDT", "SUIUSDT", "ARBUSDT", "OPUSDT", "INJUSDT",
    "STXUSDT", "CFXUSDT", "LDOUSDT", "RNDRUSDT", "IMXUSDT",
    "WOOUSDT", "GALAUSDT", "APEUSDT", "GALAUSDT", "SPELLUSDT",
    "FOOTBALLUSDT", "HOOKUSDT", "MAGICUSDT", "HIGHUSDT", "MINAUSDT",
    "ROSEUSDT", "ACHUSDT", "COMBOUSDT", "IDUSDT", "EDUUSDT",
    "XVSUSDT", "SSVUSDT", "AMBUSDT", "LEVERAGE", "CYBERUSDT",
    "ARKUSDT", "NFPUSDT", "AIUSDT", "XAIUSDT", "MANTAUSDT",
    # 121-200
    "ALTUSDT", "JUPUSDT", "DYMUSDT", "PYTHUSDT", "JTOUSDT",
    "ACEUSDT", "NOUSDT", "BEAMXUSDT", "PIXELUSDT", "PORTALUSDT",
    "PDAUSDT", "AXLUSDT", "WUSDT", "ENAUSDT", "WLDUSDT",
    "TNSRUSDT", "SAGAUSDT", "TAOUSDT", "REZUSDT", "IOUSDT",
    "ZKUSDT", "LISTAUSDT", "ZROUSDT", "RENDERUSDT", "NOTUSDT",
    "EIGENUSDT", "SCRUSDT", "HMSTRUSDT", "REIUSDT", "COWUSDT",
    "CATIUSDT", "KDAUSDT", "MOVEUSDT", "MEUSDT", "VELODROMEUSDT",
    "VIRTUALUSDT", "SPXUSDT", "PNUTUSDT", "ACTUSDT", "GRASSUSDT",
    "DEGOUSDT", "UXLINKUSDT", "KAIAUSDT", "THEUSDT", "SONICUSDT",
    "PENDLEUSDT", "HYPEUSDT", "FARTCOINUSDT", "TRUMPUSDT", "MELANIAUSDT",
]

# ── BIST — Borsa İstanbul (tüm BİST100 + seçili orta ölçek) ──────────────────
BIST_SYMBOLS = [
    # BIST30
    "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "DOHOL.IS",
    "EKGYO.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS", "GUBRF.IS",
    "HALKB.IS", "ISCTR.IS", "KCHOL.IS", "KOZAA.IS", "KOZAL.IS",
    "KRDMD.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS", "SASA.IS",
    "SISE.IS", "SKBNK.IS", "SOKM.IS", "TAVHL.IS", "TCELL.IS",
    "THYAO.IS", "TKFEN.IS", "TOASO.IS", "TTKOM.IS", "TUPRS.IS",
    "VAKBN.IS", "VESTL.IS", "YKBNK.IS",
    # BIST50 ek
    "AEFES.IS", "AGHOL.IS", "AKSEN.IS", "ALARK.IS", "ALBRK.IS",
    "ANACM.IS", "ASUZU.IS", "BASGZ.IS", "BTCIM.IS", "CIMSA.IS",
    "CLEBI.IS", "DEVA.IS", "EGEEN.IS", "ENKAI.IS", "EREGL.IS",
    "GLYHO.IS", "GOLTS.IS", "HEKTS.IS", "IPEKE.IS", "ISGYO.IS",
    "ISFIN.IS", "LOGO.IS", "MGROS.IS", "NETAS.IS", "ODAS.IS",
    "OTKAR.IS", "OYAKC.IS", "PARSN.IS", "RYSAS.IS", "SELEC.IS",
    "SODA.IS", "TATGD.IS", "TRGYO.IS", "TRILC.IS", "TURSG.IS",
    "ULKER.IS", "USDTR.IS", "VESBE.IS", "YATAS.IS", "ZOREN.IS",
]

# ── ABD — S&P 500 (seçili sektörler, yfinance ile çekilir) ───────────────────
SP500_SYMBOLS = [
    # Teknoloji
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "TSLA",
    "AVGO", "ORCL", "ADBE", "CRM", "AMD", "INTC", "QCOM", "TXN",
    "IBM", "NOW", "INTU", "AMAT", "LRCX", "KLAC", "MRVL", "SNPS",
    # Finans
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "USB",
    "PNC", "TFC", "COF", "SCHW", "CME", "ICE", "SPGI", "MCO",
    # Sağlık
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT",
    "DHR", "BMY", "AMGN", "GILD", "VRTX", "REGN", "ISRG", "SYK",
    # Enerji
    "XOM", "CVX", "COP", "EOG", "SLB", "PSX", "MPC", "VLO",
    "OXY", "KMI", "WMB", "ET", "HAL", "BKR",
    # Tüketim
    "AMZN", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX",
    "BKNG", "MAR", "HLT", "YUM", "CMG", "DPZ",
    # Sanayi
    "GE", "HON", "UPS", "CAT", "DE", "LMT", "RTX", "BA",
    "NOC", "GD", "FDX", "CSX", "UNP", "NSC",
    # Enerji & Kamu
    "NEE", "DUK", "SO", "AEP", "EXC", "SRE", "D", "PCG",
    # İletişim
    "VZ", "T", "TMUS", "NFLX", "DIS", "CMCSA", "WBD",
    # Hammadde
    "LIN", "APD", "NEM", "FCX", "NUE", "STLD", "AA", "X",
]

# ── Avrupa ────────────────────────────────────────────────────────────────────
EUROPE_SYMBOLS = [
    # Almanya DAX
    "SAP.DE", "SIE.DE", "ALV.DE", "MUV2.DE", "BMW.DE", "MBG.DE",
    "BAS.DE", "BAYN.DE", "DB1.DE", "DTE.DE", "RWE.DE", "EON.DE",
    "HEI.DE", "MTX.DE", "VOW3.DE", "ADS.DE", "1COV.DE", "IFX.DE",
    # İngiltere FTSE
    "HSBA.L", "SHEL.L", "BP.L", "GSK.L", "AZN.L", "ULVR.L",
    "DGE.L", "BATS.L", "RIO.L", "AAL.L", "BHP.L", "GLEN.L",
    "LLOY.L", "BARC.L", "NWG.L", "VOD.L", "BT-A.L", "NG.L",
    # Fransa CAC
    "MC.PA", "TTE.PA", "SAN.PA", "AIR.PA", "BNP.PA", "AXA.PA",
    "OR.PA", "RI.PA", "KER.PA", "DG.PA", "ENGI.PA", "VIE.PA",
    # İtalya, İspanya, Hollanda
    "ENI.MI", "ENEL.MI", "ISP.MI", "UCG.MI", "STM.MI",
    "IBE.MC", "ITX.MC", "SAN.MC", "BBVA.MC", "REP.MC",
    "ASML.AS", "PHIA.AS", "INGA.AS", "REN.AS", "HEIA.AS",
]

# ── Asya / Çin ────────────────────────────────────────────────────────────────
ASIA_SYMBOLS = [
    # Hong Kong / Çin (yfinance .HK)
    "0700.HK",  # Tencent
    "9988.HK",  # Alibaba
    "3690.HK",  # Meituan
    "1810.HK",  # Xiaomi
    "9999.HK",  # Netease
    "2318.HK",  # Ping An
    "0941.HK",  # China Mobile
    "1398.HK",  # ICBC
    "3988.HK",  # Bank of China
    "0005.HK",  # HSBC HK
    # Japonya (yfinance .T)
    "7203.T",   # Toyota
    "6758.T",   # Sony
    "9984.T",   # SoftBank
    "7974.T",   # Nintendo
    "8306.T",   # Mitsubishi UFJ
    "6501.T",   # Hitachi
    "9432.T",   # NTT
    "4519.T",   # Chugai Pharma
    # Güney Kore (yfinance .KS)
    "005930.KS",  # Samsung
    "000660.KS",  # SK Hynix
    "005380.KS",  # Hyundai Motor
    "051910.KS",  # LG Chem
    "035420.KS",  # Naver
    # Hindistan (yfinance .NS)
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "BAJFINANCE.NS", "WIPRO.NS",
]

# ── Forex — Major + Minor + Exotic ────────────────────────────────────────────
FOREX_PAIRS = [
    # Majors
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X",
    "AUDUSD=X", "USDCAD=X", "NZDUSD=X",
    # Minors (EUR cross)
    "EURGBP=X", "EURJPY=X", "EURCHF=X", "EURAUD=X", "EURCAD=X",
    # GBP cross
    "GBPJPY=X", "GBPCHF=X", "GBPAUD=X", "GBPCAD=X",
    # Emtia para birimleri
    "AUDJPY=X", "AUDCAD=X", "CADJPY=X", "NZDJPY=X",
    # Türk Lirası
    "USDTRY=X", "EURTRY=X", "GBPTRY=X",
    # Diğer EM
    "USDBRL=X", "USDMXN=X", "USDZAR=X", "USDINR=X",
    "USDCNH=X", "USDRUB=X", "USDKRW=X",
]

# ── Emtialar ───────────────────────────────────────────────────────────────────
COMMODITY_SYMBOLS = [
    "GC=F",   # Altın
    "SI=F",   # Gümüş
    "PL=F",   # Platin
    "PA=F",   # Paladyum
    "CL=F",   # Ham petrol (WTI)
    "BZ=F",   # Brent petrol
    "NG=F",   # Doğal gaz
    "HG=F",   # Bakır
    "ZW=F",   # Buğday
    "ZC=F",   # Mısır
    "ZS=F",   # Soya fasulyesi
    "KC=F",   # Kahve
    "CT=F",   # Pamuk
    "SB=F",   # Şeker
]

# ── Endeksler ─────────────────────────────────────────────────────────────────
INDEX_SYMBOLS = [
    "^GSPC",   # S&P 500
    "^DJI",    # Dow Jones
    "^IXIC",   # NASDAQ
    "^RUT",    # Russell 2000
    "^VIX",    # VIX (korku endeksi)
    "^FTSE",   # FTSE 100
    "^GDAXI",  # DAX
    "^FCHI",   # CAC 40
    "^N225",   # Nikkei 225
    "^HSI",    # Hang Seng
    "000001.SS", # Shanghai Composite
    "^BSESN",  # BSE Sensex
    "XU100.IS", # BIST 100
    "DX-Y.NYB", # DXY (Dolar endeksi)
    "^TNX",    # 10 yıllık ABD tahvil faizi
    "^TYX",    # 30 yıllık ABD tahvil faizi
    "^IRX",    # 3 aylık ABD hazine bonosu
]

# ── Makro Veri — FRED Serileri (genişletilmiş) ────────────────────────────────
FRED_SERIES = {
    # ── ABD Para Politikası ──
    "FEDFUNDS":     "FED Faiz Oranı",
    "DFEDTARU":     "FED Üst Hedef Faiz",
    "DFEDTARL":     "FED Alt Hedef Faiz",
    "IORB":         "FED Rezerv Faizi",
    "SOFR":         "SOFR (gecelik repo)",
    "DFF":          "Efektif FED Funds (günlük)",
    "WRESBAL":      "FED Rezerv Bakiyesi",
    "WALCL":        "FED Bilanço Büyüklüğü",
    "M1SL":         "ABD M1 Para Arzı",
    "M2SL":         "ABD M2 Para Arzı",

    # ── ABD Enflasyon ──
    "CPIAUCSL":     "ABD TÜFE (genel)",
    "CPILFESL":     "ABD Çekirdek TÜFE",
    "PCEPI":        "PCE Enflasyon",
    "PCEPILFE":     "Çekirdek PCE",
    "PPIFIS":       "Üretici Fiyat Endeksi",
    "MICH":         "Michigan Enflasyon Beklentisi",

    # ── ABD Büyüme & Aktivite ──
    "GDP":          "ABD GSYİH (çeyreklik)",
    "GDPC1":        "ABD Reel GSYİH",
    "INDPRO":       "ABD Sanayi Üretimi",
    "CAPUTLB50001": "ABD Kapasite Kullanımı",
    "RSXFS":        "ABD Perakende Satışlar",
    "DSPIC96":      "Reel Kişisel Harcanabilir Gelir",
    "PCE":          "Kişisel Tüketim Harcamaları",
    "BOPTEXP":      "ABD İhracat",
    "BOPTIMP":      "ABD İthalat",
    "NETEXP":       "ABD Net İhracat",

    # ── ABD İşgücü ──
    "UNRATE":       "ABD İşsizlik Oranı",
    "U6RATE":       "ABD Geniş İşsizlik (U6)",
    "PAYEMS":       "Tarım Dışı İstihdam",
    "MANEMP":       "İmalat İstihdamı",
    "CIVPART":      "İşgücüne Katılım Oranı",
    "ICSA":         "İşsizlik Başvuruları (haftalık)",
    "CES0500000003":"Ortalama Saatlik Kazanç",

    # ── ABD Konut & İnşaat ──
    "HOUST":        "Konut Başlangıçları",
    "PERMIT":       "İnşaat İzinleri",
    "EXHOSLUSM495S":"Mevcut Konut Satışları",
    "MSPUS":        "Medyan Konut Satış Fiyatı",
    "MORTGAGE30US": "30Y Mortgage Faizi",

    # ── ABD Tahvil & Getiri Eğrisi ──
    "DGS1MO":       "1 Aylık Hazine",
    "DGS3MO":       "3 Aylık Hazine",
    "DGS6MO":       "6 Aylık Hazine",
    "DGS1":         "1 Yıllık Hazine",
    "DGS2":         "2 Yıllık Hazine",
    "DGS5":         "5 Yıllık Hazine",
    "DGS10":        "10 Yıllık Hazine",
    "DGS30":        "30 Yıllık Hazine",
    "T10Y2Y":       "Getiri Eğrisi (10Y-2Y)",
    "T10Y3M":       "Getiri Eğrisi (10Y-3M)",
    "BAMLH0A0HYM2": "High Yield Kredi Spread",
    "BAMLC0A0CM":   "Investment Grade Spread",
    "TEDRATE":      "TED Spread",

    # ── ABD Hisse & Risk ──
    "DTWEXBGS":     "Dolar Endeksi (geniş)",
    "VIXCLS":       "VIX Korku Endeksi",
    "NIKKEI225":    "Nikkei 225",
    "SP500":        "S&P 500",
    "NASDAQCOM":    "NASDAQ Composite",

    # ── ABD Öncü Göstergeler ──
    "USSLIND":      "ABD Öncü Göstergeler",
    "UMCSENT":      "Michigan Tüketici Güveni",
    "CSCICP03USM665S": "Tüketici Güven Endeksi (OECD)",
    "BSCICP03USM665S": "İş Güven Endeksi (OECD)",
    "PMSAVE":       "Kişisel Tasarruf Oranı",

    # ── Emtia ──
    "DCOILWTICO":   "WTI Ham Petrol",
    "DCOILBRENTEU": "Brent Ham Petrol",
    "DHHNGSP":      "Henry Hub Doğal Gaz",
    "GOLDAMGBD228NLBM": "Londra Altın Fix",
    "PCOPPUSDM":    "Bakır Fiyatı",
    "PWHEAMTUSDM":  "Buğday Fiyatı",
    "PMAIZMTUSDM":  "Mısır Fiyatı",

    # ── Küresel Merkez Bankaları ──
    "ECBDFR":           "ECB Mevduat Faizi",
    "IRLTLT01EZM156N":  "Euro Bölgesi 10Y Tahvil",
    "IRSTCB01JPM156N":  "Japonya Merkez Bankası Faizi",
    "IRSTCB01GBM156N":  "İngiltere Merkez Bankası Faizi",
    "IRSTCB01CNM156N":  "Çin Merkez Bankası Faizi",

    # ── Küresel Büyüme ──
    "CLVMNACSCAB1GQEA19": "Euro Bölgesi GSYİH",
    "JPNRGDPEXP":    "Japonya Reel GSYİH",
    "CHNRGDPNQDSMEI": "Çin Reel GSYİH",
    "CPIEAMU01EZM659N": "Euro Bölgesi TÜFE",
    "LRHUTTTTEZM156S":  "Euro Bölgesi İşsizlik",

    # ── Küresel Ticaret & Likidite ──
    "BOGZ1FL073161113Q": "Para Piyasası Fonu Varlıkları",
    "TOTRESNS":     "Toplam Banka Rezervleri",
    "DPSACBW027SBOG":"Banka Mevduatları",
    "LOANS":        "Banka Kredileri",
    "BUSLOANS":     "Ticari & Sanayi Kredileri",
}

# ── Zaman Dilimleri (öncelik sırası) ─────────────────────────────────────────
TIMEFRAMES_CRYPTO = ["1d", "4h", "1h", "15m", "5m", "1m"]
TIMEFRAMES_STOCK  = ["1d", "1wk", "1mo"]
TIMEFRAMES_FOREX  = ["1d", "4h", "1h"]

# ── Tarihsel Veri Aralıkları ─────────────────────────────────────────────────
HISTORY_YEARS = {
    "crypto": 5,
    "stock": 10,
    "forex": 7,
    "commodity": 10,
    "index": 10,
    "macro": 20,
}
